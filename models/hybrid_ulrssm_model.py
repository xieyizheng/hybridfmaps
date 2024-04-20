import torch
import torch.nn.functional as F

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, trim_basis, hybrid_fmap2pointmap
from utils.geometry_util import get_all_operators

def cache_operators(data, cache_dir=None):
    data_x, data_y = data['first'], data['second']
    if 'operators' not in data_x.keys():
        cache_dir = cache_dir or data_x.get('cache_dir', None)
        _, mass, L, evals, evecs, gradX, gradY = get_all_operators(data_x['verts'].cpu(), data_x['faces'].cpu(), k=128,
                                                                    cache_dir=cache_dir)
        
        data_x['operators'] = {'mass': mass, 'L': L, 'evals': evals, 'evecs': evecs, 'gradX': gradX, 'gradY': gradY}
    if 'operators' not in data_y.keys():
        cache_dir = cache_dir or data_y.get('cache_dir', None)
        _, mass, L, evals, evecs, gradX, gradY = get_all_operators(data_y['verts'].cpu(), data_y['faces'].cpu(), k=128,
                                                                    cache_dir=cache_dir)
        data_y['operators'] = {'mass': mass, 'L': L, 'evals': evals, 'evecs': evecs, 'gradX': gradX, 'gradY': gradY}

@MODEL_REGISTRY.register()
class Hybrid_ULRSSM_Model(BaseModel):
    def __init__(self, opt):
        self.with_refine = opt.get('refine', -1)
        self.partial = opt.get('partial', False)
        self.non_isometric = opt.get('non-isometric', False)
        if self.with_refine > 0:
            opt['is_train'] = True
        super(Hybrid_ULRSSM_Model, self).__init__(opt)

    def feed_data(self, data):
        cache_dir = self.opt['networks']['feature_extractor'].get('cache_dir', None)
        cache_operators(data, cache_dir=cache_dir)
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # feature extractor for mesh
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'], data=data_x)  # [B, Nx, C]
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'], data=data_y)  # [B, Ny, C]

        # trim basis
        n_lb = self.opt.get('n_lb', 140)
        n_elas = self.opt.get('n_elas', 60)
        data_x = trim_basis(data_x, n_lb, n_elas)
        data_y = trim_basis(data_y, n_lb, n_elas)

        Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)

        # ------------------- LB Loss ------------------- #
        if n_lb > 0:
            # get spectral operators
            evals_x = data_x['evals']
            evals_y = data_y['evals']
            evecs_x = data_x['evecs']
            evecs_y = data_y['evecs']
            evecs_trans_x = data_x['evecs_trans']  # [B, K, Nx]
            evecs_trans_y = data_y['evecs_trans']  # [B, K, Ny]

            Cxy, Cyx = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

            self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, evals_x, evals_y)

            # compute C
            Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))

            self.loss_metrics['l_align'] = self.losses['align_loss'](Cxy, Cxy_est)
            if not self.partial:
                Cyx_est = torch.bmm(evecs_trans_x, torch.bmm(Pxy, evecs_y))
                self.loss_metrics['l_align'] += self.losses['align_loss'](Cyx, Cyx_est)

        # ------------------- Elas Loss ------------------- #
        if n_elas > 0:
            # get elas spectral operators
            elas_evals_x = torch.abs(data_x['elas_evals'])
            elas_evals_y = torch.abs(data_y['elas_evals'])
            elas_evecs_x = data_x['elas_evecs']
            elas_evecs_y = data_y['elas_evecs']
            elas_evecs_trans_x = data_x['elas_evecs_trans']  # [B, K, Nx]
            elas_evecs_trans_y = data_y['elas_evecs_trans']  # [B, K, Ny]
            elas_mass_x = data_x['elas_mass']
            elas_mass_y = data_y['elas_mass']
            elas_Mk_x = data_x['elas_Mk']
            elas_Mk_y = data_y['elas_Mk']
            elas_invsqrtMk_x = data_x['elas_invsqrtMk']
            elas_invsqrtMk_y = data_y['elas_invsqrtMk']
            elas_sqrtMk_x = data_x['elas_sqrtMk']
            elas_sqrtMk_y = data_y['elas_sqrtMk']
            
            elas_Cxy, elas_Cyx = self.networks['expanded_fmap_net'](feat_x, feat_y, elas_evals_x, elas_evals_y, elas_evecs_trans_x, elas_evecs_trans_y, elas_sqrtMk_x, elas_sqrtMk_y)

            hs_surfmnet_loss = self.losses['hs_surfmnet_loss'](elas_Cxy, elas_Cyx, elas_evals_x, elas_evals_y, elas_Mk_x, elas_Mk_y)
            self.loss_metrics['l_elas_bij'] = hs_surfmnet_loss['l_bij']
            self.loss_metrics['l_elas_orth'] = hs_surfmnet_loss['l_orth']

            # compute C
            elas_Cxy_est = torch.bmm(elas_evecs_trans_y, torch.bmm(Pyx, elas_evecs_x))

            self.loss_metrics['l_elas_align'] = self.losses['hs_align_loss'](elas_Cxy, elas_Cxy_est, elas_sqrtMk_x, elas_sqrtMk_y)
            if not self.partial:
                elas_Cyx_est = torch.bmm(elas_evecs_trans_x, torch.bmm(Pxy, elas_evecs_y))
                self.loss_metrics['l_elas_align'] += self.losses['hs_align_loss'](elas_Cyx, elas_Cyx_est, elas_sqrtMk_y, elas_sqrtMk_x)
        
        if 'dirichlet_loss' in self.losses:
            Lx, Ly = data_x['operators']['L'], data_y['operators']['L']
            verts_x, verts_y = data_x['verts'], data_y['verts']
            self.loss_metrics['l_d'] = self.losses['dirichlet_loss'](torch.bmm(Pxy, verts_y), Lx) + \
                                       self.losses['dirichlet_loss'](torch.bmm(Pyx, verts_x), Ly)

    def validate_single(self, data, timer):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # get previous network state dict
        if self.with_refine > 0:
            state_dict = {'networks': self._get_networks_state_dict()}

        # start record
        timer.start()

        # test-time refinement
        if self.with_refine > 0:
            self.refine(data)
        
        # trim basis
        n_lb = self.opt.get('n_lb', 140)
        n_elas = self.opt.get('n_elas', 60)
        data_x = trim_basis(data_x, n_lb, n_elas)
        data_y = trim_basis(data_y, n_lb, n_elas)

        # feature extractor
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x.get('faces'))
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y.get('faces'))

        # get spectral operators
        evecs_x = data_x['evecs'].squeeze()
        evecs_y = data_y['evecs'].squeeze()
        evecs_trans_x = data_x['evecs_trans'].squeeze()
        evecs_trans_y = data_y['evecs_trans'].squeeze()
        elas_evecs_x = data_x['elas_evecs'].squeeze()
        elas_evecs_y = data_y['elas_evecs'].squeeze()
        elas_trans_x = data_x['elas_evecs_trans'].squeeze()
        elas_trans_y = data_y['elas_evecs_trans'].squeeze()
        elas_mass_x = data_x['elas_mass'].squeeze()
        elas_mass_y = data_y['elas_mass'].squeeze()

        if self.non_isometric:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)

            # nearest neighbour query
            p2p = nn_query(feat_x, feat_y).squeeze()

            # compute Pyx from functional map
            Cxy = evecs_trans_y @ evecs_x[p2p]
            Pyx = evecs_y @ Cxy @ evecs_trans_x
        else:
            # compute Pxy
            Pyx = self.compute_permutation_matrix(feat_y, feat_x, bidirectional=False).squeeze()
            Cxy = evecs_trans_y @ (Pyx @ evecs_x)
            elas_Cxy = elas_trans_y @ (Pyx @ elas_evecs_x)

            # convert functional map to point-to-point map
            p2p = hybrid_fmap2pointmap(Cxy, evecs_x, evecs_y, elas_Cxy, elas_evecs_x, elas_evecs_y, elas_mass_x, elas_mass_y)

            # compute Pyx from functional map
            Pyx = evecs_y @ Cxy @ evecs_trans_x

        # finish record
        timer.record()

        # resume previous network state dict
        if self.with_refine > 0:
            self.resume_model(state_dict, net_only=True, verbose=False)
        return p2p, Pyx, Cxy

    def compute_permutation_matrix(self, feat_x, feat_y, bidirectional=False, normalize=True):
        if normalize:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)
        similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

        # sinkhorn normalization
        Pxy = self.networks['permutation'](similarity)

        if bidirectional:
            Pyx = self.networks['permutation'](similarity.transpose(1, 2))
            return Pxy, Pyx
        else:
            return Pxy

    def refine(self, data):
        self.networks['permutation'].hard = False
        self.networks['fmap_net'].bidirectional = True

        with torch.set_grad_enabled(True):
            for _ in range(self.with_refine):
                self.curr_iter += 1
                self.feed_data(data)
                self.optimize_parameters()
        self.curr_iter = 0

        self.networks['permutation'].hard = True
        self.networks['fmap_net'].bidirectional = False

    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        # change permutation prediction status
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = True
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = False
        super(Hybrid_ULRSSM_Model, self).validation(dataloader, tb_logger, update)
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = False
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = True
    
    def optimize_parameters(self):
        """Override for Hybrid_ULRSSM_Model"""
        n_lb = self.opt.get('n_lb', 140)
        n_elas = self.opt.get('n_elas', 60)

        # Loss normalization and weight scheduling
        if n_lb > 0 and n_elas > 0:
            # Normalize LB and Elas Loss; 
            w_lb = 20000 / (n_lb * n_lb)
            w_elas = 20000 / (n_elas * n_elas)

            # early schduler for Elas Loss
            weight_schedule = self.opt['train'].get('weight_schedule', False)
            if weight_schedule:
                def linear_anneal(current_iter, total_iters):
                    return min(max(current_iter / total_iters, 0.0), 1.0)
                curr_iter = self.curr_iter
                anneal_weight = linear_anneal(curr_iter, weight_schedule)

                w_elas = w_elas * anneal_weight

            # applying loss weights
            self.loss_metrics['l_bij'] *= w_lb
            self.loss_metrics['l_orth'] *= w_lb
            self.loss_metrics['l_align'] *= w_lb

            self.loss_metrics['l_elas_bij'] *= w_elas
            self.loss_metrics['l_elas_orth'] *= w_elas
            self.loss_metrics['l_elas_align'] *= w_elas

        # compute total loss
        loss = 0.0
        for k, v in self.loss_metrics.items():
            if k != 'l_total':
                loss += v

        # update loss metrics
        self.loss_metrics['l_total'] = loss

        # zero grad
        for name in self.optimizers:
            self.optimizers[name].zero_grad()

        # backward pass
        loss.backward()

        # clip gradient for stability
        for key in self.networks:
            torch.nn.utils.clip_grad_norm_(self.networks[key].parameters(), 1.0)

        # update weight
        for name in self.optimizers:
            self.optimizers[name].step()
