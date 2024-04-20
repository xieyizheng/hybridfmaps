import torch

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import trim_basis, hybrid_fmap2pointmap,zoomout, elas_zoomout, hybrid_zoomout


@MODEL_REGISTRY.register()
class Hybrid_GeomFmaps_Model(BaseModel):
    def __init__(self, opt):
        super(Hybrid_GeomFmaps_Model, self).__init__(opt)

    def feed_data(self, data):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # feature extractor for mesh
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'])  # [B, Nx, C]
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'])  # [B, Ny, C]

        # trim basis
        n_lb = self.opt.get('n_lb', 20)
        n_elas = self.opt.get('n_elas', 10)
        data_x = trim_basis(data_x, n_lb, n_elas)
        data_y = trim_basis(data_y, n_lb, n_elas)

        # get gt correspondence
        corr_x = data_x['corr'][0]
        corr_y = data_y['corr'][0]

        #-------------------------------LB part--------------------------------
        # get spectral operators
        evals_x = data_x['evals']
        evals_y = data_y['evals']
        evecs_x = data_x['evecs'].squeeze()
        evecs_y = data_y['evecs'].squeeze()
        evecs_trans_x = data_x['evecs_trans']  # [B, K, Nx]
        evecs_trans_y = data_y['evecs_trans']  # [B, K, Ny]

        Cxy, _ = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
        
        evecs_x_a = evecs_x[corr_x, :]
        evecs_y_a = evecs_y[corr_y, :]
        # get C_gt
        C_gt = torch.linalg.lstsq(evecs_x_a, evecs_y_a).solution.t()
        C_gt = C_gt.unsqueeze(0)

        # calculate loss
        self.loss_metrics = {}
        self.loss_metrics['l_gt_fmap'] = self.losses['gt_loss'](Cxy, C_gt) 

        #-------------------------------Elastic part--------------------------------
        # get elas spectral operators
        elas_evals_x = data_x['elas_evals']
        elas_evals_y = data_y['elas_evals']
        elas_evecs_x = data_x['elas_evecs'].squeeze()
        elas_evecs_y = data_y['elas_evecs'].squeeze()
        elas_mass_x = data_x['elas_mass'].squeeze()
        elas_mass_y = data_y['elas_mass'].squeeze()
        elas_evecs_trans_x = data_x['elas_evecs_trans']
        elas_evecs_trans_y = data_y['elas_evecs_trans']

        elas_sqrtMk_x = data_x['elas_sqrtMk']
        elas_sqrtMk_y = data_y['elas_sqrtMk']

        elas_M1k = data_x['elas_Mk'].squeeze()
        elas_M2k = data_y['elas_Mk'].squeeze()

        # 
        elas_Cxy, _ = self.networks['expanded_fmap_net'](feat_x, feat_y, elas_evals_x, elas_evals_y, elas_evecs_trans_x, elas_evecs_trans_y, elas_sqrtMk_x, elas_sqrtMk_y)

        elas_evecs_x_a = elas_evecs_x[corr_x, :]
        elas_evecs_y_a = elas_evecs_y[corr_y, :]
        sqrt_elas_mass_x_a = torch.sqrt(elas_mass_x[corr_x])
        elas_evecs_x_a = torch.diag(sqrt_elas_mass_x_a) @ elas_evecs_x_a
        elas_evecs_y_a = torch.diag(sqrt_elas_mass_x_a) @ elas_evecs_y_a
        # get elas C_gt
        elas_C_gt = torch.inverse(elas_M1k) @ torch.linalg.lstsq(elas_evecs_x_a, elas_evecs_y_a).solution.t() @ elas_M2k
        elas_C_gt = elas_C_gt.unsqueeze(0)

        self.loss_metrics['l_elas_gt_fmap'] = self.losses['elas_gt_loss'](elas_Cxy, elas_C_gt, elas_sqrtMk_x, elas_sqrtMk_y) 


    def validate_single(self, data, timer):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # start record
        timer.start()

        # feature extractor
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x.get('faces'))
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y.get('faces'))
        
        # trim basis
        n_lb = self.opt.get('n_lb', 20)
        n_elas = self.opt.get('n_elas', 10)
        data_x = trim_basis(data_x, n_lb, n_elas)
        data_y = trim_basis(data_y, n_lb, n_elas)

        # get spectral operators
        evals_x = data_x['evals']
        evals_y = data_y['evals']
        evecs_x = data_x['evecs'].squeeze()
        evecs_y = data_y['evecs'].squeeze()
        evecs_trans_x = data_x['evecs_trans']
        evecs_trans_y = data_y['evecs_trans']

        # get elas spectral operators
        elas_evals_x = data_x['elas_evals']
        elas_evals_y = data_y['elas_evals']
        elas_evecs_x = data_x['elas_evecs'].squeeze()
        elas_evecs_y = data_y['elas_evecs'].squeeze()
        elas_mass_x = data_x['elas_mass'].squeeze()
        elas_mass_y = data_y['elas_mass'].squeeze()
        elas_evecs_trans_x = data_x['elas_evecs_trans']
        elas_evecs_trans_y = data_y['elas_evecs_trans']

        elas_sqrtMk_x = data_x['elas_sqrtMk']
        elas_sqrtMk_y = data_y['elas_sqrtMk']

        #
        Cxy, _ = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
        Cxy = Cxy.squeeze()

        #
        elas_Cxy, _ = self.networks['expanded_fmap_net'](feat_x, feat_y, elas_evals_x, elas_evals_y, elas_evecs_trans_x, elas_evecs_trans_y, elas_sqrtMk_x, elas_sqrtMk_y)
        elas_Cxy = elas_Cxy.squeeze()

        # convert functional map to point-to-point map
        p2p = hybrid_fmap2pointmap(Cxy, evecs_x, evecs_y, elas_Cxy, elas_evecs_x, elas_evecs_y, elas_mass_x, elas_mass_y)

        # compute Pyx from functional map
        Pyx = evecs_y @ Cxy @ evecs_trans_x

        #-------------------------------Optional ZoomOut--------------------------------
        # perform zoomout if specified
        opt_zoomout = self.opt.get('zoomout', None)
        if opt_zoomout is not None:
            # reload so that we have full basis
            data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)
            evecs_x = data_x['evecs'].squeeze()
            evecs_y = data_y['evecs'].squeeze()
            elas_evecs_x = data_x['elas_evecs'].squeeze()
            elas_evecs_y = data_y['elas_evecs'].squeeze()
        if opt_zoomout == "lb":
            p2p = zoomout(p2p, evecs_x, evecs_y)
        if opt_zoomout == "elas":
            p2p = elas_zoomout(p2p, elas_evecs_x, elas_evecs_y, elas_mass_x, elas_mass_y)
        if opt_zoomout == "hybrid":
            p2p = hybrid_zoomout(p2p, evecs_x, evecs_y, elas_evecs_x, elas_evecs_y, elas_mass_x, elas_mass_y)

        # finish record
        timer.record()

        return p2p, Pyx, Cxy


    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        super(Hybrid_GeomFmaps_Model, self).validation(dataloader, tb_logger, update)
