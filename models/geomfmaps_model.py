import torch

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import fmap2pointmap


@MODEL_REGISTRY.register()
class GeomFmaps_Model(BaseModel):
    def __init__(self, opt):
        super(GeomFmaps_Model, self).__init__(opt)

    def feed_data(self, data):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # feature extractor for mesh
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'])  # [B, Nx, C]
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'])  # [B, Ny, C]

        # get spectral operators
        evals_x = data_x['evals']
        evals_y = data_y['evals']
        evecs_x = data_x['evecs'].squeeze()
        evecs_y = data_y['evecs'].squeeze()
        evecs_trans_x = data_x['evecs_trans']  # [B, K, Nx]
        evecs_trans_y = data_y['evecs_trans']  # [B, K, Ny]

        Cxy, _ = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
        
        # get gt correspondence
        corr_x = data_x['corr'][0]
        corr_y = data_y['corr'][0]

        evecs_x_a = evecs_x[corr_x, :]
        evecs_y_a = evecs_y[corr_y, :]
        # get C_gt
        C_gt = torch.linalg.lstsq(evecs_x_a, evecs_y_a).solution.t()
        C_gt = C_gt.unsqueeze(0)

        # calculate loss
        self.loss_metrics = {}
        self.loss_metrics['l_gt_fmap'] = self.losses['gt_loss'](Cxy, C_gt)

    def validate_single(self, data, timer):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # start record
        timer.start()

        # feature extractor
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x.get('faces'))
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y.get('faces'))
        
        # get spectral operators
        evals_x = data_x['evals']
        evals_y = data_y['evals']
        evecs_x = data_x['evecs'].squeeze()
        evecs_y = data_y['evecs'].squeeze()
        evecs_trans_x = data_x['evecs_trans']
        evecs_trans_y = data_y['evecs_trans']

        Cxy, _ = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
        Cxy = Cxy.squeeze()

        # convert functional map to point-to-point map
        p2p = fmap2pointmap(Cxy, evecs_x, evecs_y)

        # compute Pyx from functional map
        Pyx = evecs_y @ Cxy @ evecs_trans_x

        # finish record
        timer.record()

        return p2p, Pyx, Cxy


    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        super(GeomFmaps_Model, self).validation(dataloader, tb_logger, update)
