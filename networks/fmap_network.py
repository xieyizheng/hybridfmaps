import torch
import torch.nn as nn

from utils.registry import NETWORK_REGISTRY


def _get_mask(evals1, evals2, resolvant_gamma):
    scaling_factor = max(torch.max(evals1), torch.max(evals2))
    evals1, evals2 = evals1 / scaling_factor, evals2 / scaling_factor
    evals_gamma1 = (evals1 ** resolvant_gamma)[None, :]
    evals_gamma2 = (evals2 ** resolvant_gamma)[:, None]

    M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
    M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
    return M_re.square() + M_im.square()


def get_mask(evals1, evals2, resolvant_gamma):
    masks = []
    for bs in range(evals1.shape[0]):
        masks.append(_get_mask(evals1[bs], evals2[bs], resolvant_gamma))
    return torch.stack(masks, dim=0)


@NETWORK_REGISTRY.register()
class RegularizedFMNet(nn.Module):
    """Compute the functional map matrix representation in DPFM"""
    def __init__(self, lmbda=100, resolvant_gamma=0.5, bidirectional=False):
        super(RegularizedFMNet, self).__init__()
        self.lmbda = lmbda
        self.resolvant_gamma = resolvant_gamma
        self.bidirectional = bidirectional

    def compute_functional_map(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        A = torch.bmm(evecs_trans_x, feat_x)  # [B, K, C]
        B = torch.bmm(evecs_trans_y, feat_y)  # [B, K, C]

        D = get_mask(evals_x, evals_y, self.resolvant_gamma)  # [B, K, K]

        A_t = A.transpose(1, 2)  # [B, C, K]
        A_A_t = torch.bmm(A, A_t)  # [B, K, K]
        B_A_t = torch.bmm(B, A_t)  # [B, K, K]

        C_i = []
        for i in range(evals_x.shape[1]):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.shape[0])], dim=0)
            C = torch.bmm(torch.inverse(A_A_t + self.lmbda * D_i), B_A_t[:, [i], :].transpose(1, 2))
            C_i.append(C.transpose(1, 2))

        Cxy = torch.cat(C_i, dim=1)
        return Cxy

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        """
        Forward pass to compute functional map
        Args:
            feat_x (torch.Tensor): feature vector of shape x. [B, Vx, C].
            feat_y (torch.Tensor): feature vector of shape y. [B, Vy, C].
            evals_x (torch.Tensor): eigenvalues of shape x. [B, K].
            evals_y (torch.Tensor): eigenvalues of shape y. [B, K].
            evecs_trans_x (torch.Tensor): pseudo inverse of eigenvectors of shape x. [B, K, Vx].
            evecs_trans_y (torch.Tensor): pseudo inverse of eigenvectors of shape y. [B, K, Vy].

        Returns:
            C (torch.Tensor): functional map from shape x to shape y. [B, K, K].
        """
        Cxy = self.compute_functional_map(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        if self.bidirectional:
            Cyx = self.compute_functional_map(feat_y, feat_x, evals_y, evals_x, evecs_trans_y, evecs_trans_x)
        else:
            Cyx = None

        return Cxy, Cyx

@NETWORK_REGISTRY.register()
class StandardFMNet(nn.Module):
    """Compute the functional map matrix representation in GeomFmaps"""
    def __init__(self, lambda_param=1e-3, bidirectional=False):
        super(StandardFMNet, self).__init__()
        self.lambda_param = lambda_param
        self.bidirectional = bidirectional

    def compute_functional_map(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        F_hat = torch.bmm(evecs_trans_x, feat_x)
        G_hat = torch.bmm(evecs_trans_y, feat_y)
        A, B = F_hat, G_hat
        lambda_param = self.lambda_param

        D = torch.repeat_interleave(evals_x.unsqueeze(1), repeats=evals_x.size(1), dim=1)
        D = (D - torch.repeat_interleave(evals_y.unsqueeze(2), repeats=evals_x.size(1), dim=2)) ** 2

        A_t = A.transpose(1, 2)
        A_A_t = torch.bmm(A, A_t)
        B_A_t = torch.bmm(B, A_t)

        C_i = []
        for i in range(evals_x.size(1)):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
            C = torch.bmm(torch.inverse(A_A_t + lambda_param * D_i), B_A_t[:, i, :].unsqueeze(1).transpose(1, 2))
            C_i.append(C.transpose(1, 2))
        C = torch.cat(C_i, dim=1)

        return C

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        """
        Forward pass to compute functional map
        Args:
            feat_x (torch.Tensor): feature vector of shape x. [B, Vx, C].
            feat_y (torch.Tensor): feature vector of shape y. [B, Vy, C].
            evals_x (torch.Tensor): eigenvalues of shape x. [B, K].
            evals_y (torch.Tensor): eigenvalues of shape y. [B, K].
            evecs_trans_x (torch.Tensor): pseudo inverse of eigenvectors of shape x. [B, K, Vx].
            evecs_trans_y (torch.Tensor): pseudo inverse of eigenvectors of shape y. [B, K, Vy].

        Returns:
            C (torch.Tensor): functional map from shape x to shape y. [B, K, K].
        """
        Cxy = self.compute_functional_map(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        if self.bidirectional:
            Cyx = self.compute_functional_map(feat_y, feat_x, evals_y, evals_x, evecs_trans_y, evecs_trans_x)
        else:
            Cyx = None

        return Cxy, Cyx


@NETWORK_REGISTRY.register()
class ExpandedResolventFMNet(nn.Module):
    """Compute the functional map matrix representation using Resolvent Regularization and adapted to Hilbert-Schmidt Norm """
    def __init__(self, lmbda=100, resolvant_gamma=0.5, bidirectional=False, adapt_feature=True, adapt_mask=True, dev_feature=False):
        super(ExpandedResolventFMNet, self).__init__()
        self.lmbda = lmbda
        self.resolvant_gamma = resolvant_gamma
        self.bidirectional = bidirectional

        # flag for whether adapt mass to feature term
        self.adapt_feature = adapt_feature
        # flag for whether adapt mass to regularizer term
        self.adapt_mask = adapt_mask

        self.dev_feature = dev_feature

    def compute_functional_map(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, sqrtMk_x, sqrtMk_y):
        D = get_mask(evals_x, evals_y, self.resolvant_gamma)  # [B, K, K]
        # inputs are all batched
        D = D.squeeze()
        feat_x = feat_x.squeeze()
        feat_y = feat_y.squeeze()
        sqrtMk_x = sqrtMk_x.squeeze()
        sqrtMk_y = sqrtMk_y.squeeze()
        evecs_trans_x = evecs_trans_x.squeeze()
        evecs_trans_y = evecs_trans_y.squeeze()
        # inner processing are all without batch
        evals_x = evals_x.squeeze()
        evals_y = evals_y.squeeze()


        # ---------first term: features---------
        A = evecs_trans_x @ feat_x
        B = evecs_trans_y @ feat_y

        if self.adapt_feature:
            B = sqrtMk_y @ B
            if self.dev_feature:
                B = torch.inverse(sqrtMk_x) @ B
        # A and B should be same shape
        k, m = A.size(0), A.size(1)

        vec_B = B.T.reshape(m * k, 1).contiguous()

        A_t = A.T.contiguous()
        if self.adapt_feature:
            Ik = sqrtMk_y
            if self.dev_feature:
                Ik = torch.inverse(sqrtMk_x) @ Ik
        else:
            Ik = torch.eye(k, device=A.device, dtype=torch.float32)
        
        At_Ik = torch.kron(A_t, Ik)

        first = At_Ik.T @ At_Ik


        # ---------second term: regularizer(mask)---------
        if self.adapt_mask:
            resolvant_gamma = self.resolvant_gamma
            inv_sqrtMk_x = torch.inverse(sqrtMk_x)

            scaling_factor = max(torch.max(evals_x), torch.max(evals_y))
            evals_gamma1 = (evals_x / scaling_factor) ** resolvant_gamma
            evals_gamma2 = (evals_y / scaling_factor) ** resolvant_gamma

            rn1_re = evals_gamma1 / (evals_gamma1.square() + 1)
            rn1_im = 1 / (evals_gamma1.square() + 1)

            rn2_re = evals_gamma2 / (evals_gamma2.square() + 1)
            rn2_im = 1 / (evals_gamma2.square() + 1)




            lx_Ik_re = torch.kron(torch.diag(rn1_re) @ inv_sqrtMk_x, sqrtMk_y @ torch.eye(k, device=A.device))
            lx_Ik_im = torch.kron(torch.diag(rn1_im) @ inv_sqrtMk_x, sqrtMk_y @ torch.eye(k, device=A.device))


            Ik_ly_re = torch.kron(torch.eye(k, device=A.device) @ inv_sqrtMk_x, sqrtMk_y @ torch.diag(rn2_re))
            Ik_ly_im = torch.kron(torch.eye(k, device=A.device) @ inv_sqrtMk_x, sqrtMk_y @ torch.diag(rn2_im))


            Delta_re = (Ik_ly_re - lx_Ik_re)
            Delta_im = (Ik_ly_im - lx_Ik_im)

            second = Delta_re.T @ Delta_re + Delta_im.T @ Delta_im
        else:
            second = torch.diag(D.T.flatten())

        # ---------put everything together into a system---------
        rhs = At_Ik.T @ vec_B
        op = first + self.lmbda * second

        C = torch.linalg.solve(op, rhs)

        C = C.reshape(k, k).T
        
        # output should also be batched
        Cxy = C.unsqueeze(0)
        return Cxy

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, sqrtMk_x, sqrtMk_y):
        """
        Forward pass to compute functional map
        Args:
            feat_x (torch.Tensor): feature vector of shape x. [B, Vx, C].
            feat_y (torch.Tensor): feature vector of shape y. [B, Vy, C].
            evals_x (torch.Tensor): eigenvalues of shape x. [B, K].
            evals_y (torch.Tensor): eigenvalues of shape y. [B, K].
            evecs_trans_x (torch.Tensor): pseudo inverse of eigenvectors of shape x. [B, K, Vx].
            evecs_trans_y (torch.Tensor): pseudo inverse of eigenvectors of shape y. [B, K, Vy].

        Returns:
            C (torch.Tensor): functional map from shape x to shape y. [B, K, K].
        """
        Cxy = self.compute_functional_map(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, sqrtMk_x, sqrtMk_y)

        if self.bidirectional:
            Cyx = self.compute_functional_map(feat_y, feat_x, evals_y, evals_x, evecs_trans_y, evecs_trans_x, sqrtMk_y, sqrtMk_x)
        else:
            Cyx = None

        return Cxy, Cyx

@NETWORK_REGISTRY.register()
class ExpandedStandardFMNet(nn.Module):
    """Compute the functional map matrix representation using Standard Regularization and adapted to Hilbert-Schmidt Norm."""
    def __init__(self, lmbda=1e-3, bidirectional=False, adapt_feature=True, adapt_mask=True):
        super(ExpandedStandardFMNet, self).__init__()
        self.lmbda = lmbda
        self.bidirectional = bidirectional

        # flag for whether adapt mass to feature term
        self.adapt_feature = adapt_feature
        # flag for whether adapt mass to regularizer term
        self.adapt_mask = adapt_mask

    def compute_functional_map(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, sqrtMk_x, sqrtMk_y):
        D = torch.repeat_interleave(evals_x.unsqueeze(1), repeats=evals_x.size(1), dim=1)
        D = (D - torch.repeat_interleave(evals_y.unsqueeze(2), repeats=evals_x.size(1), dim=2)) ** 2
        # inputs are all batched
        D = D.squeeze()
        feat_x = feat_x.squeeze()
        feat_y = feat_y.squeeze()
        sqrtMk_x = sqrtMk_x.squeeze()
        sqrtMk_y = sqrtMk_y.squeeze()
        evecs_trans_x = evecs_trans_x.squeeze()
        evecs_trans_y = evecs_trans_y.squeeze()
        # inner processing are all without batch
        evals_x = evals_x.squeeze()
        evals_y = evals_y.squeeze()


        # ---------first term: features---------
        A = evecs_trans_x @ feat_x
        B = evecs_trans_y @ feat_y

        if self.adapt_feature:
            B = sqrtMk_y @ B
        # A and B should be same shape
        k, m = A.size(0), A.size(1)

        vec_B = B.T.reshape(m * k, 1).contiguous()

        A_t = A.T.contiguous()
        if self.adapt_feature:
            Ik = sqrtMk_y
        else:
            Ik = torch.eye(k, device=A.device, dtype=torch.float32)
        
        At_Ik = torch.kron(A_t, Ik)

        first = At_Ik.T @ At_Ik


        # ---------second term: regularizer(mask)---------
        if self.adapt_mask:
            inv_sqrtMk_x = torch.inverse(sqrtMk_x)
            lx = torch.diag(evals_x)
            ly = torch.diag(evals_y)
            lx_Ik = torch.kron(lx @ inv_sqrtMk_x, sqrtMk_y @ torch.eye(k, device=A.device))
            Ik_ly = torch.kron(torch.eye(k, device=A.device) @ inv_sqrtMk_x, sqrtMk_y @ ly)
            Delta = (lx_Ik - Ik_ly)
            second = Delta.T @ Delta
        else:
            second = torch.diag(D.T.flatten())

        # ---------put everything together into a system---------
        rhs = At_Ik.T @ vec_B
        op = first + self.lmbda * second

        C = torch.linalg.solve(op, rhs)

        C = C.reshape(k, k).T
        
        # output should also be batched
        Cxy = C.unsqueeze(0)
        return Cxy

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, sqrtMk_x, sqrtMk_y):
        """
        Forward pass to compute functional map
        Args:
            feat_x (torch.Tensor): feature vector of shape x. [B, Vx, C].
            feat_y (torch.Tensor): feature vector of shape y. [B, Vy, C].
            evals_x (torch.Tensor): eigenvalues of shape x. [B, K].
            evals_y (torch.Tensor): eigenvalues of shape y. [B, K].
            evecs_trans_x (torch.Tensor): pseudo inverse of eigenvectors of shape x. [B, K, Vx].
            evecs_trans_y (torch.Tensor): pseudo inverse of eigenvectors of shape y. [B, K, Vy].

        Returns:
            C (torch.Tensor): functional map from shape x to shape y. [B, K, K].
        """
        Cxy = self.compute_functional_map(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, sqrtMk_x, sqrtMk_y)

        if self.bidirectional:
            Cyx = self.compute_functional_map(feat_y, feat_x, evals_y, evals_x, evecs_trans_y, evecs_trans_x, sqrtMk_y, sqrtMk_x)
        else:
            Cyx = None

        return Cxy, Cyx