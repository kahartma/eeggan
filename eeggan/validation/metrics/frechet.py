#  Author: Kay Hartmann <kg.hartma@gmail.com>

from typing import Tuple

import torch

from eeggan.cuda import to_cuda


def calculate_activation_statistics(act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        act = act.reshape(act.shape[0], -1)
        fact = act.shape[0] - 1
        act = to_cuda(act)
        mu = torch.mean(act, dim=0, keepdim=True)
        act = act - mu.expand_as(act)
        sigma = act.t().mm(act) / fact
        return mu, sigma


# From https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/frechet_inception_distance.py
def calculate_frechet_distances(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor,
                                sigma2: torch.Tensor) -> torch.Tensor:
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
    Returns:
    -- dist  : The Frechet Distance.
    Raises:
    -- InvalidFIDException if nan occures.
    """
    with torch.no_grad():
        m = torch.square(mu1 - mu2).sum()
        d = torch.bmm(sigma1, sigma2)
        s = sqrtm_newton(d)
        dists = m + torch.diagonal(sigma1 + sigma2 - 2 * s, dim1=-2, dim2=-1).sum(-1)
        return dists


# https://colab.research.google.com/drive/1wSO1MFh_ZCfOnejFnW1vkD71jaJy2Olu#scrollTo=Ju79uoiTQku6&line=1&uniqifier=1
def sqrtm_newton(A: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        numIters = 20
        batchSize = A.shape[0]
        dim = A.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A));
        I = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(A.dtype).to(A)
        Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(A.dtype).to(A)
        for i in range(numIters):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
        sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
        return sA
