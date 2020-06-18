#  Author: Kay Hartmann <kg.hartma@gmail.com>

from typing import Tuple

import numpy as np
import scipy as sp
import torch

from eeggan.cuda.cuda import to_cuda, get_activate_cuda


def calculate_activation_statistics(act: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    act = act.reshape(act.shape[0], -1)
    if get_activate_cuda():
        fact = act.shape[0] - 1
        act = to_cuda(act)
        mu = torch.mean(act, dim=0, keepdim=True)
        act = act - mu.expand_as(act)
        sigma = act.t().mm(act) / fact
        mu = mu.data.cpu().numpy().squeeze()
        sigma = sigma.data.cpu().numpy().squeeze()
    else:
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
    return mu, sigma


# From https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/frechet_inception_distance.py
def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    -- dist  : The Frechet Distance.
    Raises:
    -- InvalidFIDException if nan occures.
    """
    m = np.square(mu1 - mu2).sum()
    s, _ = sp.linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)
    dist = m + np.trace(sigma1 + sigma2 - 2 * s)
    return np.real(dist).item()
