#  Author: Kay Hartmann <kg.hartma@gmail.com>
from typing import Union

import numpy as np
import torch
from torch import Tensor

from eeggan.validation.metrics.frechet import calculate_activation_statistics, calculate_frechet_distances
from eeggan.validation.metrics.inception import calculate_inception_score
from eeggan.validation.metrics.wasserstein import create_wasserstein_transform_matrix, \
    calculate_sliced_wasserstein_distance


def init_sliced_wasserstein(train_set_tmp, test_set_tmp):
    w_transform = create_wasserstein_transform_matrix(500, np.prod(train_set_tmp.shape[1:]))
    w_dist = calculate_sliced_wasserstein_distance(train_set_tmp, test_set_tmp, w_transform)
    return w_transform, w_dist


def init_inception(train_preds, test_preds):
    train_mean, train_std = calculate_inception_score(train_preds, 50, 10)
    test_mean, test_std = calculate_inception_score(test_preds, 50, 10)
    return train_mean, train_std, test_mean, test_std


def init_frechet(train_act, test_act):
    train_mu, train_sigma = calculate_activation_statistics(train_act)
    test_mu, test_sigma = calculate_activation_statistics(test_act)
    f_dist = calculate_frechet_distances(train_mu[None, :, :], train_sigma[None, :, :], test_mu[None, :, :],
                                         test_sigma[None, :, :])
    return train_mu, train_sigma, test_mu, test_sigma, f_dist


def logsoftmax_act_to_softmax(act: Union[np.ndarray, Tensor]):
    if isinstance(act, Tensor):
        return torch.exp(act)
    else:
        return np.exp(act)


def compute_spectral_amplitude(x, axis=None):
    fft = np.fft.rfft(x, axis=axis)
    return np.log(np.abs(fft))
