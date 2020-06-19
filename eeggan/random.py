#  Author: Kay Hartmann <kg.hartma@gmail.com>

import random

import torch
from numba import np

from eeggan.cuda import get_activate_cuda

SEED = 1234567


def init_random_seeds():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if get_activate_cuda():
        torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    rng = np.random.RandomState(SEED)

    return rng
