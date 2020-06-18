#  Author: Kay Hartmann <kg.hartma@gmail.com>

import numpy as np
import torch.nn.functional as fun

from eeggan.pytorch.modules.module import Module


class PhaseShuffle1d(Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x, **kwargs):
        shift = np.random.random_integers(low=-self.n, high=self.n)
        if shift == 0:
            return x
        if shift < 0:
            return fun.pad(x, pad=[np.abs(shift), 0], mode='reflect')[:, :, :shift]
        if shift > 0:
            return fun.pad(x, pad=[0, shift], mode='reflect')[:, :, shift:]
