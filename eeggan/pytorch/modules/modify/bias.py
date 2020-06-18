#  Author: Kay Hartmann <kg.hartma@gmail.com>

import torch

from eeggan.pytorch.modules.module import Module
from eeggan.pytorch.utils.bias import fill_bias_zero


class AddBias(Module):
    def __init__(self, n_features):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.Tensor(1, n_features, 1))
        fill_bias_zero(self.bias)

    def forward(self, x, **kwargs):
        return x + self.bias
