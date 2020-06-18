#  Author: Kay Hartmann <kg.hartma@gmail.com>

import torch
from torch import nn

from eeggan.pytorch.modules.module import Module
from eeggan.pytorch.utils.weights import fill_weights_normal


class WeightedNoise(Module):
    def __init__(self, n_features, n_time):
        super().__init__()
        self.weight_conv = nn.Conv1d(1, n_features, 1, bias=False)
        self.n_features = n_features
        self.n_time = n_time
        fill_weights_normal(self.weight_conv.weight)

    def forward(self, x, **kwargs):
        noise = torch.normal(0, 1, size=(x.size(0), 1, self.n_time))
        if x.is_cuda:
            noise = noise.cuda()

        noise = self.weight_conv.forward(noise)
        return x + noise
