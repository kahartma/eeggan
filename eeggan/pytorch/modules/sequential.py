#  Author: Kay Hartmann <kg.hartma@gmail.com>

import torch.nn as nn

from eeggan.pytorch.modules.module import Module


class Sequential(nn.Sequential, Module):
    def forward(self, x, **kwargs):
        for module in self:
            if isinstance(module, Module):
                x = module(x, **kwargs)
            else:
                x = module(x)
        return x
