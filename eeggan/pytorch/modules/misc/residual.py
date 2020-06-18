#  Author: Kay Hartmann <kg.hartma@gmail.com>

from eeggan.pytorch.modules.module import Module
from eeggan.pytorch.modules.sequential import Sequential


class ResidualBlock(Module):
    def __init__(self, sequential: Sequential):
        super().__init__()
        self.block = sequential

    def forward(self, x, **kwargs):
        out_block = self.block.forward(x, **kwargs)
        out = x + out_block
        return out
