#  Author: Kay Hartmann <kg.hartma@gmail.com>

from eeggan.pytorch.modules.module import Module


class MakeContiguous(Module):
    def forward(self, x, **kwargs):
        return x.contiguous()


class PrintSize(Module):
    def forward(self, x, **kwargs):
        print(x.size())
        return x


class Dummy(Module):
    def forward(self, x, **kwargs):
        return x
