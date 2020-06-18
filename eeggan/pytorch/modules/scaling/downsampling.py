# coding=utf-8

#  Author: Kay Hartmann <kg.hartma@gmail.com>

from eeggan.pytorch.modules.module import Module


class Downsample1d(Module):
    """
    1d downsampling by only taking every n-th entry

    Parameters
    ----------
    divisor : int
        Downscaling factor
    """

    def __init__(self, divisor):
        super().__init__()
        self.divisor = divisor

    def forward(self, x, **kwargs):
        x = x.contiguous().view(x.size(0), x.size(1),
                                x.size(2) / self.divisor, self.divisor)
        x = x[:, :, :, 0]
        return x


class Downsample2d(Module):
    """
    2d downsampling by only taking every n-th entry

    Parameters
    ----------
    divisor : (int,int)
        Downscaling factors
    """

    def __init__(self, divisor):
        super().__init__()
        self.divisor = divisor

    def forward(self, x, **kwargs):
        x = x.contiguous().view(x.size(0), x.size(1),
                                x.size(2) / self.divisor[0], self.divisor[0],
                                x.size(3) / self.divisor[1], self.divisor[1])
        x = x[:, :, :, 0, :, 0]
        return x
