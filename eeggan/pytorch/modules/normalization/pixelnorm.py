#  Author: Kay Hartmann <kg.hartma@gmail.com>

import torch

from eeggan.pytorch.modules.module import Module


class PixelNorm(Module):
    """
    References
    ----------
    Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
    Progressive Growing of GANs for Improved Quality, Stability, and Variation.
    Retrieved from http://arxiv.org/abs/1710.10196
    """

    def forward(self, x, eps=1e-8, **kwargs):
        tmp = torch.sqrt(torch.pow(x, 2).mean(dim=1, keepdim=True) + eps)
        return x / tmp
