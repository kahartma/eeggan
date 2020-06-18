# coding=utf-8

#  Author: Kay Hartmann <kg.hartma@gmail.com>

import torch

from eeggan.pytorch.modules.module import Module


class StatisticAppend(Module):
    """
    Calculates full standard deviation of filters and appends std as new filter

    References
    ----------
    Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
    Progressive Growing of GANs for Improved Quality, Stability,
    and Variation. Retrieved from http://arxiv.org/abs/1710.10196
    """

    def __init__(self):
        super(StatisticAppend, self).__init__()

    def forward(self, x, **kwargs):
        mean = x.mean(dim=0, keepdim=True).expand_as(x)
        std = x.std(dim=0, keepdim=True).expand_as(x)
        x = torch.cat((x, mean, std), dim=1)
        return x


class StdAppend(Module):
    """
    Calculates full standard deviation of filters and appends std as new filter

    References
    ----------
    Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
    Progressive Growing of GANs for Improved Quality, Stability,
    and Variation. Retrieved from http://arxiv.org/abs/1710.10196
    """

    def __init__(self):
        super(StdAppend, self).__init__()

    def forward(self, x, **kwargs):
        std = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt((std ** 2).mean(dim=0) + 1e-8).mean()
        std_map = std.expand(x.size(0), 1)
        x = torch.cat((x, std_map), dim=1)
        return x


class StdMap1d(Module):
    """
    Calculates full standard deviation of filters and appends std as new filter

    References
    ----------
    Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
    Progressive Growing of GANs for Improved Quality, Stability,
    and Variation. Retrieved from http://arxiv.org/abs/1710.10196
    """

    def __init__(self):
        super(StdMap1d, self).__init__()

    def forward(self, x, **kwargs):
        std = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt((std ** 2).mean(dim=0) + 1e-8).mean()
        std_map = std.expand(x.size(0), 1, x.size(2))
        x = torch.cat((x, std_map), dim=1)
        return x


class StdMap1dIndiv(Module):
    """
    Calculates full standard deviation of filters and appends std as new filter

    References
    ----------
    Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
    Progressive Growing of GANs for Improved Quality, Stability,
    and Variation. Retrieved from http://arxiv.org/abs/1710.10196
    """

    def __init__(self):
        super(StdMap1dIndiv, self).__init__()

    def forward(self, x, **kwargs):
        std_map = x.std(dim=0, keepdim=True).mean(dim=1, keepdim=True)
        std_map = std_map.expand_as(x)
        x = torch.cat((x, std_map), dim=1)
        return x


class StdMap2d(Module):
    """
    Calculates full standard deviation of filters and appends std as new filter

    References
    ----------
    Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
    Progressive Growing of GANs for Improved Quality, Stability,
    and Variation. Retrieved from http://arxiv.org/abs/1710.10196
    """

    def __init__(self):
        super(StdMap2d, self).__init__()

    def forward(self, x, **kwargs):
        std = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt((std ** 2).mean(dim=0) + 1e-8).mean()
        std_map = std.expand(x.size(0), 1, x.size(2), x.size(3))
        x = torch.cat((x, std_map), dim=1)
        return x
