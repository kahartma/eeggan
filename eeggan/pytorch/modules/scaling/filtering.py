# coding=utf-8

#  Author: Kay Hartmann <kg.hartma@gmail.com>

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from eeggan.pytorch.modules.module import Module


def exponential_sum(x, n):
    nom = 1 - np.exp(1j * n * x)
    denom = 1 - np.exp(1j * x)
    return nom / denom


def exponential_sum2(x, n1, n2):
    nom = np.exp(1j * n1 * x) - np.exp(1j * n2 * x)
    denom = 1 - np.exp(1j * x)
    return nom / denom


def filter_weights(M, m, scale_factor):
    m2 = np.arange(0, M)
    freq_start = 1. / (2 * scale_factor)
    freq_end = 1 - freq_start
    t1 = exponential_sum((m - m2) * 2 * np.pi / M, M * freq_start)
    t1[m - m2 == 0] = M * freq_start
    t3 = exponential_sum2((m - m2) * 2 * np.pi / M, M * freq_end, M)
    t3[m - m2 == 0] = M * freq_start
    weights = t1 + t3
    return weights


class UpscaleFilter1d(Module):
    """
    Experimental
    """

    def __init__(self, M, scale_factor, window_size=5, kaiser=False, kbeta=14, adaptive=False):
        super(UpscaleFilter1d, self).__init__()
        self.scale_factor = scale_factor
        self.window_size = window_size
        self.adaptive = adaptive
        weights = np.zeros((1, 1, window_size * 2 + 1)).astype(np.float32)
        start_pos = window_size
        tmp = filter_weights(M, start_pos, scale_factor)[:window_size * 2 + 1]
        if kaiser:
            tmp *= np.kaiser(window_size * 2 + 1, kbeta)
        tmp /= tmp.sum()
        weights[0, 0, :] = tmp
        if self.adaptive:
            weights = nn.Parameter(torch.from_numpy(weights), requires_grad=self.adaptive)
            self.register_parameter('weights', weights)
        else:
            weights = Variable(torch.from_numpy(weights), requires_grad=self.adaptive)
            self.register_buffer('weights', weights)

    def forward(self, x, **kwargs):
        old_size = x.size()
        x = F.pad(x, pad=[self.window_size, self.window_size], mode='reflect')
        weight = self.weights.expand(old_size[1], -1, -1).contiguous()
        output = F.conv1d(x, weight, groups=old_size[1])
        return output


class UpscaleFilter2d(UpscaleFilter1d):
    """
    Experimental
    """

    def __init__(self, M, scale_factor, window_size=5, kaiser=False, kbeta=14, adaptive=False):
        super().__init__(M, scale_factor, window_size, kaiser, kbeta, adaptive)
        if self.adaptive:
            self.register_parameter('weights', nn.Parameter(self.weights[:, :, :, None].data, requires_grad=True))
        else:
            self.register_buffer('weights', self.weights[:, :, :, None])

    def forward(self, x, **kwargs):
        old_size = x.size()
        x = F.pad(x, pad=[0, 0, self.window_size, self.window_size], mode='reflect')
        weight = self.weights.expand(old_size[1], -1, -1, -1).contiguous()
        output = F.conv2d(x, weight, groups=old_size[1])
        return output


class SimpleFilter1d(Module):
    def __init__(self, filter_window, adaptive=False):
        super().__init__()
        filter_window = np.asarray(filter_window)
        filter_window = filter_window / np.sum(filter_window)
        filter_window = torch.from_numpy(filter_window.reshape((1, 1, -1)).astype(np.float32))
        if adaptive is False:
            filter_window = Variable(filter_window, requires_grad=False)
            self.register_buffer('filter_window', filter_window)
        else:
            self.filter_window = nn.Parameter(filter_window, requires_grad=True)
        self.padding = int(self.filter_window.size(2) / 2)

    def forward(self, x, **kwargs):
        groups = x.size(1)
        x = F.pad(x, pad=[self.padding, self.padding], mode='reflect')
        weights = self.filter_window.expand(groups, 1, self.filter_window.size(2)).contiguous()
        return F.conv1d(x, weights, groups=groups)
