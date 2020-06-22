#  Author: Kay Hartmann <kg.hartma@gmail.com>

import numpy as np
from torch import nn

from eeggan.pytorch.modules.module import Module


class ConditionalBatchNorm1d(Module):
    def __init__(self, n_classes, num_features, eps=1e-05, momentum=0.1, affine=True):
        super().__init__()
        layers = list()
        for i in range(n_classes):
            layers.append(nn.BatchNorm1d(num_features, eps, momentum, affine))
        self.bn_layers = nn.ModuleList(layers)
        self.n_classes = n_classes

    def forward(self, x, y_onehot=None, **kwargs):
        for c in range(self.n_classes):
            c_indx = np.where(y_onehot.cpu().data.numpy()[:, c] == 1)[0]
            tmp = self.bn_layers[c](x[c_indx.tolist(), :, :])
            x[c_indx, :, :] = tmp
        return x


class RealFakeBNorm(Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        layers = list()
        layers.append(nn.BatchNorm1d(num_features, eps, momentum, affine=False))
        layers.append(nn.BatchNorm1d(num_features, eps, momentum, affine=False))
        self.bn_layers = nn.ModuleList(layers)

    def forward(self, inp, fake=False, **kwargs):
        if fake is False:
            inp = self.bn_layers[0](inp)
        else:
            inp = self.bn_layers[1](inp)
        return inp
