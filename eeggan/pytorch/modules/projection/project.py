#  Author: Kay Hartmann <kg.hartma@gmail.com>

import numpy as np
import torch
import torch.nn as nn

from eeggan.pytorch.modules.module import Module
from eeggan.pytorch.utils.bias import fill_bias_zero
from eeggan.pytorch.utils.weights import fill_weights_normal


class ProjectLinear(nn.Linear, Module):
    def __init__(self, n_classes, n_features):
        nn.Linear.__init__(self, n_classes, n_features)

    def forward(self, x, y_onehot=None, **kwargs):
        projection = nn.Linear.forward(self, y_onehot)[:, :, None]
        projection = projection.expand_as(x)
        projection = projection * x
        x = x + projection
        return x


class ProjectLinear_sqrt(nn.Linear, Module):
    def __init__(self, n_classes, n_features):
        nn.Linear.__init__(self, n_classes, n_features)

    def forward(self, x, y_onehot=None, **kwargs):
        projection = nn.Linear.forward(self, y_onehot)
        # projection = projection.expand_as(x)
        projection = projection * x
        x = x + projection
        x = x / np.sqrt(2)
        return x


class ProjectLinear_sqrt2(nn.Linear, Module):
    def __init__(self, n_classes, n_features):
        nn.Linear.__init__(self, n_classes, n_features)

    def forward(self, x, y_onehot=None, **kwargs):
        projection = nn.Linear.forward(self, y_onehot)[:, :, None]
        projection = projection.expand_as(x)
        projection = projection * x
        x = x + projection
        x = x / np.sqrt(2)
        return x


class ProjectLinearNoSum(nn.Linear, Module):
    def __init__(self, n_classes, n_features):
        nn.Linear.__init__(self, n_classes, n_features)

    def forward(self, x, y_onehot=None, **kwargs):
        projection = nn.Linear.forward(self, y_onehot)[:, :, None]
        projection = projection.expand_as(x)
        x_projection = projection * x
        return x_projection


class TransformLinearEmbed(nn.Linear, Module):
    def __init__(self, n_classes, n_features):
        nn.Linear.__init__(self, n_classes, n_features)

    def forward(self, x, y_onehot=None, **kwargs):
        projection = nn.Linear.forward(self, y_onehot)
        x_projection = projection * x
        return x_projection


class AppendClassFilter1d(Module):
    def __init__(self):
        super(AppendClassFilter1d, self).__init__()

    def forward(self, x, y_onehot=None, **kwargs):
        y_onehot = y_onehot.view(y_onehot.size(0), y_onehot.size(1), 1)
        filt_tmp = y_onehot.expand(-1, -1, x.size(2))
        # print(x.size(), filt_tmp.size(), y_onehot)
        return torch.cat((x, filt_tmp), dim=1)


class ProjectionLayer1(nn.Linear, Module):
    def __init__(self, n_classes, n_features):
        nn.Linear.__init__(self, n_classes, n_features)

    def forward(self, x, y_onehot=None, **kwargs):
        class_projection = nn.Linear.forward(self, y_onehot)
        return (x, class_projection)


class ProjectionLayer2(nn.Linear, Module):
    def __init__(self, n_features):
        nn.Linear.__init__(self, n_features, 1)

    def forward(self, input, **kwargs):
        x, class_projection = input
        x = x.sum(dim=2)
        output = nn.Linear.forward(self, x)
        x_proj = class_projection * x
        x_proj = x_proj.sum(dim=1, keepdim=True)
        output += x_proj
        return output


class ProjectionLayer1_new(nn.Linear, Module):
    def __init__(self, n_classes, n_features):
        nn.Linear.__init__(self, n_classes, n_features)

    def forward(self, x, y_onehot=None, **kwargs):
        class_projection = nn.Linear.forward(self, y_onehot)
        return (x, class_projection)


class ProjectionLayer2_new(nn.Linear, Module):
    def __init__(self, n_features):
        nn.Linear.__init__(self, n_features, 1)

    def forward(self, input, **kwargs):
        x, class_projection = input
        output = nn.Linear.forward(self, x)
        x_proj = class_projection * x
        x_proj = x_proj.sum(dim=1, keepdim=True)
        x_proj = x_proj / np.sqrt(np.prod(list(x.size())[1:]))
        output += x_proj
        output = output / np.sqrt(2)
        return output


class ProjectionLayer2_new2(nn.Linear, Module):
    def __init__(self, n_features):
        nn.Linear.__init__(self, n_features, 1)

    def forward(self, input, **kwargs):
        x, class_projection = input
        class_projection = class_projection[:, :, None]
        class_projection = class_projection.expand_as(x)
        output = nn.Linear.forward(self, x.view(x.size(0), -1))
        x_proj = class_projection * x
        x_proj = x_proj.sum(dim=2)
        x_proj = x_proj.sum(dim=1, keepdim=True)
        x_proj = x_proj / np.sqrt(np.prod(list(x.size())[1:]))
        output += x_proj
        output = output / np.sqrt(2)
        return output


class EmbeddedClassFilter(nn.Linear, Module):
    def __init__(self, n_classes, n_features):
        nn.Linear.__init__(self, n_classes, n_features)
        fill_weights_normal(self.weight)
        fill_bias_zero(self.bias)

    def forward(self, x, y_onehot=None, **kwargs):
        embed = nn.Linear.forward(self, y_onehot)[:, :, None]
        embed = embed.expand_as(x)
        embed = embed * x
        return embed


class EmbeddedClassStyle(nn.Linear, Module):
    def __init__(self, n_classes, n_features):
        nn.Linear.__init__(self, n_classes, n_features * 2, bias=True)
        self.n_classes = n_classes
        self.n_features = n_features
        fill_weights_normal(self.weight)
        fill_bias_zero(self.bias)

    def forward(self, x, y_onehot=None, **kwargs):
        style = nn.Linear.forward(self, y_onehot)
        style = style.view(2, x.size(0), self.n_features, 1)
        return style[0] * x + style[1]


class EmbedAndConcatLabels(nn.Linear, Module):
    def __init__(self, n_classes, n_features, bias):
        nn.Linear.__init__(self, n_classes, n_features, bias=bias)
        fill_weights_normal(self.weight)
        fill_bias_zero(self.bias)

    def forward(self, x, y_onehot=None, **kwargs):
        embed = nn.Linear.forward(self, y_onehot)
        return torch.cat((x, embed), 1)


class StyleMod(nn.Linear, Module):
    def __init__(self, n_mappings, n_features):
        nn.Linear.__init__(self, n_mappings, n_features * 2, bias=True)
        self.n_mappings = n_mappings
        self.n_features = n_features
        fill_weights_normal(self.weight)
        fill_bias_zero(self.bias)

    def forward(self, input, mappings=None, **kwargs):
        style = nn.Linear.forward(self, mappings)
        style = style.view(2, input.size(0), self.n_features, 1)
        return style[0] * input + style[1]
