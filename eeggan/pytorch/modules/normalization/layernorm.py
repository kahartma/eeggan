#  Author: Kay Hartmann <kg.hartma@gmail.com>

import torch
import torch.nn.functional as fun
from torch.nn import Parameter

from eeggan.pytorch.modules.module import Module
from eeggan.pytorch.utils.bias import fill_bias_zero
from eeggan.pytorch.utils.weights import fill_weights_normal


class LayerNorm(Module):
    """
    References
    ----------
    Ba, J. L., Kiros, J. R., & Hinton, G. E. (n.d.). Layer Normalization.
    Retrieved from https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, num_features, n_dim, eps=1e-5, affine=True):
        assert (n_dim > 1)

        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.n_dim = n_dim

        tmp_ones = [1] * (n_dim - 2)
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.weight = Parameter(torch.Tensor(1, num_features, *tmp_ones))
            self.bias = Parameter(torch.Tensor(1, num_features, *tmp_ones))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            fill_weights_normal(self.weight)
            fill_bias_zero(self.bias)

    def forward(self, x, **kwargs):
        orig_size = x.size()
        b = orig_size[0]
        tmp_dims = range(self.n_dim)

        trash_mean = torch.zeros(b)
        trash_var = torch.ones(b)
        if x.is_cuda:
            trash_mean = trash_mean.cuda()
            trash_var = trash_var.cuda()

        input_reshaped = x.contiguous().permute(1, 0, *tmp_dims[2:]).contiguous()

        out = fun.batch_norm(
            input_reshaped, trash_mean, trash_var, None, None,
            True, 0., self.eps).permute(1, 0, *tmp_dims[2:]).contiguous()

        if self.affine:
            weight = self.weight
            bias = self.bias
            out = weight * out + bias

        return out
