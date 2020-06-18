#  Author: Kay Hartmann <kg.hartma@gmail.com>

import torch.nn as nn
from torch import Tensor


class Module(nn.Module):
    def forward(self, x: Tensor, **kwargs):
        raise NotImplementedError
