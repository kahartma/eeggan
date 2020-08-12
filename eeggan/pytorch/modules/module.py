#  Author: Kay Hartmann <kg.hartma@gmail.com>
from abc import ABCMeta
from typing import Callable, Any

import torch.nn as nn
from torch import Tensor


class Module(nn.Module, metaclass=ABCMeta):
    def forward(self, x: Tensor, **kwargs) -> Any:
        raise NotImplementedError


class LambdaModule(Module):

    def __init__(self, fn: Callable[[Tensor, Any], Any]):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs):
        return self.fn(x, **kwargs)
