#  Author: Kay Hartmann <kg.hartma@gmail.com>
from typing import Iterable, Union

from torch import Tensor, nn

from eeggan.pytorch.modules.module import Module


class Interpolate(Module):

    def __init__(self, scale_factor: Union[float, Iterable[float]], mode: str):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return nn.functional.interpolate(x, size=None, scale_factor=self.scale_factor, mode=self.mode)
