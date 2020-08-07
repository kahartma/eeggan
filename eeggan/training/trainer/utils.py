#  Author: Kay Hartmann <kg.hartma@gmail.com>
from typing import Union

from torch import Tensor

from eeggan.data.dataset import Data


def detach_all(*elements: Union[Tensor, Data[Tensor]]):
    if len(elements) == 1:
        return detach(elements[0])
    else:
        return (detach(e) for e in elements)


def detach_data(element: Data[Tensor]):
    return Data(element.X.detach(), element.y.detach(), element.y_onehot.detach())


def detach(element: Union[Tensor, Data]):
    if isinstance(element, Tensor):
        return element.detach()
    elif isinstance(element, Data):
        return detach_data(element)
