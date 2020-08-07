#  Author: Kay Hartmann <kg.hartma@gmail.com>
import numpy as np


def normalize_data(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    x = x / x.std()
    return x
