#  Author: Kay Hartmann <kg.hartma@gmail.com>

import numpy as np

from eeggan.data.data import Data
from eeggan.data.preprocess.normalize import normalize_data


def create_onehot_vector(targets, n_classes) -> np.ndarray:
    targets_onehot = np.zeros((targets.shape[0], n_classes))
    for c in range(n_classes):
        targets_onehot[targets == c, c] = 1

    return targets_onehot


def prepare_data(X, y, n_classes, input_length, normalize=True) -> Data[np.ndarray]:
    X = X[:, :, :input_length]

    if normalize:
        X = normalize_data(X)

    data = Data[np.ndarray](X,
                            y,
                            create_onehot_vector(y, n_classes))

    return data
