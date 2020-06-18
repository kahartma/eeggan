#  Author: Kay Hartmann <kg.hartma@gmail.com>

import numpy as np


def create_wasserstein_transform_matrix(n_projections, n_features):
    return np.random.normal(size=(n_projections, n_features))


def calculate_sliced_wasserstein_distance(input1, input2, w_transform):
    if input1.shape[0] != input2.shape[0]:
        n_inputs = input1.shape[0] if input1.shape[0] < input2.shape[0] else input2.shape[0]
        input1 = np.random.permutation(input1)[:n_inputs]
        input2 = np.random.permutation(input2)[:n_inputs]

    input1 = input1.reshape(input1.shape[0], -1)
    input2 = input2.reshape(input2.shape[0], -1)

    transformed1 = np.matmul(w_transform, input1.T)
    transformed2 = np.matmul(w_transform, input2.T)

    for i in np.arange(w_transform.shape[0]):
        transformed1[i] = np.sort(transformed1[i], -1)
        transformed2[i] = np.sort(transformed2[i], -1)

    diff = transformed1 - transformed2
    diff = np.power(diff, 2).mean()
    return np.sqrt(diff)
