#  Author: Kay Hartmann <kg.hartma@gmail.com>

import numpy as np


def calculate_inception_score(preds, splits, repititions=1):
    """

    :param split_preds: set of softmax predictions
    :return:
    """
    stepsize = int(np.ceil(preds.shape[0] / splits))
    steps = np.arange(0, preds.shape[0], stepsize)
    scores = []
    for rep in np.arange(repititions):
        preds_tmp = np.random.permutation(preds)
        for i in np.arange(len(steps)):
            pred = preds_tmp[steps[i]:steps[i] + stepsize]
            kl = pred * (np.log(pred) - np.log(np.expand_dims(np.mean(pred, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)
