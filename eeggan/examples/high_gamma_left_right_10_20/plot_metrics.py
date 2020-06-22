#  Author: Kay Hartmann <kg.hartma@gmail.com>
import os

import numpy as np
import torch
from matplotlib import pyplot

from eeggan.plotting.plots import labeled_plot, labeled_tube_plot

SUBJ_IND = 1
RESULT_PATH = "/home/khartmann/projects/eeggandata/results/%d/style_wgan_gp" % SUBJ_IND
STAGE = 0

if __name__ == '__main__':
    metrics = torch.load(os.path.join(RESULT_PATH, 'metrics_stage_%d.pt' % STAGE))
    metric_wasserstein = np.asarray(metrics["wasserstein"], dtype=float)
    metric_inception = np.asarray(metrics["inception"])
    metric_frechet = np.asarray(metrics["frechet"])
    metric_loss = metrics["loss"]

    fig = pyplot.figure()
    labeled_plot(
        metric_wasserstein[:, 0],
        [metric_wasserstein[:, 1]],
        ["fake to train"],
        "Sliced Wasserstein Distance", "Epochs", "SWD", fig.gca()
    )
    pyplot.show()

    fig = pyplot.figure()
    labeled_tube_plot(
        metric_inception[:, 0].astype(float),
        [np.asarray([np.asarray(t, dtype=float) for t in metric_inception[:, 1]], dtype=float)[:, 0]],
        [np.asarray([np.asarray(t, dtype=float) for t in metric_inception[:, 1]], dtype=float)[:, 1]],
        ["fake"],
        "Inception scores", "Epochs", "score", fig.gca()
    )
    pyplot.show()

    fig = pyplot.figure()
    labeled_tube_plot(
        metric_frechet[:, 0].astype(float),
        [np.asarray([np.asarray(t, dtype=float) for t in metric_frechet[:, 1]], dtype=float)[:, 0]],
        [np.asarray([np.asarray(t, dtype=float) for t in metric_frechet[:, 1]], dtype=float)[:, 1]],
        ["fake"],
        "Frechet Distance", "Epochs", "distance", fig.gca()
    )
    pyplot.show()
