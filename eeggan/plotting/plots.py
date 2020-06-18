#  Author: Kay Hartmann <kg.hartma@gmail.com>

import matplotlib.pyplot as plt
import numpy as np

from eeggan.validation.validation_helper import compute_spectral_amplitude


def labeled_plot(x, data_y, labels,
                 title="", xlabel="", ylabel="", axes=None):
    x = np.asarray(x)
    data_y = np.asarray(data_y)
    if axes is None:
        axes = plt.gca()

    for i, label in enumerate(labels):
        axes.plot(x, data_y[i], lw=2, label=label)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xlim(x.min(), x.max())
    axes.set_ylim(data_y.min(), data_y.max())
    axes.legend()


def labeled_tube_plot(x, data_y, tube_y, labels,
                      title="", xlabel="", ylabel="", axes=None):
    x = np.asarray(x)
    data_y = np.asarray(data_y)
    tube_y = np.asarray(tube_y)
    if axes is None:
        axes = plt.gca()

    colors = []
    for i, label in enumerate(labels):
        y_tmp = data_y[i]
        tube_tmp = tube_y[i]
        p = axes.fill_between(x, y_tmp + tube_tmp, y_tmp - tube_tmp, alpha=0.5, label=labels[i])
        colors.append(p._original_facecolor)

    for i, label in enumerate(labels):
        axes.plot(x, data_y[i], lw=2, color=colors[i])

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xlim(x.min(), x.max())
    axes.set_ylim((data_y - tube_y).min(), (data_y + tube_y).max())
    axes.legend()


def spectral_plot(X_real: np.ndarray, X_fake: np.ndarray, fs, axes=None):
    n_samples = X_real.shape[2]
    freqs = np.fft.rfftfreq(n_samples, 1. / fs)
    amps_real = compute_spectral_amplitude(X_real, axis=2)
    amps_real_mean = amps_real.mean(axis=(0, 1)).squeeze()
    amps_real_std = amps_real.std(axis=(0, 1)).squeeze()
    amps_fake = compute_spectral_amplitude(X_fake, axis=2)
    amps_fake_mean = amps_fake.mean(axis=(0, 1)).squeeze()
    amps_fake_std = amps_fake.std(axis=(0, 1)).squeeze()
    labeled_tube_plot(freqs,
                      [amps_real_mean, amps_fake_mean],
                      [amps_real_std, amps_fake_std],
                      ["Real", "Fake"],
                      "Mean spectral log amplitude", "Hz", "log(Amp)", axes)
