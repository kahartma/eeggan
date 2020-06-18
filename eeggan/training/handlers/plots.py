#  Author: Kay Hartmann <kg.hartma@gmail.com>

import os
from abc import ABCMeta

import numpy as np
from matplotlib.figure import Figure

from eeggan.plotting.plots import labeled_tube_plot
from eeggan.training.trainer.trainer import Trainer, BatchOutput
from eeggan.validation.validation_helper import compute_spectral_amplitude


class EpochPlot(metaclass=ABCMeta):
    def __init__(self, figure: Figure, plot_path: str, prefix: str):
        self.figure = figure
        self.path = plot_path
        self.prefix = prefix
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

    def __call__(self, trainer: Trainer):
        self.plot(trainer)
        self.figure.savefig(os.path.join(self.path, self.prefix + str(trainer.state.epoch)))
        self.figure.clear()

    def plot(self, trainer: Trainer):
        raise NotImplementedError


class SpectralPlot(EpochPlot):

    def __init__(self, figure: Figure, plot_path: str, prefix: str, n_samples: int, fs: float):
        self.n_samples = n_samples
        self.fs = fs
        self.freqs = np.fft.rfftfreq(self.n_samples, 1. / self.fs)
        super().__init__(figure, plot_path, prefix)

    def plot(self, trainer: Trainer):
        batch_output: BatchOutput = trainer.state.output

        amps_real = compute_spectral_amplitude(batch_output.batch_real.X.data.cpu().numpy(), axis=2)
        amps_real_mean = amps_real.mean(axis=(0, 1)).squeeze()
        amps_real_std = amps_real.std(axis=(0, 1)).squeeze()
        amps_fake = compute_spectral_amplitude(batch_output.batch_fake.X.data.cpu().numpy(), axis=2)
        amps_fake_mean = amps_fake.mean(axis=(0, 1)).squeeze()
        amps_fake_std = amps_fake.std(axis=(0, 1)).squeeze()
        labeled_tube_plot(self.freqs,
                          [amps_real_mean, amps_fake_mean],
                          [amps_real_std, amps_fake_std],
                          ["Real", "Fake"],
                          "Mean spectral log amplitude", "Hz", "log(Amp)", self.figure.gca())
