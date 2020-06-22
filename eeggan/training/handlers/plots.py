#  Author: Kay Hartmann <kg.hartma@gmail.com>

import os
from abc import ABCMeta

from matplotlib.figure import Figure

from eeggan.plotting.plots import spectral_plot
from eeggan.training.trainer.trainer import Trainer, BatchOutput


class EpochPlot(metaclass=ABCMeta):
    def __init__(self, figure: Figure, plot_path: str, prefix: str):
        self.figure = figure
        self.path = plot_path
        self.prefix = prefix

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
        super().__init__(figure, plot_path, prefix)

    def plot(self, trainer: Trainer):
        batch_output: BatchOutput = trainer.state.output
        spectral_plot(batch_output.batch_real.X.data.cpu().numpy(), batch_output.batch_fake.X.data.cpu().numpy(),
                      self.fs, self.figure.gca())
