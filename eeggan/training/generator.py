#  Author: Kay Hartmann <kg.hartma@gmail.com>

from abc import ABCMeta

from numpy.random.mtrand import RandomState
from torch import Tensor

from eeggan.data.preprocess.util import create_onehot_vector
from eeggan.pytorch.modules.module import Module


class Generator(Module, metaclass=ABCMeta):
    def __init__(self, n_samples, n_channels, n_classes, n_latent):
        super().__init__()
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_latent = n_latent

    def create_latent_input(self, rng: RandomState, n_trials) -> (Tensor, Tensor, Tensor):
        z_latent = rng.normal(0, 1, size=(n_trials, self.n_latent))
        y_fake = rng.randint(0, self.n_classes, size=n_trials)
        y_fake_onehot = create_onehot_vector(y_fake, self.n_classes)
        return Tensor(z_latent), Tensor(y_fake), Tensor(y_fake_onehot)
