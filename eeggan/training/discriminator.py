#  Author: Kay Hartmann <kg.hartma@gmail.com>

from abc import ABCMeta

from eeggan.pytorch.modules.module import Module


class Discriminator(Module, metaclass=ABCMeta):
    def __init__(self, n_samples, n_channels, n_classes):
        super().__init__()
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.n_classes = n_classes
