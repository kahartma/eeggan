#  Author: Kay Hartmann <kg.hartma@gmail.com>
from abc import ABCMeta

from eeggan.pytorch.modules.sequential import Sequential
from eeggan.training.discriminator import Discriminator
from eeggan.training.generator import Generator
from eeggan.training.progressive.discriminator import ProgressiveDiscriminator
from eeggan.training.progressive.generator import ProgressiveGenerator


class ModelBuilder(metaclass=ABCMeta):

    def build_discriminator(self) -> Discriminator:
        raise NotImplementedError

    def build_generator(self) -> Generator:
        raise NotImplementedError


class ProgressiveModelBuilder(ModelBuilder, metaclass=ABCMeta):
    def __init__(self, n_stages: int):
        self.n_stages = n_stages

    def build_disc_conv_sequence(self, i_stage: int) -> Sequential:
        raise NotImplementedError

    def build_disc_in_sequence(self) -> Sequential:
        raise NotImplementedError

    def build_disc_fade_sequence(self) -> Sequential:
        raise NotImplementedError

    def build_discriminator(self) -> ProgressiveDiscriminator:
        raise NotImplementedError

    def build_gen_conv_sequence(self, i_stage: int) -> Sequential:
        raise NotImplementedError

    def build_gen_out_sequence(self) -> Sequential:
        raise NotImplementedError

    def build_gen_fade_sequence(self) -> Sequential:
        raise NotImplementedError

    def build_generator(self) -> ProgressiveGenerator:
        raise NotImplementedError
