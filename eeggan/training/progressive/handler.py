#  Author: Kay Hartmann <kg.hartma@gmail.com>

from eeggan.training.progressive.discriminator import ProgressiveDiscriminator
from eeggan.training.progressive.generator import ProgressiveGenerator


class ProgressionHandler:
    def __init__(self, discriminator: ProgressiveDiscriminator, generator: ProgressiveGenerator, n_blocks: int,
                 use_fade: bool, epochs_fade: int = None, current_stage: int = 0, current_alpha: float = 1.):
        self.discriminator = discriminator
        self.generator = generator
        self.n_blocks = n_blocks
        self.use_fade = use_fade
        self.epochs_fade = epochs_fade
        self.current_stage = current_stage
        self.current_alpha = current_alpha

    def set_progression(self, current_stage: int, current_alpha: float):
        self.current_stage = current_stage
        self.current_alpha = current_alpha
        self.generator.cur_block = current_stage
        self.generator.alpha = current_alpha
        self.discriminator.cur_block = self.n_blocks - current_stage - 1
        self.discriminator.alpha = current_alpha

    def advance_stage(self):
        alpha = 0.
        if not self.use_fade:
            alpha = 1.
        self.set_progression(self.current_stage + 1, alpha)

    def advance_alpha(self):
        if self.current_alpha < 1. and self.epochs_fade is not None:
            alpha_increase = 1. / self.epochs_fade
            self.set_progression(self.current_stage, self.current_alpha + alpha_increase)
