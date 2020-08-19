#  Author: Kay Hartmann <kg.hartma@gmail.com>

import numpy as np
from torch import nn
from torch.nn.init import calculate_gain

from eeggan.model.builder import ProgressiveModelBuilder
from eeggan.pytorch.modules.conv.multiconv import MultiConv1d
from eeggan.pytorch.modules.normalization.pixelnorm import PixelNorm
from eeggan.pytorch.modules.projection.project import EmbeddedClassStyle
from eeggan.pytorch.modules.reshape.permute import Permute
from eeggan.pytorch.modules.reshape.pixelshuffle import PixelShuffle2d
from eeggan.pytorch.modules.reshape.reshape import Reshape
from eeggan.pytorch.modules.scaling.interpolate import Interpolate
from eeggan.pytorch.modules.sequential import Sequential
from eeggan.pytorch.modules.weights.weight_scaling import weight_scale
from eeggan.training.progressive.discriminator import ProgressiveDiscriminatorBlock, ProgressiveDiscriminator
from eeggan.training.progressive.generator import ProgressiveGeneratorBlock, ProgressiveGenerator


class BaselineV2(ProgressiveModelBuilder):
    def __init__(self, n_stages: int, n_latent: int, n_time: int, n_channels: int, n_classes: int, n_filters: int,
                 upsampling: str = 'linear', downsampling: str = 'linear', discfading: str = 'linear',
                 genfading: str = 'linear'):
        super().__init__(n_stages)
        self.n_latent = n_latent
        self.n_time = n_time
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.n_time_last_layer = int(np.floor(n_time / 2 ** n_stages))
        self.upsampling = upsampling
        self.downsampling = downsampling
        self.discfading = discfading
        self.genfading = genfading

    def build_disc_downsample_sequence(self) -> nn.Module:
        if self.downsampling in ['nearest', 'linear', 'area', 'cubic']:
            return build_interpolate(0.5, self.downsampling)
        if self.downsampling == 'conv':
            return Sequential(
                nn.ReflectionPad1d(1),
                weight_scale(nn.Conv1d(self.n_filters, self.n_filters, 4, stride=2),
                             gain=calculate_gain('leaky_relu')),
                nn.LeakyReLU(0.2)
            )

    def build_gen_upsample_sequence(self) -> nn.Module:
        if self.upsampling in ['nearest', 'linear', 'area', 'cubic']:
            return build_interpolate(2, self.upsampling)
        if self.upsampling == 'conv':
            return Sequential(
                weight_scale(nn.ConvTranspose1d(self.n_filters, self.n_filters, 4, stride=2, padding=1),
                             gain=calculate_gain('leaky_relu')),
                nn.LeakyReLU(0.2)
            )

    def build_disc_conv_sequence(self, i_stage: int):
        return Sequential(
            weight_scale(create_multiconv_for_stage(self.n_filters, i_stage),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            weight_scale(nn.Conv1d(self.n_filters, self.n_filters, kernel_size=1),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            self.build_disc_downsample_sequence(),
            weight_scale(EmbeddedClassStyle(self.n_classes, self.n_filters),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2)
        )

    def build_disc_in_sequence(self):
        return Sequential(
            weight_scale(nn.Conv1d(self.n_channels, self.n_filters, 1),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2)
        )

    def build_disc_fade_sequence(self):
        return build_interpolate(0.5, self.discfading)

    def build_discriminator(self) -> ProgressiveDiscriminator:
        blocks = []
        for i in range(self.n_stages - 1):
            block = ProgressiveDiscriminatorBlock(
                self.build_disc_conv_sequence(self.n_stages - 1 - i),
                self.build_disc_in_sequence(),
                self.build_disc_fade_sequence()
            )
            blocks.append(block)

        last_block = ProgressiveDiscriminatorBlock(
            Sequential(
                self.build_disc_conv_sequence(0),
                Reshape([[0], self.n_filters * self.n_time_last_layer]),
                weight_scale(nn.Linear(self.n_filters * self.n_time_last_layer, 1),
                             gain=calculate_gain('linear'))
            ),
            self.build_disc_in_sequence(),
            None
        )
        blocks.append(last_block)
        return ProgressiveDiscriminator(self.n_time, self.n_channels, self.n_classes, blocks)

    def build_gen_conv_sequence(self, i_stage: int):
        return Sequential(
            self.build_gen_upsample_sequence(),
            weight_scale(create_multiconv_for_stage(self.n_filters, i_stage),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            weight_scale(nn.Conv1d(self.n_filters, self.n_filters, kernel_size=1),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            weight_scale(EmbeddedClassStyle(self.n_classes, self.n_filters),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

    def build_gen_out_sequence(self):
        return Sequential(
            weight_scale(nn.Conv1d(self.n_filters, self.n_channels, 1),
                         gain=calculate_gain('linear')),
            Reshape([[0], [1], [2], 1]),
            PixelShuffle2d((1, self.n_channels)),
            Reshape([[0], [2], [3]]),
            Permute([0, 2, 1])
        )

    def build_gen_fade_sequence(self):
        return build_interpolate(2, self.discfading)

    def build_generator(self) -> ProgressiveGenerator:
        blocks = []
        first_block = ProgressiveGeneratorBlock(
            Sequential(
                weight_scale(nn.Linear(self.n_latent, self.n_filters * self.n_time_last_layer),
                             gain=calculate_gain('leaky_relu')),
                Reshape([[0], self.n_filters, -1]),
                nn.LeakyReLU(0.2),
                PixelNorm(),
                self.build_gen_conv_sequence(0)
            ),
            self.build_gen_out_sequence(),
            self.build_gen_fade_sequence()
        )
        blocks.append(first_block)

        for i in range(1, 6):
            block = ProgressiveGeneratorBlock(
                self.build_gen_conv_sequence(i),
                self.build_gen_out_sequence(),
                self.build_gen_fade_sequence()
            )
            blocks.append(block)
        return ProgressiveGenerator(self.n_time, self.n_channels, self.n_classes, self.n_latent, blocks)


def build_interpolate(scale_factor: float, mode: str):
    if mode in ['nearest', 'linear', 'area']:
        return Interpolate(scale_factor=scale_factor, mode=mode)
    if mode == 'cubic':
        return Sequential(
            Reshape([[0], [1], [2], 1]),
            Interpolate(scale_factor=(scale_factor, 1), mode='bicubic'),
            Reshape([[0], [1], [2]])
        )


def create_multiconv_for_stage(n_filters: int, i_stage: int):
    groups = int(n_filters / ((i_stage + 1) * 2))
    conv_configs = list()
    conv_configs.append({'kernel_size': 3, 'padding': 1, 'groups': groups})
    conv_configs.append({'kernel_size': 5, 'padding': 2, 'groups': groups})
    if i_stage >= 1:
        conv_configs.append({'kernel_size': 7, 'padding': 3, 'groups': groups})
        conv_configs.append({'kernel_size': 9, 'padding': 4, 'groups': groups})
    if i_stage >= 2:
        conv_configs.append({'kernel_size': 11, 'padding': 5, 'groups': groups})
        conv_configs.append({'kernel_size': 13, 'padding': 6, 'groups': groups})
    if i_stage >= 3:
        conv_configs.append({'kernel_size': 15, 'padding': 7, 'groups': groups})
        conv_configs.append({'kernel_size': 17, 'padding': 8, 'groups': groups})
    if i_stage >= 4:
        conv_configs.append({'kernel_size': 19, 'padding': 9, 'groups': groups})
        conv_configs.append({'kernel_size': 21, 'padding': 10, 'groups': groups})
    if i_stage >= 5:
        conv_configs.append({'kernel_size': 23, 'padding': 11, 'groups': groups})
        conv_configs.append({'kernel_size': 25, 'padding': 12, 'groups': groups})
    return MultiConv1d(conv_configs, n_filters, n_filters, split_in_channels=True, reflective=True)
