#  Author: Kay Hartmann <kg.hartma@gmail.com>

from torch import nn
from torch.nn.init import calculate_gain

from eeggan.pytorch.modules.conv.multiconv import MultiConv1d
from eeggan.pytorch.modules.misc.helper import Dummy
from eeggan.pytorch.modules.modify.stdmap import StdMap1d
from eeggan.pytorch.modules.normalization.pixelnorm import PixelNorm
from eeggan.pytorch.modules.projection.project import EmbeddedClassFilter
from eeggan.pytorch.modules.reshape.permute import Permute
from eeggan.pytorch.modules.reshape.pixelshuffle import PixelShuffle2d
from eeggan.pytorch.modules.reshape.reshape import Reshape
from eeggan.pytorch.modules.scaling.upsampling import CubicUpsampling1d, CubicUpsampling2d
from eeggan.pytorch.modules.sequential import Sequential
from eeggan.pytorch.modules.weights.weight_scaling import weight_scale
from eeggan.training.progressive.discriminator import ProgressiveDiscriminatorBlock
from eeggan.training.progressive.generator import ProgressiveGeneratorBlock


def create_disc_blocks(n_chans, n_time, n_classes):
    def create_conv_sequence(in_filters, out_filters, i_layer, stdmap=False):
        conv_configs = list()
        if i_layer == 6:
            conv_configs.append({'kernel_size': 3, 'padding': 1, 'groups': 60})
            conv_configs.append({'kernel_size': 5, 'padding': 2, 'groups': 60})
        if i_layer == 5:
            conv_configs.append({'kernel_size': 3, 'padding': 1, 'groups': 30})
            conv_configs.append({'kernel_size': 5, 'padding': 2, 'groups': 30})
            conv_configs.append({'kernel_size': 7, 'padding': 3, 'groups': 30})
            conv_configs.append({'kernel_size': 9, 'padding': 4, 'groups': 30})
        if i_layer == 4:
            conv_configs.append({'kernel_size': 3, 'padding': 1, 'groups': 20})
            conv_configs.append({'kernel_size': 5, 'padding': 2, 'groups': 20})
            conv_configs.append({'kernel_size': 7, 'padding': 3, 'groups': 20})
            conv_configs.append({'kernel_size': 9, 'padding': 4, 'groups': 20})
            conv_configs.append({'kernel_size': 11, 'padding': 5, 'groups': 20})
            conv_configs.append({'kernel_size': 13, 'padding': 6, 'groups': 20})
        if i_layer == 3:
            conv_configs.append({'kernel_size': 3, 'padding': 1, 'groups': 15})
            conv_configs.append({'kernel_size': 5, 'padding': 2, 'groups': 15})
            conv_configs.append({'kernel_size': 7, 'padding': 3, 'groups': 15})
            conv_configs.append({'kernel_size': 9, 'padding': 4, 'groups': 15})
            conv_configs.append({'kernel_size': 11, 'padding': 5, 'groups': 15})
            conv_configs.append({'kernel_size': 13, 'padding': 6, 'groups': 15})
            conv_configs.append({'kernel_size': 15, 'padding': 7, 'groups': 15})
            conv_configs.append({'kernel_size': 17, 'padding': 8, 'groups': 15})
        if i_layer == 2:
            conv_configs.append({'kernel_size': 3, 'padding': 1, 'groups': 12})
            conv_configs.append({'kernel_size': 5, 'padding': 2, 'groups': 12})
            conv_configs.append({'kernel_size': 7, 'padding': 3, 'groups': 12})
            conv_configs.append({'kernel_size': 9, 'padding': 4, 'groups': 12})
            conv_configs.append({'kernel_size': 11, 'padding': 5, 'groups': 12})
            conv_configs.append({'kernel_size': 13, 'padding': 6, 'groups': 12})
            conv_configs.append({'kernel_size': 15, 'padding': 7, 'groups': 12})
            conv_configs.append({'kernel_size': 17, 'padding': 8, 'groups': 12})
            conv_configs.append({'kernel_size': 19, 'padding': 9, 'groups': 12})
            conv_configs.append({'kernel_size': 21, 'padding': 10, 'groups': 12})
        if i_layer == 1:
            conv_configs.append({'kernel_size': 3, 'padding': 1, 'groups': 10})
            conv_configs.append({'kernel_size': 5, 'padding': 2, 'groups': 10})
            conv_configs.append({'kernel_size': 7, 'padding': 3, 'groups': 10})
            conv_configs.append({'kernel_size': 9, 'padding': 4, 'groups': 10})
            conv_configs.append({'kernel_size': 11, 'padding': 5, 'groups': 10})
            conv_configs.append({'kernel_size': 13, 'padding': 6, 'groups': 10})
            conv_configs.append({'kernel_size': 15, 'padding': 7, 'groups': 10})
            conv_configs.append({'kernel_size': 17, 'padding': 8, 'groups': 10})
            conv_configs.append({'kernel_size': 19, 'padding': 9, 'groups': 10})
            conv_configs.append({'kernel_size': 21, 'padding': 10, 'groups': 10})
            conv_configs.append({'kernel_size': 23, 'padding': 11, 'groups': 10})
            conv_configs.append({'kernel_size': 25, 'padding': 12, 'groups': 10})

        filters_tmp = out_filters
        tmp_layer = Dummy()
        if stdmap:
            filters_tmp = in_filters + 1
            tmp_layer = StdMap1d()

        return Sequential(
            weight_scale(MultiConv1d(conv_configs, in_filters, out_filters, split_in_channels=True,
                                     reflective=True),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            weight_scale(MultiConv1d(conv_configs, out_filters, out_filters, split_in_channels=True,
                                     reflective=True),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            # tmp_layer,
            weight_scale(nn.Conv1d(out_filters, out_filters, kernel_size=1),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(1),
            weight_scale(nn.Conv1d(out_filters, out_filters, 4, stride=2),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            weight_scale(EmbeddedClassFilter(n_classes, out_filters),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2)
        )

    def create_in_sequence(n_chans, out_filters):
        return Sequential(Permute([0, 2, 1]),
                          Reshape([[0], 1, [1], [2]]),
                          weight_scale(nn.Conv2d(1, out_filters, (1, n_chans)),
                                       gain=calculate_gain('leaky_relu')),
                          Reshape([[0], [1], [2]]),
                          nn.LeakyReLU(0.2))

    def create_fade_sequence(factor):
        return Sequential(Permute([0, 2, 1]),
                          Reshape([[0], 1, [1], [2]]),
                          nn.AvgPool2d((factor, 1), stride=(factor, 1)),
                          Reshape([[0], [2], [3]]),
                          Permute([0, 2, 1]))

    blocks = []
    tmp_block = ProgressiveDiscriminatorBlock(
        create_conv_sequence(120, 120, 1),
        create_in_sequence(n_chans, 120),
        create_fade_sequence(2)
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveDiscriminatorBlock(
        create_conv_sequence(120, 120, 2),
        create_in_sequence(n_chans, 120),
        create_fade_sequence(2)
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveDiscriminatorBlock(
        create_conv_sequence(120, 120, 3),
        create_in_sequence(n_chans, 120),
        create_fade_sequence(2)
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveDiscriminatorBlock(
        create_conv_sequence(120, 120, 4),
        create_in_sequence(n_chans, 120),
        create_fade_sequence(2)
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveDiscriminatorBlock(
        create_conv_sequence(120, 120, 5),
        create_in_sequence(n_chans, 120),
        create_fade_sequence(2)
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveDiscriminatorBlock(
        Sequential(create_conv_sequence(120, 120, 6, stdmap=False),
                   Reshape([[0], 120 * n_time]),
                   weight_scale(nn.Linear(120 * n_time, 1),
                                gain=calculate_gain('linear'))),
        create_in_sequence(n_chans, 120),
        None
    )
    blocks.append(tmp_block)
    return blocks


def create_gen_blocks(n_chans, n_z, n_time, n_classes):
    def create_conv_sequence(in_filters, out_filters, i_layer):
        conv_configs = list()
        if i_layer == 6:
            conv_configs.append({'kernel_size': 3, 'padding': 1, 'groups': 60})
            conv_configs.append({'kernel_size': 5, 'padding': 2, 'groups': 60})
        if i_layer == 5:
            conv_configs.append({'kernel_size': 3, 'padding': 1, 'groups': 30})
            conv_configs.append({'kernel_size': 5, 'padding': 2, 'groups': 30})
            conv_configs.append({'kernel_size': 7, 'padding': 3, 'groups': 30})
            conv_configs.append({'kernel_size': 9, 'padding': 4, 'groups': 30})
        if i_layer == 4:
            conv_configs.append({'kernel_size': 3, 'padding': 1, 'groups': 20})
            conv_configs.append({'kernel_size': 5, 'padding': 2, 'groups': 20})
            conv_configs.append({'kernel_size': 7, 'padding': 3, 'groups': 20})
            conv_configs.append({'kernel_size': 9, 'padding': 4, 'groups': 20})
            conv_configs.append({'kernel_size': 11, 'padding': 5, 'groups': 20})
            conv_configs.append({'kernel_size': 13, 'padding': 6, 'groups': 20})
        if i_layer == 3:
            conv_configs.append({'kernel_size': 3, 'padding': 1, 'groups': 15})
            conv_configs.append({'kernel_size': 5, 'padding': 2, 'groups': 15})
            conv_configs.append({'kernel_size': 7, 'padding': 3, 'groups': 15})
            conv_configs.append({'kernel_size': 9, 'padding': 4, 'groups': 15})
            conv_configs.append({'kernel_size': 11, 'padding': 5, 'groups': 15})
            conv_configs.append({'kernel_size': 13, 'padding': 6, 'groups': 15})
            conv_configs.append({'kernel_size': 15, 'padding': 7, 'groups': 15})
            conv_configs.append({'kernel_size': 17, 'padding': 8, 'groups': 15})
        if i_layer == 2:
            conv_configs.append({'kernel_size': 3, 'padding': 1, 'groups': 12})
            conv_configs.append({'kernel_size': 5, 'padding': 2, 'groups': 12})
            conv_configs.append({'kernel_size': 7, 'padding': 3, 'groups': 12})
            conv_configs.append({'kernel_size': 9, 'padding': 4, 'groups': 12})
            conv_configs.append({'kernel_size': 11, 'padding': 5, 'groups': 12})
            conv_configs.append({'kernel_size': 13, 'padding': 6, 'groups': 12})
            conv_configs.append({'kernel_size': 15, 'padding': 7, 'groups': 12})
            conv_configs.append({'kernel_size': 17, 'padding': 8, 'groups': 12})
            conv_configs.append({'kernel_size': 19, 'padding': 9, 'groups': 12})
            conv_configs.append({'kernel_size': 21, 'padding': 10, 'groups': 12})
        if i_layer == 1:
            conv_configs.append({'kernel_size': 3, 'padding': 1, 'groups': 10})
            conv_configs.append({'kernel_size': 5, 'padding': 2, 'groups': 10})
            conv_configs.append({'kernel_size': 7, 'padding': 3, 'groups': 10})
            conv_configs.append({'kernel_size': 9, 'padding': 4, 'groups': 10})
            conv_configs.append({'kernel_size': 11, 'padding': 5, 'groups': 10})
            conv_configs.append({'kernel_size': 13, 'padding': 6, 'groups': 10})
            conv_configs.append({'kernel_size': 15, 'padding': 7, 'groups': 10})
            conv_configs.append({'kernel_size': 17, 'padding': 8, 'groups': 10})
            conv_configs.append({'kernel_size': 19, 'padding': 9, 'groups': 10})
            conv_configs.append({'kernel_size': 21, 'padding': 10, 'groups': 10})
            conv_configs.append({'kernel_size': 23, 'padding': 11, 'groups': 10})
            conv_configs.append({'kernel_size': 25, 'padding': 12, 'groups': 10})

        return Sequential(
            CubicUpsampling1d(2),
            weight_scale(MultiConv1d(conv_configs, in_filters, out_filters, split_in_channels=True, reflective=True),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            weight_scale(MultiConv1d(conv_configs, out_filters, out_filters, split_in_channels=True, reflective=True),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            weight_scale(nn.Conv1d(out_filters, out_filters, kernel_size=1),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            weight_scale(EmbeddedClassFilter(n_classes, out_filters),
                         gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2),
            PixelNorm())

    def create_out_sequence(n_chans, in_filters):
        return Sequential(weight_scale(nn.Conv1d(in_filters, n_chans, 1),
                                       gain=calculate_gain('linear')),
                          Reshape([[0], [1], [2], 1]),
                          PixelShuffle2d((1, n_chans)),
                          Reshape([[0], [2], [3]]),
                          Permute([0, 2, 1]))

    def create_fade_sequence(factor):
        return Sequential(Permute([0, 2, 1]),
                          Reshape([[0], 1, [1], [2]]),
                          CubicUpsampling2d(factor),
                          Reshape([[0], [2], [3]]),
                          Permute([0, 2, 1]))

    blocks = []
    tmp_block = ProgressiveGeneratorBlock(
        Sequential(weight_scale(nn.Linear(n_z, 120 * n_time),
                                gain=calculate_gain('leaky_relu')),
                   Reshape([[0], 120, -1]),
                   nn.LeakyReLU(0.2),
                   PixelNorm(),
                   create_conv_sequence(120, 120, 6)),
        create_out_sequence(n_chans, 120),
        create_fade_sequence(2)
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveGeneratorBlock(
        create_conv_sequence(120, 120, 5),
        create_out_sequence(n_chans, 120),
        create_fade_sequence(2)
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveGeneratorBlock(
        create_conv_sequence(120, 120, 4),
        create_out_sequence(n_chans, 120),
        create_fade_sequence(2)
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveGeneratorBlock(
        create_conv_sequence(120, 120, 3),
        create_out_sequence(n_chans, 120),
        create_fade_sequence(2)
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveGeneratorBlock(
        create_conv_sequence(120, 120, 2),
        create_out_sequence(n_chans, 120),
        create_fade_sequence(2)
    )
    blocks.append(tmp_block)
    tmp_block = ProgressiveGeneratorBlock(
        create_conv_sequence(120, 120, 1),
        create_out_sequence(n_chans, 120),
        None
    )
    blocks.append(tmp_block)
    return blocks
