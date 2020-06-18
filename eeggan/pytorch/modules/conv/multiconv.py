#  Author: Kay Hartmann <kg.hartma@gmail.com>

import torch
import torch.nn as nn

from eeggan.pytorch.modules.module import Module
from eeggan.pytorch.utils.bias import fill_bias_zero
from eeggan.pytorch.utils.weights import fill_weights_normal


class MultiConv1d(Module):
    def __init__(self, conv_configs, in_channels, out_channels, split_in_channels=False, reflective=False):
        super().__init__()
        assert (out_channels % len(conv_configs) == 0)
        self.conv_configs = conv_configs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_channels_per_conv = in_channels
        self.out_channels_per_conv = out_channels / len(conv_configs)
        self.split_in_channels = split_in_channels
        self.reflective = reflective

        if split_in_channels:
            assert (in_channels % len(conv_configs) == 0)
            self.in_channels_per_conv = in_channels / len(conv_configs)

        self.convs = nn.ModuleList()
        for config in conv_configs:
            config = config.copy()
            config['padding'] = 0
            conv = nn.Conv1d(
                int(self.in_channels_per_conv),
                int(self.out_channels_per_conv),
                **config
            )
            fill_weights_normal(conv.weight)
            fill_bias_zero(conv.bias)
            self.convs.append(conv)

    def forward(self, x, **kwargs):
        tmp_outputs = list()
        for i, conv in enumerate(self.convs):
            tmp_input = x
            if self.split_in_channels:
                tmp_input = tmp_input[:, int(i * self.in_channels_per_conv):int(
                    i * self.in_channels_per_conv + self.in_channels_per_conv)]

            if self.reflective:
                tmp_input = nn.functional.pad(tmp_input,
                                              [self.conv_configs[i]['padding'], self.conv_configs[i]['padding']],
                                              mode='reflect')
            else:
                tmp_input = nn.functional.pad(tmp_input,
                                              [self.conv_configs[i]['padding'], self.conv_configs[i]['padding']])
            tmp_outputs.append(conv(tmp_input))

        return torch.cat(tmp_outputs, dim=1)
