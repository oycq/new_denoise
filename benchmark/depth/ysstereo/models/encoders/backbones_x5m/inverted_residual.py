# Copyright (c) Horizon Robotics. All rights reserved.

import torch
import torch.nn as nn

from .conv_module import ConvModule2d

__all__ = ["InvertedResidual"]


class InvertedResidual(nn.Module):
    """
    A module of inverted residual.

    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        stride (int): The stride of depthwise conv.
        expand_ratio (int): Expand ratio of channels.
        bn_kwargs (dict): Dict for BN layers.
        bias (bool): Whether to use bias in module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
        bn_kwargs: dict,
        bias: bool = True,
    ):
        super(InvertedResidual, self).__init__()
        mid_channels = int(in_channels * expand_ratio)
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            ConvModule2d(
                in_channels,
                mid_channels,
                1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(mid_channels, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                mid_channels,
                mid_channels,
                3,
                padding=1,
                stride=stride,
                groups=mid_channels,
                bias=bias,
                norm_layer=nn.BatchNorm2d(mid_channels, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                mid_channels,
                out_channels,
                1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            ),
        )
        self.skip_add = nn.quantized.FloatFunctional()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_shortcut:
            x = self.skip_add.add(self.conv(x), x)
        else:
            x = self.conv(x)
        return self.act(x)
