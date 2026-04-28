# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import math

import torch
import torch.nn as nn
from torch.nn.quantized import FloatFunctional

from .conv_module import ConvModule2d

__all__ = ["SEBlock", "MBConvBlock"]


class SEBlock(nn.Module):
    """Basic Squeeze-and-Excitation block for EfficientNet.

    Args:
        in_channels: Input channels.
        num_squeezed_channels: Squeezed channels.
        out_channels: Output channels.
        act_layer: Config dict for activation layer.
        adapt_pooling: For historical reason, please set True to make it align
            with standard SEBlock

    """

    def __init__(
        self,
        in_channels: int,
        num_squeezed_channels: int,
        out_channels: int,
        act_layer: torch.nn.Module,
        adapt_pooling: bool = False,
    ):
        super(SEBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.AvgPool2d(1) if not adapt_pooling else nn.AdaptiveAvgPool2d(1),
            ConvModule2d(
                in_channels=in_channels,
                out_channels=num_squeezed_channels,
                kernel_size=1,
                bias=True,
                norm_layer=None,
                act_layer=act_layer,
            ),
            ConvModule2d(
                in_channels=num_squeezed_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=True,
                norm_layer=None,
                act_layer=nn.Sigmoid(),
            ),
        )
        self.float_func = FloatFunctional()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.float_func.mul(x, inputs)

        return x

    def fuse_model(self):
        getattr(self.conv, "1").fuse_model()


class MBConvBlock(nn.Module):
    """Basic MBConvBlock for EfficientNet.

    Args:
        block_args (dict): Dict for block parameters.
        bn_kwargs (dict): Dict for Bn layer.
        act_layer (torch.nn.Module): activation layer.
        use_se_block (bool): Whether to use SEBlock in module.
        split_expand_conv (bool): Whether split expand conv into two conv. Set
            to true when expand conv is too large to deploy on xj3.
    """

    def __init__(
        self,
        block_args: dict,
        bn_kwargs: dict,
        act_layer: torch.nn.Module,
        use_se_block: bool = False,
        split_expand_conv: bool = False,
    ):
        super(MBConvBlock, self).__init__()

        self._block_args = block_args
        self.expand_ratio = self._block_args.expand_ratio
        self.in_planes = self._block_args.in_filters
        self.out_planes = self.in_planes * self.expand_ratio
        self.final_out_planes = self._block_args.out_filters

        self.kernel_size = self._block_args.kernel_size
        self.stride = self._block_args.strides
        self.id_skip = self._block_args.id_skip
        self.split_expand_conv = split_expand_conv
        if split_expand_conv:
            self.out_planes1 = self.out_planes // 2
            self.out_planes2 = self.out_planes - self.out_planes1

        self.has_se = (
            use_se_block
            and (self._block_args.se_ratio is not None)
            and (0 < self._block_args.se_ratio <= 1)
        )

        if self.expand_ratio != 1:
            # only conv of which the channel <= 2048 is supported by xj3
            if not self.split_expand_conv or self.out_planes <= 2048:
                self._expand_conv = ConvModule2d(
                    in_channels=self.in_planes,
                    out_channels=self.out_planes,
                    kernel_size=1,
                    bias=False,
                    norm_layer=nn.SyncBatchNorm(self.out_planes, **bn_kwargs),
                    act_layer=copy.deepcopy(act_layer),
                )
            else:
                self._expand_conv1 = ConvModule2d(
                    in_channels=self.in_planes,
                    out_channels=self.out_planes1,
                    kernel_size=1,
                    bias=False,
                    norm_layer=nn.SyncBatchNorm(self.out_planes1, **bn_kwargs),
                    act_layer=copy.deepcopy(act_layer),
                )
                self._expand_conv2 = ConvModule2d(
                    in_channels=self.in_planes,
                    out_channels=self.out_planes2,
                    kernel_size=1,
                    bias=False,
                    norm_layer=nn.SyncBatchNorm(self.out_planes2, **bn_kwargs),
                    act_layer=copy.deepcopy(act_layer),
                )

        # only conv of which the channel <= 2048 is supported by xj3
        if not self.split_expand_conv or self.out_planes <= 2048:
            self._depthwise_conv = ConvModule2d(
                in_channels=self.out_planes,
                out_channels=self.out_planes,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=math.ceil((self.kernel_size - 1) // 2)
                if self.stride == 1
                else math.ceil(self.kernel_size // self.stride),
                bias=False,
                groups=self.out_planes,
                norm_layer=nn.SyncBatchNorm(self.out_planes, **bn_kwargs),
                act_layer=copy.deepcopy(act_layer),
            )
            self._project_conv = ConvModule2d(
                in_channels=self.out_planes,
                out_channels=self.final_out_planes,
                kernel_size=1,
                bias=False,
                norm_layer=nn.SyncBatchNorm(self.final_out_planes, **bn_kwargs),
                act_layer=None,
            )
        else:
            self._depthwise_conv1 = ConvModule2d(
                in_channels=self.out_planes1,
                out_channels=self.out_planes1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=math.ceil((self.kernel_size - 1) // 2)
                if self.stride == 1
                else math.ceil(self.kernel_size // self.stride),
                bias=False,
                groups=self.out_planes1,
                norm_layer=nn.SyncBatchNorm(self.out_planes1, **bn_kwargs),
                act_layer=copy.deepcopy(act_layer),
            )
            self._depthwise_conv2 = ConvModule2d(
                in_channels=self.out_planes2,
                out_channels=self.out_planes2,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=math.ceil((self.kernel_size - 1) // 2)
                if self.stride == 1
                else math.ceil(self.kernel_size // self.stride),
                bias=False,
                groups=self.out_planes2,
                norm_layer=nn.SyncBatchNorm(self.out_planes2, **bn_kwargs),
                act_layer=copy.deepcopy(act_layer),
            )
            self._project_conv1 = ConvModule2d(
                in_channels=self.out_planes1,
                out_channels=self.final_out_planes,
                kernel_size=1,
                bias=False,
                norm_layer=nn.SyncBatchNorm(self.final_out_planes, **bn_kwargs),
                act_layer=None,
            )
            self._project_conv2 = ConvModule2d(
                in_channels=self.out_planes2,
                out_channels=self.final_out_planes,
                kernel_size=1,
                bias=False,
                norm_layer=nn.SyncBatchNorm(self.final_out_planes, **bn_kwargs),
                act_layer=None,
            )
            self.float_func_proj_conv = FloatFunctional()

        if self.has_se:
            num_squeezed_num = max(
                1, int(self.in_planes * self._block_args.se_ratio)
            )
            self._se_block = SEBlock(
                in_channels=self.out_planes,
                num_squeezed_channels=num_squeezed_num,
                out_channels=self.out_planes,
                act_layer=copy.deepcopy(act_layer),
            )

        self.use_shortcut = (
            self.id_skip
            and self.stride == 1
            and self.in_planes == self.final_out_planes
        )

        if self.use_shortcut:
            self.float_func = FloatFunctional()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs

        if self.expand_ratio != 1:
            # only conv of which the channel <= 2048 is supported by xj3
            if not self.split_expand_conv or self.out_planes <= 2048:
                x = self._expand_conv(inputs)
            else:
                x1 = self._expand_conv1(inputs)
                x2 = self._expand_conv2(inputs)

        # only conv of which the channel <= 2048 is supported by xj3
        if not self.split_expand_conv or self.out_planes <= 2048:
            x = self._depthwise_conv(x)
        else:
            x1 = self._depthwise_conv1(x1)
            x2 = self._depthwise_conv2(x2)

        if self.has_se:
            x = self._se_block(x)

        # only conv of which the channel <= 2048 is supported by xj3
        if not self.split_expand_conv or self.out_planes <= 2048:
            x = self._project_conv(x)
            if self.use_shortcut:
                if drop_connect_rate:
                    x = self.drop_connect(x, drop_connect_rate)
                x = self.float_func.add(x, inputs)
        else:
            x1 = self._project_conv1(x1)
            x2 = self._project_conv2(x2)
            if self.use_shortcut:
                if drop_connect_rate:
                    x1 = self.drop_connect(x1, drop_connect_rate)
                x1 = self.float_func.add(x1, inputs)
                x = self.float_func_proj_conv.add(x2, x1)

        return x

    def drop_connect(self, x, drop_connect_rate):
        if self.training is True:
            keep_prob = 1.0 - drop_connect_rate
            batch_size = x.shape[0]
            random_tensor = keep_prob
            x_dtype = x.dtype
            random_tensor += torch.rand(
                [batch_size, 1, 1, 1],
                dtype=x_dtype,
                device=x.device,
            )
            binary_mask = torch.floor(random_tensor)
            x = (x / keep_prob) * binary_mask
        return x


class MBConvBlock_prune(nn.Module):
    """Basic MBConvBlock for EfficientNet.

    Args:
        block_args (dict): Dict for block parameters.
        bn_kwargs (dict): Dict for Bn layer.
        act_layer (torch.nn.Module): activation layer.
        use_se_block (bool): Whether to use SEBlock in module.
        split_expand_conv (bool): Whether split expand conv into two conv. Set
            to true when expand conv is too large to deploy on xj3.
    """

    def __init__(
        self,
        block_args: dict,
        bn_kwargs: dict,
        act_layer: torch.nn.Module,
        prune_channels: int,
        use_se_block: bool = False,
        split_expand_conv: bool = False,
    ):
        super(MBConvBlock_prune, self).__init__()

        self._block_args = block_args
        self.expand_ratio = self._block_args.expand_ratio
        self.in_planes = self._block_args.in_filters
        #self.out_planes = self.in_planes * self.expand_ratio
        self.out_planes = prune_channels
        self.final_out_planes = self._block_args.out_filters

        self.kernel_size = self._block_args.kernel_size
        self.stride = self._block_args.strides
        self.id_skip = self._block_args.id_skip
        self.split_expand_conv = split_expand_conv
        if split_expand_conv:
            self.out_planes1 = self.out_planes // 2
            self.out_planes2 = self.out_planes - self.out_planes1

        self.has_se = (
            use_se_block
            and (self._block_args.se_ratio is not None)
            and (0 < self._block_args.se_ratio <= 1)
        )

        if self.expand_ratio != 1:
            # only conv of which the channel <= 2048 is supported by xj3
            if not self.split_expand_conv or self.out_planes <= 2048:
                self._expand_conv = ConvModule2d(
                    in_channels=self.in_planes,
                    out_channels=self.out_planes,
                    kernel_size=1,
                    bias=False,
                    norm_layer=nn.SyncBatchNorm(self.out_planes, **bn_kwargs),
                    act_layer=copy.deepcopy(act_layer),
                )
            else:
                self._expand_conv1 = ConvModule2d(
                    in_channels=self.in_planes,
                    out_channels=self.out_planes1,
                    kernel_size=1,
                    bias=False,
                    norm_layer=nn.SyncBatchNorm(self.out_planes1, **bn_kwargs),
                    act_layer=copy.deepcopy(act_layer),
                )
                self._expand_conv2 = ConvModule2d(
                    in_channels=self.in_planes,
                    out_channels=self.out_planes2,
                    kernel_size=1,
                    bias=False,
                    norm_layer=nn.SyncBatchNorm(self.out_planes2, **bn_kwargs),
                    act_layer=copy.deepcopy(act_layer),
                )

        # only conv of which the channel <= 2048 is supported by xj3
        if not self.split_expand_conv or self.out_planes <= 2048:
            self._depthwise_conv = ConvModule2d(
                in_channels=self.out_planes,
                out_channels=self.out_planes,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=math.ceil((self.kernel_size - 1) // 2)
                if self.stride == 1
                else math.ceil(self.kernel_size // self.stride),
                bias=False,
                groups=self.out_planes,
                norm_layer=nn.SyncBatchNorm(self.out_planes, **bn_kwargs),
                act_layer=copy.deepcopy(act_layer),
            )
            self._project_conv = ConvModule2d(
                in_channels=self.out_planes,
                out_channels=self.final_out_planes,
                kernel_size=1,
                bias=False,
                norm_layer=nn.SyncBatchNorm(self.final_out_planes, **bn_kwargs),
                act_layer=None,
            )
        else:
            self._depthwise_conv1 = ConvModule2d(
                in_channels=self.out_planes1,
                out_channels=self.out_planes1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=math.ceil((self.kernel_size - 1) // 2)
                if self.stride == 1
                else math.ceil(self.kernel_size // self.stride),
                bias=False,
                groups=self.out_planes1,
                norm_layer=nn.SyncBatchNorm(self.out_planes1, **bn_kwargs),
                act_layer=copy.deepcopy(act_layer),
            )
            self._depthwise_conv2 = ConvModule2d(
                in_channels=self.out_planes2,
                out_channels=self.out_planes2,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=math.ceil((self.kernel_size - 1) // 2)
                if self.stride == 1
                else math.ceil(self.kernel_size // self.stride),
                bias=False,
                groups=self.out_planes2,
                norm_layer=nn.SyncBatchNorm(self.out_planes2, **bn_kwargs),
                act_layer=copy.deepcopy(act_layer),
            )
            self._project_conv1 = ConvModule2d(
                in_channels=self.out_planes1,
                out_channels=self.final_out_planes,
                kernel_size=1,
                bias=False,
                norm_layer=nn.SyncBatchNorm(self.final_out_planes, **bn_kwargs),
                act_layer=None,
            )
            self._project_conv2 = ConvModule2d(
                in_channels=self.out_planes2,
                out_channels=self.final_out_planes,
                kernel_size=1,
                bias=False,
                norm_layer=nn.SyncBatchNorm(self.final_out_planes, **bn_kwargs),
                act_layer=None,
            )
            self.float_func_proj_conv = FloatFunctional()

        if self.has_se:
            num_squeezed_num = max(
                1, int(self.in_planes * self._block_args.se_ratio)
            )
            self._se_block = SEBlock(
                in_channels=self.out_planes,
                num_squeezed_channels=num_squeezed_num,
                out_channels=self.out_planes,
                act_layer=copy.deepcopy(act_layer),
            )

        self.use_shortcut = (
            self.id_skip
            and self.stride == 1
            and self.in_planes == self.final_out_planes
        )

        if self.use_shortcut:
            self.float_func = FloatFunctional()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs

        if self.expand_ratio != 1:
            # only conv of which the channel <= 2048 is supported by xj3
            if not self.split_expand_conv or self.out_planes <= 2048:
                x = self._expand_conv(inputs)
            else:
                x1 = self._expand_conv1(inputs)
                x2 = self._expand_conv2(inputs)

        # only conv of which the channel <= 2048 is supported by xj3
        if not self.split_expand_conv or self.out_planes <= 2048:
            x = self._depthwise_conv(x)
        else:
            x1 = self._depthwise_conv1(x1)
            x2 = self._depthwise_conv2(x2)

        if self.has_se:
            x = self._se_block(x)

        # only conv of which the channel <= 2048 is supported by xj3
        if not self.split_expand_conv or self.out_planes <= 2048:
            x = self._project_conv(x)
            if self.use_shortcut:
                if drop_connect_rate:
                    x = self.drop_connect(x, drop_connect_rate)
                x = self.float_func.add(x, inputs)
        else:
            x1 = self._project_conv1(x1)
            x2 = self._project_conv2(x2)
            if self.use_shortcut:
                if drop_connect_rate:
                    x1 = self.drop_connect(x1, drop_connect_rate)
                x1 = self.float_func.add(x1, inputs)
                x = self.float_func_proj_conv.add(x2, x1)

        return x

    def drop_connect(self, x, drop_connect_rate):
        if self.training is True:
            keep_prob = 1.0 - drop_connect_rate
            batch_size = x.shape[0]
            random_tensor = keep_prob
            x_dtype = x.dtype
            random_tensor += torch.rand(
                [batch_size, 1, 1, 1],
                dtype=x_dtype,
                device=x.device,
            )
            binary_mask = torch.floor(random_tensor)
            x = (x / keep_prob) * binary_mask
        return x
