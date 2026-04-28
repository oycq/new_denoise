# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


__all__ = [
    "ConvModule2d",
    "ConvTransposeModule2d",
    "ConvUpsample2d",
    "FixedConvModule2d",
    "FusedConv2d",
]


class ConvModule2d(nn.Sequential):
    """
    A conv block that bundles conv/norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool): Same as nn.Conv2d.
        padding_mode (str): Same as nn.Conv2d.
        norm_layer (nn.Module): Normalization layer.
        act_layer (nn.Module): Activation layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        conv_list = [conv, norm_layer, act_layer]
        self.conv_list = [layer for layer in conv_list if layer is not None]
        super(ConvModule2d, self).__init__(*self.conv_list)
        self.has_norm_layer = norm_layer is not None
        self.has_act_layer = act_layer is not None
        self.st_fn = None
        self.cp = False
        if os.environ.get("HAT_USE_CHECKPOINT") is not None:
            self.cp = bool(int(os.environ.get("HAT_USE_CHECKPOINT", "0")))

    def forward(self, x):
        if self.cp and torch.is_tensor(x) and x.requires_grad:
            out = checkpoint.checkpoint(super().forward, x)
        elif self.st_fn is not None and self.training and torch.is_tensor(x):
            out = self.st_fn(x)
        else:
            out = super().forward(x)
        return out


class ConvTransposeModule2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: bool = 1,
        padding_mode: str = "zeros",
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
    ):
        """Transposed convolution, followed by normalization and activation.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (Union[int, Tuple[int, int]]): kernel size.
            stride (Union[int, Tuple[int, int]], optional): conv stride.
                Defaults to 1.
            padding (Union[int, Tuple[int, int]], optional): conv padding.
                dilation * (kernel_size - 1) - padding zero-padding will be
                added to the input. Defaults to 0.
            output_padding (Union[int, Tuple[int, int]], optional):
                additional size added to the output. Defaults to 0.
            groups (int, optional): number of blocked connections from input
                to output. Defaults to 1.
            bias (bool, optional): whether to add learnable bias.
                Defaults to True.
            dilation (bool, optional): kernel dilation. Defaults to 1.
            padding_mode (str, optional): same as conv2d. Defaults to 'zeros'.
            norm_layer (Optional[nn.Module], optional): normalization layer.
                Defaults to None.
            act_layer (Optional[nn.Module], optional): activation layer.
                Defaults to None.
        """
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
        )
        self.norm = norm_layer
        self.act = act_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x

class ConvUpsample2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
    ):
        """Conv upsample module.

        Different from ConvTransposeModule2d, this module does conv2d,
        followed by an upsample layer. The final effect is almost the same,
        but one should pay attention to the output size.

        Args:
            in_channels (int): same as nn.Conv2d.
            out_channels (int): same as nn.Conv2d.
            kernel_size (Union[int, Tuple[int, int]]): same as nn.Conv2d.
            stride (Union[int, Tuple[int, int]], optional): Upsample stride.
                Defaults to 1.
            padding (Union[int, Tuple[int, int]], optional): same as nn.Conv2d.
                Defaults to 0.
            dilation (Union[int, Tuple[int, int]], optional): same as
                nn.Conv2d. Defaults to 1.
            groups (int, optional): same as nn.Conv2d. Defaults to 1.
            bias (bool, optional): same as nn.Conv2d. Defaults to True.
            padding_mode (str, optional): same as nn.Conv2d.
                Defaults to "zeros".
            norm_layer (Optional[nn.Module], optional): normalization layer.
                Defaults to None.
            act_layer (Optional[nn.Module], optional): activation layer.
                Defaults to None.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            1,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.norm = norm_layer
        self.act = act_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        h, w = x.shape[2:]
        x = F.interpolate(x, size=(h*self.stride, w*self.stride), scale_factor=None, mode='bilinear', align_corners=True)
        return x

class FixedConvModule2d(ConvModule2d):
    """
    We use depthwise1x1 + bn + relu to replace nn.BatchNorm2d + nn.Relu.

    depthwise1x1's weight is constant 1, and parameters' requires_grad = False.

    Args:
        in_channels: Same as nn.Conv2d.
        bn_kwargs: Dict for BN layer.
    """

    def __init__(self, in_channels: int, bn_kwargs: dict = None):

        if bn_kwargs is None:
            bn_kwargs = {}

        super(FixedConvModule2d, self).__init__(
            in_channels,
            in_channels,
            kernel_size=1,
            groups=in_channels,
            bias=False,
            norm_layer=nn.BatchNorm2d(in_channels, **bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.conv_list[0].weight, 1)
        self.conv_list[0].weight.requires_grad = False
        if self.conv_list[0].bias is not None:
            nn.init.constant_(self.conv_list[0].bias, 0)
            self.conv_list[0].bias.requires_grad = False

    def forward(self, x):
        return super().forward(x)


class FusedConv2d(nn.Module):
    """
    A conv2d fused with BatchNorm, eltwiseadd and ReLU.

    Args:
        in_channels: In channels of Fused Conv.
        out_channels: Out channels of Fused Conv.
        kernel_size: Kernel size of Fused Conv.
        stride: Stride of Fused Conv.
        group: Group of Fused Conv.
        bias: Whether to use bias in Fused Conv.
        bn_kwargs: Dict for BN layer.
        elementwise: Whether to use elementwise add.
        with_relu: Whether to use relu in Fused Conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        group: int = 1,
        bias: bool = False,
        bn_kwargs: dict = None,
        elementwise: bool = False,
        with_relu: bool = False,
    ):
        super(FusedConv2d, self).__init__()
        if bn_kwargs is None:
            bn_kwargs = {}
        self.elementwise = elementwise
        self.with_relu = with_relu

        self.mod = ConvModule2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=group,
            bias=bias,
            norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
        )

        if with_relu:
            self.relu = nn.ReLU(inplace=True)
        if self.elementwise:
            pass
            # self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, data, edata=None):
        output = self.mod(data)
        if self.elementwise:
            # output = self.skip_add.add(output, edata)
            output = output + edata
        if self.with_relu:
            output = self.relu(output)
        return output
