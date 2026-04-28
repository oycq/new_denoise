# Copyright (c) Horizon Robotics. All rights reserved.

import torch.nn as nn

from ysstereo.models.encoders.backbones_x5m.conv_module import ConvModule2d
from ysstereo.models.encoders.backbones_x5m.inverted_residual import InvertedResidual

__all__ = ["MobileNetV2"]

BASE_MOBV2_CONFIG = dict(
    include_top=False,
    num_classes=1000,
    bn_kwargs={},
    alpha=1.0,
)

ALPHA0P75_MOBV2_CONFIG = dict(
    include_top=False,
    num_classes=1000,
    bn_kwargs={},
    alpha=0.75,
)

ALPHA0P5_MOBV2_CONFIG = dict(
    include_top=False,
    num_classes=1000,
    bn_kwargs={},
    alpha=0.5,
)

ALPHA0P65_MOBV2_CONFIG = dict(
    include_top=False,
    num_classes=1000,
    bn_kwargs={},
    alpha=0.65,
)

TUNED_MOBV2_CONFIG = dict(
    include_top=False,
    num_classes=1000,
    bn_kwargs={},
    alpha=1.0,
    in_chls = [
        [24],
        [12, 16],
        [16, 24, 24],
        [32] + [48] * 4 + [64, 80],
        [80] + [128] * 3,
    ],
    out_chls = [
        [12],
        [16, 16],
        [24, 24, 32],
        [48] * 4 + [64] * 1 + [80] * 2,
        [128] * 3 + [160],
    ],
)

MOBV2_ARCH_CFGS = dict(
    base=BASE_MOBV2_CONFIG,
    tuned=TUNED_MOBV2_CONFIG,
    alpha0p5=ALPHA0P5_MOBV2_CONFIG,
    alpha0p65=ALPHA0P65_MOBV2_CONFIG,
    alpha0p75=ALPHA0P75_MOBV2_CONFIG,
)

class MobileNetV2(nn.Module):
    """
    A module of mobilenetv2.

    Args:
        num_classes (int): Num classes of output layer.
        bn_kwargs (dict): Dict for BN layer.
        alpha (float): Alpha for mobilenetv2.
        bias (bool): Whether to use bias in module.
        include_top (bool): Whether to include output layer.
        flat_output (bool): Whether to view the output tensor.
        use_dw_as_avgpool (bool): Whether to replace AvgPool with DepthWiseConv
    """

    def __init__(
        self,
        num_classes,
        bn_kwargs: dict,
        alpha: float = 1.0,
        bias: bool = True,
        include_top: bool = True,
        flat_output: bool = True,
        use_dw_as_avgpool: bool = False,
        in_chls = [
            [32],
            [16, 24],
            [24, 32, 32],
            [32] + [64] * 4 + [96] * 2,
            [96] + [160] * 3,
        ],
        out_chls = [
            [16],
            [24, 24],
            [32, 32, 32],
            [64] * 4 + [96] * 3,
            [160] * 3 + [320],
        ],
    ):
        super(MobileNetV2, self).__init__()
        self.alpha = alpha
        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.num_classes = num_classes
        self.include_top = include_top
        self.flat_output = flat_output
        self.use_dw_as_avgpool = use_dw_as_avgpool

        self.mod1 = self._make_stage(in_chls[0], out_chls[0], 1, True)
        self.mod2 = self._make_stage(in_chls[1], out_chls[1], 6)
        self.mod3 = self._make_stage(in_chls[2], out_chls[2], 6)
        self.mod4 = self._make_stage(in_chls[3], out_chls[3], 6)
        self.mod5 = self._make_stage(in_chls[4], out_chls[4], 6)

        if self.use_dw_as_avgpool:
            pool_layer = ConvModule2d(
                in_channels=max(1280, int(1280 * alpha)),
                out_channels=max(1280, int(1280 * alpha)),
                kernel_size=7,
                stride=1,
                padding=0,
                groups=max(1280, int(1280 * alpha)),
            )
        else:
            pool_layer = nn.AvgPool2d(7)

        if self.include_top:
            self.output = nn.Sequential(
                ConvModule2d(
                    int(out_chls[4][-1] * alpha),
                    max(1280, int(1280 * alpha)),
                    1,
                    bias=self.bias,
                    norm_layer=nn.BatchNorm2d(
                        max(1280, int(1280 * alpha)), **bn_kwargs
                    ),
                    act_layer=nn.ReLU(inplace=True),
                ),
                pool_layer,
                ConvModule2d(
                    max(1280, int(1280 * alpha)),
                    num_classes,
                    1,
                    bias=self.bias,
                    norm_layer=nn.BatchNorm2d(num_classes, **bn_kwargs),
                ),
            )
        else:
            self.output = None

    def _make_stage(self, in_chls, out_chls, expand_t, first_layer=False):
        layers = []
        in_chls = [int(chl * self.alpha) for chl in in_chls]
        out_chls = [int(chl * self.alpha) for chl in out_chls]
        for i, in_chl, out_chl in zip(range(len(in_chls)), in_chls, out_chls):
            stride = 2 if i == 0 else 1
            if first_layer:
                layers.append(
                    ConvModule2d(
                        3,
                        in_chls[0],
                        3,
                        stride,
                        1,
                        bias=self.bias,
                        norm_layer=nn.BatchNorm2d(
                            in_chls[0], **self.bn_kwargs
                        ),
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
                stride = 1
            layers.append(
                InvertedResidual(
                    in_chl,
                    out_chl,
                    stride,
                    expand_t,
                    self.bn_kwargs,
                    self.bias,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        output = []
        for module in [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]:
            x = module(x)
            output.append(x)
        if not self.include_top:
            return output
        x = self.output(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x
