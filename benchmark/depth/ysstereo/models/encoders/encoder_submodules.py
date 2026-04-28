from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModel

from ysstereo.models.builder import ENCODERS, COMPONENTS
from ysstereo.models.utils import BasicBlock, Bottleneck, ResLayer, DwiseResidual, InvertedResidual


def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


@ENCODERS.register_module()
class MultiBasicEncoder(BaseModel):
    """The feature extraction sub-module in RAFTStereo.

    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        num_layers (int): Number of layers. Defaults to 2.
        net_type (str): The type of this sub-module, if net_type is Basic, the
            residual block is BasicBlock, if net_type is Small, the residual
            block is Bottleneck. Defaults to 'Basic'.
        stem_channels (int, optional): Number of stem channels. If
            stem_channels is None, it will be set based on net_type. If the
            net_type is Basic, the stem_channels is 64, otherwise the
            stem_channels is 32. Defaults to None.
        base_channels (Sequence[int], optional):  Number of base channels of
            res layer. If base_channels is None, it will be set based on
            net_type. If the net_type is Basic, the base_channels is
            (64, 96, 128), otherwise the base_channels is (8, 16, 24).
            Defaults to None.
        num_stages (int, optional): Resnet stages, if it is None, set
            num_stages as length of base_channels. Defaults to None.
        strides (Sequence[int], optional): Strides of the first block of each
            stage. If it is None, it will be (1, 2, 2). Defaults to None.
        dilations (Sequence[int], optional): Dilation of each stage. If it is
            None, it will be (1, 1, 1). Defaults to None.
        deep_stem (bool): Whether Replace 7x7 conv in input stem with 3 3x3
            conv. Defaults to False.
        avg_down (bool): Whether use AvgPool instead of stride conv when
            downsampling in the bottleneck. Defaults to False.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None..
        norm_cfg (dict, optional): Config dict for each normalization layer.
            Defaults to dict(type='BN', requires_grad=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, list, optional): Config of weights initialization.
            Default: None.
    """

    _arch_settings = {
        'Basic': (BasicBlock, (2, 2, 2, 2)),
        'Small': (Bottleneck, (2, 2, 2, 2))
    }

    _stem_channels = {'Basic': 16, 'Small': 32}

    _base_channels = {'Basic': (8, 12, 16, 16), 'Small': (8, 16, 24, 24)}

    _strides = {'Basic': (1, 2, 2, 2), 'Small': (1, 2, 2, 2)}

    _dilations = {'Basic': (1, 1, 1, 1), 'Small': (1, 1, 1, 1)}

    def __init__(self,
                 in_channels: int,
                 num_layers: int,
                 net_type: str = 'Basic',
                 stem_channels: Optional[int] = None,
                 base_channels: Optional[Sequence[int]] = None,
                 num_stages: Optional[int] = None,
                 strides: Optional[Sequence[int]] = None,
                 dilations: Optional[Sequence[int]] = None,
                 deep_stem: bool = False,
                 avg_down: bool = False,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: dict = dict(type='BN', requires_grad=True),
                 norm_eval: bool = False,
                 with_cp: bool = False,
                 init_cfg: Optional[Union[dict, list]] = None) -> None:

        super().__init__(init_cfg=init_cfg)

        if net_type not in self._arch_settings:
            raise KeyError(f'invalid net type {net_type} for RAFTStereo')
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.stem_channels = (
            stem_channels
            if stem_channels is not None else self._stem_channels[net_type])
        self.base_channels = (
            base_channels
            if base_channels is not None else self._base_channels[net_type])
        self.num_stages = (
            num_stages
            if num_stages is not None else len(self._base_channels[net_type]))
        assert self.num_stages >= 1 and self.num_stages <= 5
        self.strides = (
            strides if strides is not None else self._strides[net_type])
        self.dilations = (
            dilations if dilations is not None else self._dilations[net_type])
        assert len(self.strides) == len(self.dilations) == self.num_stages

        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.block, stage_blocks = self._arch_settings[net_type]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = self.stem_channels

        self._make_stem_layer(self.in_channels, self.stem_channels)

        self.res_layers = []

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]

            planes = self.base_channels[i]
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes
            layer_name = f'res_layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        output_list = []
        for i in range(self.num_layers):
            outputs_layer = self.make_outputs_layer(self.stem_channels, self.stem_channels)
            output_list.append(outputs_layer)
        self.outputs08 = nn.ModuleList(output_list)
        self.outputs16 = nn.ModuleList(output_list)

        # last_channel = self.base_channels[-1]
        # self.conv2 = build_conv_layer(
        #     self.conv_cfg, last_channel, out_channels, kernel_size=3)

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=True)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)

    def make_outputs_layer(self, in_channels: int, out_channels: int) -> None:
        """Make stem layer for ResNet."""
        return nn.Sequential(
            ResLayer(
                block=self.block,
                inplanes=in_channels,
                planes=in_channels,
                num_blocks=1,
            ),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

    def make_res_layer(self, **kwargs) -> torch.nn.Module:
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self) -> torch.nn.Module:
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): Input feature.

        Returns:
            torch.Tensor: Output feature.
        """
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        for i in range(3):
            layer_name = f'res_layer{i + 1}'
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        v = x
        x = x[:(x.shape[0] // 2)]

        outputs08 = [f(x) for f in self.outputs08]

        res_layer4 = getattr(self, 'res_layer4')
        y = res_layer4(x)
        outputs16 = [f(y) for f in self.outputs16]

        return (outputs08, outputs16, v)


@COMPONENTS.register_module()
class Conv2(BaseModel):
    """The feature extraction sub-module in RAFTStereo.

    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        out_channels (int): Number of output channels. Defaults to 128.
        hidden_channels (int): Number of hidden channels. Defaults to 128.
        init_cfg (dict, list, optional): Config of weights initialization.
            Default: None.
    """

    _arch_settings = {
        'Basic': BasicBlock,
        'Small': Bottleneck
    }

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 block_type: str = 'Basic',
                 init_cfg: Optional[Union[dict, list]] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.block = self._arch_settings[block_type]
        self.layer = self.make_layer(in_channels, hidden_channels, out_channels)

    def make_layer(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        """Make stem layer for ResNet."""
        return nn.Sequential(
            ResLayer(
                block=self.block,
                inplanes=in_channels,
                planes=hidden_channels,
                num_blocks=1,
            ),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): Input feature.

        Returns:
            torch.Tensor: Output feature.
        """
        out = self.layer(x)

        return out


@ENCODERS.register_module()
class BasicEncoder(BaseModel):
    _arch_settings = {
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck,
        'InvertedResidual': InvertedResidual,
    }

    def __init__(self,
                 in_channels: int,
                 stem_channels: int,
                 out_channels: int,
                 block_name: str,
                 num_blocks: Optional[Sequence[int]] = None,
                 strides: Optional[Sequence[int]] = None,
                 base_channels: Optional[Sequence[int]] = None,
                 num_stages: Optional[int] = None,
                 deep_stem: bool = False,
                 dilations: Optional[Sequence[int]] = (1, 1, 1),
                 conv_cfg: Optional[dict] = None,
                 with_cp: bool = False,
                 norm_cfg: dict = dict(type='BN', requires_grad=True),
                 init_cfg: Optional[Union[dict, list]] = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.stem_channels = stem_channels
        self.strides = strides
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.deep_stem = deep_stem
        self.block = self._arch_settings[block_name]
        self.num_blocks = num_blocks
        self.dilations = dilations
        self.inplanes = self.stem_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg

        self._make_stem_layer(self.in_channels, self.stem_channels)

        self.res_layers = []

        for i in range(self.num_stages):
            stride = self.strides[i]
            dilation = self.dilations[i]

            planes = self.base_channels[i]
            res_layer = self._make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=self.num_blocks[i],
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes
            layer_name = f'res_layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.out_conv = self._make_outputs_layer(self.inplanes, self.out_channels)

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=True)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)

    def _make_res_layer(self, **kwargs) -> torch.nn.Module:
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    def _make_outputs_layer(self, in_channels: int, out_channels: int) -> torch.nn.Module:
        """Make stem layer for ResNet."""
        return nn.Conv2d(in_channels, out_channels, 1)

    @property
    def norm1(self) -> torch.nn.Module:
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def forward(self, x, dual_inp=False):

        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)

        for i in range(3):
            layer_name = f'res_layer{i + 1}'
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

        x = self.out_conv(x)

        return x


@ENCODERS.register_module()
class NewBasicEncoder(BaseModel):
    _arch_settings = {
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck,
        'InvertedResidual': InvertedResidual,
    }

    def __init__(self,
                 in_channels: int,
                 stem_channels: int,
                 out_channels: int,
                 block_name: str,
                 num_blocks: Optional[Sequence[int]] = None,
                 strides: Optional[Sequence[int]] = None,
                 base_channels: Optional[Sequence[int]] = None,
                 num_stages: Optional[int] = None,
                 deep_stem: bool = False,
                 dilations: Optional[Sequence[int]] = (1, 1, 1),
                 conv_cfg: Optional[dict] = None,
                 with_cp: bool = False,
                 norm_cfg: dict = dict(type='BN', requires_grad=True),
                 init_cfg: Optional[Union[dict, list]] = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.stem_channels = stem_channels
        self.strides = strides
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.deep_stem = deep_stem
        self.block = self._arch_settings[block_name]
        self.num_blocks = num_blocks
        self.dilations = dilations
        self.inplanes = self.stem_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg

        self._make_stem_layer(self.in_channels, self.stem_channels)

        self.res_layers = []

        for i in range(len(self.base_channels)):
            stride = self.strides[i]
            dilation = self.dilations[i]

            planes = self.base_channels[i]
            res_layer = self._make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=self.num_blocks[i],
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes
            layer_name = f'res_layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        for i in range(self.num_stages):
            out_conv = self._make_outputs_layer(self.base_channels[-i-1], self.out_channels)
            layer_name = f'out_conv{i}'
            self.add_module(layer_name, out_conv)

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=True)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)

    def _make_res_layer(self, **kwargs) -> torch.nn.Module:
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    def _make_outputs_layer(self, in_channels: int, out_channels: int) -> torch.nn.Module:
        """Make stem layer for ResNet."""
        return nn.Conv2d(in_channels, out_channels, 1)

    @property
    def norm1(self) -> torch.nn.Module:
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def forward(self, x, dual_inp=False):
        feats = []
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        feats.append(x)
        for i in range(len(self.base_channels)):
            layer_name = f'res_layer{i + 1}'
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            feats.append(x)
        
        out_feats = []
        for i in range(self.num_stages):
            layer_name = f'out_conv{i}'
            out_conv = getattr(self, layer_name)
            out_feats.append(out_conv(feats[-i-1]))

        return out_feats


@ENCODERS.register_module()
class NewUBasicEncoder(BaseModel):
    _arch_settings = {
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck,
        'InvertedResidual': InvertedResidual,
    }

    def __init__(self,
                 in_channels: int,
                 stem_channels: int,
                 out_channels: int,
                 block_name: str,
                 num_blocks: Optional[Sequence[int]] = None,
                 strides: Optional[Sequence[int]] = None,
                 base_channels: Optional[Sequence[int]] = None,
                 num_stages: Optional[int] = None,
                 deep_stem: bool = False,
                 dilations: Optional[Sequence[int]] = (1, 1, 1, 1, 1),
                 conv_cfg: Optional[dict] = None,
                 with_cp: bool = False,
                 norm_cfg: dict = dict(type='BN', requires_grad=True),
                 init_cfg: Optional[Union[dict, list]] = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.stem_channels = stem_channels
        self.strides = strides
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.deep_stem = deep_stem
        self.block = self._arch_settings[block_name]
        self.num_blocks = num_blocks
        self.dilations = dilations
        self.inplanes = self.stem_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg

        self._make_stem_layer(self.in_channels, self.stem_channels)

        self.res_layers = []

        for i in range(len(self.base_channels)):
            stride = self.strides[i]
            dilation = self.dilations[i]

            planes = self.base_channels[i]
            res_layer = self._make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=self.num_blocks[i],
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes
            layer_name = f'res_layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.inplanes = 0
        for i in range(len(self.base_channels)-1):
            stride = 1
            dilation = 1

            inplanes = self.base_channels[-1 - i] + self.inplanes
            planes = self.base_channels[-1 - i]
            res_layer = self._make_res_layer(
                block=self.block,
                inplanes=inplanes,
                planes=planes,
                num_blocks=1,
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes
            layer_name = f'res_layer{i + 1 + len(self.base_channels)}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        for i in range(self.num_stages):
            out_conv = self._make_outputs_layer(self.base_channels[2-i], self.out_channels)
            layer_name = f'out_conv{i}'
            self.add_module(layer_name, out_conv)

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=True)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)

    def _make_res_layer(self, **kwargs) -> torch.nn.Module:
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    def _make_outputs_layer(self, in_channels: int, out_channels: int) -> torch.nn.Module:
        """Make stem layer for ResNet."""
        return nn.Conv2d(in_channels, out_channels, 1)

    @property
    def norm1(self) -> torch.nn.Module:
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def forward(self, x, dual_inp=False):

        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)

        feats = []
        for i in range(len(self.base_channels)):
            layer_name = f'res_layer{i + 1}'
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            feats.append(x)

        dec_feats = []
        for i in range(len(self.base_channels)-1):
            if i == 0:
                x = feats[-1]
            else:
                x = torch.cat([x, feats[-1 - i]], dim=1)

            layer_name = f'res_layer{i + 1 + len(self.base_channels)}'
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            dec_feats.append(x)
            if i != len(self.base_channels)-2:
                x = interp(x, feats[-2 - i])

        out_feats = []
        for i in range(self.num_stages):
            layer_name = f'out_conv{i}'
            out_conv = getattr(self, layer_name)
            out_feats.append(out_conv(dec_feats[i-self.num_stages]))

        return out_feats




@ENCODERS.register_module()
class MobileEncoder(BaseModel):
    _arch_settings = {
        'DwiseResidual': DwiseResidual,
        'InvertedResidual': InvertedResidual,
    }

    def __init__(self,
                 in_channels: int,
                 stem_channels: int,
                 out_channels: int,
                 block_name: str,
                 num_blocks: Optional[Sequence[int]] = None,
                 strides: Optional[Sequence[int]] = None,
                 base_channels: Optional[Sequence[int]] = None,
                 num_stages: Optional[int] = None,
                 deep_stem: bool = False,
                 dilations: Optional[Sequence[int]] = (1, 1, 1),
                 conv_cfg: Optional[dict] = None,
                 with_cp: bool = False,
                 norm_cfg: dict = dict(type='BN', requires_grad=True),
                 init_cfg: Optional[Union[dict, list]] = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.stem_channels = stem_channels
        self.strides = strides
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.deep_stem = deep_stem
        self.block = self._arch_settings[block_name]
        self.num_blocks = num_blocks
        self.dilations = dilations
        self.inplanes = self.stem_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg

        self._make_stem_layer(self.in_channels, self.stem_channels)

        self.res_layers = []

        for i in range(self.num_stages):
            stride = self.strides[i]
            dilation = self.dilations[i]

            planes = self.base_channels[i]
            res_layer = self._make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=self.num_blocks[i],
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes
            layer_name = f'res_layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.out_conv = self._make_outputs_layer(self.inplanes, self.out_channels)

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                InvertedResidual(in_channels,
                                 stem_channels,
                                 stride=2,
                                 expand_ratio=3),
                nn.ReLU6(inplace=True),
                InvertedResidual(stem_channels,
                                 stem_channels,
                                 stride=1,
                                 expand_ratio=3),
                nn.ReLU6(inplace=True),
                InvertedResidual(stem_channels,
                                 stem_channels,
                                 stride=1,
                                 expand_ratio=3),
                nn.ReLU6(inplace=True))
        else:
            self.stem = InvertedResidual(
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                expand_ratio=3)

    def _make_res_layer(self, **kwargs) -> torch.nn.Module:
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    def _make_outputs_layer(self, in_channels: int, out_channels: int) -> torch.nn.Module:
        """Make stem layer for ResNet."""
        return nn.Conv2d(in_channels, out_channels, 1)

    @property
    def norm1(self) -> torch.nn.Module:
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def forward(self, x, dual_inp=False):
        x = self.stem(x)

        for i in range(3):
            layer_name = f'res_layer{i + 1}'
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

        x = self.out_conv(x)

        return x


@ENCODERS.register_module()
class MobileUBasicEncoder(BaseModel):
    _arch_settings = {
        'DwiseResidual': DwiseResidual,
        'InvertedResidual': InvertedResidual,
    }

    def __init__(self,
                 in_channels: int,
                 stem_channels: int,
                 out_channels: int,
                 block_name: str,
                 num_blocks: Optional[Sequence[int]] = None,
                 strides: Optional[Sequence[int]] = None,
                 base_channels: Optional[Sequence[int]] = None,
                 num_stages: Optional[int] = None,
                 deep_stem: bool = False,
                 dilations: Optional[Sequence[int]] = (1, 1, 1, 1, 1),
                 conv_cfg: Optional[dict] = None,
                 with_cp: bool = False,
                 norm_cfg: dict = dict(type='BN', requires_grad=True),
                 init_cfg: Optional[Union[dict, list]] = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.stem_channels = stem_channels
        self.strides = strides
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.deep_stem = deep_stem
        self.block = self._arch_settings[block_name]
        self.num_blocks = num_blocks
        self.dilations = dilations
        self.inplanes = self.stem_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg

        self._make_stem_layer(self.in_channels, self.stem_channels)

        self.res_layers = []

        for i in range(len(self.base_channels)):
            stride = self.strides[i]
            dilation = self.dilations[i]

            planes = self.base_channels[i]
            res_layer = self._make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=self.num_blocks[i],
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes
            layer_name = f'res_layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.inplanes = 0
        for i in range(self.num_stages):
            stride = 1
            dilation = 1

            inplanes = self.base_channels[-1 - i] + self.inplanes
            planes = self.base_channels[-1 - i]
            res_layer = self._make_res_layer(
                block=self.block,
                inplanes=inplanes,
                planes=planes,
                num_blocks=1,
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes
            layer_name = f'res_layer{i + 1 + len(self.base_channels)}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.out_conv = self._make_outputs_layer(self.inplanes, self.out_channels)

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                InvertedResidual(in_channels,
                                 stem_channels // 2,
                                 stride=2,
                                 expand_ratio=3),
                nn.ReLU6(inplace=True),
                InvertedResidual(stem_channels // 2,
                                 stem_channels // 2,
                                 stride=1,
                                 expand_ratio=3),
                nn.ReLU6(inplace=True),
                InvertedResidual(stem_channels // 2,
                                 stem_channels,
                                 stride=1,
                                 expand_ratio=3),
                nn.ReLU6(inplace=True))
        else:
            self.conv1 = InvertedResidual(
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                expand_ratio=3)
            self.relu = nn.ReLU6(inplace=True)

    def _make_res_layer(self, **kwargs) -> torch.nn.Module:
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    def _make_outputs_layer(self, in_channels: int, out_channels: int) -> torch.nn.Module:
        """Make stem layer for ResNet."""
        return nn.Conv2d(in_channels, out_channels, 1)

    @property
    def norm1(self) -> torch.nn.Module:
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def forward(self, x, dual_inp=False):

        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.relu(x)

        feats = []
        for i in range(len(self.base_channels)):
            layer_name = f'res_layer{i + 1}'
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            feats.append(x)

        for i in range(self.num_stages):
            if i == 0:
                x = feats[-1]
            else:
                x = torch.cat([x, feats[-1 - i]], dim=1)

            layer_name = f'res_layer{i + 1 + len(self.base_channels)}'
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            if i != self.num_stages - 1:
                x = interp(x, feats[-2 - i])

        x = self.out_conv(x)

        return x


@ENCODERS.register_module()
class UBasicEncoder(BaseModel):
    _arch_settings = {
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck,
        'InvertedResidual': InvertedResidual,
    }

    def __init__(self,
                 in_channels: int,
                 stem_channels: int,
                 out_channels: int,
                 block_name: str,
                 num_blocks: Optional[Sequence[int]] = None,
                 strides: Optional[Sequence[int]] = None,
                 base_channels: Optional[Sequence[int]] = None,
                 num_stages: Optional[int] = None,
                 deep_stem: bool = False,
                 dilations: Optional[Sequence[int]] = (1, 1, 1, 1, 1),
                 conv_cfg: Optional[dict] = None,
                 with_cp: bool = False,
                 norm_cfg: dict = dict(type='BN', requires_grad=True),
                 init_cfg: Optional[Union[dict, list]] = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.stem_channels = stem_channels
        self.strides = strides
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.deep_stem = deep_stem
        self.block = self._arch_settings[block_name]
        self.num_blocks = num_blocks
        self.dilations = dilations
        self.inplanes = self.stem_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg

        self._make_stem_layer(self.in_channels, self.stem_channels)

        self.res_layers = []

        for i in range(len(self.base_channels)):
            stride = self.strides[i]
            dilation = self.dilations[i]

            planes = self.base_channels[i]
            res_layer = self._make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=self.num_blocks[i],
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes
            layer_name = f'res_layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.inplanes = 0
        for i in range(self.num_stages):
            stride = 1
            dilation = 1

            inplanes = self.base_channels[-1 - i] + self.inplanes
            planes = self.base_channels[-1 - i]
            res_layer = self._make_res_layer(
                block=self.block,
                inplanes=inplanes,
                planes=planes,
                num_blocks=1,
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes
            layer_name = f'res_layer{i + 1 + len(self.base_channels)}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.out_conv = self._make_outputs_layer(self.inplanes, self.out_channels)

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=True)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)

    def _make_res_layer(self, **kwargs) -> torch.nn.Module:
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    def _make_outputs_layer(self, in_channels: int, out_channels: int) -> torch.nn.Module:
        """Make stem layer for ResNet."""
        return nn.Conv2d(in_channels, out_channels, 1)

    @property
    def norm1(self) -> torch.nn.Module:
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def forward(self, x, dual_inp=False):

        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)

        feats = []
        for i in range(len(self.base_channels)):
            layer_name = f'res_layer{i + 1}'
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            feats.append(x)

        for i in range(self.num_stages):
            if i == 0:
                x = feats[-1]
            else:
                x = torch.cat([x, feats[-1 - i]], dim=1)

            layer_name = f'res_layer{i + 1 + len(self.base_channels)}'
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            if i != self.num_stages - 1:
                x = interp(x, feats[-2 - i])

        x = self.out_conv(x)

        return x


@ENCODERS.register_module()
class AAFAEncoder0(BaseModel):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 base_channels: Optional[Sequence[int]] = None,
                 strides: Optional[Sequence[int]] = None,
                 num_stages: Optional[int] = None,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: dict = dict(type='BN', requires_grad=True),
                 init_cfg: Optional[Union[dict, list]] = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.base_channels = base_channels
        self.strides = strides
        self.num_stages = num_stages
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg

        self.conv_layers = []
        stem_layer = self._make_stem_layer(self.in_channels, self.base_channels[0])
        self.add_module('layer1', stem_layer)
        self.conv_layers.append(stem_layer)

        for i in range(1, len(self.base_channels)):
            in_planes = self.base_channels[i - 1]
            planes = self.base_channels[i]
            stride = self.strides[i]
            layer = self._make_layer(in_planes, planes, stride)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, layer)
            self.conv_layers.append(layer_name)

        for i in range(1, self.num_stages):
            upconv = nn.Sequential(nn.Conv2d(self.base_channels[i - self.num_stages],
                                             self.base_channels[-self.num_stages],
                                             1, 1, 0, bias=False),
                                   build_norm_layer(self.norm_cfg, self.base_channels[-self.num_stages])[1])
            layer_name = f'upconv{i}'
            self.add_module(layer_name, upconv)

        self.pooling = nn.Sequential(nn.ReLU(),
                                     nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(planes, planes // 2, 1, 1, 0, bias=True),
                                     nn.ReLU(),
                                     nn.Conv2d(planes // 2, self.base_channels[-self.num_stages], 1, 1, 0, bias=True),
                                     nn.Sigmoid())

        self.fusion = nn.Sequential(nn.ReLU(),
                                    self._make_conv(self.base_channels[-self.num_stages],
                                                    self.base_channels[-self.num_stages],
                                                    3, 1, 1, 1, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(self.base_channels[-self.num_stages],
                                              self.base_channels[-self.num_stages],
                                              3, 1, 1, bias=False))

        self.out_conv = self._make_outputs_layer(self.base_channels[-self.num_stages], self.out_channels)

    def _make_conv(self, in_planes, out_planes, kernel_size, stride, pad, dilation, groups=1):
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                       stride=stride, padding=dilation if dilation > 1 else pad,
                                       dilation=dilation, groups=groups, bias=False),
                             build_norm_layer(self.norm_cfg, out_planes)[1])

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        firstconv = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False),
                                  nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False),
                                  build_norm_layer(self.norm_cfg, in_channels)[1],
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels, stem_channels, 1, 1, 0, bias=False),
                                  self._make_conv(stem_channels, stem_channels, 3, 1, 1, 1, stem_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(stem_channels, stem_channels, 1, 1, 0, bias=False),
                                  self._make_conv(stem_channels, stem_channels, 3, 1, 1, 1, stem_channels))  # 1/4
        return firstconv

    def _make_layer(self, in_channels: int, out_channels: int, stride: int) -> torch.nn.Module:
        layer = nn.Sequential(nn.ReLU(),
                              nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                              self._make_conv(out_channels, out_channels, 3, stride, 1, 1, out_channels),
                              nn.ReLU(),
                              nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
                              self._make_conv(out_channels, out_channels, 3, 1, 1, 1, out_channels))

        return layer

    def _make_outputs_layer(self, in_channels: int, out_channels: int) -> torch.nn.Module:
        """Make stem layer for ResNet."""
        return nn.Sequential(nn.ReLU(), nn.Conv2d(in_channels, out_channels, 1))

    @property
    def norm1(self) -> torch.nn.Module:
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def forward(self, x, dual_inp=False):

        feat_list1 = []
        for i in range(len(self.base_channels)):
            layer_name = f'layer{i + 1}'
            layer = getattr(self, layer_name)
            x = layer(x)
            feat_list1.append(x)

        attention = self.pooling(x)

        feat_list2 = []
        feat_list2.append(feat_list1[-self.num_stages])
        for i in range(1, self.num_stages):
            layer_name = f'upconv{i}'
            upconv = getattr(self, layer_name)
            x = upconv(interp(feat_list1[i - self.num_stages], feat_list1[-self.num_stages]))
            feat_list2.append(x)
        y = torch.stack(feat_list2, dim=0).sum(dim=0)
        y = self.fusion(y) * attention + y
        y = self.out_conv(y)
        return y


@ENCODERS.register_module()
class AAFABasicEncoder(BaseModel):
    _arch_settings = {
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck
    }

    def __init__(self,
                 in_channels: int,
                 stem_channels: int,
                 out_channels: int,
                 block_name: str,
                 num_blocks: Optional[Sequence[int]] = None,
                 strides: Optional[Sequence[int]] = None,
                 base_channels: Optional[Sequence[int]] = None,
                 num_stages: Optional[int] = None,
                 deep_stem: bool = False,
                 dilations: Optional[Sequence[int]] = (1, 1, 1, 1, 1),
                 conv_cfg: Optional[dict] = None,
                 with_cp: bool = False,
                 norm_cfg: dict = dict(type='BN', requires_grad=True),
                 init_cfg: Optional[Union[dict, list]] = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.stem_channels = stem_channels
        self.strides = strides
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.deep_stem = deep_stem
        self.block = self._arch_settings[block_name]
        self.num_blocks = num_blocks
        self.dilations = dilations
        self.inplanes = self.stem_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg

        self._make_stem_layer(self.in_channels, self.stem_channels)

        self.res_layers = []

        for i in range(len(self.base_channels)):
            stride = self.strides[i]
            dilation = self.dilations[i]

            planes = self.base_channels[i]
            res_layer = self._make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=self.num_blocks[i],
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes
            layer_name = f'res_layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.pooling = nn.Sequential(nn.ReLU(),
                                     nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(planes, planes // 2, 1, 1, 0, bias=True),
                                     nn.ReLU(),
                                     nn.Conv2d(planes // 2, self.base_channels[-self.num_stages], 1, 1, 0, bias=True),
                                     nn.Sigmoid())

        for i in range(1, self.num_stages):
            stride = 1
            dilation = 1
            inplanes = self.base_channels[i - self.num_stages]
            planes = self.base_channels[-self.num_stages]
            upconv = self._make_res_layer(
                block=self.block,
                inplanes=inplanes,
                planes=planes,
                num_blocks=1,
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            layer_name = f'upconv{i}'
            self.add_module(layer_name, upconv)

        self.fusion = nn.Sequential(nn.Conv2d(self.base_channels[-self.num_stages],
                                              self.base_channels[-self.num_stages],
                                              3, 1, 1, bias=False))

        self.out_conv = self._make_outputs_layer(self.base_channels[-self.num_stages], self.out_channels)

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=True)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)

    def _make_res_layer(self, **kwargs) -> torch.nn.Module:
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    def _make_outputs_layer(self, in_channels: int, out_channels: int) -> torch.nn.Module:
        """Make stem layer for ResNet."""
        return nn.Conv2d(in_channels, out_channels, 1)

    @property
    def norm1(self) -> torch.nn.Module:
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def forward(self, x, dual_inp=False):

        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)

        feat_list1 = []
        for i in range(len(self.base_channels)):
            layer_name = f'res_layer{i + 1}'
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            feat_list1.append(x)

        attention = self.pooling(x)

        feat_list2 = []
        feat_list2.append(feat_list1[-self.num_stages])
        for i in range(1, self.num_stages):
            layer_name = f'upconv{i}'
            upconv = getattr(self, layer_name)
            x = upconv(interp(feat_list1[i - self.num_stages], feat_list1[-self.num_stages]))
            feat_list2.append(x)

        y = torch.stack(feat_list2, dim=0).sum(dim=0)
        y = self.fusion(y) * attention + y
        y = self.out_conv(y)
        return y


@ENCODERS.register_module()
class AAFAUBasicEncoder(BaseModel):
    _arch_settings = {
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck
    }

    def __init__(self,
                 in_channels: int,
                 stem_channels: int,
                 out_channels: int,
                 block_name: str,
                 num_blocks: Optional[Sequence[int]] = None,
                 strides: Optional[Sequence[int]] = None,
                 base_channels: Optional[Sequence[int]] = None,
                 num_stages: Optional[int] = None,
                 deep_stem: bool = False,
                 dilations: Optional[Sequence[int]] = (1, 1, 1, 1, 1),
                 conv_cfg: Optional[dict] = None,
                 with_cp: bool = False,
                 norm_cfg: dict = dict(type='BN', requires_grad=True),
                 init_cfg: Optional[Union[dict, list]] = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.stem_channels = stem_channels
        self.strides = strides
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.deep_stem = deep_stem
        self.block = self._arch_settings[block_name]
        self.num_blocks = num_blocks
        self.dilations = dilations
        self.inplanes = self.stem_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg

        self._make_stem_layer(self.in_channels, self.stem_channels)

        self.res_layers = []

        for i in range(len(self.base_channels)):
            stride = self.strides[i]
            dilation = self.dilations[i]

            planes = self.base_channels[i]
            res_layer = self._make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=self.num_blocks[i],
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes
            layer_name = f'res_layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.inplanes = 0
        for i in range(self.num_stages):
            stride = 1
            dilation = 1

            inplanes = self.base_channels[-1 - i] + self.inplanes
            planes = self.base_channels[-1 - i]
            res_layer = self._make_res_layer(
                block=self.block,
                inplanes=inplanes,
                planes=planes,
                num_blocks=1,
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            self.inplanes = planes
            layer_name = f'res_layer{i + 1 + len(self.base_channels)}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.pooling = nn.Sequential(nn.ReLU(),
                                     nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.base_channels[-1], planes // 2, 1, 1, 0, bias=True),
                                     nn.ReLU(),
                                     nn.Conv2d(planes // 2, self.base_channels[-self.num_stages], 1, 1, 0, bias=True),
                                     nn.Sigmoid())

        for i in range(1, self.num_stages):
            stride = 1
            dilation = 1
            inplanes = self.base_channels[i - self.num_stages]
            planes = self.base_channels[-self.num_stages]
            upconv = self._make_res_layer(
                block=self.block,
                inplanes=inplanes,
                planes=planes,
                num_blocks=1,
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
            layer_name = f'upconv{i}'
            self.add_module(layer_name, upconv)

        self.fusion = nn.Sequential(nn.Conv2d(self.base_channels[-self.num_stages],
                                              self.base_channels[-self.num_stages],
                                              3, 1, 1, bias=False))

        self.out_conv = self._make_outputs_layer(self.base_channels[-self.num_stages], self.out_channels)

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=True)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)

    def _make_res_layer(self, **kwargs) -> torch.nn.Module:
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    def _make_outputs_layer(self, in_channels: int, out_channels: int) -> torch.nn.Module:
        """Make stem layer for ResNet."""
        return nn.Conv2d(in_channels, out_channels, 1)

    @property
    def norm1(self) -> torch.nn.Module:
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def forward(self, x, dual_inp=False):

        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)

        feat_list1 = []
        for i in range(len(self.base_channels)):
            layer_name = f'res_layer{i + 1}'
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            feat_list1.append(x)

        feat_list2 = []
        for i in range(self.num_stages):
            if i == 0:
                x = feat_list1[-1]
            else:
                x = torch.cat([x, feat_list1[-1 - i]], dim=1)

            layer_name = f'res_layer{i + 1 + len(self.base_channels)}'
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            feat_list2.append(x)

            if i != self.num_stages - 1:
                x = interp(x, feat_list1[-2 - i])

        attention = self.pooling(feat_list2[0])

        feat_list3 = []
        feat_list3.append(feat_list2[-1])
        for i in range(1, self.num_stages):
            layer_name = f'upconv{i}'
            upconv = getattr(self, layer_name)
            x = upconv(interp(feat_list2[-1 - i], feat_list2[-1]))
            feat_list3.append(x)

        y = torch.stack(feat_list3, dim=0).sum(dim=0)
        y = self.fusion(y) * attention + y
        y = self.out_conv(y)
        return y

