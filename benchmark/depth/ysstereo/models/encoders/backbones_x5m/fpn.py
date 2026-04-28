# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from ysstereo.models.encoders.backbones_x5m.conv_module import ConvModule2d

__all__ = ["FPN"]

def _normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class FPN(nn.Module):
    def __init__(
        self,
        in_strides: List[int],
        in_channels: List[int],
        out_strides: List[int],
        out_channels: List[int],
        fix_out_channel: Optional[int] = None,
        bn_kwargs: Optional[Dict] = None,
    ):
        """FPN neck.

        Args:
            in_strides (list): strides of each input feature map
            in_channels (list): channels of each input feature map,
                the length of in_channels should be equal to in_strides
            out_strides (list): strides of each output feature map,
                should be a subset of in_strides, and continuous (any
                subsequence of 2, 4, 8, 16, 32, 64 ...). The largest
                stride in in_strides and out_strides should be equal
            out_channels (list): channels of each output feature maps
                the length of out_channels should be equal to out_strides
            fix_out_channel (:obj:`int`, optional): if set, there will be
                a 1x1 conv following each output feature map so that each
                final output has fix_out_channel channels
            bn_kwargs (dict): Dict for Bn layer. No Bn layer if
                bn_kwargs=None
        """

        super(FPN, self).__init__()
        self._valid_strides = [2, 4, 8, 16, 32, 64, 128, 256]
        self.bn_kwargs = bn_kwargs
        # in_strides check
        assert len(in_strides) == len(in_channels)
        for stride_i in in_strides:
            assert stride_i in self._valid_strides

        min_idx = self._valid_strides.index(in_strides[0])
        max_idx = self._valid_strides.index(in_strides[-1])

        assert tuple(in_strides) == tuple(
            self._valid_strides[min_idx : max_idx + 1]
        ), "Input strides must be continuous and in ascending order"
        self.in_strides = in_strides

        # out_strides check
        assert len(out_strides) == len(out_channels)
        #out_channels = [64, 96, 96]

        min_idx = self._valid_strides.index(out_strides[0])
        max_idx = self._valid_strides.index(out_strides[-1])

        assert tuple(out_strides) == tuple(
            self._valid_strides[min_idx : max_idx + 1]
        ), "Output strides must be continuous"

        assert all(
            [stride in in_strides for stride in out_strides]
        ), "all stride of output stride must be in input stride"

        assert (
            out_strides[-1] == in_strides[-1]
        ), "The largest stride in in_strides and out_strides should be equal"

        self.out_strides = out_strides
        #self.out_channels = out_channels

        self.src_min_stride_idx = self.in_strides.index(self.out_strides[0])

        # init modules
        self.conv_extract = nn.ModuleList()
        self.conv_add = nn.ModuleList()
        self.conv1x1_up = nn.ModuleList()

        for idx in range(len(out_channels)):
            if idx == 0:
                self.conv_extract.append(
                    ConvModule2d(
                        in_channels=in_channels[-1],
                        out_channels=out_channels[-1],
                        kernel_size=1,
                        padding=0,
                        stride=1,
                        bias=True,
                        norm_layer=None
                        if bn_kwargs is None
                        else nn.SyncBatchNorm(out_channels[-1], **bn_kwargs),
                    )
                )
            else:
                if len(out_channels) > 1:
                    self.conv1x1_up.append(
                        ConvModule2d(
                            in_channels=out_channels[-idx],
                            out_channels=out_channels[-1 - idx],
                            kernel_size=1,
                            padding=0,
                            stride=1,
                            bias=True,
                            norm_layer=None
                            if bn_kwargs is None
                            else nn.SyncBatchNorm(
                                out_channels[-1 - idx], **bn_kwargs
                            ),
                        )
                    )

                self.conv_extract.append(
                    ConvModule2d(
                        in_channels=in_channels[-1 - idx],
                        out_channels=out_channels[-1 - idx],
                        kernel_size=1,
                        padding=0,
                        stride=1,
                        bias=True,
                        norm_layer=None
                        if bn_kwargs is None
                        else nn.SyncBatchNorm(
                            out_channels[-1 - idx], **bn_kwargs
                        ),
                    )
                )

        # optionally map the output feature maps to fix_out_channel
        if fix_out_channel is not None:
            self.conv1x1 = nn.ModuleList()
            for idx, _ in enumerate(self.out_strides[::-1]):
                self.conv1x1.append(
                    ConvModule2d(
                        in_channels=out_channels[-1 - idx],
                        out_channels=fix_out_channel,
                        kernel_size=1,
                        padding=0,
                        stride=1,
                        bias=True,
                        norm_layer=None
                        if bn_kwargs is None
                        else nn.SyncBatchNorm(fix_out_channel, **bn_kwargs),
                    )
                )

    def _init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _normal_init(m, mean=0, std=0.01, bias=0)

    def forward_onnx(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(features) == len(self.in_strides)

        # slice features
        in_features = features[self.src_min_stride_idx :][::-1]
        strides = self.in_strides[self.src_min_stride_idx :][::-1]

        fpn_fuse = {}
        for idx, stride in enumerate(strides):
            if idx > 0:
                cur_feat = self.conv1x1_up[idx - 1](fpn_fuse[strides[idx - 1]])
                cur_feat = F.interpolate(cur_feat, size=None, scale_factor=2, mode='bilinear', align_corners=False)
            else:
                cur_feat = None

            fpn_fuse[stride] = self.conv_extract[idx](in_features[idx])
            if idx > 0:
                fpn_fuse[stride] = fpn_fuse[stride] + cur_feat

        if hasattr(self, "conv1x1"):
            for idx, stride in enumerate(strides):
                fpn_fuse[stride] = self.conv1x1[idx](fpn_fuse[stride])

        return [fpn_fuse[stride] for stride in self.out_strides]

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # make fpn always run in fp32 mode
        with autocast('cuda', enabled=False):
            assert len(features) == len(self.in_strides)
            features = [f.float() for f in features]
            # slice features
            in_features = features[self.src_min_stride_idx :][::-1]
            strides = self.in_strides[self.src_min_stride_idx :][::-1]

            fpn_fuse = {}
            for idx, stride in enumerate(strides):
                if idx > 0:
                    cur_feat = self.conv1x1_up[idx - 1](fpn_fuse[strides[idx - 1]])
                    tH, tW = in_features[idx].shape[2:]
                    cur_feat = F.interpolate(cur_feat, size=(tH, tW), scale_factor=None, mode='bilinear', align_corners=False)
                else:
                    cur_feat = None

                fpn_fuse[stride] = self.conv_extract[idx](in_features[idx])
                if idx > 0:
                    fpn_fuse[stride] = fpn_fuse[stride] + cur_feat

            if hasattr(self, "conv1x1"):
                for idx, stride in enumerate(strides):
                    fpn_fuse[stride] = self.conv1x1[idx](fpn_fuse[stride])

            return [fpn_fuse[stride] for stride in self.out_strides]
