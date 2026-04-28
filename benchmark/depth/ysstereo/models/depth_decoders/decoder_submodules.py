import math
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModel
from ysstereo.models.utils import InvertedResidual, HardTanh, HardSigmoid


class CorrBlock1D(BaseModel):
    def __init__(self, num_levels: int = 4) -> None:
        super().__init__()
        self.num_levels = num_levels

    def forward(self, feat1: torch.Tensor,
                feat2: torch.Tensor) -> Sequence[torch.Tensor]:
        """Forward function for Correlation pyramid.

        Args:
            feat1 (Tensor): The feature from first input image.
            feat2 (Tensor): The feature from second input image.

        Returns:
            Sequence[Tensor]: The list of correlation which is pooled using
                average pooling with kernel sizes {1, 2, 4, 8}.
        """

        B, D, H, W1 = feat1.shape
        _, _, _, W2 = feat2.shape
        feat1 = feat1.view(B, D, H, W1)
        feat2 = feat2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', feat1, feat2)

        # feat2_list = list(torch.split(feat2, 1, dim=3))
        # corr_list = []
        # for i in range(W2):
        #     corr_list.append((feat1 * feat2_list[i]).sum(dim=1).unsqueeze(3))
        # corr = torch.cat(corr_list, dim = 3)

        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        corr = corr / torch.sqrt(torch.tensor(D).float())
        corr = corr.reshape(B * H * W1, 1, 1, W2)
        corr_pyramid = []
        corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
            corr_pyramid.append(corr)

        return corr_pyramid


class FisheyeCorrBlock1D(BaseModel):
    def __init__(self, num_levels: int = 4) -> None:
        super().__init__()
        self.num_levels = num_levels

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor,
                valid_grids: torch.Tensor, corr_grids: torch.Tensor, ) -> Sequence[torch.Tensor]:
        """Forward function for Correlation pyramid.

        Args:
            feat1 (Tensor): The feature from first input image.
            feat2 (Tensor): The feature from second input image.
            valid_grids (Tensor): The valid fisheye grids. 
            corr_grids (Tensor): The correlation of right fisheye grids.

        Returns:
            Sequence[Tensor]: The list of correlation which is pooled using
                average pooling with kernel sizes {1, 2, 4, 8}.
        """
        N, C, _, _ = feat1.shape
        valid_fmap1 = F.grid_sample(feat1, valid_grids, align_corners=True)
        valid_fmap1 = valid_fmap1.reshape(N, C, -1, 1)
        corr_fmap2s = [F.grid_sample(feat2, corr_grids[:, d, ...], align_corners=True) for d in range(0, 96)]

        valid_corr_fmap2s_list = []
        for i in range(len(corr_fmap2s)):
            valid_corr_fmap2 = F.grid_sample(corr_fmap2s[i], valid_grids, align_corners=True)
            valid_corr_fmap2 = valid_corr_fmap2.reshape(N, C, -1, 1)
            valid_corr_fmap2s_list.append(valid_corr_fmap2)
        valid_corr_fmap2s = torch.cat(valid_corr_fmap2s_list, dim=-1)

        B, D, H, W1 = valid_fmap1.shape
        _, _, _, W2 = valid_corr_fmap2s.shape
        valid_fmap1 = valid_fmap1.view(B, D, H, W1)
        valid_corr_fmap2s = valid_corr_fmap2s.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', valid_fmap1, valid_corr_fmap2s)

        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        corr = corr / torch.sqrt(torch.tensor(D).float())
        corr = corr.reshape(B * H * W1, 1, 1, W2)
        corr_pyramid = []
        corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
            corr_pyramid.append(corr)

        return corr_pyramid


class MotionEncoder(BaseModel):
    """The module of motion encoder.

    An encoder which consists of several convolution layers and outputs
    features as GRU's input.

    Args:
        num_levels (int): Number of levels used when calculating correlation
            tensor. Default: 4.
        radius (int): Radius used when calculating correlation tensor.
            Default: 4.
        net_type (str): Type of the net. Choices: ['Basic', 'Full'].
            Default: 'Basic'.
    """
    _corr_channels = {'Basic': (16, 16), 'Full': (256, 192)}
    _corr_kernel = {'Basic': (1, 3), 'Full': (1, 3)}
    _corr_padding = {'Basic': (0, 1), 'Full': (0, 1)}

    _flow_channels = {'Basic': (16, 16), 'Full': (128, 64)}
    _flow_kernel = {'Basic': (7, 3), 'Full': (7, 3)}
    _flow_padding = {'Basic': (3, 1), 'Full': (3, 1)}

    _out_channels = {'Basic': 30, 'Full': 126}
    _out_kernel = {'Basic': 3, 'Full': 3}
    _out_padding = {'Basic': 1, 'Full': 1}

    _conv_settings = {'Conv': ConvModule,
                      'InvertedResidual': InvertedResidual}

    def __init__(self,
                 num_levels: int = 1,
                 radius: int = 4,
                 corr_channels: Optional[Sequence[int]] = None,
                 flow_channels: Optional[Sequence[int]] = None,
                 out_channels: int = None,  # fix conflict bewteen fisheyestereo and others
                 net_type: str = 'Basic',
                 conv_type: str = 'Conv',
                 **kwargs) -> None:
        super().__init__()
        assert net_type in ['Basic', 'Full']

        self.conv_module = self._conv_settings[conv_type]

        corr_channels = (corr_channels
                         if corr_channels is not None else self._corr_channels[net_type])
        corr_kernel = self._corr_kernel.get(net_type) if isinstance(
            self._corr_kernel.get(net_type),
            (tuple, list)) else [self._corr_kernel.get(net_type)]
        corr_padding = self._corr_padding.get(net_type) if isinstance(
            self._corr_padding.get(net_type),
            (tuple, list)) else [self._corr_padding.get(net_type)]

        flow_channels = (flow_channels
                         if flow_channels is not None else self._flow_channels[net_type])
        flow_kernel = self._flow_kernel.get(net_type)
        flow_padding = self._flow_padding.get(net_type)

        self.out_channels = ([out_channels]
                             if out_channels is not None else [
            self._out_channels[net_type]])  # fix conflict bewteen fisheyestereo and others
        out_kernel = self._out_kernel.get(net_type) if isinstance(
            self._out_kernel.get(net_type),
            (tuple, list)) else [self._out_kernel.get(net_type)]
        out_padding = self._out_padding.get(net_type) if isinstance(
            self._out_padding.get(net_type),
            (tuple, list)) else [self._out_padding.get(net_type)]

        corr_inch = num_levels * (2 * radius + 1)
        corr_net = self._make_encoder(corr_inch, corr_channels, corr_kernel,
                                      corr_padding, **kwargs)
        self.corr_net = nn.Sequential(*corr_net)

        flow_inch = 2
        flow_net = self._make_encoder(flow_inch, flow_channels, flow_kernel,
                                      flow_padding, **kwargs)
        self.flow_net = nn.Sequential(*flow_net)

        out_inch = corr_channels[-1] + flow_channels[-1]
        out_net = self._make_encoder(out_inch, self.out_channels, out_kernel,
                                     out_padding, **kwargs)
        self.out_net = nn.Sequential(*out_net)

    def _make_encoder(self, in_channel: int, channels: Optional[Sequence[int]], kernels: Optional[Sequence[int]],
                      paddings: Optional[Sequence[int]], conv_cfg: dict, norm_cfg: dict,
                      act_cfg: dict):
        encoder = []

        for ch, k, p in zip(channels, kernels, paddings):
            if k == 1:
                encoder.append(
                    ConvModule(
                        in_channels=in_channel,
                        out_channels=ch,
                        kernel_size=1,
                        padding=p,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            else:
                encoder.append(
                    self.conv_module(
                        in_channels=in_channel,
                        out_channels=ch,
                        kernel_size=k,
                        padding=p,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))

            in_channel = ch
        return encoder

    def forward(self, corr: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Forward function for MotionEncoder.

        Args:
            corr (Tensor): The correlation feature.
            flow (Tensor): The last estimated disparity.

        Returns:
            Tensor: The output feature of motion encoder.
        """
        corr_feat = self.corr_net(corr)
        flow_feat = self.flow_net(flow)

        out = self.out_net(torch.cat([corr_feat, flow_feat], dim=1))
        return torch.cat([out, flow], dim=1)


class DispEncoder(BaseModel):
    """The module of disparity encoder.

    An encoder which consists of several convolution layers and outputs
    features as GRU's input.

    Args:
        num_levels (int): Number of levels used when calculating correlation
            tensor. Default: 4.
        radius (int): Radius used when calculating correlation tensor.
            Default: 4.
        net_type (str): Type of the net. Choices: ['Basic', 'Full'].
            Default: 'Basic'.
    """
    _corr_channels = {'Basic': (16, 16), 'Full': (256, 192), 'Middle': (128, 96)}
    _corr_kernel = {'Basic': (1, 3), 'Full': (1, 3), 'Middle': (1, 3)}
    _corr_padding = {'Basic': (0, 1), 'Full': (0, 1), 'Middle': (0, 1)}

    _disp_channels = {'Basic': (16, 16), 'Full': (128, 64), 'Middle': (64, 48)}
    _disp_kernel = {'Basic': (7, 3), 'Full': (7, 3), 'Middle': (7, 3)}
    _disp_padding = {'Basic': (3, 1), 'Full': (3, 1), 'Middle': (3, 1)}

    _out_channels = {'Basic': 31, 'Full': 127, 'Middle': 63}
    _out_kernel = {'Basic': 3, 'Full': 3, 'Middle': 3}
    _out_padding = {'Basic': 1, 'Full': 1, 'Middle': 1}

    _conv_settings = {'Conv': ConvModule,
                      'InvertedResidual': InvertedResidual}

    def __init__(self,
                 num_levels: int = 1,
                 radius: int = 4,
                 corr_channels: Optional[Sequence[int]] = None,
                 disp_channels: Optional[Sequence[int]] = None,
                 out_channels: int = None,  # fix conflict bewteen fisheyestereo and others
                 net_type: str = 'Basic',
                 conv_type: str = 'Conv',
                 **kwargs) -> None:
        super().__init__()
        assert net_type in ['Basic', 'Full', 'Middle']

        self.conv_module = self._conv_settings[conv_type]

        corr_channels = (corr_channels
                         if corr_channels is not None else self._corr_channels[net_type])
        corr_kernel = self._corr_kernel.get(net_type) if isinstance(
            self._corr_kernel.get(net_type),
            (tuple, list)) else [self._corr_kernel.get(net_type)]
        corr_padding = self._corr_padding.get(net_type) if isinstance(
            self._corr_padding.get(net_type),
            (tuple, list)) else [self._corr_padding.get(net_type)]

        disp_channels = (disp_channels
                         if disp_channels is not None else self._disp_channels[net_type])
        disp_kernel = self._disp_kernel.get(net_type)
        disp_padding = self._disp_padding.get(net_type)

        self.out_channels = ([out_channels]
                             if out_channels is not None else [
            self._out_channels[net_type]])  # fix conflict bewteen fisheyestereo and others
        out_kernel = self._out_kernel.get(net_type) if isinstance(
            self._out_kernel.get(net_type),
            (tuple, list)) else [self._out_kernel.get(net_type)]
        out_padding = self._out_padding.get(net_type) if isinstance(
            self._out_padding.get(net_type),
            (tuple, list)) else [self._out_padding.get(net_type)]

        corr_inch = num_levels * (2 * radius + 1)
        corr_net = self._make_encoder(corr_inch, corr_channels, corr_kernel,
                                      corr_padding, **kwargs)
        self.corr_net = nn.Sequential(*corr_net)

        disp_inch = 1
        disp_net = self._make_encoder(disp_inch, disp_channels, disp_kernel,
                                      disp_padding, **kwargs)
        self.disp_net = nn.Sequential(*disp_net)

        out_inch = corr_channels[-1] + disp_channels[-1]
        out_net = self._make_encoder(out_inch, self.out_channels, out_kernel,
                                     out_padding, **kwargs)
        self.out_net = nn.Sequential(*out_net)

    def _make_encoder(self, in_channel: int, channels: Optional[Sequence[int]], kernels: Optional[Sequence[int]],
                      paddings: Optional[Sequence[int]], conv_cfg: dict, norm_cfg: dict,
                      act_cfg: dict):
        encoder = []

        for ch, k, p in zip(channels, kernels, paddings):
            if k == 1:
                encoder.append(
                    ConvModule(
                        in_channels=in_channel,
                        out_channels=ch,
                        kernel_size=1,
                        padding=p,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            else:
                encoder.append(
                    self.conv_module(
                        in_channels=in_channel,
                        out_channels=ch,
                        kernel_size=k,
                        padding=p,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))

            in_channel = ch
        return encoder

    def forward(self, corr: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        """Forward function for MotionEncoder.

        Args:
            corr (Tensor): The correlation feature.
            flow (Tensor): The last estimated disparity.

        Returns:
            Tensor: The output feature of motion encoder.
        """
        corr_feat = self.corr_net(corr)
        disp_feat = self.disp_net(disp)

        out = self.out_net(torch.cat([corr_feat, disp_feat], dim=1))
        return torch.cat([out, disp], dim=1)


class ConvGRU(BaseModel):
    """GRU with convolution layers.

    GRU cell with fully connected layers replaced with convolutions.

    Args:
        h_channels (int): Number of channels of hidden feature.
        x_channels (int): Number of channels of the concatenation of motion
            feature and context features.
        net_type (str):  Type of the GRU module. Choices: ['Conv', 'SeqConv'].
            Default: 'SeqConv'.
    """
    _kernel = {'Conv': 3, 'SeqConv': ((1, 5), (5, 1))}
    _padding = {'Conv': 1, 'SeqConv': ((0, 2), (2, 0))}

    def __init__(self,
                 h_channels: int,
                 x_channels: int,
                 use_hard_sigmoid: bool = False,
                 use_hard_tanh: bool = False,
                 net_type: str = 'Conv') -> None:
        super().__init__()
        assert net_type in ['Conv', 'SeqConv']
        kernel_size = self._kernel.get(net_type) if isinstance(
            self._kernel.get(net_type),
            (tuple, list)) else [self._kernel.get(net_type)]
        padding = self._padding.get(net_type) if isinstance(
            self._padding.get(net_type),
            (tuple, list)) else [self._padding.get(net_type)]

        self.use_hard_sigmoid = use_hard_sigmoid
        self.use_hard_tanh = use_hard_tanh
        if self.use_hard_sigmoid:
            self.sigmoid = HardSigmoid()
        else:
            self.sigmoid = nn.Sigmoid()
        if self.use_hard_tanh:
            self.tanh = HardTanh()
        else:
            self.tanh = nn.Tanh()

        conv_z = []
        conv_r = []
        conv_q = []

        for k, p in zip(kernel_size, padding):
            conv_z.append(
                nn.Conv2d(h_channels + x_channels,
                          h_channels,
                          k,
                          padding=p))
            conv_r.append(
                nn.Conv2d(h_channels + x_channels,
                          h_channels,
                          k,
                          padding=p))
            conv_q.append(
                nn.Conv2d(h_channels + x_channels,
                          h_channels,
                          k,
                          padding=p))
        self.conv_z = nn.ModuleList(conv_z)
        self.conv_r = nn.ModuleList(conv_r)
        self.conv_q = nn.ModuleList(conv_q)

    def init_weights(self) -> None:

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.orthogonal_(m.weight)

        self.apply(weights_init)

    def forward(self, h, cz, cr, cq, *x_list) -> torch.Tensor:
        """Forward function for ConvGRU.

        Args:
            h (Tensor): The last hidden state for GRU block.
            x (Tensor): The current input feature for GRU block

        Returns:
            Tensor: The current hidden state.
        """
        for conv_z, conv_r, conv_q in zip(self.conv_z, self.conv_r,
                                          self.conv_q):
            x = torch.cat(x_list, dim=1)
            hx = torch.cat([h, x], dim=1)
            z = self.sigmoid(conv_z(hx) + cz)
            r = self.sigmoid(conv_r(hx) + cr)
            q = self.tanh(conv_q(torch.cat([r * h, x], dim=1)) + cq)
            h = (1 - z) * h + z * q
        return h


class SeqConvGRU(BaseModel):
    """GRU with convolution layers.

    GRU cell with fully connected layers replaced with convolutions.

    Args:
        h_channels (int): Number of channels of hidden feature.
        x_channels (int): Number of channels of the concatenation of motion
            feature and context features.
        net_type (str):  Type of the GRU module. Choices: ['Conv', 'SeqConv'].
            Default: 'SeqConv'.
    """
    _kernel = {'Conv': 3, 'SeqConv': ((1, 5), (5, 1))}
    _padding = {'Conv': 1, 'SeqConv': ((0, 2), (2, 0))}

    def __init__(self,
                 h_channels: int,
                 x_channels: int,
                 net_type: str = 'SeqConv') -> None:
        super().__init__()
        assert net_type in ['Conv', 'SeqConv']
        kernel_size = self._kernel.get(net_type) if isinstance(
            self._kernel.get(net_type),
            (tuple, list)) else [self._kernel.get(net_type)]
        padding = self._padding.get(net_type) if isinstance(
            self._padding.get(net_type),
            (tuple, list)) else [self._padding.get(net_type)]

        conv_z = []
        conv_r = []
        conv_q = []

        for k, p in zip(kernel_size, padding):
            conv_z.append(
                ConvModule(
                    in_channels=h_channels + x_channels,
                    out_channels=h_channels,
                    kernel_size=k,
                    padding=p,
                    act_cfg=dict(type='Sigmoid')))
            conv_r.append(
                ConvModule(
                    in_channels=h_channels + x_channels,
                    out_channels=h_channels,
                    kernel_size=k,
                    padding=p,
                    act_cfg=dict(type='Sigmoid')))
            conv_q.append(
                ConvModule(
                    in_channels=h_channels + x_channels,
                    out_channels=h_channels,
                    kernel_size=k,
                    padding=p,
                    act_cfg=dict(type='Tanh')))
        self.conv_z = nn.ModuleList(conv_z)
        self.conv_r = nn.ModuleList(conv_r)
        self.conv_q = nn.ModuleList(conv_q)

    def init_weights(self) -> None:

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.orthogonal_(m.weight)

        self.apply(weights_init)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward function for ConvGRU.

        Args:
            h (Tensor): The last hidden state for GRU block.
            x (Tensor): The current input feature for GRU block

        Returns:
            Tensor: The current hidden state.
        """
        for conv_z, conv_r, conv_q in zip(self.conv_z, self.conv_r,
                                          self.conv_q):
            hx = torch.cat([h, x], dim=1)
            z = conv_z(hx)
            r = conv_r(hx)
            q = conv_q(torch.cat([r * h, x], dim=1))
            h = (1 - z) * h + z * q
        return h


class XHead(BaseModel):
    """A module for flow or mask prediction.

    Args:
        in_channels (int): Input channels of first convolution layer.
        feat_channels (Sequence(int)): List of features channels of different
            convolution layers.
        x_channels (int): Final output channels of predict layer.
        x (str): Type of predict layer. Choice: ['flow', 'mask']
    """

    def __init__(self, in_channels: int, feat_channels: Sequence[int],
                 x_channels: int, x: str) -> None:
        super().__init__()
        conv_layers = []
        for ch in feat_channels:
            conv_layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=ch,
                    kernel_size=3,
                    padding=1))
            in_channels = ch
        self.layers = nn.Sequential(*conv_layers)
        if x == 'flow' or x == 'disp':
            self.predict_layer = nn.Conv2d(
                feat_channels[-1], x_channels, kernel_size=3, padding=1)
        elif x == 'mask':
            self.predict_layer = nn.Conv2d(
                feat_channels[-1], x_channels, kernel_size=1, padding=0)
        else:
            raise ValueError(f'x must be \'flow\', \'disp\' or \'mask\', but got {x}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.predict_layer(x)
