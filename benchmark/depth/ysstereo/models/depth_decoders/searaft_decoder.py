import math
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ysstereo.ops.AGCL import AGCL, get_correlation, corr_iter, onnx_corr_iter
from ysstereo.models.builder import DECODERS, build_loss, build_encoder, build_components
from ysstereo.models.depth_decoders.base_decoder import BaseDecoder
from ysstereo.models.depth_decoders.decoder_submodules import DispEncoder, ConvGRU, SeqConvGRU, XHead
from ysstereo.utils.utils import convex_upsample, coords_grid
from ysstereo.models.utils import HardTanh

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, output_dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * output_dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * output_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.final = nn.Conv2d(dim, output_dim, kernel_size=1, padding=0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.final(input + x)
        return x

class CntUpdateBlock(nn.Module):
    def __init__(self,
                 radius: int,
                 extra_channels: int = 0,
                 gru_type: str = 'SeqConv',
                 conv_type: str = 'Conv',
                 h_channels: int = 32,
                 cxt_channels: int = 32,
                 feat_channels: int = 64,
                 mask_channels: int = 64,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None):
        super().__init__()

        feat_channels = feat_channels if isinstance(tuple,
                                                    list) else [feat_channels]

        self.feat_channels = feat_channels
        self.h_channels = h_channels
        self.cxt_channels = cxt_channels
        self.mask_channels = mask_channels * (2 * radius + 1)

        self.context_zqr_convs = nn.Conv2d(self.cxt_channels, self.h_channels*3, 3, padding=3//2)

        self.encoder = DispEncoder(
            radius=radius,
            net_type='Basic',
            conv_type=conv_type,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,)
        self.gru_type = gru_type

        self.refine = []
        for i in range(2):
            self.refine.append(ConvNextBlock(2*cxt_channels+h_channels+extra_channels, h_channels))
        self.refine = nn.ModuleList(self.refine)

        self.disp_pred = XHead(self.h_channels, self.feat_channels, 1, x='disp')

        self.mask_pred = XHead(
            self.h_channels, self.feat_channels, self.mask_channels, x='mask')

    def make_gru_block(self):
        return ConvGRU(
            self.h_channels,
            self.encoder.out_channels[0] + 1,
            use_hard_sigmoid=self.use_hard_sigmoid,
            use_hard_tanh=self.use_hard_tanh,
            net_type=self.gru_type)

    def forward(self, net, inp, corr=None, disp=None, mask_flag=True, output_all=False, extra_feats=None):

        if output_all:
            motion_features = self.encoder(corr, disp)
            if extra_feats is not None:
                inp = torch.cat([inp, motion_features, extra_feats], dim=1)
            else:
                inp = torch.cat([inp, motion_features], dim=1)
            for blk in self.refine:
                net = blk(torch.cat([net, inp], dim=1))
            delta_disp = self.disp_pred(net)

            # scale mask to balence gradients
            mask = .25 * self.mask_pred(net)

            return net, mask, delta_disp
        if mask_flag:
            # scale mask to balence gradients
            mask = .25 * self.mask_pred(net)
            return mask
        motion_features = self.encoder(corr, disp)
        if extra_feats is not None:
            inp = torch.cat([inp, motion_features, extra_feats], dim=1)
        else:
            inp = torch.cat([inp, motion_features], dim=1)
        for blk in self.refine:
            net = blk(torch.cat([net, inp], dim=1))
        delta_disp = self.disp_pred(net)

        return net, delta_disp


@DECODERS.register_module()
class SeaRaftStereoSlimDecoder(BaseDecoder):
    """The decoder of RumStereo Net.

    The decoder of RumStereo Net, which outputs list of upsampled stereo estimation.

    Args:
        net_type (str): Type of the net. Choices: ['Basic', 'Small'].
        radius (int): Radius used when calculating correlation tensor.
        gru_type (str): Type of the GRU module. Choices: ['Conv', 'SeqConv'].
            Default: 'SeqConv'.
        conv_type (str): Type of the Conv. Choices: ['Conv', 'InvertedResidual'].
            Default: 'Conv'.
        feat_channels (Sequence(int)): features channels of prediction module.
        mask_channels (int): Output channels of mask prediction layer.
            Default: 64.
        conv_cfg (dict, optional): Config dict of convolution layers in motion
            encoder. Default: None.
        norm_cfg (dict, optional): Config dict of norm layer in motion encoder.
            Default: None.
        act_cfg (dict, optional): Config dict of activation layer in motion
            encoder. Default: None.
    """

    def __init__(
            self,
            radius: int,
            use_ref_cost: bool = False,
            use_disp16_pred:bool = False,
            gru_type: str = 'SeqConv',
            conv_type: str = 'Conv',
            cxt_channels: int = 32,
            h_channels: int = 32,
            feat_channels: int = 64,
            mask_channels: int = 64,
            iters_d16: int = 5,
            iters: int = 10,
            psize: Optional[Sequence[int]] = (1, 9),
            psize_d8: Optional[Sequence[int]] = (1, 9),
            frozen_features: bool = False,
            conv_cfg: Optional[dict] = None,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None,
            disp_loss: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.h_channels = h_channels
        self.iters_d16 = iters_d16
        self.iters = iters
        self.frozen_features = frozen_features

        self.use_ref_cost = use_ref_cost
        if self.use_ref_cost:
            self.ref_conv_dw16 = nn.Sequential(
                nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.ref_conv_dw8 = nn.Sequential(
                nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            extra_channels = 16
        else:
            extra_channels = 0
        self.use_disp16_pred = use_disp16_pred
        if self.use_disp16_pred:
            self.init_conv_dw16 = nn.Conv2d(self.h_channels*2, self.h_channels*2, kernel_size=3, stride=1, padding=1)
            self.init_conv_dw8 = nn.Conv2d(self.h_channels*2+1, self.h_channels*2, kernel_size=3, stride=1, padding=1)
        else:
            self.init_conv_dw16 = nn.Conv2d(self.h_channels*2, self.h_channels*2, kernel_size=3, stride=1, padding=1)
            self.init_conv_dw8 = nn.Conv2d(self.h_channels*2, self.h_channels*2, kernel_size=3, stride=1, padding=1)
        self.psize = psize
        self.psize_d8 = psize_d8

        # adaptive search
        self.search_num = 9
        self.update_block_8 = CntUpdateBlock(radius, extra_channels, gru_type, conv_type,
                                               h_channels, cxt_channels, feat_channels, mask_channels,
                                               conv_cfg, norm_cfg, act_cfg)
        self.update_block_16 = CntUpdateBlock(radius, extra_channels, gru_type, conv_type,
                                               h_channels, cxt_channels, feat_channels, mask_channels,
                                               conv_cfg, norm_cfg, act_cfg)

        if disp_loss is not None:
            self.loss_names = []
            self.loss_types = []
            for name, loss_setting in disp_loss.items():
                setattr(self, name, build_loss(loss_setting))
                self.loss_names.append(name)
                self.loss_types.append(loss_setting['type'])

    def zero_init(self, fmap):
        N, C, H, W = fmap.shape
        _x = torch.zeros([N, 1, H, W], dtype=torch.float32)
        zero_disp = _x.to(fmap.device)
        return zero_disp

    def forward(self, fmaps1, fmaps2, disp_init=None, test_mode=False, contexts = None, ref_depth = None):
        # feature
        fmap1_16 = fmaps1[0]
        fmap2_16 = fmaps2[0]

        fmap1_8 = fmaps1[1]
        fmap2_8 = fmaps2[1]

        if self.use_ref_cost:
            b = ref_depth.shape[0]
            ref_median, _ = torch.median(ref_depth.view(b, -1), dim=1, keepdim=False)
            ref_median = ref_median.view(b, 1, 1, 1)
            ref_depth = ref_depth - ref_median
            ref_std = torch.std(torch.abs(ref_depth).view(b, -1), dim=1).view(b, 1, 1, 1)
            ref_depth = ref_depth / ref_std

        # outputs
        predictions = []
        disp = None
        disp_up = None
        disp_16 = None
        # context
        if contexts is None:
            cnet_dw16 = self.init_conv_dw16(fmap1_16)
            net_16, inp_16 = torch.split(cnet_dw16, [self.h_channels, self.h_channels], dim=1)

            cnet_dw8 = self.init_conv_dw16(fmap1_8) # TODO: here has a bug
            net_8, inp_8 = torch.split(cnet_dw8, [self.h_channels, self.h_channels], dim=1)
        else:
            cnet_dw16 = self.init_conv_dw16(contexts[0])
            net_16, inp_16 = torch.split(cnet_dw16, [self.h_channels, self.h_channels], dim=1)

            if not self.use_disp16_pred:
                cnet_dw8 = self.init_conv_dw8(fmap1_8) # TODO: here has a bug
                net_8, inp_8 = torch.split(cnet_dw8, [self.h_channels, self.h_channels], dim=1)

            disp_16 = self.update_block_16.disp_pred(net_16)
            up_mask = 0.25 * self.update_block_16.mask_pred(net_16)
            disp = convex_upsample(disp_16, up_mask, rate=8)
            disp_up = -2 * F.interpolate(
                disp,
                size=(2 * disp.shape[2], 2 * disp.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
            predictions.append(disp_up)

        dilate = (1, 1)
        # Cascaded refinement (1/16 + 1/8)
        if disp_init is not None:
            scale = fmap1_8.shape[2] / disp_init.shape[2]
            disp = -scale * F.interpolate(
                disp_init,
                size=(fmap1_8.shape[2], fmap1_8.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
        else:
            # zero initialization
            disp_16 = self.zero_init(fmap1_16) if disp_16 is None else disp_16

            # Recurrent Update Module
            # RUM: 1/16
            # for itr in range(self.iters // 2):
            for itr in range(self.iters_d16):
                disp_16 = disp_16.detach()
                out_corrs = corr_iter(fmap1_16, fmap2_16, disp_16, self.psize, dilate)
                if self.use_ref_cost:
                    prev_disp = predictions[-1].detach()
                    median, _ = torch.median(prev_disp.view(b, -1), dim=1, keepdim=False)
                    median = median.view(b, 1, 1, 1)
                    std = torch.std(torch.abs(prev_disp - median).view(b, -1), dim=1).view(b, 1, 1, 1)
                    aligned_ref_depth = ref_depth * std + median
                    cost = torch.abs(prev_disp - aligned_ref_depth)
                    cost = F.interpolate(cost, size=(cost.shape[2]//2, cost.shape[3]//2), mode='bilinear', align_corners=True)
                    b, _, h, w = cost.shape
                    cost_feat = cost.view(b, h//8, 8, w//8, 8).permute(0, 2, 4, 1, 3).contiguous().view(b, 64, h//8, w//8)
                    cost_feat = self.ref_conv_dw16(cost_feat)
                    net_16, up_mask, delta_disp = self.update_block_16(
                        net_16, inp_16, out_corrs, disp_16, output_all=True, extra_feats=cost_feat)
                else:
                    net_16, up_mask, delta_disp = self.update_block_16(
                        net_16, inp_16, out_corrs, disp_16, output_all=True)

                disp_16 = disp_16 + delta_disp
                disp = convex_upsample(disp_16, up_mask, rate=8)
                disp_up = -2 * F.interpolate(
                    disp,
                    size=(2 * disp.shape[2], 2 * disp.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                predictions.append(disp_up)

            scale = fmap1_8.shape[2] / disp.shape[2]
            disp = scale * F.interpolate(
                disp,
                size=(fmap1_8.shape[2], fmap1_8.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
        
        if self.use_disp16_pred and contexts is not None:
            disp = disp.detach()
            cxt_dw8_in = torch.cat([contexts[1], disp], dim=1)
            cnet_dw8 = self.init_conv_dw8(cxt_dw8_in)
            net_8, inp_8 = torch.split(cnet_dw8, [self.h_channels, self.h_channels], dim=1)
            disp = disp + self.update_block_8.disp_pred(net_8)
            up_mask = 0.25 * self.update_block_8.mask_pred(net_8)
            disp_up = -convex_upsample(disp, up_mask, rate=8)
            predictions.append(disp_up)


        # RUM: 1/8
        for itr in range(self.iters):
            disp = disp.detach()
            out_corrs = corr_iter(fmap1_8, fmap2_8, disp, self.psize_d8, dilate)
            if self.use_ref_cost:
                    prev_disp = predictions[-1].detach()
                    median, _ = torch.median(prev_disp.view(b, -1), dim=1, keepdim=False)
                    median = median.view(b, 1, 1, 1)
                    std = torch.std(torch.abs(prev_disp - median).view(b, -1), dim=1).view(b, 1, 1, 1)
                    aligned_ref_depth = ref_depth * std + median
                    cost = torch.abs(prev_disp - aligned_ref_depth)
                    b, _, h, w = cost.shape
                    cost_feat = cost.view(b, h//8, 8, w//8, 8).permute(0, 2, 4, 1, 3).contiguous().view(b, 64, h//8, w//8)
                    cost_feat = self.ref_conv_dw8(cost_feat)
                    net_8, up_mask, delta_disp = self.update_block_8(net_8, inp_8, out_corrs, disp, output_all=True, extra_feats=cost_feat)
            else:
                net_8, up_mask, delta_disp = self.update_block_8(net_8, inp_8, out_corrs, disp, output_all=True)

            disp = disp + delta_disp
            disp_up = -convex_upsample(disp, up_mask, rate=8)
            predictions.append(disp_up)

        if test_mode:
            return disp_up

        return predictions

    def forward_train(self,
                      fmaps1: torch.Tensor,
                      fmaps2: torch.Tensor,
                      flow_init: torch.Tensor,
                      test_mode: bool,
                      return_preds: bool,
                      disp_gt: torch.Tensor,
                      valid: Optional[torch.Tensor] = None,
                      contexts = None,
                      ref_depth = None,
                      *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward function when model training.

        Args:
            feat1 (Tensor): The feature from the left input image.
            feat2 (Tensor): The feature from the right input image.
            flow_init (Tensor): The init estimated flow from GRU cell.
            test_mode (bool): The flag of test mode.
            disp_gt (Tensor): The ground truth of disparity.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.

        Returns:
            Dict[str, Tensor]: The losses of model.
        """
        if self.frozen_features:
            fmaps1 = [fmap.detach() for fmap in fmaps1]
            fmaps2 = [fmap.detach() for fmap in fmaps2]
        disp_pred = self.forward(fmaps1, fmaps2, flow_init, test_mode, contexts, ref_depth)

        if return_preds:
            return self.losses(disp_pred, disp_gt, valid=valid, *args, **kwargs), disp_pred
        else:
            return self.losses(disp_pred, disp_gt, valid=valid, *args, **kwargs)

    def forward_test(self,
                     fmaps1: torch.Tensor,
                     fmaps2: torch.Tensor,
                     flow_init: torch.Tensor,
                     test_mode: bool = False,
                     contexts = None,
                     ref_depth = None,
                     img_metas=None) -> Sequence[Dict[str, np.ndarray]]:
        """Forward function when model training.

        Args:
            fmap1 (Tensor): The feature from the left input image.
            fmap2 (Tensor): The feature from the right input image.
            flow_init (Tensor): The init estimated flow from GRU cell.
            test_mode (bool): The flag of test mode.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the disparity to original ground truth size. Defaults to None.
        Returns:
            Sequence[Dict[str, ndarray]]: The batch of predicted disparity
                with the same size of images before augmentation.
        """
        disp_pred = self.forward(fmaps1, fmaps2, flow_init, test_mode, contexts, ref_depth)
        disp_result = disp_pred.permute(0, 2, 3, 1).cpu().data.numpy()
        #disp_result = [d.permute(0, 2, 3, 1).cpu().data.numpy() for d in disp_pred]
        # unravel batch dim
        disp_result = list(disp_result)
        disp_result = [dict(disp=f) for f in disp_result]
        #
        # disp_preds = self.forward(fmap1, fmap2, flow_init, test_mode)
        # disp_results = []
        # for disp_pred in disp_preds:
        #     disp = disp_pred.permute(0, 2, 3, 1).cpu().data.numpy()
        #     disp_results.append(disp)
        # disp_result = [dict(disp=f) for f in disp_results]

        return self.get_disp(disp_result, img_metas=img_metas)

    def forward_onnx_feat(self,
                          fmap1: torch.Tensor,
                          fmap2: torch.Tensor):
        """Forward function part 1.

        Args:
            fmap1 (Tensor): The feature from the left input image.
            fmap2 (Tensor): The feature from the right input image.
        """
        # context
        fmap1_dw16 = F.avg_pool2d(fmap1, 2, stride=2, padding=0, count_include_pad=False)
        fmap2_dw16 = F.avg_pool2d(fmap2, 2, stride=2, padding=0, count_include_pad=False)

        # context
        net, inp = torch.split(fmap1, [self.h_channels, self.h_channels], dim=1)
        net = torch.tanh(net)
        inp = F.relu(inp)
        net_dw16 = F.avg_pool2d(net, 2, stride=2, padding=0, count_include_pad=False)
        inp_dw16 = F.avg_pool2d(inp, 2, stride=2, padding=0, count_include_pad=False)


        return fmap1, fmap2, net, inp, fmap1_dw16, fmap2_dw16, net_dw16, inp_dw16

    def forward_onnx_dw16corr_update(self,
                                     fmap1_dw16: torch.Tensor,
                                     fmap2_dw16: torch.Tensor,
                                     net_dw16: torch.Tensor,
                                     inp_dw16: torch.Tensor,
                                     flow_dw16: torch.Tensor,
                                     coords0_dw16: torch.Tensor):
        """Forward function part 3.

        Args:
            fmap1_dw8 (Tensor): The small resolution feature from the left input image.
            fmap2_dw8 (Tensor): The small resolution feature from the right input image.
        """
        dw16_coords1 = coords0_dw16 + flow_dw16
        out_corrs = onnx_corr_iter(fmap1_dw16, fmap2_dw16, dw16_coords1)
        net_dw16, delta_flow = self.update_block(
            net_dw16, inp_dw16, out_corrs, flow_dw16, mask_flag=False)

        flow_dw16 = flow_dw16 + delta_flow

        return flow_dw16, net_dw16

    def forward_onnx_dw16flow_upsample(self, flow_dw16: torch.Tensor):
        scale = 2
        up_flow = -scale * F.interpolate(
            flow_dw16,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        return up_flow

    def forward_onnx_dw8corr_update(self,
                                    fmap1: torch.Tensor,
                                    fmap2: torch.Tensor,
                                    net: torch.Tensor,
                                    inp: torch.Tensor,
                                    flow: torch.Tensor,
                                    coords0: torch.Tensor):
        """Forward function part 4.

        Args:
            fmap1 (Tensor): The feature from the left input image.
            fmap2 (Tensor): The feature from the right input image.
        """
        for itr in range(self.iters):
            dw8_coords1 = coords0 + flow
            out_corrs = onnx_corr_iter(fmap1, fmap2, dw8_coords1)
            net, delta_flow = self.update_block(net, inp, out_corrs, flow, mask_flag=False)

            flow = flow + delta_flow

        # mask = self.update_block(net, mask_flag=True)
        return flow, net

    def forward_onnx_mask(self, net: torch.Tensor):
        """Forward function when model training.

        Args:
            feat1 (Tensor): The feature from the left input image.
            feat2 (Tensor): The feature from the right input image.
        Returns:
            Sequence[Dict[str, ndarray]]: The batch of predicted disparity
                with the same size of images before augmentation.
        """
        mask = self.update_block(net, mask_flag=True)
        return mask

    def forward_onnx_rumstereo(self,
                               fmap1: torch.Tensor,
                               fmap2: torch.Tensor):
        # context
        fmap1_dw8 = F.avg_pool2d(fmap1, 2, stride=2, padding=0, count_include_pad=False)
        fmap2_dw8 = F.avg_pool2d(fmap2, 2, stride=2, padding=0, count_include_pad=False)

        # context
        # net, inp = torch.split(fmap1, [self.h_channels, self.h_channels], dim=1)
        net = fmap1[:, 0:self.h_channels, :, :]
        inp = fmap1[:, self.h_channels:(self.h_channels + self.h_channels), :, :]
        net = torch.tanh(net)
        inp = F.relu(inp)
        net_dw8 = F.avg_pool2d(net, 2, stride=2, padding=0, count_include_pad=False)
        inp_dw8 = F.avg_pool2d(inp, 2, stride=2, padding=0, count_include_pad=False)

        flow_dw16 = self.zero_init(fmap1_dw8)
        dw16_coords0 = coords_grid(1, fmap1_dw8.shape[2], fmap1_dw8.shape[3], fmap1_dw8.device)

        for i in range(5):
            dw16_coords1 = dw16_coords0 + flow_dw16
            out_corrs = onnx_corr_iter(fmap1_dw8, fmap2_dw8, dw16_coords1)
            net_dw8, delta_flow = self.update_block(
                net_dw8, inp_dw8, out_corrs, flow_dw16, mask_flag=False)

            flow_dw16 = flow_dw16 + delta_flow

        # return flow_dw16, net_dw8
        scale = 2
        flow = scale * F.interpolate(
            flow_dw16,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )

        dw8_coords0 = coords_grid(fmap1_8.shape[0], fmap1_8.shape[2], fmap1_8.shape[3], fmap1_8.device)
        for itr in range(self.iters):
            dw8_coords1 = dw8_coords0 + flow
            out_corrs = onnx_corr_iter(fmap1, fmap2, dw8_coords1)
            net, delta_flow = self.update_block(net, inp, out_corrs, flow, mask_flag=False)

            flow = flow + delta_flow

        return flow, net

    def losses(self,
               disp_pred: Sequence[torch.Tensor],
               disp_gt: torch.Tensor,
               valid: torch.Tensor = None, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute disparity loss.

        Args:
            disp_pred (Sequence[Tensor]): The list of predicted disparity.
            disp_gt (Tensor): The ground truth of disparity.
            valid (Tensor, optional): The valid mask. Defaults to None.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """
        if disp_gt is None:
            return None
        loss = dict()
        for loss_name, loss_type in zip(self.loss_names, self.loss_types):
            loss_function = getattr(self, loss_name)
            if loss_type == 'SequenceFSLoss':
                loss[loss_name] = loss_function(disp_pred, kwargs['keysets'], kwargs['lambda_sets'])
            elif loss_type == 'SequencePseudoLoss':
                loss[loss_name] = loss_function(disp_pred, kwargs['pseudo_gt'], kwargs['pseudo_valid'])
            else:
                loss[loss_name] = loss_function(disp_pred, disp_gt, valid)
        return loss
