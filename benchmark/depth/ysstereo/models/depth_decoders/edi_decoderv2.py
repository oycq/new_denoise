# hardware-friendly implementation of EDI-Stereo decoder

import math
from typing import Dict, Optional, Sequence, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from ysstereo.ops.AGCL import AGCL, get_correlation, corr_iter, onnx_corr_iter
from ysstereo.models.builder import DECODERS, build_loss, build_encoder, build_components
from ysstereo.models.depth_decoders.base_decoder import BaseDecoder
from ysstereo.models.depth_decoders.decoder_submodules import DispEncoder, ConvGRU, SeqConvGRU, XHead
from ysstereo.models.depth_decoders.igev_decoder import ConvNextBlock, CostVolumeReg
from ysstereo.utils.utils import convex_upsample, coords_grid
from ysstereo.models.utils import HardTanh
from ysstereo.models.depth_losses.sequence_pseudo_loss import sequence_pseudo_loss
from ysstereo.models.depth_losses.sequence_fs_loss import sequence_fs_loss

class EffConvNextBlock(nn.Module):
    """
    1. use ReLU instead of GELU (no accuracy degration according to the paper)
    2. use BN instead of LN (only a slight degration according to the paper)
    """
    def __init__(self, dim, out_dim, exp_ratio:int=4, dw_kernel:Tuple[int] = (7,7),
                 layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=dw_kernel, stride=1,
            padding=(dw_kernel[0]//2, dw_kernel[1]//2), groups=dim,
        )
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, exp_ratio * out_dim, kernel_size=1)
        self.act = nn.ReLU()
        self.pwconv2 = nn.Conv2d(out_dim * exp_ratio, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.final = nn.Conv2d(dim, out_dim, kernel_size=1, padding=0)

    def forward(self, x):
        raw = x
        x = self.norm(self.dwconv(x))
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            # here can be fused into conv when export to onnx
            x = self.gamma * x
        x = self.final(raw + x)
        return x

class MultiUpdateBlock(nn.Module):
    UB_TYPE_DCNN = "dcnn"
    UB_TYPE_DCNNLITE = "dcnnlite"
    UB_TYPE_DCNNLITE2 = "dcnnlite2"
    UB_TYPE_CNBLOCK = "cnt"
    UB_TYPE_TUNED = "tuned"
    UB_TYPE_GRU = "gru"
    AVAILABLE_UB_TYPES = [
        UB_TYPE_CNBLOCK, UB_TYPE_DCNN,
        UB_TYPE_GRU, UB_TYPE_TUNED, UB_TYPE_DCNNLITE,
        UB_TYPE_DCNNLITE2,
    ]
    def __init__(self,
                 radius: int,
                 update_block_type:str,
                 net_type: str = 'Basic',
                 extra_channels: int = 0,
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

        # encode geo feature from corr & disp
        self.geo_encoder = DispEncoder(
            radius=radius, net_type=net_type,
            conv_type=conv_type, conv_cfg=conv_cfg,
            norm_cfg=norm_cfg, act_cfg=act_cfg)

        # init main update block
        self.update_block_type = update_block_type
        self._init_update_block(update_block_type)

        self.disp_pred = XHead(self.h_channels, self.feat_channels, 1, x='disp')

        self.mask_pred = XHead(
            self.h_channels, self.feat_channels, self.mask_channels, x='mask')
        
        # update func mapping
        self.update_funcs = {
            self.UB_TYPE_CNBLOCK : self._update_cnt,
            self.UB_TYPE_GRU : self._update_gru,
            self.UB_TYPE_DCNN : self._update_dcnn,
            self.UB_TYPE_TUNED : self._update_tuned,
            self.UB_TYPE_DCNNLITE : self._update_dcnn,
            self.UB_TYPE_DCNNLITE2 : self._update_dcnn,
        }

    def _init_update_block(self, update_block_type: str):
        cxt_channel = self.cxt_channels
        hidden_channel = self.h_channels
        motion_channel = self.geo_encoder.out_channels[0] + 1
        if update_block_type == self.UB_TYPE_CNBLOCK:
            self.refine = nn.ModuleList(
                [ConvNextBlock(hidden_channel + cxt_channel + motion_channel, hidden_channel)
                 for _ in range(2)])
        elif update_block_type == self.UB_TYPE_GRU:
            self.context_zqr_convs = nn.Sequential(
                nn.Conv2d(cxt_channel, hidden_channel*3, 3, padding=3//2),
                nn.BatchNorm2d(hidden_channel*3),
            )
            self.refine = ConvGRU(
                hidden_channel, motion_channel, False, False, 'SeqConv',
            )
        elif update_block_type == self.UB_TYPE_DCNN:
            self.conv1 = nn.Sequential(
                nn.Conv2d(hidden_channel + cxt_channel + motion_channel, hidden_channel * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(hidden_channel * 2),
                nn.LeakyReLU(negative_slope=0.1),
            )
            self.refine = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(hidden_channel * 2, hidden_channel * 2, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(hidden_channel * 2),
                        nn.LeakyReLU(negative_slope=0.1),
                    ) for _ in range(4)
                ]
            )
            self.net_out = nn.Sequential(
                nn.Conv2d(hidden_channel * 2, hidden_channel, 3, 1, 1),
                nn.Hardtanh(min_val=-4.0, max_val=4.0)   
            )
        elif update_block_type == self.UB_TYPE_DCNNLITE2:
            in_ch = int(hidden_channel * 1.5) // 8 * 8
            self.conv1 = nn.Sequential(
                nn.Conv2d(hidden_channel + cxt_channel + motion_channel, in_ch, 3, 1, 1, bias=False, groups=1),
                nn.BatchNorm2d(in_ch),
                nn.LeakyReLU(negative_slope=0.1),
            )
            self.refine = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(in_ch, hidden_channel * 2, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(hidden_channel * 2),
                        nn.LeakyReLU(negative_slope=0.1),
                    ),
                    nn.Sequential(
                        nn.Conv2d(hidden_channel * 2, hidden_channel*2, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(hidden_channel * 2),
                        nn.LeakyReLU(negative_slope=0.1),
                    ),
                    nn.Sequential(
                        nn.Conv2d(hidden_channel * 2, in_ch, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(in_ch),
                        nn.LeakyReLU(negative_slope=0.1),
                    ),
                ]
            )
            self.net_out = nn.Sequential(
                nn.Conv2d(in_ch, hidden_channel, 3, 1, 1),
                nn.Hardtanh(min_val=-4.0, max_val=4.0)   
            )
        elif update_block_type == self.UB_TYPE_DCNNLITE:
            # use group conv & relu for efficiency (x5m can fuse conv-bn-relu as a single op on hardware)
            self.conv1 = nn.Sequential(
                nn.Conv2d(hidden_channel + cxt_channel + motion_channel, hidden_channel * 2, 3, 1, 1, bias=False, groups=2),
                nn.BatchNorm2d(hidden_channel * 2),
                nn.ReLU(),
            )
            # use relu for efficiency (x5m can fuse conv-bn-relu as a single op on hardware)
            self.refine = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(hidden_channel * 2, hidden_channel * 2, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(hidden_channel * 2),
                        nn.ReLU(),
                    ) for _ in range(4)
                ]
            )
            # use clip instead of hardtanh
            self.net_out = nn.Sequential(
                nn.Conv2d(hidden_channel * 2, hidden_channel, 3, 1, 1),
                nn.Hardtanh(min_val=-4.0, max_val=4.0)
            )
        elif update_block_type == self.UB_TYPE_TUNED:
            self.conv1 = nn.Sequential(
                nn.Conv2d(cxt_channel + motion_channel, hidden_channel, 3, 1, 1, bias=False, groups=2),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU(),
            )
            self.refine = nn.ModuleList(
                [EffConvNextBlock(hidden_channel*2, hidden_channel, exp_ratio=4, dw_kernel=(5,7)) for _ in range(2)]
            )
        else:
            raise NotImplementedError

    def _update_cnt(self, net, cxt, motion_feat):
        cxt = torch.cat([cxt, motion_feat], dim=1)
        for blk in self.refine:
            net = blk(torch.cat([net, cxt], dim=1))
        return net

    def _update_gru(self, net, cxt, motion_feat):
        cz, cr, cq = list(self.context_zqr_convs(cxt).split(split_size=self.hidden_channel, dim=1))
        net = self.refine(net, cz, cr, cq, motion_feat)
        return net

    def _update_dcnn(self, net, cxt, motion_feat):
        mid_feat = self.conv1(
            torch.cat([net, cxt, motion_feat], dim=1),
        )
        for blk in self.refine:
            mid_feat = blk(mid_feat)
        net = self.net_out(mid_feat)
        return net
    
    def _update_tuned(self, net, cxt, motion_feat):
        # similar cnt update, but remove layernorm and
        # adjust kernel size & mid-channel for better efficiency
        motion_cxt = self.conv1(torch.cat([cxt, motion_feat], dim=1))
        for blk in self.refine:
            net = blk(torch.cat([net, motion_cxt], dim=1))
        return net

    def forward(self, net, inp, corr=None, disp=None, mask_flag=True, output_all=False):
        if output_all:
            motion_features = self.geo_encoder(corr, disp)
            net = self.update_funcs[self.update_block_type](net, inp, motion_features)
            delta_disp = self.disp_pred(net)
            # scale mask to balence gradients
            mask = .25 * self.mask_pred(net)
            return net, mask, delta_disp
        if mask_flag:
            # scale mask to balence gradients
            mask = .25 * self.mask_pred(net)
            return mask
        motion_features = self.encoder(corr, disp)
        net = self.update_funcs[self.update_block_type](net, inp, motion_features)
        delta_disp = self.disp_pred(net)

        return net, delta_disp

@DECODERS.register_module()
class EdiStereoV2Decoder(BaseDecoder):
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
            update_block_type: str = 'cnt',
            net_type: str = 'Basic',
            x5m_support: bool = False,
            use_3dcnn:bool=True,
            radius_d16: int = 24,
            lazy_mode:bool = False,
            sub_pixel_sample_nums: Sequence[int] = None,
            groups_d16: int = 4,
            gru_type: str = 'SeqConv',
            conv_type: str = 'Conv',
            cxt_channels: int = 32,
            h_channels: int = 32,
            feat_channels: int = 64,
            mask_channels: int = 64,
            iters: int = 10,
            psize: Optional[Sequence[int]] = (1, 9),
            psize_d8: Optional[Sequence[int]] = (1, 9),
            frozen_features: bool = False,
            conv_cfg: Optional[dict] = None,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None,
            disp_loss: Optional[dict] = None,
            pseudo_loss_for_init: bool = False,
    ) -> None:
        super().__init__()
        self.update_block_type = update_block_type
        self.pseudo_loss_for_init = pseudo_loss_for_init
        self.h_channels = h_channels
        self.iters = iters
        self.frozen_features = frozen_features
        self.init_conv_dw8 = nn.Conv2d(self.h_channels*2+1, self.h_channels*2, kernel_size=3, stride=1, padding=1)
        self.psize = psize
        self.psize_d8 = psize_d8

        # 1/16 cost volume
        self.sub_pixel_sample_nums = sub_pixel_sample_nums
        self.dw16_reg = CostVolumeReg(
            h_channels+cxt_channels, h_channels+cxt_channels, self.sub_pixel_sample_nums, x5m_support,
            lazy_mode, radius_d16, groups_d16, use_3dcnn, True,
        )
        # 1/16 convex mask head
        self.d16mask_corr_coder = nn.Sequential(
            nn.Conv2d(self.dw16_reg.corr_dims, cxt_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cxt_channels),
            nn.ReLU(), 
        )
        self.d16mask_feat1_coder = nn.Sequential(
            nn.Conv2d(h_channels+cxt_channels, cxt_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cxt_channels),
            nn.ReLU(), 
        )
        self.d16_maskhead = XHead(in_channels=cxt_channels, feat_channels=[feat_channels], x_channels= (3**2) * mask_channels, x='mask')
        # adaptive search
        self.search_num = 9
        self.update_block_8 = MultiUpdateBlock(
            radius, update_block_type, net_type, 0, conv_type,
            h_channels, cxt_channels, feat_channels, mask_channels,
            conv_cfg, norm_cfg, act_cfg
        )

        self.update_block_8_1 = MultiUpdateBlock(
            radius, update_block_type, net_type, 0, conv_type,
            h_channels, cxt_channels, feat_channels, mask_channels,
            conv_cfg, norm_cfg, act_cfg
        )
        
        self.d16_net_act = nn.Identity() # for cnt&tuned
        if "dcnn" in update_block_type:
            # just like dcnn update block's final activation
            self.d16_net_act = nn.HardTanh(min_val=-4.0, max_val=4.0)
        elif 'gru' in update_block_type:
            # just like gru 
            self.d16_net_act = nn.Tanh()

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

    def forward(self, fmaps1, fmaps2, contexts, disp_init=None, test_mode=False):
        # dw 16 reg
        fmap1_16, fmap2_16 = fmaps1[0], fmaps2[0]
        disp_16, corr_feat = self.dw16_reg(fmap1_16, fmap2_16, True)
        mask_feat = self.d16mask_corr_coder(corr_feat) + self.d16mask_feat1_coder(fmap1_16)
        mask = 0.25 * self.d16_maskhead(mask_feat)
        disp = convex_upsample(disp_16, mask, rate=8)
        disp_up = 2.0 * F.interpolate(
            disp,
            size=(disp.shape[2]*2, disp.shape[3]*2),
            mode="bilinear",
            align_corners=True,
        )
        # dw8 iteration
        predictions = [disp_up]
        fmap1_8 = fmaps1[1]
        fmap2_8 = fmaps2[1]

        dilate = (1, 1)
        # Cascaded refinement (1/16 + 1/8)
        disp = -2.0 * F.interpolate(
            disp_16, size=(fmap1_8.shape[2], fmap1_8.shape[3]),
            mode="bilinear", align_corners=True,
        )

        disp = disp.detach()
        cxt_dw8_in = torch.cat([contexts[1], disp], dim=1)
        cnet_dw8 = self.init_conv_dw8(cxt_dw8_in)
        net_8, inp_8 = torch.split(cnet_dw8, [self.h_channels, self.h_channels], dim=1)
        net_8 = self.d16_net_act(net_8) # act net
        disp = disp + self.update_block_8_1.disp_pred(net_8)
        up_mask = 0.25 * self.update_block_8_1.mask_pred(net_8)
        disp_up = -convex_upsample(disp, up_mask, rate=8)
        predictions.append(disp_up)

        # RUM: 1/8
        for itr in range(self.iters):
            disp = disp.detach()
            out_corrs = corr_iter(fmap1_8, fmap2_8, disp, self.psize_d8, dilate)
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
                      contexts: torch.Tensor,
                      flow_init: torch.Tensor,
                      test_mode: bool,
                      return_preds: bool,
                      disp_gt: torch.Tensor,
                      valid: Optional[torch.Tensor] = None,
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
        if self.frozen_features or kwargs.get("frozen_features", False):
            fmaps1 = [fmap.detach() for fmap in fmaps1]
            fmaps2 = [fmap.detach() for fmap in fmaps2]
        with autocast('cuda', enabled=False):
            fmaps1 = [f.float() for f in fmaps1]
            fmaps2 = [f.float() for f in fmaps2]
            if contexts is not None:
                contexts = [c.float() for c in contexts]
            # make sure decoder always run in fp32 mode
            disp_pred = self.forward(fmaps1, fmaps2, contexts, flow_init, test_mode)

        if return_preds:
            return self.losses(disp_pred, disp_gt, valid=valid, *args, **kwargs), disp_pred
        else:
            return self.losses(disp_pred, disp_gt, valid=valid, *args, **kwargs)

    def forward_test(self,
                     fmaps1: torch.Tensor,
                     fmaps2: torch.Tensor,
                     contexts: torch.Tensor,
                     flow_init: torch.Tensor,
                     test_mode: bool = False,
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
        with autocast('cuda', enabled=False):
            fmaps1 = [f.float() for f in fmaps1]
            fmaps2 = [f.float() for f in fmaps2]
            if contexts is not None:
                contexts = [c.float() for c in contexts]
            # make sure decoder always run in fp32 mode
            disp_pred = self.forward(fmaps1, fmaps2, contexts, flow_init, test_mode)
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

        loss = dict()
        reg_disp = disp_pred[0]
        disp_pred = disp_pred[1:]
        for loss_name, loss_type in zip(self.loss_names, self.loss_types):
            loss_function = getattr(self, loss_name)
            if loss_type == 'SequenceFSLoss':
                keysets, lambda_sets = kwargs['keysets'], kwargs['lambda_sets']
                loss[loss_name] = loss_function(disp_pred, keysets, lambda_sets)
                fs_weight = self.loss_weights[loss_name]
                loss['init_fs_loss'] = fs_weight * sequence_fs_loss([reg_disp], keysets, lambda_sets, gamma=1.0)
            elif loss_type == 'SequencePseudoLoss':
                loss[loss_name] = loss_function(disp_pred, kwargs['pseudo_gt'], kwargs['pseudo_valid'])
            else:
                loss[loss_name] = loss_function(disp_pred, disp_gt, valid)
        # use smooth l1 loss for regression disp from dw16 feat map
        disp_gt = disp_gt.squeeze(1)
        mag = (disp_gt**2).sqrt()
        if valid is None:
            valid = torch.ones_like(disp_gt)
        else:
            valid = valid.squeeze(1)
            valid = ((valid >= 0.5) & (mag < 400)).to(disp_gt)
        pred = reg_disp.squeeze(1)
        i_weight = 1.0
        oa_param = 1.0 # 0.5 for previous exps
        suppress_swing = False # True for previous exps
        factor = 0
        i_loss = F.smooth_l1_loss(disp_gt, pred, reduction='none')
        with torch.no_grad():
            oa_mask = (pred-disp_gt)>0.0
            oa_mask = 1.0 + (oa_param - 1.0) * oa_mask.float()
        i_loss = i_loss * oa_mask
        if suppress_swing:
            i_weight = i_weight + 2 ** (-disp_gt[valid.bool()]/3 + factor)
        disp_loss = (i_weight * i_loss[valid.bool()]).mean()
        loss['init_disp_loss'] = disp_loss.mean()
        if self.pseudo_loss_for_init:
            pseudo_loss = sequence_pseudo_loss(
                [reg_disp], kwargs['pseudo_gt'], 1.0, kwargs['pseudo_valid'], 0.2
            )
            loss['init_pseudo_loss'] = 0.2 * pseudo_loss
        return loss
