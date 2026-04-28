import math
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from ysstereo.ops.AGCL import AGCL, get_correlation, corr_iter, onnx_corr_iter, corr_iter_raft
from ysstereo.models.builder import DECODERS, build_loss, build_encoder, build_components
from ysstereo.models.depth_decoders.base_decoder import BaseDecoder
from ysstereo.models.depth_decoders.decoder_submodules import DispEncoder, ConvGRU, SeqConvGRU, XHead
from ysstereo.utils.utils import convex_upsample, coords_grid
from ysstereo.models.utils import HardTanh
from ysstereo.models.depth_losses.sequence_pseudo_loss import sequence_pseudo_loss
from ysstereo.models.depth_losses.sequence_fs_loss import sequence_fs_loss

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
        self.norm = torch.nn.LayerNorm(dim, eps=1e-6) #ckkqat
        self.pwconv1 = nn.Conv2d(dim, 128, kernel_size=1)
        self.norm1 = nn.SyncBatchNorm(128)
        self.act1 = nn.ReLU()
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(128, dim, kernel_size=1)
        self.norm2 = nn.SyncBatchNorm(dim)
        self.act2 = nn.ReLU()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.final = nn.Conv2d(dim, output_dim, kernel_size=1, padding=0)
        self.eps = 9.999999974752427e-7

    def forward(self, x):
        raw_x = x
        x = self.dwconv(x)
        x = x - torch.mean(x, dim=1, keepdim=True)
        x = x / torch.sqrt(torch.mean(torch.pow(x,2), dim=1, keepdim=True)+self.eps)
        x = self.pwconv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.final(raw_x + x)
        return x


class CntUpdateBlock(nn.Module):
    def __init__(self,
                 radius: int,
                 net_type: str = 'Basic',
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
            net_type=net_type,
            conv_type=conv_type,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,)
        self.gru_type = gru_type

        self.refine = []
        for i in range(2):
            self.refine.append(ConvNextBlock(cxt_channels+h_channels+self.encoder.out_channels[0]+1+extra_channels, h_channels))
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


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    # assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    # assert cost.shape == (B, num_groups, H, W)
    return cost

def groupwise_corrlection_via_group_conv(fea1, fea2, num_groups):
    # TODO: just a sample code for ck
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    g_conv = nn.Conv2d(C, out_channels=num_groups, kernel_size=1, groups=num_groups, bias=False)
    g_conv.weight.data = (1.0/(C //num_groups)) * torch.ones_like(g_conv.weight.data)
    cost = g_conv(fea1 * fea2)
    return cost

def build_gwc_volume_onnx(refimg_fea, targetimg_fea, maxdisp, num_groups, conv1x1, pad):
    B, C, H, W = 1, 64, 1280//16, 1024//16
    cost_list = []
    if pad is not None:
        targetimg_fea_pad = pad(targetimg_fea)
        for i in range(maxdisp):
            feat1 = refimg_fea
            feat2 = targetimg_fea_pad[:, :, :, i:(i+W)]
            cost = feat1 * feat2
            if conv1x1 is not None:
                cost = conv1x1(cost)
            else:
                cost = cost.view(B, num_groups, C//num_groups, H, W)
                cost = torch.mean(cost, dim=2)
            cost_list.append(cost)
    else:
        for i in range(maxdisp):
            if i==0:
                feat1 = refimg_fea
                feat2 = targetimg_fea
            else:
                feat1 = refimg_fea[:, :, :, i:]
                feat2 = targetimg_fea[:, :, :, :-i]
            cost = feat1 * feat2
            if conv1x1 is not None:
                cost = conv1x1(cost)
            else:
                cost = cost.view(B, num_groups, C//num_groups, H, W-i)
                cost = torch.mean(cost, dim=2)
            cost_list.append(cost)
    return cost_list

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, sub_pixel_nums=None):
    B, C, H, W = refimg_fea.shape
    if sub_pixel_nums is None:
        volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
        for i in range(maxdisp):
            if i > 0:
                volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                            num_groups)
            else:
                volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
        volume = volume.contiguous()
        return volume
    else:
        corr_dim = sum(sub_pixel_nums)+maxdisp
        volume = refimg_fea.new_zeros([B, num_groups, corr_dim, H, W])
        base_itr = 0
        for idx in range(len(sub_pixel_nums)+1):
            if idx>0:
                pre_sub_num = sub_pixel_nums[idx-1]
                right_idx = base_itr+pre_sub_num
                left_idx = base_itr-1
                volume[:, :, right_idx, : ,idx:] = groupwise_correlation(refimg_fea[:, :, :, idx:], targetimg_fea[:, :, :, :-idx],
                                                num_groups)
                weight_left =np.array([(p+1)/(pre_sub_num+1) for p in range(pre_sub_num)]).astype(np.float32)
                weight_left =torch.from_numpy(weight_left).to(volume.device).view(1, 1, -1, 1, 1)
                volume[:, :, (left_idx+1):right_idx] = volume[:, :, left_idx:(left_idx+1)] * weight_left + (1.0 - weight_left) * volume[:, :, right_idx:(right_idx+1)]
                base_itr = right_idx + 1
            else:
                volume[:, :, base_itr, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
                base_itr+=1
        for i in range(maxdisp):
            if i <=len(sub_pixel_nums):
                continue
            if base_itr > 0:
                volume[:, :, base_itr, :, i:] = groupwise_correlation(
                    refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups,
                )
            else:
                volume[:, :, base_itr, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
            base_itr+=1
        assert base_itr == corr_dim
        volume = volume.contiguous()
        return volume

class FeatAtt(nn.Module):
    def __init__(self, cv_ch, feat_ch):
        super().__init__()
        self.feat_att = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch//2, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(feat_ch//2), nn.ReLU(),
            nn.Conv2d(feat_ch//2, cv_ch, 1))

    def forward(self, cv, feat):
        feat_guide = torch.sigmoid(self.feat_att(feat))
        cv = cv * feat_guide.unsqueeze(1)
        return cv

class CostVolumeReg(nn.Module):
    def __init__(self, in_feat_ch:int, corr_feat_ch:int=None, sub_pixel_sample_nums:Sequence[int]=None, x5m_support:bool=False,
                 lazy_mode:bool=False, srange:int=24, ngroups:int=4, use_3dcnn:bool=True, use_softmaxout:bool=True):
        super().__init__()
        self.srange = srange
        self.x5m_support = x5m_support
        assert self.x5m_support==True or self.x5m_support==False or self.x5m_support=='gn0' or self.x5m_support=='gn1'
        if self.x5m_support == False:
            self.x5m_support = None
        self.corr_dims = srange
        self.lazy_mode = lazy_mode
        if sub_pixel_sample_nums is not None:
            self.corr_dims += sum(sub_pixel_sample_nums)
        self.sub_pixel_sample_nums = sub_pixel_sample_nums
        self.ngroups = ngroups
        self.g_conv = nn.Conv2d(
            in_feat_ch, self.ngroups, 1, 1, 0, groups=self.ngroups, bias=False
        )
        #self.g_conv.weight.fill_(1.0/(in_feat_ch//self.ngroups))
        self.gc_pad = nn.ModuleList()
        for i in range(self.srange):
            self.gc_pad.append(nn.ZeroPad2d(padding=(i, 0, 0, 0)))
        corr_feat_ch = corr_feat_ch if corr_feat_ch is not None else in_feat_ch
        if corr_feat_ch != in_feat_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_feat_ch, corr_feat_ch, kernel_size=1, padding=0, stride=1),
                nn.SyncBatchNorm(corr_feat_ch),
                nn.ReLU()
            )
        else:
            self.proj = lambda x:x
        # 3d cnn for cost encode
        self.use_3dcnn = use_3dcnn
        if use_3dcnn:
            self.corr_stem = nn.Sequential(
                nn.Conv3d(ngroups, out_channels=ngroups, bias=False, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(ngroups),
                nn.ReLU(),
            )
            self.corr_stem_att = FeatAtt(self.corr_dims, corr_feat_ch)
            # simple unet
            self.conv1 = nn.Sequential(
                nn.Conv3d(ngroups, out_channels=ngroups*2, bias=False, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(ngroups*2),
                nn.ReLU(),
                nn.Conv3d(ngroups*2, out_channels=ngroups*2, bias=False, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(ngroups*2),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv3d(ngroups*2, out_channels=ngroups*4, bias=False, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(ngroups*4),
                nn.ReLU(),
                nn.Conv3d(ngroups*4, out_channels=ngroups*4, bias=False, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(ngroups*4),
                nn.ReLU(),
            )
            
            guide_feat_ch1 = (int(corr_feat_ch * 1.5) // 8) * 8
            self.guide_feat_proj1 = nn.Sequential(
                nn.Conv2d(corr_feat_ch, guide_feat_ch1, kernel_size=3, stride=2, padding=1),
                nn.SyncBatchNorm(guide_feat_ch1), nn.ReLU(),
            )
            self.feat_att1 = FeatAtt(self.corr_dims//2, guide_feat_ch1)

            guide_feat_ch2 = (int(corr_feat_ch * 2) // 8) * 8
            self.guide_feat_proj2 = nn.Sequential(
                nn.Conv2d(guide_feat_ch1, guide_feat_ch2, kernel_size=3, stride=2, padding=1),
                nn.SyncBatchNorm(guide_feat_ch2), nn.ReLU(),
            )
            self.feat_att2 = FeatAtt(self.corr_dims//4, guide_feat_ch2)

            if self.x5m_support:
                if self.x5m_support=='gn0':
                    group_num = ngroups*2
                elif self.x5m_support=='gn1':
                    group_num = ngroups
                else:
                    group_num = 1
                self.conv2_up = nn.Sequential(
                    nn.ConvTranspose2d(ngroups*4*(self.corr_dims//4), ngroups*2 * (self.corr_dims//2),
                                       kernel_size=(4,4), stride=(2,2), padding=(1,1), groups=group_num),
                    nn.SyncBatchNorm(ngroups*2 * (self.corr_dims//2)), nn.ReLU(),
                )
                self.conv2_up_proj = nn.Sequential(
                    nn.Conv3d(ngroups*2, ngroups*2, kernel_size=1),
                    nn.BatchNorm3d(ngroups*2), nn.ReLU(),
                )
            else:
                self.conv2_up = nn.Sequential(
                    nn.ConvTranspose3d(ngroups*4, ngroups*2, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1)),
                    nn.BatchNorm3d(ngroups*2), nn.ReLU(),
                )
            self.agg_0 = nn.Sequential(
                nn.Conv3d(ngroups*4, ngroups*2, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(ngroups*2), nn.ReLU(),
                nn.Conv3d(ngroups*2, ngroups*2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(ngroups*2), nn.ReLU(),
            )
            if self.x5m_support is not None:
                if self.x5m_support=='gn0':
                    group_num = ngroups
                elif self.x5m_support=='gn1':
                    group_num = ngroups
                else:
                    group_num = 1
                self.conv1_up = nn.Sequential(
                    nn.ConvTranspose2d(ngroups*2*(self.corr_dims//2), ngroups * (self.corr_dims),
                                       kernel_size=(4,4), stride=(2,2), padding=(1,1), groups=group_num),
                    nn.SyncBatchNorm(ngroups*self.corr_dims), nn.ReLU(),
                )
                self.conv1_up_proj = nn.Sequential(
                    nn.Conv3d(ngroups, ngroups, kernel_size=1),
                    nn.BatchNorm3d(ngroups), nn.ReLU(),
                )
            else:
                self.conv1_up = nn.Sequential(
                    nn.ConvTranspose3d(ngroups*2, ngroups*1, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1)),
                    nn.BatchNorm3d(ngroups*1), nn.ReLU(),
                )
            self.agg_1 = nn.Sequential(
                nn.Conv3d(ngroups*2, ngroups*1, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(ngroups*1), nn.ReLU(),
                nn.Conv3d(ngroups*1, ngroups*1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(ngroups*1), nn.ReLU(),
            )
            self.out_att = FeatAtt(self.corr_dims, corr_feat_ch)
            self.corr_out = nn.Conv3d(ngroups, 1, 3, 1, 1, bias=False)
            if self.lazy_mode and use_softmaxout:
                self.corr_lazy = nn.Sequential(
                    nn.Conv2d(self.corr_dims, corr_feat_ch//2, kernel_size=3, padding=1, bias=False),
                    nn.SyncBatchNorm(corr_feat_ch//2),
                    nn.ReLU(),
                )
                self.feat_lazy = nn.Sequential(
                    nn.Conv2d(corr_feat_ch, corr_feat_ch//2, kernel_size=3, padding=1, bias=False),
                    nn.SyncBatchNorm(corr_feat_ch//2),
                    nn.ReLU(),
                )
                self.lazy_out = nn.Sequential(
                    nn.Conv2d(corr_feat_ch//2, 1, kernel_size=3, padding=1),
                    nn.ReLU()
                )
        else:
            ngroups = self.ngroups
            feat_ch = ngroups*self.srange
            self.corr_stem = nn.Sequential(
                nn.Conv2d(feat_ch, out_channels=feat_ch, bias=False, kernel_size=3, stride=1, padding=1),
                nn.SyncBatchNorm(feat_ch),
                nn.ReLU(),
            )
            self.corr_stem_att = FeatAtt(self.corr_dims, corr_feat_ch)
            self.att_map = nn.Conv2d(self.corr_dims, feat_ch, kernel_size=1)
            # simple unet
            self.conv1 = nn.Sequential(
                nn.Conv2d(feat_ch, out_channels=feat_ch, bias=False, kernel_size=3, stride=2, padding=1),
                nn.SyncBatchNorm(feat_ch),
                nn.ReLU(),
                nn.Conv2d(feat_ch, out_channels=feat_ch, bias=False, kernel_size=3, stride=1, padding=1),
                nn.SyncBatchNorm(feat_ch),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(feat_ch, out_channels=feat_ch, bias=False, kernel_size=3, stride=2, padding=1),
                nn.SyncBatchNorm(feat_ch),
                nn.ReLU(),
                nn.Conv2d(feat_ch, out_channels=feat_ch, bias=False, kernel_size=3, stride=1, padding=1),
                nn.SyncBatchNorm(feat_ch),
                nn.ReLU(),
            )
            
            guide_feat_ch1 = (int(corr_feat_ch * 1.5) // 8) * 8
            self.guide_feat_proj1 = nn.Sequential(
                nn.Conv2d(corr_feat_ch, guide_feat_ch1, kernel_size=3, stride=2, padding=1),
                nn.SyncBatchNorm(guide_feat_ch1), nn.ReLU(),
            )
            self.feat_att1 = FeatAtt(self.corr_dims//2, guide_feat_ch1)
            self.att1_map = nn.Conv2d(self.corr_dims//2, feat_ch, kernel_size=1)
            guide_feat_ch2 = (int(corr_feat_ch * 2) // 8) * 8
            self.guide_feat_proj2 = nn.Sequential(
                nn.Conv2d(guide_feat_ch1, guide_feat_ch2, kernel_size=3, stride=2, padding=1),
                nn.SyncBatchNorm(guide_feat_ch2), nn.ReLU(),
            )
            self.feat_att2 = FeatAtt(self.corr_dims//4, guide_feat_ch2)
            self.att2_map = nn.Conv2d(self.corr_dims//4, feat_ch, kernel_size=1)

            self.conv1_up = nn.Sequential(
                nn.ConvTranspose2d(feat_ch, feat_ch,
                                    kernel_size=(4,4), stride=(2,2), padding=(1,1), groups=ngroups),
                nn.SyncBatchNorm(feat_ch), nn.ReLU(),
            )
            self.conv1_up_proj = nn.Sequential(
                nn.Conv2d(feat_ch, feat_ch, kernel_size=1),
                nn.SyncBatchNorm(feat_ch), nn.ReLU(),
            )

            self.conv2_up = nn.Sequential(
                nn.ConvTranspose2d(feat_ch, feat_ch,
                                    kernel_size=(4,4), stride=(2,2), padding=(1,1), groups=ngroups),
                nn.SyncBatchNorm(feat_ch), nn.ReLU(),
            )
            self.conv2_up_proj = nn.Sequential(
                nn.Conv2d(feat_ch, feat_ch, kernel_size=1),
                nn.SyncBatchNorm(feat_ch), nn.ReLU(),
            )

            self.agg_0 = nn.Sequential(
                nn.Conv2d(feat_ch*2, feat_ch, kernel_size=1, stride=1, padding=0),
                nn.SyncBatchNorm(feat_ch), nn.ReLU(),
                nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=1, padding=1),
                nn.SyncBatchNorm(feat_ch), nn.ReLU(),
            )

            self.agg_1 = nn.Sequential(
                nn.Conv2d(feat_ch*2, feat_ch, kernel_size=1, stride=1, padding=0),
                nn.SyncBatchNorm(feat_ch), nn.ReLU(),
                nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=1, padding=1),
                nn.SyncBatchNorm(feat_ch), nn.ReLU(),
            )
            self.out_att = FeatAtt(self.corr_dims, corr_feat_ch)
            self.atto_map = nn.Conv2d(self.corr_dims, feat_ch, 1)
            self.corr_out = nn.Conv2d(feat_ch, self.srange, 3, 1, 1, bias=False)
        self.use_softmaxout = use_softmaxout
        if not self.use_softmaxout:
            self.out_layer = nn.Sequential(
                nn.Conv2d(self.corr_dims, self.corr_dims*2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.corr_dims*2, 1, kernel_size=3, stride=1, padding=0),
            )
        else:
            if self.sub_pixel_sample_nums is None:
                self.softmax_bins = nn.Parameter(
                    torch.linspace(0, srange-1, steps=srange, dtype=torch.float32).view(1, -1, 1, 1),
                    requires_grad=False,
                )
            else:
                bins = []
                for idx, sub_num in enumerate(sub_pixel_sample_nums):
                    bins.append(float(idx))
                    for j in range(sub_num):
                        bins.append(float(idx)+float(j+1) * (1.0/(sub_num+1)))
                bins.extend(list(range(len(sub_pixel_sample_nums), self.srange, 1)))
                assert len(bins) == self.corr_dims
                self.softmax_bins = nn.Parameter(
                    torch.from_numpy(np.array(bins).astype(np.float32)).view(1, -1, 1, 1),
                    requires_grad=False,
                )

    def add_depoly_modules(self, b=1, h=1280//16, w=1024//16):
        self.b = b
        self.h = h
        self.w = w
        self.gc_pad = nn.ModuleList()
        for i in range(self.srange):
            self.gc_pad.append(nn.ZeroPad2d(padding=(i, 0, 0, 0)))
        self.cost_conv_1x1 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, groups=4)
        self.global_pad = nn.ZeroPad2d(padding=(self.srange, 0, 0, 0))

    def _get_disp_from_corr(self, corr):
        if self.use_softmaxout:
            prob = F.softmax(corr, dim=1)
            disp = torch.sum(self.softmax_bins * prob, dim=1, keepdim=True)
            return disp
        else:
            return self.out_layer(corr)

    def build_gwc_via_gcn(self, refimg_fea, targetimg_fea):
        corr_list = []
        for i in range(self.srange):
            if i==0:
                feat1 = refimg_fea
                feat2 = targetimg_fea
            else:
                feat1 = refimg_fea[:, :, :, i:]
                feat2 = targetimg_fea[:, :, :, :-i]
            cost = self.g_conv(feat1 * feat2)
            corr_list.append(self.gc_pad[i](cost))
        return corr_list

    def forward_onnx(self, feat1, feat2, return_corr_feat:bool=False, 
                     use_cost_conv:bool=False, use_global_pad:bool=False):
        feat1 = self.proj(feat1)
        feat2 = self.proj(feat2)
        if use_global_pad:
            corr_list = build_gwc_volume_onnx(
                feat1, feat2, self.srange, 4,
                self.cost_conv_1x1 if use_cost_conv else None, self.global_pad,
            )
            corr = torch.stack(corr_list, dim=2)
        else:
            corr_list = build_gwc_volume_onnx(
                feat1, feat2, self.srange, 4,
                self.cost_conv_1x1 if use_cost_conv else None,
            )
            paded_list = [self.gc_pad[i](corr_list[i]) for i in range(self.srange)]
            corr = torch.stack(paded_list, dim=2)
        stem_feat = self.corr_stem(corr)
        stem_feat = self.corr_stem_att(corr, feat1)
        enc_feats = [stem_feat]
        guide_feat1 = self.guide_feat_proj1(feat1)
        enc_feats.append(self.feat_att1(self.conv1(enc_feats[-1]), guide_feat1))
        guide_feat2 = self.guide_feat_proj2(guide_feat1)
        enc_feats.append(self.feat_att2(self.conv2(enc_feats[-1]), guide_feat2))
        dec_feat = []
        b, c, d, h, w = 1, 4*4, 24//4, 80//4, 64//4
        c1, d1, h1, w1 = 4*2, 24//2, 80//2, 64//2
        c2, d2, h2, w2 = 4, 24, 80, 64
        dec_feat0 = self.conv2_up(enc_feats[-1].view(b, c*d, h, w))
        dec_feat0 = self.conv2_up_proj(dec_feat0.view(b, c1, d1, h1, w1))
        dec_feat1 = self.agg_0(torch.cat((dec_feat0, enc_feats[-2]), dim=1))
        dec_feat1 = self.conv1_up(dec_feat1.view(b, c1*d1, h1, w1))
        dec_feat1 = self.conv1_up_proj(dec_feat1.view(b, c2, d2, h2, w2))
        dec_feat.append(dec_feat0)
        dec_feat.append(dec_feat1)
        dec_feat.append(self.agg_1(torch.cat((dec_feat[-1], enc_feats[0]), dim=1)))
        out_corr_feat = self.corr_out(self.out_att(dec_feat[-1], feat1)).squeeze(1)
        out_disp = self._get_disp_from_corr(out_corr_feat)
        return out_disp

    def forward(self, feat1, feat2, return_corr_feat:bool=False):
        feat1 = self.proj(feat1)
        feat2 = self.proj(feat2)
        #corr = self._compute_corr(feat1, feat2)
        corr_list = self.build_gwc_via_gcn(feat1, feat2)
        corr = torch.concat(corr_list, dim=1)
        if self.use_3dcnn:
            stem_feat = self.corr_stem(corr)
            stem_feat = self.corr_stem_att(corr, feat1)
            enc_feats = [stem_feat]
            guide_feat1 = self.guide_feat_proj1(feat1)
            enc_feats.append(self.feat_att1(self.conv1(enc_feats[-1]), guide_feat1))
            guide_feat2 = self.guide_feat_proj2(guide_feat1)
            enc_feats.append(self.feat_att2(self.conv2(enc_feats[-1]), guide_feat2))

            dec_feat = []
            if self.x5m_support:
                b, c, d, h, w = enc_feats[-1].shape
                _, c1, d1, h1, w1 = enc_feats[-2].shape
                _, c2, d2, h2, w2 = enc_feats[-3].shape
                dec_feat0 = self.conv2_up(enc_feats[-1].view(b, c*d, h, w))
                dec_feat0 = self.conv2_up_proj(dec_feat0.view(b, c1, d1, h1, w1))
                dec_feat1 = self.agg_0(torch.cat((dec_feat0, enc_feats[-2]), dim=1))
                dec_feat1 = self.conv1_up(dec_feat1.view(b, c1*d1, h1, w1))
                dec_feat1 = self.conv1_up_proj(dec_feat1.view(b, c2, d2, h2, w2))
                dec_feat.append(dec_feat0)
                dec_feat.append(dec_feat1)
            else:
                dec_feat.append(self.conv2_up(enc_feats[-1]))
                dec_feat.append(self.conv1_up(self.agg_0(torch.cat((dec_feat[-1], enc_feats[-2]), dim=1))))
            dec_feat.append(self.agg_1(torch.cat((dec_feat[-1], enc_feats[0]), dim=1)))
            out_corr_feat = self.corr_out(self.out_att(dec_feat[-1], feat1)).squeeze(1)
            if self.lazy_mode:
                lazy = self.lazy_out(self.corr_lazy(out_corr_feat) + self.feat_lazy(feat1))
                # inject position lazy prediction (serve as very small disp segmentation)
                zero_disp_feat = out_corr_feat[:,0:1]+lazy
                out_corr_feat = torch.cat((zero_disp_feat, out_corr_feat[:,1:]), dim=1)
            out_disp = self._get_disp_from_corr(out_corr_feat)
            if return_corr_feat:
                return out_disp, out_corr_feat
            else:
                return out_disp
        else:
            # conver 5d corr tensor to 4d corr
            # b, n, d, h, w = corr.shape
            # corr = corr.view(b, n*d, h, w)

            # encode
            stem_feat = self.corr_stem(corr)
            attn = self.att_map(torch.sigmoid(self.corr_stem_att.feat_att(feat1)))
            stem_feat = stem_feat * attn
            enc_feats = [stem_feat]

            guide_feat1 = self.guide_feat_proj1(feat1)
            efeat1 = self.conv1(enc_feats[-1])
            attn1 = self.att1_map(torch.sigmoid(self.feat_att1.feat_att(guide_feat1)))
            enc_feats.append(attn1*efeat1)

            guide_feat2 = self.guide_feat_proj2(guide_feat1)
            efeat2 = self.conv2(enc_feats[-1])
            attn2 = self.att2_map(torch.sigmoid(self.feat_att2.feat_att(guide_feat2)))
            enc_feats.append(attn2*efeat2)

            dec_feat = []
            dec_feat0 = self.conv2_up_proj(self.conv2_up(enc_feats[-1]))
            dec_feat1 = self.agg_0(torch.cat((dec_feat0, enc_feats[-2]), dim=1))
            dec_feat1 = self.conv1_up_proj(self.conv1_up(dec_feat1))
            dec_feat.append(dec_feat0)
            dec_feat.append(dec_feat1)
            dec_feat.append(
                self.agg_1(torch.cat((dec_feat[-1], enc_feats[0]), dim=1))
            )
            out_corr_feat = self.atto_map(torch.sigmoid(self.out_att.feat_att(feat1))) * dec_feat[-1]
            out_corr_feat = self.corr_out(out_corr_feat)
            out_disp = self._get_disp_from_corr(out_corr_feat)
            if return_corr_feat:
                return out_disp, out_corr_feat
            else:
                return out_disp

    def _compute_corr(self, feat1, feat2):
        return build_gwc_volume(feat1, feat2, maxdisp=self.srange, num_groups=self.ngroups,
                                sub_pixel_nums=self.sub_pixel_sample_nums)

@DECODERS.register_module()
class IGEVStereoSlimDecoder(BaseDecoder):
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
            use_depth_head: bool = False, # add a depth head for d16, wish a inf mask & a depth head
            use_remap_corr: bool = False, # use nonregular corr, 0, 1.0,1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 14, 20, 8*2+16*3+32*3+64+128
            use_raft_corr: bool = False,
            net_type: str = 'Basic',
            use_cvx_scale: bool = True,
            x5m_support: bool = False,
            use_3dcnn:bool=True,
            radius_d16: int = 24,
            lazy_mode:bool = False,
            sub_pixel_sample_nums: Sequence[int] = None,
            groups_d16: int = 4,
            use_disp16_pred:bool = False,
            gru_type: str = 'SeqConv',
            conv_type: str = 'Conv',
            cxt_channels: int = [32, 32],
            h_channels: int = [32, 32],
            in_index: int = [1, 2],
            feat_channels: int = [64, 64],
            mask_channels: int = 64,
            iters: int = 10,
            psize: Optional[Sequence[int]] = (1, 9),
            psize_d8: Optional[Sequence[int]] = (1, 9),
            truncated_grad: bool = False,
            conv_cfg: Optional[dict] = None,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None,
            disp_loss: Optional[dict] = None,
            pseudo_loss_for_init: bool = False,
    ) -> None:
        super().__init__()
        self.use_depth_head = use_depth_head
        self.use_remap_corr = use_remap_corr
        self.use_raft_corr = use_raft_corr
        self.use_cvx_scale = use_cvx_scale
        self.pseudo_loss_for_init = pseudo_loss_for_init
        self.h_channels = h_channels
        self.in_index = in_index,
        self.iters = iters
        self.truncated_grad  = truncated_grad
        self.use_disp16_pred = use_disp16_pred
        if self.use_disp16_pred:
            self.init_conv_dw8 = nn.Conv2d(self.h_channels[-2]*2+1, self.h_channels[-2]*2, kernel_size=3, stride=1, padding=1)
        else:
            self.init_conv_dw8 = nn.Conv2d(self.h_channels[-2]*2, self.h_channels[-2]*2, kernel_size=3, stride=1, padding=1)
        self.psize = psize
        self.psize_d8 = psize_d8
        self.small_res_regw = 1.0
        self.init_max_disp = 400.0

        # 1/16 cost volume
        self.sub_pixel_sample_nums = sub_pixel_sample_nums
        self.dw16_reg = CostVolumeReg(
            h_channels[-1]+cxt_channels[-1], h_channels[-1]+cxt_channels[-1], self.sub_pixel_sample_nums, x5m_support,
            lazy_mode, radius_d16, groups_d16, use_3dcnn, True,
        )
        # 1/16 convex mask head
        self.d16mask_corr_coder = nn.Sequential(
            nn.Conv2d(self.dw16_reg.corr_dims, cxt_channels[-1], kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(cxt_channels[-1]),
            nn.ReLU(), 
        )
        self.d16mask_feat1_coder = nn.Sequential(
            nn.Conv2d(h_channels[-1]+cxt_channels[-1], cxt_channels[-1], kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(cxt_channels[-1]),
            nn.ReLU(), 
        )
        self.d16_maskhead = XHead(in_channels=cxt_channels[-1], feat_channels=[feat_channels[-1]], x_channels= (3**2) * mask_channels, x='mask')
        # adaptive search
        self.search_num = 9
        self.update_block_8 = CntUpdateBlock(radius, net_type, 0, gru_type, conv_type,
                                               h_channels[-2], cxt_channels[-2], feat_channels[-2], mask_channels,
                                               conv_cfg, norm_cfg, act_cfg)
        self.update_block_8_1 = CntUpdateBlock(radius, net_type, 0, gru_type, conv_type,
                                               h_channels[-2], cxt_channels[-2], feat_channels[-2], mask_channels,
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
        # context
        if contexts is None:
            cnet_dw8 = self.init_conv_dw8(fmap1_8) # TODO: here has a bug
            net_8, inp_8 = torch.split(cnet_dw8, [self.h_channels[-2], self.h_channels[-2]], dim=1)
        else:
            if not self.use_disp16_pred:
                cnet_dw8 = self.init_conv_dw8(fmap1_8) # TODO: here has a bug
                net_8, inp_8 = torch.split(cnet_dw8, [self.h_channels[-2], self.h_channels[-2]], dim=1)

        dilate = (1, 1)
        # Cascaded refinement (1/16 + 1/8)
        use_cvx_scale = self.use_cvx_scale
        if use_cvx_scale:
            scale = fmap1_8.shape[2] / disp.shape[2]
            disp = -1.0 * scale * F.interpolate(
                disp,
                size=(fmap1_8.shape[2], fmap1_8.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
        else:
            disp = -2.0 * F.interpolate(
                disp_16,
                size=(fmap1_8.shape[2], fmap1_8.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
        
        if self.use_disp16_pred and contexts is not None:
            disp = disp.detach()
            cxt_dw8_in = torch.cat([contexts[1], disp], dim=1)
            cnet_dw8 = self.init_conv_dw8(cxt_dw8_in)
            net_8, inp_8 = torch.split(cnet_dw8, [self.h_channels[-2], self.h_channels[-2]], dim=1)
            disp = disp + self.update_block_8_1.disp_pred(net_8)
            up_mask = 0.25 * self.update_block_8_1.mask_pred(net_8)

            disp_up = -convex_upsample(disp, up_mask, rate=8)
            predictions.append(disp_up)

        # RUM: 1/8
        for itr in range(self.iters):
            disp = disp.detach()
            if not self.use_raft_corr:
                out_corrs = corr_iter(fmap1_8, fmap2_8, disp, self.psize_d8, dilate)
            else:
                out_corrs = corr_iter_raft(fmap1_8, fmap2_8, disp, self.psize_d8, dilate)
            net_8, up_mask, delta_disp = self.update_block_8(net_8, inp_8, out_corrs, disp, output_all=True)

            disp = disp + delta_disp
            disp_up = -convex_upsample(disp, up_mask, rate=8)
            predictions.append(disp_up)
            if self.small_res_regw >0.0:
                predictions.append(-8 * disp)

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
        if self.truncated_grad or kwargs.get("truncated_grad", False):
            fmaps1 = [fmap.detach() for fmap in fmaps1]
            fmaps2 = [fmap.detach() for fmap in fmaps2]
            if contexts is not None:
                contexts = [context.detach() for context in contexts]

        with autocast('cuda', enabled=False):
            fmaps1 = [f.float() for f in fmaps1]
            fmaps2 = [f.float() for f in fmaps2]
            if contexts is not None:
                contexts = [c.float() for c in contexts]

            # make sure decoder always run in fp32 mode
            disp_pred = self.forward(fmaps1, fmaps2, flow_init, test_mode, contexts, ref_depth)

        if return_preds:
            return self.losses(disp_pred, disp_gt, valid=valid, *args, **kwargs), disp_pred
        else:
            return self.losses(disp_pred, disp_gt, valid=valid, *args, **kwargs), None

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
        with autocast('cuda', enabled=False):
            fmaps1 = [f.float() for f in fmaps1]
            fmaps2 = [f.float() for f in fmaps2]
            if contexts is not None:
                contexts = [c.float() for c in contexts]
            # make sure decoder always run in fp32 mode
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
        small_disp_loss = None
        if self.small_res_regw > 0.0:
            small_disp = disp_pred[-1]
            disp_pred.pop(-1)
            # compute loss for small_disp
            ## use max pooling for disp_gt
            small_disp_gt = F.max_pool2d(disp_gt, kernel_size=8, stride=8).squeeze()
            small_valid = F.max_pool2d(valid.float(), kernel_size=8, stride=8) > 0.5
            small_valid = small_valid.squeeze() & (small_disp_gt < self.init_max_disp)
            small_valid = small_valid.float()
            small_disp_loss = torch.abs(small_disp_gt - small_disp.squeeze()) * small_valid
            small_disp_loss = small_disp_loss.sum() / torch.clamp(small_valid.sum(), min=1.0)
            small_disp_loss = self.small_res_regw * small_disp_loss
            loss['small_disp_loss'] = small_disp_loss
        
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
        valid_mask = valid.bool()
        if valid_mask.any():
            disp_loss = (i_weight * i_loss[valid.bool()]).mean()
        else:
            disp_loss = torch.tensor(0.0, device=i_loss.device)
        loss['init_disp_loss'] = disp_loss.mean()
        if self.pseudo_loss_for_init:
            pseudo_loss = sequence_pseudo_loss(
                [reg_disp], kwargs['pseudo_gt'], 1.0, kwargs['pseudo_valid'], 0.2
            )
            loss['init_pseudo_loss'] = 0.2 * pseudo_loss
        return loss
