from typing import Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None

def merge_cnn_feats(cnn_feats:Sequence[torch.Tensor], cnn_strides:Sequence[int], tgt_stride:int,
                    tgt_shape:Sequence[int]):
    aH, aW = tgt_shape
    resized_cnn_feat = []
    for s, cnn_feat in zip(cnn_strides, cnn_feats):
        if s == tgt_stride:
            resized_cnn_feat.append(cnn_feat)
        else:
            rH, rW = cnn_feat.shape[2:]
            if s<tgt_stride:
                assert (rH * s) // tgt_stride == aH and (rW * s) // tgt_stride == aW, 'shape don\'t align'
            resized_cnn_feat.append(
                F.interpolate(cnn_feat, size=(aH, aW), scale_factor=None, mode='bilinear', align_corners=True)
            )
    cnn_feat = torch.cat(resized_cnn_feat, dim=1)
    return cnn_feat

class PatchMiner(nn.Module):
    def __init__(self, cnn_channels:Sequence[int], vit_channel:int, attn_channel:int, out_mlp_cfg:dict = {}, use_xformers:bool=False,
                 head_dim:int=1, cnn_strides:Sequence[int]=[4, 8, 16, 32], vit_stride:int=16, mine_stride:int=4):
        super().__init__()
        assert len(cnn_channels) == len(cnn_strides) and vit_stride % mine_stride == 0 and attn_channel%head_dim==0
        if use_xformers:
            func_name = 'memory_efficient_attention'
            if not (func_name in globals() or func_name in locals()):
                raise ImportError('xformers is not availble. model failed to init')
        self.head_dim = head_dim
        self.cnn_strides = cnn_strides
        self.vit_stride = vit_stride
        self.mine_stride = mine_stride
        self.mine_scale = vit_stride // mine_stride
        self.patch_region_num = (vit_stride // mine_stride) ** 2
        self.use_xformers = use_xformers

        self.cnn_channels = cnn_channels # channels of input cnn feats
        self.vit_channel = vit_channel # channels of input vit feature
        self.attn_channel = attn_channel # channels for computing attention
        self.merged_cnn_ch = sum(cnn_channels)

        # k, v matrix for cnn feat. proj cnn feat to attn channel
        self.K_proj = nn.Sequential(
            nn.LayerNorm(self.merged_cnn_ch),
            nn.Linear(self.merged_cnn_ch, attn_channel),
        )
        self.V_proj = nn.Sequential(
            nn.LayerNorm(self.merged_cnn_ch),
            nn.Linear(self.merged_cnn_ch, attn_channel),
        )

        # q matrix for vit feat. proj vit feat to attn channel
        self.vit_Q_proj = nn.Sequential(
            nn.LayerNorm(self.vit_channel),
            nn.Linear(self.vit_channel, attn_channel),
        )
        if attn_channel == vit_channel:
            self.cattn_out_unproj = nn.Identity()
        else:
            self.cattn_out_unproj = nn.Linear(attn_channel, vit_channel)
        # mlp:ffn
        self.MLP = nn.Sequential(
            nn.Linear(vit_channel, vit_channel),
            nn.GELU(), nn.Linear(vit_channel, vit_channel),
        )

    def forward(self, cnn_feats:Union[torch.Tensor, Sequence[torch.Tensor]], vit_feat:torch.Tensor, cnn_processed:bool=False):
        """
        Args:
            cnn_feats: cnn features with shape [n,c,h,w]
            vit_feat: vit features with shape [n, h, w, c]
            cnn_merged: cnn feats has been merged outside
        """
        b, pH, pW = vit_feat.shape[:3]
        aH, aW = pH * self.mine_scale, pW * self.mine_scale
        if not cnn_processed:
            assert len(cnn_feats) == len(self.cnn_strides)
            cnn_feat = merge_cnn_feats(cnn_feats, self.cnn_strides, self.mine_stride, (aH, aW))
            b, c, h, w = cnn_feat.shape
            cnn_feat = cnn_feat.view(b, c, h//self.mine_scale, self.mine_scale, w//self.mine_scale, self.mine_scale)
            cnn_feat = cnn_feat.permute(0, 2, 4, 3, 5, 1).contiguous()
        else:
            cnn_feat = cnn_feats

        # q, k, v projection
        cnn_feat = cnn_feat.view(b, pH*pW, self.patch_region_num, -1)
        vit_feat = vit_feat.view(b, pH*pW, -1)
        q = self.vit_Q_proj(vit_feat).unsqueeze(-2) # b, N, 1, c
        k = self.K_proj(cnn_feat) # b, N, R, c
        v = self.V_proj(cnn_feat) # b, N, R, c

        # cross att
        if not self.use_xformers:
            h, attn_c = self.head_dim, self.attn_channel//self.head_dim
            q = q.view(b, pH*pW, 1, h, attn_c).transpose(2,3)
            k = k.view(b, pH*pW, self.patch_region_num, h, attn_c).transpose(2,3)
            v = v.view(b, pH*pW, self.patch_region_num, h, attn_c).transpose(2,3)
            scale = 1.0 / (attn_c**0.5)
            attn:torch.Tensor = (q @ k.transpose(-1, -2)) * scale
            attn = attn.nan_to_num() # b,N, H, 1, R
            feat = (attn.softmax(-1) @ v).squeeze(3).view(b, pH*pW, self.attn_channel) # b, n, h, c
        else:
            q = q.view(b*pH*pW,1,self.head_dim, self.attn_channel//self.head_dim)
            k = k.view(b*pH*pW,self.patch_region_num,self.head_dim, self.attn_channel//self.head_dim)
            v = v.view(b*pH*pW,self.patch_region_num,self.head_dim, self.attn_channel//self.head_dim)
            feat = memory_efficient_attention(q, k, v, scale=1.0/(self.attn_channel**0.5))
            feat = feat.view(b, pH*pW, -1)
        # unproj to vit ch & add
        feat = self.cattn_out_unproj(feat) + vit_feat

        # out
        out = self.MLP(feat).view(b, pH, pW, -1)

        return out
