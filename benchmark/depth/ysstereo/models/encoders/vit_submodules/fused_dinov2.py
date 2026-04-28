import torch
from typing import Sequence
import torch.nn as nn
import torch.nn.functional as F
from ysstereo.models.encoders.vit_submodules.dinov2 import fetch_dinov2
from ysstereo.models.encoders.vit_submodules.patch_cross_attn_modules import (
    PatchMiner, merge_cnn_feats,
)

class FusedDinoEncoder(nn.Module):
    _FUSE_NONE='none' # raw dino
    _FUSE_RADD = 'radd' # resize & add
    _FUSE_VIT_COMER = 'vit_comer' # bidirection fusion, refer to vit comer
    _FUSE_LATE_CROSS_ATTN = 'late_cattn' # mini high res info via cross attn
    _FUSE_MIDDLE_CROSS_ATTN = 'mid_cattn' # like late_cattn, but process during blocks
    _AVAILABLE_FUSION_METHODS = [
        _FUSE_NONE, _FUSE_RADD, _FUSE_VIT_COMER,
        _FUSE_LATE_CROSS_ATTN, _FUSE_MIDDLE_CROSS_ATTN,
    ]
    def __init__(
        self,
        use_deep_resize: bool = False,
        # dino cfg
        dino_arch:str = 'base',
        dino_pretrained:str = None,
        vit_layers:Sequence[int] = None,
        # cnn highres feature cfg
        cnn_net:nn.Module = None,
        cnn_out_channels:Sequence[int] = None,
        cnn_out_strides:Sequence[int] = None,
        # lora config
        lora_mode:bool = False,
        # fusion config
        fusion_method:str=_FUSE_NONE,
        fuse_kwargs:dict = {},
        **kwargs,
    ):
        super().__init__()
        self.dino = fetch_dinov2(dino_arch, dino_pretrained)
        self.vit_layers = vit_layers

        self.use_deep_resize = use_deep_resize
        if self.use_deep_resize:
            raise NotImplementedError("only support input image resize to align feature space now.")
        self.lora_mode = lora_mode
        self._set_trainable_dino_params()

        self.cnn_net = cnn_net
        self.cnn_out_channels = cnn_out_channels
        self.cnn_out_strides = cnn_out_strides
        self.stride_feat_map = {s:i for i,s in enumerate(self.cnn_out_strides)}

        self.fusion_funcs = {
            self._FUSE_NONE:self._fuse_none, self._FUSE_RADD:self._fuse_radd,
            self._FUSE_VIT_COMER:self._fuse_vit_comer, self._FUSE_LATE_CROSS_ATTN:self._fuse_late_cattn,
            self._FUSE_MIDDLE_CROSS_ATTN:self._fuse_mid_cattn,
        }
        assert fusion_method in self._AVAILABLE_FUSION_METHODS
        assert fusion_method in self.fusion_funcs
        self.fusion_method = fusion_method
        if self.fusion_method != self._FUSE_NONE:
            assert self.cnn_net is not None
        if self.fusion_method == self._FUSE_LATE_CROSS_ATTN:
            self.miner = PatchMiner(
                cnn_channels=self.cnn_out_channels,
                cnn_strides=self.cnn_out_strides,
                vit_channel=self.dino.embed_dim,
                **fuse_kwargs,
            )
        elif self.fusion_method == self._FUSE_MIDDLE_CROSS_ATTN:
            miner_dict = {}
            mine_stride = None
            for vit_layer in vit_layers:
                if mine_stride is None:
                    mine_stride = fuse_kwargs[vit_layer].get('mine_stride', 4)
                else:
                    cur_mine_stride = fuse_kwargs[vit_layer].get('mine_stride', 4)
                    if cur_mine_stride != mine_stride:
                        raise ValueError("only support same mine stride for each miners")
                    mine_stride = cur_mine_stride

                miner_dict[str(vit_layer)] = PatchMiner(
                    cnn_channels=self.cnn_out_channels,
                    cnn_strides=self.cnn_out_strides,
                    vit_channel=self.dino.embed_dim,
                    **fuse_kwargs[vit_layer],
                )
            self.miner_dict = nn.ModuleDict(miner_dict)
    def _set_trainable_dino_params(self):
        if not self.lora_mode:
            for params in self.dino.parameters():
                params.requires_grad = False
        else:
            raise NotImplementedError("Lora dino finetuning is not supported now")
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        return self

    def get_fused_features(
            self, x:torch.Tensor, vit_layers=None, **kwargs):
        """
        Args:
            x: input image tensor
            vit_layers: the vit layers you want to extract
        Returns:
            multiple feature maps in shape b, h, w, c
        """
        if vit_layers is None:
            vit_layers = self.vit_layers
        assert len(vit_layers) == 3, "we only support extract 3 layer features now"
        return self.fusion_funcs[self.fusion_method](x, vit_layers, **kwargs)

    def _fuse_none(self, x, vit_layers, **kwargs):
        rH, rW = x.shape[2:]
        assert rH % 16 == 0 and rW % 16 == 0, "height & width should be times of 16"
        dino_h, dino_w = (rH//16)*14, (rW//16)*14
        x = F.interpolate(
            x, size=(dino_h, dino_w), mode='bicubic', align_corners=True,
            scale_factor=None, recompute_scale_factor=False, antialias=False,
        )
        ret = self.dino.get_intermediate_layers(
            x, vit_layers, return_class_token=False, reshape=False,
        )
        ret = [feat.view(-1,rH//16, rW//16,self.dino.embed_dim) for feat in ret]
        return ret
    
    def _fuse_radd(self, x, vit_layers, **kwargs):
        pass

    def _fuse_late_cattn(self, x, vit_layers, **kwargs):
        dino_feats = self._fuse_none(x, vit_layers) # return 1/16 global features
        cnn_feats = self.cnn_net(x)[-len(self.cnn_out_channels):]
        # mine 1/8 cnn feature into dino vit_layers[-1]
        # TODO: bug, we do duplicate layernorms in miner, need to be deleted
        dino_feats[-1] = self.miner(cnn_feats, dino_feats[-1])
        return dino_feats

    def _fuse_mid_cattn(self, x, vit_layers, **kwargs):
        rH, rW = x.shape[2:]
        assert rH % 16 == 0 and rW % 16 == 0, "height & width should be times of 16"
        pH, pW = rH//16, rW//16
        cnn_feats = self.cnn_net(x)[-len(self.cnn_out_channels):]
        miner0 = self.miner_dict[str(vit_layers[0])]
        cnn_feat = merge_cnn_feats(
            cnn_feats, miner0.cnn_strides, miner0.mine_stride,
            (pH*miner0.mine_scale, pW*miner0.mine_scale),
        )
        b, c, h, w = cnn_feat.shape
        cnn_feat = cnn_feat.view(b, c, h//miner0.mine_scale, miner0.mine_scale, w//miner0.mine_scale, miner0.mine_scale)
        cnn_feat = cnn_feat.permute(0, 2, 4, 3, 5, 1).contiguous()
        # miner
        dino_h, dino_w = pH*14, pW*14
        x_for_dino = F.interpolate(
            x, size=(dino_h, dino_w), mode='bicubic', align_corners=True,
            scale_factor=None, recompute_scale_factor=False, antialias=False,
        )
        i_tokens = self.dino.prepare_tokens_with_masks(x_for_dino)
        output = []
        for i, blk in enumerate(self.dino.blocks):
            i_tokens = blk(i_tokens)
            if i in vit_layers:
                extra_tokens = i_tokens[:, :(1+self.dino.num_register_tokens)]
                patch_tokens = i_tokens[:, (1+self.dino.num_register_tokens):].view(
                    -1,pH, pW,self.dino.embed_dim
                )
                patch_tokens = self.miner_dict[str(i)](cnn_feat, patch_tokens, True)
                output.append(patch_tokens)
                if i < len(self.dino.blocks) - 1:
                    patch_tokens = patch_tokens.view(-1, (pH * pW), self.dino.embed_dim)
                    i_tokens = torch.cat([extra_tokens, patch_tokens], dim=1)
        return output

    def _fuse_vit_comer(self, x, vit_layers, **kwargs):
        pass

def fetch_fused_dino_vit(dino_cfg:dict, cnn_cfg:dict,
                         fuse_cfg:dict={}, lora_cfg:dict = {},
                         **kwargs) -> FusedDinoEncoder:
    encoder_kwargs = {}
    encoder_kwargs.update(kwargs)
    encoder_kwargs.update(dino_cfg)
    encoder_kwargs.update(cnn_cfg)
    encoder_kwargs.update(fuse_cfg)
    encoder_kwargs.update(lora_cfg)

    fused_dino = FusedDinoEncoder(
        **encoder_kwargs,
    )
    return fused_dino