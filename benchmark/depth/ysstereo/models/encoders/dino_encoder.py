
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from ysstereo.models.builder import ENCODERS
from ysstereo.models.encoders.backbones_x5m import (
    fetch_cnn, efficientnet_lite, FPN
)
from ysstereo.models.encoders.vit_submodules import fetch_dinov2, fetch_fused_dino_vit
from ysstereo.models.encoders.vit_submodules.vit_heads import DPTFeatHead, fetch_dptfeat_head

@ENCODERS.register_module()
class DeltaDinov2DPTFeat(BaseModel):
    _FUSE_RESIZE_ADD = 'resize_add'
    _FUSE_DPT_ADD = 'dpt_add'
    _FUSE_RADD_CP = 'radd_cp'
    def __init__(self, out_channels, dinov2_arch, delta_dino_arch:str='nass', layer_idx = None,
                 dino_pretrained_path:str=None, delta_dino_pretrained_path:str=None,
                 fuse_method:str=_FUSE_RESIZE_ADD, cp_pool_type='max', **kwargs):
        super().__init__()
        self.dino = fetch_dinov2(dinov2_arch, dino_pretrained_path)
        if layer_idx is None:
            self.intermediate_layer_idx = {
                'vits': [2, 5, 8, 11],
                'vitb': [2, 5, 8, 11], 
                'vitl': [4, 11, 17, 23], 
                'vitg': [9, 19, 29, 39]
            }[dinov2_arch]
        else:
            self.intermediate_layer_idx = layer_idx
        assert fuse_method in [self._FUSE_DPT_ADD, self._FUSE_RESIZE_ADD, self._FUSE_RADD_CP]
        for params in self.dino.parameters():
            # fix dino encoder
            params.requires_grad = False
        self.delta_dino = efficientnet_lite(delta_dino_arch)
        if delta_dino_pretrained_path is not None:
            ckpt = torch.load(delta_dino_pretrained_path, map_location='cpu')
            sd = ckpt['state_dict']
            new_sd = {}
            for k in sd:
                new_sd[k[len('backbone.'):]] = sd[k]
            self.delta_dino.load_state_dict(new_sd, strict=False)
            print("Load Pretrained ImageNet Efficient Lite")
        # config fusion layer-cnn part
        self.delta_dino_out_feat_chs =dict(
            lite0=[16, 24, 40, 112, 320],
            lite0t=[8, 16, 24, 56, 160],
            lite0tm=[8, 16, 24, 56, 160],
            lite0m=[16, 24, 32, 88, 240],
            lite0tb=[8, 16, 24, 56, 160],
            lite0tuned=[16, 16, 24, 64, 192],
            nass=[16, 24, 48, 128, 352], # 1/2, 1/4, 1/8, 1/16, 1/32
        )[delta_dino_arch]

        # fuse dino feat to cnn via
        self.fuse_method = fuse_method
        if self.fuse_method == self._FUSE_RESIZE_ADD:
            # use simple grid sample & 1x1 conv to align size & channels
            vit_ch = self.dino.embed_dim
            self.fuse_1x1convs = nn.ModuleList([
                nn.Conv2d(vit_ch, self.delta_dino_out_feat_chs[i+2], 1, 1, 0)
                for i in range(3)
            ])
            self.fuse_3x3convs = nn.ModuleList([
                nn.Conv2d(self.delta_dino_out_feat_chs[i+2], self.delta_dino_out_feat_chs[i+2], 1, 1, 0)
                for i in range(3)
            ])
        elif self.fuse_method == self._FUSE_DPT_ADD:
            fuse_cfg = {
                'vits': {'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'features': 128, 'out_channels': [96, 192, 384, 768]},
            }
            # use a dpt style fusion
            vit_ch = self.dino.embed_dim
            expand_proj_chs = fuse_cfg[dinov2_arch]['out_channels']
            ## 1. projection & transpose_conv
            self.fuse_inner_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=vit_ch,
                        out_channels=expand_proj_chs[1],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                    nn.ConvTranspose2d(
                        in_channels=expand_proj_chs[1],
                        out_channels=expand_proj_chs[1],
                        kernel_size=2,
                        stride=2,
                        padding=0), # 2.0
                ),
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=vit_ch,
                        out_channels=expand_proj_chs[2],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                    nn.Identity(), # 1.0
                ),
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=vit_ch,
                        out_channels=expand_proj_chs[3],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                    nn.Conv2d(
                        in_channels=expand_proj_chs[3],
                        out_channels=expand_proj_chs[3],
                        kernel_size=3,
                        stride=2,
                        padding=1) # 1/2
                )
            ])
            ## 2. down to top feature fusion
            ### we use only simple fpn for feature fusion
            self.dino_fpn = FPN(
                in_strides=[8, 16, 32],
                in_channels=expand_proj_chs[1:],
                out_strides=[8, 16, 32],
                out_channels=self.delta_dino_out_feat_chs[2:],
            )
            ## 3. resize to cnn size & 3x3 feature adjustment
            self.fuse_3x3convs = nn.ModuleList([
                nn.Conv2d(self.delta_dino_out_feat_chs[i+2], self.delta_dino_out_feat_chs[i+2], 3, 1, 1)
                for i in range(3)
            ])
        elif self.fuse_method == self._FUSE_RADD_CP:
            vit_ch = self.dino.embed_dim
            # use channel pooling & resize add
            self.fuse_1x1convs = nn.ModuleList([
                nn.Conv2d(vit_ch, self.delta_dino_out_feat_chs[i+2], 1, 1, 0)
                for i in range(3)
            ])
            self.fuse_3x3convs = nn.ModuleList([
                # nn.Sequential(
                nn.Conv2d(self.delta_dino_out_feat_chs[i+2], self.delta_dino_out_feat_chs[i+2], 1, 1, 0)
                    # nn.ReLU()
                # )
                for i in range(3)
            ])
            # channel max pooling
            if cp_pool_type=='max':
                self.channel_mpool = nn.ModuleList([
                    nn.AdaptiveMaxPool1d(output_size=self.delta_dino_out_feat_chs[i+2])
                    for i in range(3)
                ])
            else:
                self.channel_mpool = nn.ModuleList([
                    nn.AdaptiveAvgPool1d(output_size=self.delta_dino_out_feat_chs[i+2])
                    for i in range(3)
                ])
        else:
            raise NotImplementedError
        # final use a simple fpn for output
        out_channels = [out_channels]*3 if isinstance(out_channels, int) else out_channels
        self.fpn = FPN(
            in_strides=[8, 16, 32],
            in_channels=self.delta_dino_out_feat_chs[-3:],
            out_strides=[8, 16, 32],
            out_channels=out_channels,
        )
        self.tH = self.tW = 560

    @torch.no_grad()    
    def _fetch_dino_features(self, x):
        # rH, rW = x.shape[2:]
        # self.tH, self.tW = (rH // 14) * 14, (rW // 14) * 14
        pH, pW = self.tH//14, self.tW//14
        # we resize x to 560x560
        x = F.interpolate(x, size=(self.tH, self.tW), mode='bilinear', align_corners=True)
        # feats with cls token, 
        feats = self.dino.get_intermediate_layers(x, self.intermediate_layer_idx[-3:], return_class_token=True)
        # only save latest 
        out_feats = [
            it[0] for it in feats
        ]
        out_feats = [
            it.permute(0, 2, 1).reshape((it.shape[0], it.shape[-1], pH, pW))
            for it in out_feats
        ]
        return out_feats

    def _align_vit_feature_to_cnn(self, dino_feats, cnn_feats):
        if self.fuse_method == self._FUSE_DPT_ADD:
            dino_feats = [net(f) for net, f in zip(self.fuse_inner_convs, dino_feats)]
            dino_feats = self.dino_fpn(dino_feats)
            dino_feats = [
                self.fuse_3x3convs[i](
                    F.interpolate(dino_feats[i], cnn_feats[i].shape[2:],
                                  scale_factor=None, mode='bilinear', align_corners=True)
                ) for i in range(len(dino_feats))
            ]
            return dino_feats
        elif self.fuse_method == self._FUSE_RESIZE_ADD:
            dino_feats = [net(f) for net, f in zip(self.fuse_1x1convs, dino_feats)]
            dino_feats = [
                self.fuse_3x3convs[i](
                    F.interpolate(dino_feats[i], cnn_feats[i].shape[2:],
                                  scale_factor=None, mode='bilinear', align_corners=True)
                ) for i in range(len(dino_feats))
            ]
            return dino_feats
        elif self.fuse_method == self._FUSE_RADD_CP:
            tH, tW = self.tH, self.tW
            pH, pW = tH//14, tW//14
            # compress dino feats via 1x1 conv
            conv_dino_feats = [net(f) for net, f in zip(self.fuse_1x1convs, dino_feats)]
            conv_dino_feats = [
                self.fuse_3x3convs[i](
                    F.interpolate(conv_dino_feats[i], cnn_feats[i].shape[2:],
                                  scale_factor=None, mode='bilinear', align_corners=True)
                ) for i in range(len(conv_dino_feats))
            ]
            # compress dino feats via channel max pooling
            pool_dino_feats = [f.permute(0,2,3,1).reshape((f.shape[0], pH*pW, -1)) for f in dino_feats]
            pool_dino_feats = [net(f).permute(0, 2, 1).reshape(f.shape[0], -1, pH, pW) for net, f in zip(self.channel_mpool, pool_dino_feats)]
            pool_dino_feats =  [
                F.interpolate(pool_dino_feats[i], cnn_feats[i].shape[2:],
                              scale_factor=None, mode='bilinear', align_corners=True)
                for i in range(len(pool_dino_feats))
            ]
            # fuse via add
            dino_feats = [pf + cf for pf, cf in zip(pool_dino_feats, conv_dino_feats)]
            return dino_feats
        else:
            raise NotImplementedError

    def forward(self, x):
        dino_feats = self._fetch_dino_features(x)
        delta_cnn_feats = self.delta_dino(x)[-3:]
        aligned_dino_feats = self._align_vit_feature_to_cnn(dino_feats, delta_cnn_feats)
        fused_feats = [df+cf for df,cf in zip(aligned_dino_feats, delta_cnn_feats)]
        out_feats = self.fpn(fused_feats)
        return out_feats[:2][::-1]

@ENCODERS.register_module()
class Dinov2DPTFeat(BaseModel):
    def __init__(self, out_channels, arch_type, backbone_pretrain_path:str=None, **kwargs):
        super().__init__()
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            # 'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        assert arch_type in model_configs
        self.backbone = fetch_dinov2(arch_type, backbone_pretrain_path)
        out_channels = [out_channels]*2 if isinstance(out_channels, int) else out_channels
        assert len(out_channels) == 2
        self.feat_head = DPTFeatHead(
            in_channel=self.backbone.embed_dim,
            proj_out_channels=model_configs[arch_type]['out_channels'][1:],
            features=max(model_configs[arch_type]['features'], 96),
            out_channels=out_channels,
        )

    def forward(self, x):
        patch_h, patch_w = x.shape[2:]
        patch_h, patch_w = patch_h//14, patch_w//14
        feats = self.backbone.get_intermediate_layers(x, [2, 5, 8, 11], return_class_token=True)
        feats = self.feat_head(feats, patch_h, patch_w)
        return feats[::-1]

@ENCODERS.register_module()
class FusedDinov2Encoder(BaseModel):
    def __init__(self, fused_dino_cfg, head_cfg, out_channels:int=64, **kwargs):
        super().__init__()
        if fused_dino_cfg.get('cnn_cfg', None) is not None:
            cnn_start_stride = fused_dino_cfg.pop('cnn_start_stride', 4)
            cnn, cnn_out_chs, cnn_out_strides = fetch_cnn(**fused_dino_cfg['cnn_cfg'])
            in_cnn_out_chs, in_cnn_out_strides = [], []
            for ch, s in zip(cnn_out_chs, cnn_out_strides):
                if s>cnn_start_stride:
                    in_cnn_out_chs.append(ch)
                    in_cnn_out_strides.append(s)
            new_cnn_cfg = dict(
                cnn_net=cnn, cnn_out_channels=in_cnn_out_chs, cnn_out_strides=in_cnn_out_strides,
            )
            fused_dino_cfg['cnn_cfg'] = new_cnn_cfg
        self.backbone = fetch_fused_dino_vit(**fused_dino_cfg)
        head_cfg['in_channel'] = self.backbone.dino.embed_dim
        self.head, self.head_out_chs = fetch_dptfeat_head(**head_cfg)
        # final out head
        out_channels = [out_channels]*2 if isinstance(out_channels, int) else out_channels
        self.out_s8 = nn.Conv2d(self.head_out_chs[0], out_channels[0], kernel_size=1)
        self.out_s16 = nn.Conv2d(self.head_out_chs[1], out_channels[1], kernel_size=1)

    def forward(self, x:torch.Tensor):
        encode_feats = self.backbone.get_fused_features(x) # seq of n, h, w, c
        _, ph, pw, _ = encode_feats[0].shape
        encode_feats = [feat.permute(0, 3, 1, 2) for feat in encode_feats]
        out_feats = self.head(
            encode_feats, patch_h=ph, patch_w=pw, feats_prepared=True,
        ) # n,c,h,w. high res first
        ret = [self.out_s16(out_feats[1]), self.out_s16(out_feats[0])]
        return ret
