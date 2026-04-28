from mmengine.model import BaseModel
from ysstereo.models.builder import ENCODERS
from ysstereo.models.encoders.backbones_x5m import (
    fetch_cnn, fetch_cnn_prune, FPN, CNN_TYPE_EFFICIENTNETLITE,
    CNN_TYPE_MIXVARGENET, CNN_TYPE_MOBILEV2,
    CNN_TYPE_EFFICIENTNET, ALL_CNN_TYPES,
)

@ENCODERS.register_module()
class PureCNNClsEncoder(BaseModel):
    def __init__(self, cnn_type:str, arch_type:str, pretrained_path:str=None, include_top:bool=True):
        super().__init__()
        assert cnn_type in ALL_CNN_TYPES, f"unsupported cnn type:{cnn_type}"
        self.cnn, _, _ = fetch_cnn(cnn_type, arch_type, pretrained_path, include_top=include_top)
    def forward(self, x):
        return self.cnn(x)

@ENCODERS.register_module()
class EffNetFPN(BaseModel):
    def __init__(self, out_channels: int = 64,
                 return_resolutions: int = 2,
                 arch_type:str='b2',
                 frozen_encoder: bool = False,
                 backbone_pretrain_path:str=None, **kwargs):
        super().__init__()
        self.return_resolutions = return_resolutions
        self.backbone, encoder_out_feat_chs, _ = fetch_cnn(
            CNN_TYPE_EFFICIENTNET, arch_type, backbone_pretrain_path, **kwargs,
        )
        out_channels = [out_channels]*3 if isinstance(out_channels, int) else out_channels
        self.fpn =  FPN(
            in_strides=[8, 16, 32],
            in_channels=encoder_out_feat_chs[-3:],
            out_strides=[8, 16, 32],
            out_channels=out_channels,
        )

        if frozen_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        feats = self.backbone(x)
        fpn_feats = self.fpn(feats[-3:])
        return fpn_feats[:self.return_resolutions][::-1]

@ENCODERS.register_module()
class MixVargeNetFPN(BaseModel):
    def __init__(self, out_channels: int = 64,
                 return_resolutions: int = 2,
                 arch_type:str='base',
                 frozen_encoder: bool = False,
                 backbone_pretrain_path:str=None, **kwargs):
        super().__init__()

        self.return_resolutions = return_resolutions
        # backbone out
        # resolutions [1/2, 1/4, 1/8, 1/16, 1/32]
        # channels [32, 32, 64, 96, 160]
        self.backbone, self.backbone_chs, _ = fetch_cnn(
            CNN_TYPE_MIXVARGENET, arch_type, backbone_pretrain_path
        )
        self.out_channels = out_channels
        # use fpn to extract multi-scale features
        self.fpn = FPN(
            in_strides=[8, 16, 32],
            in_channels=self.backbone_chs[-3:],
            out_strides=[8, 16, 32],
            out_channels=[out_channels] * 3,
        )

        if frozen_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        feats = self.backbone(x)
        fpn_feats = self.fpn(feats[-3:])
        return fpn_feats[:self.return_resolutions][::-1]


@ENCODERS.register_module()
class Mobv2FPN(BaseModel):
    def __init__(self, out_channels: int = 64,
                 return_resolutions: int = 2,
                 arch_type:str='base',
                 frozen_encoder: bool = False,
                 backbone_pretrain_path:str=None, **kwargs):
        super().__init__()
        self.return_resolutions = return_resolutions
        self.backbone, encoder_out_feat_chs, _ = fetch_cnn(
            CNN_TYPE_MOBILEV2, arch_type, backbone_pretrain_path,
        )
        out_channels = [out_channels]*3 if isinstance(out_channels, int) else out_channels
        self.fpn =  FPN(
            in_strides=[8, 16, 32],
            in_channels=encoder_out_feat_chs[-3:],
            out_strides=[8, 16, 32],
            out_channels=out_channels,
        )
        if frozen_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        feats = self.backbone(x)
        fpn_feats = self.fpn(feats[-3:])
        return fpn_feats[:self.return_resolutions][::-1]

@ENCODERS.register_module()
class EffliteFPN(BaseModel):
    def __init__(self, out_channels: int = 64,
                 return_resolutions: int = 2,
                 arch_type:str='base',
                 frozen_encoder: bool = False,
                 backbone_pretrain_path:str=None, **kwargs):
        super().__init__()
        self.return_resolutions = return_resolutions
        self.backbone, encoder_out_feat_chs, _ = fetch_cnn(
            CNN_TYPE_EFFICIENTNETLITE, arch_type, backbone_pretrain_path,
        )
        out_channels = [out_channels]*3 if isinstance(out_channels, int) else out_channels
        self.fpn =  FPN(
            in_strides=[8, 16, 32],
            in_channels=encoder_out_feat_chs[-3:],
            out_strides=[8, 16, 32],
            out_channels=out_channels,
        )
        if frozen_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        feats = self.backbone(x)
        fpn_feats = self.fpn(feats[-3:])
        return fpn_feats[:self.return_resolutions][::-1]

    

@ENCODERS.register_module()
class EffliteFPN_prune(BaseModel):
    def __init__(self, out_channels: int = 64,
                 return_resolutions: int = 2,
                 arch_type:str='base',
                 frozen_encoder: bool = False,
                 backbone_pretrain_path:str=None,
                 prune_cfg:list=[48, 96, 144, 240, 480, 480, 672, 672, 1152, 1152], **kwargs):
        super().__init__()
        self.return_resolutions = return_resolutions
        self.backbone, encoder_out_feat_chs, _ = fetch_cnn_prune(
            CNN_TYPE_EFFICIENTNETLITE, arch_type, prune_cfg, backbone_pretrain_path,
        )
        out_channels = [out_channels]*3 if isinstance(out_channels, int) else out_channels
        for out_channel in out_channels:
            print("out_channel", out_channel)
        self.fpn =  FPN(
            in_strides=[8, 16, 32],
            in_channels=encoder_out_feat_chs[-3:],
            out_strides=[8, 16, 32],
            out_channels=out_channels,
        )
        if frozen_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.fpn.parameters():
                param.requires_grad = False

    def forward(self, x):
        feats = self.backbone(x)
        fpn_feats = self.fpn(feats[-3:])
        return fpn_feats[:self.return_resolutions][::-1]