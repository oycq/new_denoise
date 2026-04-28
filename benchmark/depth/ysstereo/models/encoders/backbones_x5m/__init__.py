from ysstereo.models.encoders.backbones_x5m.mixvargenet import (
    MixVarGENetConfig, MixVarGENet, get_mixvargenet_stride2channels, MIXVARGENet_ARCH_CFGS
)
from ysstereo.models.encoders.backbones_x5m.mobilenetv2 import MOBV2_ARCH_CFGS, MobileNetV2
from ysstereo.models.encoders.backbones_x5m.efficientnet import efficientnet, efficientnet_lite
from ysstereo.models.encoders.backbones_x5m.efficientnet_prune import efficientnet_lite_prune
from ysstereo.models.encoders.backbones_x5m.fpn import FPN
from ysstereo.models.encoders.backbones_x5m.effcientnet_torchvision import EfficientNetTorchVision
import torchvision.models as tvm
import torch

CNN_TYPE_EFFICIENTNET = 'effnet'
CNN_TYPE_EFFICIENTNETLITE = 'efficientnet_lite'
CNN_TYPE_MOBILEV2 = 'mobv2'
CNN_TYPE_MIXVARGENET = 'mixvargenet'
ALL_CNN_TYPES = [
    CNN_TYPE_MOBILEV2, CNN_TYPE_EFFICIENTNET,
    CNN_TYPE_EFFICIENTNETLITE, CNN_TYPE_MIXVARGENET,
]

def fetch_cnn(cnn_type:str, cnn_arch:str, pretrained_path:str=None, include_top:bool = False,
              **kwargs):
    assert cnn_type in ALL_CNN_TYPES
    if cnn_type == CNN_TYPE_EFFICIENTNET:
        cnn = EfficientNetTorchVision(cnn_arch,
            use_torch_pretrained=kwargs.get("use_torch_pretrained", True))
        cnn_out_chs = cnn.out_chs
        cnn_out_strides = cnn.out_strides
    elif cnn_type == CNN_TYPE_EFFICIENTNETLITE:
        eff_out_chs_map =dict(
            lite2=[16, 24, 48, 120, 352],
            lite3=[24, 32, 48, 136, 384],
            lite0=[16, 24, 40, 112, 320],
            lite0t=[8, 16, 24, 56, 160],
            lite0tfat = [16, 24, 40, 112, 320],
            lite0tm=[8, 16, 24, 56, 160],
            lite0m=[16, 24, 32, 88, 240],
            lite0tb=[8, 16, 24, 56, 160],
            lite0tuned=[16, 16, 24, 64, 192],
            nass=[16, 24, 48, 128, 352],
        )
        assert cnn_arch in eff_out_chs_map
        cnn_out_chs = eff_out_chs_map[cnn_arch]
        cnn_out_strides = [2**(6-len(cnn_out_chs)+i) for i in range(len(cnn_out_chs))]
        cnn = efficientnet_lite(cnn_arch, num_classes=1000, include_top=include_top)
    elif cnn_type == CNN_TYPE_MOBILEV2:
        assert cnn_arch in MOBV2_ARCH_CFGS
        backbone_cfg = MOBV2_ARCH_CFGS[cnn_arch]
        backbone_cfg['include_top'] = include_top
        cnn = MobileNetV2(**backbone_cfg)
        backbone_cfg['in_chls'] = backbone_cfg.get('in_chls', [
            [32], [16, 24], [24, 32, 32],
            [32] + [64] * 4 + [96] * 2, [96] + [160] * 3,
        ])
        backbone_cfg['out_chls'] = backbone_cfg.get('out_chls',  [
            [16], [24, 24], [32, 32, 32],
            [64] * 4 + [96] * 3, [160] * 3 + [320],
        ])
        base_out_chs = [c[-1] for c in backbone_cfg['out_chls']]
        cnn_out_chs = [int(c * backbone_cfg.get('alpha', 1.0)) for c in base_out_chs]
        cnn_out_strides = [2**(6-len(cnn_out_chs)+i) for i in range(len(cnn_out_chs))]
    elif cnn_type == CNN_TYPE_MIXVARGENET:
        assert cnn_arch in MIXVARGENet_ARCH_CFGS
        backbone_cfg = MIXVARGENet_ARCH_CFGS[cnn_arch]
        backbone_cfg['include_top'] = include_top
        cnn = MixVarGENet(**backbone_cfg)
        cnn_out_chs, cnn_out_strides= [], []
        net_config = backbone_cfg['net_config']
        for i in range(len(net_config)):
            cnn_out_chs.append(net_config[i][-1].out_channels)
            cnn_out_strides.append(2**(6-len(net_config)+i))
    else:
        raise NotImplementedError
    # load pretrained weights
    if pretrained_path is not None:
        ckpt = torch.load(pretrained_path, map_location='cpu')
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                sd = ckpt['state_dict']
            elif 'model' in ckpt:
                sd = ckpt['model']
            else:
                sd = ckpt
        else:
            sd = ckpt

        new_sd = {}
        for k in sd:
            if k.startswith('backbone.'):
                new_sd[k[len('backbone.'):]] = sd[k]
            elif k.startswith('module.'):
                new_sd[k[len('module.'):]] = sd[k]
            else:
                new_sd[k] = sd[k]
        
        missing_keys, unexpected_keys = cnn.load_state_dict(new_sd, strict=False)
        if missing_keys:
            print(f"⚠️  缺失的键 ({len(missing_keys)} 个): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(f"⚠️  意外的键 ({len(unexpected_keys)} 个): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        print(f"✅ Load Pretrained Weights for {cnn_type}-{cnn_arch} from {pretrained_path}")
    # if pretrained_path is not None:
    #     ckpt = torch.load(pretrained_path, map_location='cpu')
    #     sd = sd if 'state_dict' not in ckpt else ckpt['state_dict']
    #     new_sd = {}
    #     for k in sd:
    #         if k.startswith('backbone.'):
    #             new_sd[k[len('backbone.'):]] = sd[k]
    #         elif k.startswith('module.'):
    #             new_sd[k[len('module.'):]] = sd[k]
    #         else:
    #             new_sd[k] = sd[k]
    #     cnn.load_state_dict(new_sd, strict=False)
    #     print(f"Load Pretrained Weights for {cnn_type}-{cnn_arch} from {pretrained_path}")
    return cnn, cnn_out_chs, cnn_out_strides

def fetch_cnn_prune(cnn_type:str, cnn_arch:str, prune_cfg:list, pretrained_path:str=None, include_top:bool = False,
              **kwargs):
    assert cnn_type in ALL_CNN_TYPES
    if cnn_type == CNN_TYPE_EFFICIENTNET:
        cnn = EfficientNetTorchVision(cnn_arch,
            use_torch_pretrained=kwargs.get("use_torch_pretrained", True))
        cnn_out_chs = cnn.out_chs
        cnn_out_strides = cnn.out_strides
    elif cnn_type == CNN_TYPE_EFFICIENTNETLITE:
        eff_out_chs_map =dict(
            lite2=[16, 24, 48, 120, 352],
            lite3=[24, 32, 48, 136, 384],
            lite0=[16, 24, 40, 112, 320],
            lite0t=[8, 16, 24, 56, 160],
            lite0tfat = [16, 24, 40, 112, 320],
            lite0tfatopt2 = [16, 16, 24, 56, 160],
            lite0tfatopt4 = [16, 24, 40, 96, 216],
            lite0thuge=[24, 32, 48, 136, 384], #ckkprune
            lite0tm=[8, 16, 24, 56, 160],
            lite0m=[16, 24, 32, 88, 240],
            lite0tb=[8, 16, 24, 56, 160],
            lite0tuned=[16, 16, 24, 64, 192],
            nass=[16, 24, 48, 128, 352],
        )
        assert cnn_arch in eff_out_chs_map
        cnn_out_chs = eff_out_chs_map[cnn_arch]
        cnn_out_strides = [2**(6-len(cnn_out_chs)+i) for i in range(len(cnn_out_chs))]
        cnn = efficientnet_lite_prune(cnn_arch, prune_cfg, num_classes=1000, include_top=include_top)
    elif cnn_type == CNN_TYPE_MOBILEV2:
        assert cnn_arch in MOBV2_ARCH_CFGS
        backbone_cfg = MOBV2_ARCH_CFGS[cnn_arch]
        backbone_cfg['include_top'] = include_top
        cnn = MobileNetV2(**backbone_cfg)
        backbone_cfg['in_chls'] = backbone_cfg.get('in_chls', [
            [32], [16, 24], [24, 32, 32],
            [32] + [64] * 4 + [96] * 2, [96] + [160] * 3,
        ])
        backbone_cfg['out_chls'] = backbone_cfg.get('out_chls',  [
            [16], [24, 24], [32, 32, 32],
            [64] * 4 + [96] * 3, [160] * 3 + [320],
        ])
        base_out_chs = [c[-1] for c in backbone_cfg['out_chls']]
        cnn_out_chs = [int(c * backbone_cfg.get('alpha', 1.0)) for c in base_out_chs]
        cnn_out_strides = [2**(6-len(cnn_out_chs)+i) for i in range(len(cnn_out_chs))]
    elif cnn_type == CNN_TYPE_MIXVARGENET:
        assert cnn_arch in MIXVARGENet_ARCH_CFGS
        backbone_cfg = MIXVARGENet_ARCH_CFGS[cnn_arch]
        backbone_cfg['include_top'] = include_top
        cnn = MixVarGENet(**backbone_cfg)
        cnn_out_chs, cnn_out_strides= [], []
        net_config = backbone_cfg['net_config']
        for i in range(len(net_config)):
            cnn_out_chs.append(net_config[i][-1].out_channels)
            cnn_out_strides.append(2**(6-len(net_config)+i))
    else:
        raise NotImplementedError
    # load pretrained weights
    if pretrained_path is not None:
        ckpt = torch.load(pretrained_path, map_location='cpu')
        sd = sd if 'state_dict' not in ckpt else ckpt['state_dict']
        new_sd = {}
        for k in sd:
            if k.startswith('backbone.'):
                new_sd[k[len('backbone.'):]] = sd[k]
            elif k.startswith('module.'):
                new_sd[k[len('module.'):]] = sd[k]
            else:
                new_sd[k] = sd[k]
        cnn.load_state_dict(new_sd, strict=False)
        print(f"Load Pretrained Weights for {cnn_type}-{cnn_arch} from {pretrained_path}")
    return cnn, cnn_out_chs, cnn_out_strides

__all__ = [
    "MixVarGENetConfig", "MixVarGENet", "get_mixvargenet_stride2channels", "FPN",
    "MIXVARGENet_ARCH_CFGS", "MOBV2_ARCH_CFGS", "MobileNetV2",
    "efficientnet", "efficientnet_lite", "efficientnet_lite_prune", "fetch_cnn", "fetch_cnn_prune"
]
