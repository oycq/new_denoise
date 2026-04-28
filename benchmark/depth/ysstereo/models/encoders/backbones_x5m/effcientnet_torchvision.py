import copy
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models.efficientnet import _efficientnet_conf, MBConvConfig, EfficientNet

class EfficientNetTorchVision(nn.Module):
    AVAILBLE_ARCHS = dict(
        b0 = (
            tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.DEFAULT,
            _efficientnet_conf('efficientnet_b0', width_mult=1.0, depth_mult=1.0)[0],
        ),
        b1 = (
            tvm.efficientnet_b1, tvm.EfficientNet_B1_Weights.DEFAULT,
            _efficientnet_conf('efficientnet_b1', width_mult=1.0, depth_mult=1.1)[0],
        ),
        b2 = (
            tvm.efficientnet_b2, tvm.EfficientNet_B2_Weights.DEFAULT,
            _efficientnet_conf('efficientnet_b2', width_mult=1.1, depth_mult=1.2)[0],
        ),
        b3 = (
            tvm.efficientnet_b3, tvm.EfficientNet_B3_Weights.DEFAULT,
            _efficientnet_conf('efficientnet_b3', width_mult=1.2, depth_mult=1.4)[0],
        ),
        b4 = (
            tvm.efficientnet_b4, tvm.EfficientNet_B4_Weights.DEFAULT,
            _efficientnet_conf('efficientnet_b4', width_mult=1.4, depth_mult=1.8)[0],
        ),
        b5 = (
            tvm.efficientnet_b5, tvm.EfficientNet_B5_Weights.DEFAULT,
            _efficientnet_conf('efficientnet_b5', width_mult=1.6, depth_mult=2.2)[0],
        ),
    )
    def __init__(self, arch, use_torch_pretrained:bool=True):
        super().__init__()
        assert arch in self.AVAILBLE_ARCHS
        model_cfg = self.AVAILBLE_ARCHS[arch]
        if use_torch_pretrained:
            cnn:EfficientNet = model_cfg[0](model_cfg[1])
        else:
            cnn:EfficientNet = model_cfg[0]()
        # transfer
        features = cnn.features
        current_layers = [copy.deepcopy(features[0])] # first conv, stride 2
        inter_layers_cfgs = model_cfg[2]
        current_out_ch = inter_layers_cfgs[0].input_channels
        blks = []
        strides = []
        out_chs = []
        current_stride = 2
        for idx, cfg in enumerate(inter_layers_cfgs):
            cur_layer = features[1+idx]
            cfg:MBConvConfig
            if cfg.stride == 2:
                blks.append(current_layers)
                strides.append(current_stride)
                out_chs.append(current_out_ch)
                current_layers = []
                current_stride *= 2
            else:
                pass
            current_layers.append(copy.deepcopy(cur_layer))
            current_out_ch = cfg.out_channels
        if len(current_layers)>0:
            blks.append(current_layers)
            strides.append(current_stride)
            out_chs.append(current_out_ch)
        self.out_chs = out_chs
        self.out_strides = strides
        self.blks = nn.ModuleList([nn.Sequential(*layers) for layers in blks])
    
    def forward(self, x):
        feats = []
        for blk in self.blks:
            x = blk(x)
            feats.append(x)
        return feats
