import copy
import torch.nn as nn
import torch.nn.functional as F
from ysstereo.models.encoders.vit_submodules.vit_heads.dpt_blocks import FeatureFusionBlockWoUp
from ysstereo.models.encoders.vit_submodules.utils import _get_round

class DPTFeatHead(nn.Module):
    def __init__(self, in_channel, proj_out_channels, out_channels=None, features:int = 256):
        super().__init__()
        self.act_levels = len(proj_out_channels)
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in proj_out_channels
        ])
        expand_proj_chs = copy.deepcopy(proj_out_channels)
        if len(expand_proj_chs) < 4:
            expand_proj_chs = [expand_proj_chs[0]] * (4-len(expand_proj_chs)) + expand_proj_chs
        full_resize_layers = [
            nn.ConvTranspose2d(
                in_channels=expand_proj_chs[0],
                out_channels=expand_proj_chs[0],
                kernel_size=4,
                stride=4,
                padding=0), # 4.0
            nn.ConvTranspose2d(
                in_channels=expand_proj_chs[1],
                out_channels=expand_proj_chs[1],
                kernel_size=2,
                stride=2,
                padding=0), # 2.0
            nn.Identity(), # 1.0
            nn.Conv2d(
                in_channels=expand_proj_chs[3],
                out_channels=expand_proj_chs[3],
                kernel_size=3,
                stride=2,
                padding=1) # 1/2
        ]
        self.resize_layers = nn.ModuleList(full_resize_layers[-len(proj_out_channels):])

        # skip connections
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(out_ch, features, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
            for out_ch in proj_out_channels
        ])

        # fusion connections(just a res-unit conv fpn)
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlockWoUp(features, nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True,size=None)
            for _ in proj_out_channels
        ])

        # head for out channels
        self.out_channels = out_channels
        if out_channels is not None:
            inter_chs = [max(features//2, _get_round(out_ch*1.3)) for out_ch in out_channels]
            self.heads1 = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(features, inter_ch, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                ) for inter_ch in inter_chs
            ])
            self.heads2 = nn.ModuleList([
                nn.Sequential(
                    # depth wise conv
                    nn.Conv2d(inter_ch, inter_ch, kernel_size=5, stride=1, padding=2, groups=inter_ch),
                    nn.GELU(True),
                    nn.Conv2d(inter_ch, out_ch, kernel_size=1, stride=1, padding=0),
                )
                for inter_ch, out_ch in zip(inter_chs, out_channels)
            ])

    def forward(self, feats, patch_h:int, patch_w:int, feats_prepared:bool=False):
        out = []
        skip_info = []
        # tokens to patch image
        for i, x in enumerate(feats[-self.act_levels:]):
            if not feats_prepared:
                x = x[0]
                x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
            skip_info.append(self.skip_convs[i](x))
        # skip fusion
        skip_info = skip_info[::-1] # high res first -> low res first
        fused_info = [] # low res first
        up_fused_info = [] # low res first
        for idx, sinfo in enumerate(skip_info):
            if idx==0:
                fused_info.append(self.fusion_blocks[idx](sinfo))
                up_fused_info.append(
                    F.interpolate(fused_info[-1], size=skip_info[1].shape[2:], mode='bilinear', align_corners=True)
                )
            elif idx<len(skip_info)-1:
                fused_info.append(
                    self.fusion_blocks[idx](sinfo, up_fused_info[-1])
                )
                up_fused_info.append(
                    F.interpolate(
                        fused_info[-1],
                        size=skip_info[idx+1].shape[2:],
                        mode='bilinear',
                        align_corners=True,
                    )
                )
            else:
                fused_info.append(
                    self.fusion_blocks[idx](sinfo, up_fused_info[-1])
                )

        # resize and output features
        fused_info = fused_info[::-1] # low res first -> high res first
        if self.out_channels is not None:
            raw_h, raw_w = int(patch_h * 14), int(patch_w * 14)
            out_sizes = [(raw_h//s, raw_w//s) for s in [4, 8, 16, 32]]
            out_sizes = out_sizes[-(self.act_levels):][:len(self.heads1)]
            # print(out_sizes)
            outs = [
                self.heads2[i](
                    F.interpolate(
                        self.heads1[i](fused_info[i]),
                        size=out_sizes[i], mode='bilinear', align_corners=True,
                    )
                ) for i in range(0, len(self.heads1))
            ]
        else:
            outs = fused_info
        return outs

def fetch_dptfeat_head(arch_type, in_channel, proj_out_channels=None, features=None):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    proj_out_channels = model_configs[arch_type]['out_channels'][1:] if proj_out_channels is None else proj_out_channels
    out_features = model_configs[arch_type]['features'] if features is None else features
    head = DPTFeatHead(
        in_channel,
        proj_out_channels,
        None,
        out_features,
    )
    return head, [out_features]*len(proj_out_channels)