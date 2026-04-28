import torch
import torch.nn as nn
import torch.nn.functional as F
from ysstereo.models.utils.seg_model_utils import UAFM_SpAtten
from ysstereo.models.builder import DECODERS, build_loss
from .mmseg.models.decode_heads.decode_head import BaseDecodeHead

class ConvBNReLU(nn.Sequential):
    """Equivalent to Paddle's layers.ConvBNReLU"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class PPContextModule(nn.Module):
    """
    PyTorch port of PPLiteSeg's PPContextModule
    """
    def __init__(self, in_channels, inter_channels, out_channels,
                 bin_sizes, align_corners=False):
        super().__init__()
        # pyramid pooling stages
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=bs),
                ConvBNReLU(in_channels, inter_channels, kernel_size=1)
            )
            for bs in bin_sizes
        ])
        # final conv to mix pooled features
        self.conv_out = ConvBNReLU(inter_channels, out_channels,
                                   kernel_size=3, padding=1)
        self.align_corners = align_corners

    def forward(self, x):
        H, W = x.shape[2:]
        out = None
        for pool_conv in self.stages:
            y = pool_conv(x)                       # [B, inter, bs, bs]
            y = F.interpolate(y, size=(H, W),
                              mode='bilinear',
                              align_corners=self.align_corners)
            out = y if out is None else out + y
        out = self.conv_out(out)
        return out

class PPLiteSegHead(nn.Module):
    """
    PyTorch port of PPLiteSegHead
    """
    def __init__(self,
                 backbone_out_chs,    # list of ints
                 arm_out_chs,         # list of ints
                 cm_bin_sizes,        # list/tuple of ints
                 cm_out_ch,
                 arm_type,            # nn.Module class for ARM
                 resize_mode='bilinear'):
        super().__init__()
        # context module
        self.cm = PPContextModule(
            in_channels=backbone_out_chs[-1],
            inter_channels=cm_out_ch,
            out_channels=cm_out_ch,
            bin_sizes=cm_bin_sizes,
            align_corners=(resize_mode=='bilinear')
        )
        # build ARM modules for each stage
        assert arm_type == "UAFM_SpAtten"
        ArmClass = UAFM_SpAtten

        self.arm_list = nn.ModuleList()
        N = len(backbone_out_chs)
        for i in range(N):
            low_chs  = backbone_out_chs[i]
            high_chs = cm_out_ch if i==N-1 else arm_out_chs[i+1]
            out_ch   = arm_out_chs[i]
            arm = ArmClass(low_chs, high_chs, out_ch,
                           ksize=3, resize_mode=resize_mode)
            self.arm_list.append(arm)

    def forward(self, feats):
        """
        feats: list of Tensors [x2, x4, x8, x16, x32]
        returns list of Tensors same length
        """
        # context on highest stage
        high = self.cm(feats[-1])
        outs = []
        # from top to bottom
        for i in reversed(range(len(feats))):
            low = feats[i]
            arm = self.arm_list[i]
            # each ARM returns fused upsampled feature
            high = arm(low, high)
            outs.insert(0, high)
        return outs

class SegHead(nn.Module):
    """
    Simple segmentation head: ConvBNReLU -> Conv2d
    """
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.conv_out = nn.Conv2d(mid_chan, n_classes,
                                  kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


@DECODERS.register_module()
class PPLiteSegDecodeHead(BaseDecodeHead):
    """PPLiteSeg 用到的 Decode Head，直接继承 BaseDecodeHead"""
    def __init__(self,
                 in_channels,            # list[int], 多路输入
                 in_index,               # list[int], 索引
                 input_transform,        # 'multiple_select'
                 channels,               # int, 主分支通道数
                 num_classes,            # int
                 arm_out_channels,       # list[int]
                 cm_bin_sizes=[1, 2, 4],
                 cm_out_ch=128,
                 arm_type='UAFM_SpAtten',
                 resize_mode='bilinear',
                 loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 align_corners=False,
                 dropout_ratio=0.1,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            in_index=in_index,
            input_transform=input_transform,
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            loss_decode=loss_decode,
            align_corners=align_corners,
            **kwargs)

        self.ppseg_head = PPLiteSegHead(
            backbone_out_chs=in_channels,
            arm_out_chs=arm_out_channels,
            cm_bin_sizes=cm_bin_sizes,
            cm_out_ch=cm_out_ch,
            arm_type=arm_type,
            resize_mode=resize_mode
        )

        self.seg_heads = nn.ModuleList()
        for ch in arm_out_channels:
            self.seg_heads.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch, num_classes, kernel_size=1)
                )
            )

    def forward(self, inputs):
        feats = self.ppseg_head(inputs)   # e.g. [x8, x16, x32]
        logits_list = [head(feat) for head, feat in zip(self.seg_heads, feats)]
        return logits_list[0]

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logit = self.forward(inputs)
        losses = self.losses(seg_logit, gt_semantic_seg)
        # 可对 logits_list[1:] 再加 loss
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        return self.forward(inputs)