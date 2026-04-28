# head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ysstereo.models.utils.seg_model_utils import segmenthead_c, segmenthead, SPASPP, BasicBlock
from ysstereo.models.builder import DECODERS
from .mmseg.models.decode_heads import FCNHead as _FCNHead

@DECODERS.register_module()
class DSNetHead(_FCNHead):
    def __init__(self, backbone_out_channels, num_classes, loss_decode = None, planes=64, name='s128', align_corners=False, augment=True):
        in_channels = backbone_out_channels  # 输入通道数来自backbone的输出
        channels = planes  
        super(DSNetHead, self).__init__(in_channels=in_channels, 
                                        channels=channels, 
                                        loss_decode = loss_decode,
                                        num_classes=num_classes,
                                        align_corners=align_corners)

        self.augment = augment
        self.align_corners = align_corners
        self.num_classes = num_classes
        self.out_channels = num_classes
        
        # Final convolutional layers and upsampling
        self.spp = SPASPP(backbone_out_channels, backbone_out_channels, backbone_out_channels)
        self.layer1_a = self._make_layer(BasicBlock, planes, planes, 1)
        self.up8 = nn.Sequential(
            nn.Conv2d(backbone_out_channels, planes * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        
        # Last layer
        self.lastlayer = segmenthead_c(planes * 4, planes * 4, num_classes)
        
        # Additional heads if needed (for augmentations, etc.)
        if augment:
            self.seghead_p = segmenthead(planes * 4, planes * 4, num_classes)
            self.seghead_d = segmenthead(planes * 4, planes, num_classes)
    
    def _make_layer(self, block, inplanes, planes, stride=1):
        # Similar to the backbone, but for the head if needed
        layers = []
        layers.append(block(inplanes, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        width_output = x.shape[-1]
        height_output = x.shape[-2]
        
        # Apply SPASPP, upsample, and concatenate with other features
        x = self.spp(x)
        x = self.up8(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # Final segmentation output
        x = self.lastlayer(x)

        # Optionally return multiple outputs for augmentation
        if self.augment:
            x_extra_p = self.seghead_p(x)
            x_extra_d = self.seghead_d(x)
            x_extra_1 = F.interpolate(x_extra_p, size=[height_output, width_output], mode='bilinear', align_corners=False)
            x_extra_2 = F.interpolate(x_extra_d, size=[height_output, width_output], mode='bilinear', align_corners=False)
            return [x_extra_1, x, x_extra_2]
        else:
            return x
    
    