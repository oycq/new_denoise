from typing import Dict, Optional, Sequence, Tuple

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy import ndarray
import numpy as np

from ysstereo.registry import MODELS
from ysstereo.models.builder import build_encoder, build_decoder, build_components
from ysstereo.models.stereo_estimators.base import StereoEstimator
from ysstereo.utils.misc import colorMap


@MODELS.register_module()
class HresStereo(StereoEstimator):
    """HresStereo model.

    Args:
        num_levels (int): Number of levels in .
        cxt_channels (int): Number of channels of context feature.
        h_channels (int): Number of channels of hidden feature in .
        freeze_bn (bool, optional): Whether to freeze batchnorm layer or not.
            Default: False.
    """

    def __init__(self,
                 encoder: dict,
                 decoder: dict,
                 feat_component: dict,
                 h_channels: int,
                 num_layers: int,
                 onnx_cfg: dict,
                 freeze_bn: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.h_channels = h_channels

        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)
        self.conv2 = build_components(feat_component)
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(h_channels, h_channels*3, 3, padding=1) for i in range(num_layers)])

        assert self.h_channels == self.decoder.h_channels
        # assert self.h_channels + self.cxt_channels == self.context.out_channels

        self.onnx_cfg = onnx_cfg
        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def extract_feat(
        self, imgs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        """Extract features from images.

        Args:
            imgs (Tensor): The concatenated input images.

        Returns:
            Tuple[Tensor, Tensor, list[Tensor], list[Tensor]]: The feature from the first
                image, the feature from the second image, the hidden state
                feature for GRU cell and the contextual feature.
        """
        imgl = imgs[:,:3,:]
        imgr = imgs[:,3:,:]
        imgs = torch.cat([imgl, imgr], 0)

        fmap = self.encoder(imgs)
        fmap1, fmap2 = torch.split(fmap, fmap.shape[0]//2, dim=0)

        net, inp = torch.split(fmap1, [self.h_channels,self.h_channels], dim=1)
        net = torch.tanh(net)
        inp = F.relu(inp)

        net_dw8 = F.avg_pool2d(net, 2, stride=2)
        inp_dw8 = F.avg_pool2d(inp, 2, stride=2)

        net_list = [net, net_dw8]
        inp_list = [inp, inp_dw8]

        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        return fmap1, fmap2, net_list, inp_list

    def forward_train(
            self,
            imgs: torch.Tensor,
            disp_gt: torch.Tensor,
            valid: Optional[torch.Tensor] = None,
            flow_init: Optional[torch.Tensor] = None,
            img_metas: Optional[Sequence[dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward function for RAFTStereo when model training.

        Args:
            imgs (Tensor): The concatenated input images.
            disp_gt (Tensor): The ground truth of disparity.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Dict[str, Tensor]: The losses of output.
        """
        vis = torch.cat([imgs[:,:3,:], imgs[:,3:,:]], axis=3)
        vis_in = colorMap('inferno', vis[0][0].detach())
        vis_gt = colorMap('inferno', disp_gt[0][0].detach())
        vis_in = np.concatenate([vis_in, vis_gt], axis=1)
        cv2.imwrite('temp/vis_raft_inputs.png', vis_in)

        feat1, feat2, net_list, inp_list = self.extract_feat(imgs)

        return self.decoder.forward_train(
            feat1,
            feat2,
            flow=flow_init,
            net_list=net_list,
            inp_list=inp_list,
            disp_gt=disp_gt,
            valid=valid)

    def forward_test(
            self,
            imgs: torch.Tensor,
            flow_init: Optional[torch.Tensor] = None,
            img_metas: Optional[Sequence[dict]] = None) -> Sequence[ndarray]:
        """Forward function for RAFTStereo when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the disparity to original ground truth size. Defaults to None.

        Returns:
            Sequence[Dict[str, ndarray]]: the batch of predicted disparity
                with the same size of images after augmentation.
        """
        train_iter = self.decoder.iters
        if self.test_cfg is not None and self.test_cfg.get(
                'iters') is not None:
            self.decoder.iters = self.test_cfg.get('iters')

        feat1, feat2, net_list, inp_list = self.extract_feat(imgs)
        B, _, H, W = feat1.shape

        if flow_init is None:
            flow_init = torch.zeros((B, 2, H, W), device=feat1.device)

        results = self.decoder.forward_test(
            feat1=feat1,
            feat2=feat2,
            flow=flow_init,
            net_list=net_list,
            inp_list=inp_list,
            img_metas=img_metas)
        # recover iter in train
        self.decoder.iters = train_iter

        return results

    def forward_onnx(self, sub_type, imgs = None, *args, **kwargs):
        """Forward function for RAFTStereo when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
        """
        pass