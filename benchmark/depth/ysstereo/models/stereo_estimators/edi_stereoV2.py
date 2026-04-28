# use hard-ware friendly implementation of EDI-Stereo

from typing import Dict, Optional, Sequence, Tuple
import cv2

import torch, os
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from copy import deepcopy
from ysstereo.models.builder import STEREO_ESTIMATORS, build_encoder, build_decoder
from ysstereo.models.stereo_estimators.base import StereoEstimator

__all__ = ['EdiStereoV2']

@STEREO_ESTIMATORS.register_module()
class EdiStereoV2(StereoEstimator):
    """CREStereo model.

    Args:
        freeze_bn (bool, optional): Whether to freeze batchnorm layer or not.
        Default: False.
    """

    def __init__(self,
                 encoder: dict,
                 decoder: dict,
                 onnx_cfg: dict = None,
                 freeze_bn: bool = False,
                 pretrained_weight: str = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)
        self.in_channels = encoder.in_channels
        self.fuse_lconv_dw16 = nn.Sequential(
            nn.Conv2d(encoder.out_channels, encoder.out_channels, 3, 1, 1),
            nn.BatchNorm2d(encoder.out_channels), nn.ReLU(),
        )
        self.fuse_rconv_dw16 = nn.Sequential(
            nn.Conv2d(encoder.out_channels, encoder.out_channels, 3, 1, 1),
            nn.BatchNorm2d(encoder.out_channels), nn.ReLU(),
        )
        self.fuse_conv_dw16 = nn.Conv2d(encoder.out_channels, encoder.out_channels, 3, 1, 1)
        self.fuse_lconv_dw8 = nn.Sequential(
            nn.Conv2d(encoder.out_channels, encoder.out_channels, 3, 1, 1),
            nn.BatchNorm2d(encoder.out_channels), nn.ReLU(),
        )
        self.fuse_rconv_dw8 = nn.Sequential(
            nn.Conv2d(encoder.out_channels, encoder.out_channels, 3, 1, 1),
            nn.BatchNorm2d(encoder.out_channels), nn.ReLU(),
        )
        self.fuse_conv_dw8 = nn.Conv2d(encoder.out_channels, encoder.out_channels, 3, 1, 1)

        self.onnx_cfg = onnx_cfg
        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward_train(
            self,
            imgs: torch.Tensor,
            disp_gt: torch.Tensor,
            valid: Optional[torch.Tensor] = None,
            flow_init: Optional[torch.Tensor] = None,
            img_metas: Optional[Sequence[dict]] = None,
            return_preds: bool = False, 
            *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward function for RAFTStereo when model training.

        Args:
            imgs (Tensor): The concatenated input images.
            disp_gt (Tensor): The ground truth of disparity.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.
        Returns:
            Dict[str, Tensor]: The losses of output.
        
        vis = torch.cat([imgs[:,:self.in_channels,:], imgs[:,self.in_channels:,:]], axis=3)
        vis_in = colorMap('inferno', vis[0][0].detach())
        vis_gt = colorMap('inferno', disp_gt[0][0].detach())
        vis_in = np.concatenate([vis_in, vis_gt], axis=1)
        cv2.imwrite('temp/vis_cre_inputs.png', vis_in)
        """
        imgl = imgs[:, :self.in_channels, :]
        imgr = imgs[:, self.in_channels:, :]
        imgs = torch.cat([imgl, imgr], 0)

        fmaps = self.encoder(imgs)
        fmaps1, fmaps2 = [], []
        for fmap in fmaps:
            fmap1, fmap2 = torch.split(fmap, fmap.shape[0]//2, dim=0)
            fmaps1.append(fmap1)
            fmaps2.append(fmap2)
        contexts = []
        # dw16 context
        contexts.append(
            self.fuse_conv_dw16(
                self.fuse_lconv_dw16(fmaps1[0]) + self.fuse_rconv_dw16(fmaps2[0])
            )
        )
        # dw8 context
        contexts.append(
            self.fuse_conv_dw8(
                self.fuse_lconv_dw8(fmaps1[1]) + self.fuse_rconv_dw8(fmaps2[1])
            )
        )
        return self.decoder.forward_train(
                        fmaps1, fmaps2,
                        contexts = contexts,
                        flow_init=flow_init,
                        test_mode=False,
                        return_preds=return_preds,
                        disp_gt=disp_gt,
                        valid=valid,
                        *args, **kwargs)

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
        assert len(imgs.shape) > 2
        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)
        imgl = imgs[:,:self.in_channels,:]
        imgr = imgs[:,self.in_channels:,:]
        imgs = torch.cat([imgl, imgr], 0)

        fmaps = self.encoder(imgs)
        fmaps1, fmaps2 = [], []
        for fmap in fmaps:
            fmap1, fmap2 = torch.split(fmap, fmap.shape[0]//2, dim=0)
            fmaps1.append(fmap1)
            fmaps2.append(fmap2)

        contexts = []
        # dw16 context
        contexts.append(
            self.fuse_conv_dw16(
                self.fuse_lconv_dw16(fmaps1[0]) + self.fuse_rconv_dw16(fmaps2[0])
            )
        )
        # dw8 context
        contexts.append(
            self.fuse_conv_dw8(
                self.fuse_lconv_dw8(fmaps1[1]) + self.fuse_rconv_dw8(fmaps2[1])
            )
        )

        results = self.decoder.forward_test(
            fmaps1, fmaps2,
            contexts=contexts,
            flow_init=flow_init,
            test_mode=True,
            img_metas=img_metas)
        # recover iter in train
        self.decoder.iters = train_iter

        return results

    def forward_onnx(self, sub_type, imgs = None, *args, **kwargs):
        """Forward function for RumStereo when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
        """
        raise NotImplementedError
