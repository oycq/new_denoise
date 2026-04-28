from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from numpy import ndarray

from ysstereo.registry import MODELS
from ysstereo.models.builder import build_encoder, build_decoder
from ysstereo.models.stereo_estimators.base import StereoEstimator

@MODELS.register_module()
class CREStereo(StereoEstimator):
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
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)

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
        Returns:
            Dict[str, Tensor]: The losses of output.
        """
        imgl = imgs[:,:3,:]
        imgr = imgs[:,3:,:]
        imgs = torch.cat([imgl, imgr], 0)

        fmap = self.encoder(imgs)
        fmap1, fmap2 = torch.split(fmap, fmap.shape[0]//2, dim=0)

        return self.decoder.forward_train(
                        fmap1,
                        fmap2,
                        flow_init=flow_init,
                        test_mode=False,
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

        imgl = imgs[:,:3,:]
        imgr = imgs[:,3:,:]
        imgs = torch.cat([imgl, imgr], 0)

        fmap = self.encoder(imgs)
        fmap1, fmap2 = torch.split(fmap, fmap.shape[0]//2, dim=0)

        results = self.decoder.forward_test(
            fmap1,
            fmap2,
            flow_init=flow_init,
            test_mode=True,
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