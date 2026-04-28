from typing import Dict, Optional, Sequence, Tuple
import cv2
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    DepthAnythingV2 = None

import torch, os
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from copy import deepcopy
from ysstereo.models.builder import STEREO_ESTIMATORS, build_encoder, build_decoder
from ysstereo.models.stereo_estimators.base import StereoEstimator
from mmengine.logging import print_log
from mmengine.model import initialize

def convert_color(input: Tensor, weight: Tensor, offset: Tensor) -> Tensor:
    """Convert RGB to YUV BT.601.

    (https://en.wikipedia.org/wiki/YUV#Studio_swing_for_YCbCr_BT.601)
    """
    input = input.to(torch.float32)
    weight = weight.to(torch.float32).to(input.device)
    bias = (torch.ones(3) * 128).to(input.device)
    weight = weight.unsqueeze(2).unsqueeze(3)  # weight shape: [3,3,1,1]
    res = torch.nn.functional.conv2d(input, weight, bias) / 256
    offset = offset.to(torch.float32).to(input.device)
    offset = offset.reshape(1, 3, 1, 1)
    res += offset
    return res.to(torch.int32)

def bgr2yuv(input: Tensor, swing: str = "studio") -> Tensor:
    """Convert color space.

    Convert images from BGR format to YUV444 BT.601

    Args:
        input: input image in BGR format, ranging 0~255
        swing: "studio" for YUV studio swing (Y: 16~235,
                U, V: 16~240).
                "full" for YUV full swing (Y, U, V: 0~255).
                default is "studio"

    Returns:
        Tensor: YUV image
    """
    weight_map = {
        "studio": [[25, 129, 66], [112, -74, -38], [-18, -94, 112]],
        "full": [[29, 150, 77], [127, -84, -43], [-21, -106, 127]],
    }
    offset_map = {
        "studio": [16, 128, 128],
        "full": [0, 128, 128],
    }
    assert (
        swing in weight_map
    ), '`swing` is not valid! must be "full" or "studio"!'
    return convert_color(
        input,
        torch.tensor(weight_map[swing]),
        torch.tensor(offset_map[swing]),
    )

def bgr2centered_yuv(input: Tensor, swing: str = "studio") -> Tensor:
    """Convert color space.

    Convert images from BGR format to centered YUV444 BT.601

    Args:
        input: input image in BGR format, ranging 0~255
        swing: "studio" for YUV studio swing (Y: -112~107,
                U, V: -112~112).
                "full" for YUV full swing (Y, U, V: -128~127).
                default is "studio"

    Returns:
        Tensor: centered YUV image
    """
    return bgr2yuv(input, swing) - 128


def _fetch_da_v2(encoder='vitl'):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[encoder])
    da_pth_path = f'/home/louzhiqiang/area/stereo/Depth-Anything-V2/depth_anything_v2_{encoder}.pth'
    if not os.path.exists(da_pth_path):
        da_pth_path = f'/media/{da_pth_path}'
    model.load_state_dict(torch.load(da_pth_path, map_location='cpu'))
    model.eval()
    return model

@STEREO_ESTIMATORS.register_module()
class SepCxtNewRumStereo(StereoEstimator):
    """CREStereo model.

    Args:
        freeze_bn (bool, optional): Whether to freeze batchnorm layer or not.
        Default: False.
    """

    def __init__(self,
                 encoder: dict,
                 depth_decoder: dict,
                 seg_decoder: dict,
                 fused_context: bool = False,
                 onnx_cfg: dict = None,
                 freeze_bn: bool = False,
                 use_dav2_out: bool = False,
                 limit_infer_size = False,
                 fronzen_encoder_features: bool = False,
                 fronzen_depth_features: bool = False,
                 fronzen_seg_features: bool = False,
                 pretrained_weight: str = None,
                 train_depth: bool = True,
                 train_seg: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.train_depth = train_depth
        self.train_seg = train_seg
        self.limit_infer_size = limit_infer_size
        self.encoder = build_encoder(encoder)
        self.depth_decoder_dict = depth_decoder
        self.depth_decoder = build_decoder(depth_decoder)
        self.seg_decoder = build_decoder(seg_decoder)
        self.in_channels = encoder.in_channels
        self.depth_in_index = depth_decoder.in_index
        self.seg_in_index = seg_decoder.in_index
        self.fused_context = fused_context
        self.fronzen_depth_features = fronzen_depth_features
        self.fronzen_encoder_features = fronzen_encoder_features
        self.fronzen_seg_features = fronzen_seg_features
        if self.fused_context:
            self.fuse_lconv_dw16 = nn.Sequential(
                nn.Conv2d(encoder.out_channels[-2], encoder.out_channels[-2], 3, 1, 1),
                nn.SyncBatchNorm(encoder.out_channels[-2]),
                nn.ReLU(),
            )
            self.fuse_rconv_dw16 = nn.Sequential(
                nn.Conv2d(encoder.out_channels[-2], encoder.out_channels[-2], 3, 1, 1),
                nn.SyncBatchNorm(encoder.out_channels[-2]),
                nn.ReLU(),
            )
            self.fuse_conv_dw16 = nn.Conv2d(encoder.out_channels[-2], encoder.out_channels[-2], 3, 1, 1)
            self.fuse_lconv_dw8 = nn.Sequential(
                nn.Conv2d(encoder.out_channels[-3], encoder.out_channels[-3], 3, 1, 1),
                nn.SyncBatchNorm(encoder.out_channels[-3]),
                nn.ReLU(),
            )
            self.fuse_rconv_dw8 = nn.Sequential(
                nn.Conv2d(encoder.out_channels[-3], encoder.out_channels[-3], 3, 1, 1),
                nn.SyncBatchNorm(encoder.out_channels[-3]),
                nn.ReLU(),
            )
            self.fuse_conv_dw8 = nn.Conv2d(encoder.out_channels[-3], encoder.out_channels[-3], 3, 1, 1)
        else:
            sep_encoder = deepcopy(encoder)
            self.use_dav2_out = use_dav2_out
            sep_encoder.in_channels = self.in_channels * 2 + (1 if self.use_dav2_out else 0)
            self.cxt_encoder = build_encoder(sep_encoder)
            if self.use_dav2_out:
                self.da = _fetch_da_v2()
                for params in self.da.parameters():
                    params.requires_grad = False

        if self.fronzen_encoder_features or kwargs.get("fronzen_encoder_features", False):
            print_log("fronzen_encoder_features...")
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.fronzen_depth_features or kwargs.get("fronzen_depth_features", False):
            print_log("fronzen_depth_features...")
            for param in self.depth_decoder.parameters():
                param.requires_grad = False
            for param in self.depth_decoder.parameters():
                param.requires_grad = False

        if self.fronzen_seg_features or kwargs.get("fronzen_seg_features", False):
            print_log("fronzen_seg_features...")
            for param in self.seg_decoder.parameters():
                param.requires_grad = False
            for param in self.seg_decoder.parameters():
                param.requires_grad = False

        self.onnx_cfg = onnx_cfg
        if freeze_bn:
            self.freeze_bn()

    @torch.no_grad()
    def _get_da_depth(self, img, img_size=(518, 518)):
        h, w = img.shape[2:]
        img = F.interpolate(img, img_size, mode="bilinear", align_corners=True)
        depth = self.da(img[:, [2, 1, 0], :, :]).unsqueeze(1) # bgr to rgb
        depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)
        return depth

    def train(self, mode:bool=True):
        self.encoder.train(mode)
        self.depth_decoder.train(mode)
        self.seg_decoder.train(mode)
        if not self.fused_context:
            self.cxt_encoder.train(mode)
            if self.use_dav2_out:
                self.da.eval()
        else:
            self.fuse_conv_dw16.train(mode)
            self.fuse_lconv_dw16.train(mode)
            self.fuse_rconv_dw16.train(mode)
            self.fuse_conv_dw8.train(mode)
            self.fuse_lconv_dw8.train(mode)
            self.fuse_rconv_dw8.train(mode)

    def freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def reinit_depth_decoder(self) -> None:
        print_log("reinit_depth_decoder...")
        self.depth_decoder = build_decoder(self.depth_decoder_dict)

    def forward_train(
            self,
            imgs: torch.Tensor,
            disp_gt: torch.Tensor,
            seg_gt: torch.Tensor,
            depth_valid: Optional[torch.Tensor] = None,
            seg_valid: Optional[torch.Tensor] = None,
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
        imgl = bgr2centered_yuv(imgl, swing="full")/128.0
        imgr = bgr2centered_yuv(imgr, swing="full")/128.0
        da_depth = None
        if not self.fused_context:
            if self.use_dav2_out:
                da_depth = self._get_da_depth(imgl)
                contexts_input = torch.cat([imgs, da_depth], 1)
                contexts = self.cxt_encoder(contexts_input)
            else:
                contexts = self.cxt_encoder(imgs)
        imgs = torch.cat([imgl, imgr], 0)

        fmaps = self.encoder(imgs)
        fmaps1, fmaps2 = [], []
        for fmap in fmaps:
            fmap1, fmap2 = torch.split(fmap, fmap.shape[0]//2, dim=0)
            fmaps1.append(fmap1)
            fmaps2.append(fmap2)

        d_fmaps1 = [fmaps1[i] for i in self.depth_in_index]
        d_fmaps2 = [fmaps2[i] for i in self.depth_in_index]
        s_fmaps1 = [fmaps1[i] for i in self.seg_in_index[::-1]]
    
        if self.fused_context:
            contexts = []
            # dw16 context
            contexts.append(
                self.fuse_conv_dw16(
                    self.fuse_lconv_dw16(d_fmaps1[0]) + self.fuse_rconv_dw16(d_fmaps2[0])
                )
            )
            # dw8 context
            contexts.append(
                self.fuse_conv_dw8(
                    self.fuse_lconv_dw8(d_fmaps1[1]) + self.fuse_rconv_dw8(d_fmaps2[1])
                )
            )

        disp_loss, disp_preds = None, None
        if self.train_depth and disp_gt is not None:
            disp_loss, disp_preds = self.depth_decoder.forward_train(
                                                        d_fmaps1,
                                                        d_fmaps2,
                                                        flow_init=flow_init,
                                                        test_mode=False,
                                                        return_preds=return_preds,
                                                        disp_gt=disp_gt,
                                                        valid=depth_valid,
                                                        contexts = contexts,
                                                        ref_depth = da_depth,
                                                        *args, **kwargs)

        seg_loss, seg_preds = None, None
        if self.train_seg and seg_gt is not None:
            seg_loss, seg_preds = self.seg_decoder.forward_train(s_fmaps1,
                                                                return_preds=return_preds,
                                                                seg_gt=seg_gt,
                                                                train_cfg=self.train_cfg,
                                                                valid=seg_valid)
        
        if return_preds:
            return disp_loss, seg_loss, disp_preds, seg_preds
        else:
            return disp_loss, seg_loss

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
        train_iter = self.depth_decoder.iters
        if self.test_cfg is not None and self.test_cfg.get(
                'iters') is not None:
            self.depth_decoder.iters = self.test_cfg.get('iters')
        if self.limit_infer_size:
            h, w = imgs.shape[2:]
            nh, nw = (h // 112) * 112, (w//112)*112
            if nw > nh:
                nw = min(nw, 1120)
                nh = (int(nw*h/w) // 112 + 1)*112
            else:
                nh = min(nh, 1120)
                nw = (int(nh*w/h) // 112 + 1)*112
            disp_factor = w / nw
            imgs = F.interpolate(
                imgs,
                size=(nh, nw),
                mode='bicubic',
                align_corners=True,
            )
        assert len(imgs.shape) > 2
        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)

        imgl = imgs[:,:self.in_channels,:]
        imgr = imgs[:,self.in_channels:,:]
        imgl = bgr2centered_yuv(imgl, swing="full")/128.0
        imgr = bgr2centered_yuv(imgr, swing="full")/128.0
        da_depth = None
        if not self.fused_context:
            if self.use_dav2_out:
                da_depth = self._get_da_depth(imgl)
                contexts_input = torch.cat([imgs, da_depth], 1)
                contexts = self.cxt_encoder(contexts_input)
            else:
                contexts = self.cxt_encoder(imgs)
        imgs = torch.cat([imgl, imgr], 0)

        fmaps = self.encoder(imgs)
        fmaps1, fmaps2 = [], []
        for fmap in fmaps:
            fmap1, fmap2 = torch.split(fmap, fmap.shape[0]//2, dim=0)
            fmaps1.append(fmap1)
            fmaps2.append(fmap2)

        d_fmaps1 = [fmaps1[i] for i in self.depth_in_index]
        d_fmaps2 = [fmaps2[i] for i in self.depth_in_index]
        s_fmaps1 = [fmaps1[i] for i in self.seg_in_index[::-1]]

        if self.fused_context:
            contexts = []
            # dw16 context
            contexts.append(
                self.fuse_conv_dw16(
                    self.fuse_lconv_dw16(d_fmaps1[0]) + self.fuse_rconv_dw16(d_fmaps2[0])
                )
            )
            # dw8 context
            contexts.append(
                self.fuse_conv_dw8(
                    self.fuse_lconv_dw8(d_fmaps1[1]) + self.fuse_rconv_dw8(d_fmaps2[1])
                )
            )
        if self.limit_infer_size:
            disp_pred = self.depth_decoder.forward(d_fmaps1, d_fmaps2, flow_init, True, contexts, da_depth)
            disp_pred = disp_factor * F.interpolate(
                disp_pred,
                size=(h, w),
                mode='bilinear',
                align_corners=True,
            )
            result = disp_pred.permute(0, 2, 3, 1).cpu().data.numpy()
            #disp_result = [d.permute(0, 2, 3, 1).cpu().data.numpy() for d in disp_pred]
            # unravel batch dim
            result = list(result)
            result = [dict(disp=f) for f in result]
            disp_results = self.depth_decoder.get_disp(result, img_metas=img_metas)
        else:
            disp_results = self.depth_decoder.forward_test(
                                                d_fmaps1,
                                                d_fmaps2,
                                                flow_init=flow_init,
                                                test_mode=True,
                                                contexts=contexts,
                                                ref_depth=da_depth,
                                                img_metas=img_metas)

            seg_results = self.seg_decoder.forward_test(s_fmaps1)

        assert len(disp_results) == len(seg_results), "the disp is inconsistent with the length of the seg result"
        test_results = disp_results
        for idx in range(len(test_results)):
            test_results[idx]['disp'] = disp_results[idx]['disp']
            test_results[idx]['seg'] = seg_results[idx]['seg']

        # recover iter in train
        self.depth_decoder.iters = train_iter
        self.seg_decoder.iters   = train_iter

        return test_results

    def forward_onnx(self, sub_type, imgs = None, *args, **kwargs):
        """Forward function for RumStereo when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
        """
        onnx_submodule = {
            'feat_submodule': self.decoder.forward_onnx_feat,
            # 'transform_submodule': self.decoder.forward_onnx_feat,
            'dw16corr_update_submodule': self.decoder.forward_onnx_dw16corr_update,
            'dw16flow_upsample_submodule': self.decoder.forward_onnx_dw16flow_upsample,
            'dw8corr_update_submodule': self.decoder.forward_onnx_dw8corr_update,
            'mask_submodule': self.decoder.forward_onnx_mask,
            # 'gridsample_submodule': self.decoder.forward_onnx_gridsample,
            # 'generatecorrs_submodule': self.decoder.forward_onnx_generatecorrs,
            # 'updateblock_submodule': self.decoder.forward_onnx_updateblock,
        }

        if sub_type == 'feat_submodule':
            # imgs = torch.cat([imgs[0], imgs[1]], dim=0)
            # fmap = self.encoder(imgs)
            # fmap1, fmap2 = torch.split(fmap, fmap.shape[0]//2, dim=0)
            left_input = 2 * (imgs[0] / 255.0) - 1.0
            right_input = 2 * (imgs[1] / 255.0) - 1.0

            fmap1 = self.encoder(left_input)
            fmap2 = self.encoder(right_input)
            results = self.decoder.forward_onnx_feat(fmap1, fmap2)
            # results = (fmap1, fmap2)
        elif sub_type == 'rumstereo':
            left_input = 2 * (imgs[0] / 255.0) - 1.0
            right_input = 2 * (imgs[1] / 255.0) - 1.0

            fmap1 = self.encoder(left_input)
            fmap2 = self.encoder(right_input)
            # images = torch.cat([imgs[0], imgs[1]], dim=0)
            # fmap = self.encoder(images)
            # fmap1, fmap2 = torch.split(fmap, fmap.shape[0]//2, dim=0)

            results = self.decoder.forward_onnx_rumstereo(fmap1, fmap2)
        else:
            results = onnx_submodule[sub_type](*args, **kwargs)
        return results
