from typing import Dict, List, Optional

import cv2
import mmcv
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.structures import PixelData
from mmengine.visualization import Visualizer
from ysstereo.registry import VISUALIZERS

@VISUALIZERS.register_module()
class StereoDepthVisualizer(Visualizer):
    """Stereo Depth Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        alpha (int, float): The transparency of segmentation mask.
                Defaults to 0.8.
    """  # noqa

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 alpha: float = 0.8,
                 **kwargs):
        super().__init__(name, image, vis_backends, save_dir, **kwargs)
        self.alpha: float = alpha

    def _draw_disp(self, disp: np.ndarray,
                    valid: np.ndarray=None) -> np.ndarray:
        """Disparity visualization function.
        Args:
            disp (ndarray): The disparity will be render
        Returns:
            ndarray: disparity map image with RGB order.
        """
        # return value from mmcv.flow2rgb is [0, 1.] with type np.float32
        disp = disp.squeeze()
        if valid is not None:
            disp[valid<1] = 0
        max_disp = max(disp.max(), 5.0)
        flo = disp/max_disp
        flo[flo > 1.0] = 1.0
        flo[flo < 0.01] = 0.01
        flo = 1.0 - flo
        B = flo * 255
        R = 255 - B
        G = 255 - np.fabs(flo - 0.5) * 2 * 255
        merged = cv2.merge([B, G, R])
        if valid is not None:
            merged[valid<1] = 0
        disp_map = merged.astype(np.uint8)

        return disp_map

    def _draw_feat(self, image: np.ndarray,
                   feat_map) -> np.ndarray:
        if feat_map is None:
            return None
        feat_map = feat_map.cpu().data
        if isinstance(feat_map, np.ndarray):
            feat_map = torch.from_numpy(feat_map)
        if feat_map.ndim == 2:
            feat_map = feat_map[None]

        feat_map = self.draw_featmap(feat_map, resize_shape=image.shape[:2])
        self.set_image(feat_map)
        return feat_map

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample,
            predict: np.ndarray = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. it is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DataSample`, optional): Prediction
                DataSample and gt DataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT DataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """

        gt_disp_data = None
        pred_disp_data = None
        feat_map0_data = None
        feat_map1_data = None

        if draw_gt and data_sample is not None:
            if 'disp_gt' in data_sample:
                gt_disp_data = self._draw_disp(data_sample.disp_gt, data_sample.valid)

        if draw_pred and predict is not None:
            pred_disp_data = predict['disp'] if predict['disp'] is not None else image
            pred_disp_data = self._draw_disp(pred_disp_data, data_sample.valid)
            feat_map0_data = predict['fmaps0'] if predict['fmaps0'] is not None else None
            feat_map0_data = self._draw_feat(image, feat_map0_data)

            feat_map1_data = predict['fmaps1'] if predict['fmaps1'] is not None else None
            feat_map1_data = self._draw_feat(image, feat_map1_data)

        if gt_disp_data is not None and pred_disp_data is not None:
            drawn_img = np.concatenate((gt_disp_data, pred_disp_data), axis=1)
        elif gt_disp_data is not None:
            drawn_img = gt_disp_data
        else:
            drawn_img = pred_disp_data

        if feat_map0_data is not None and feat_map1_data is not None:
            drawn_feat_img = np.concatenate((feat_map0_data, feat_map1_data), axis=1)
        else:
            drawn_feat_img = np.concatenate((image, image), axis=1)
        drawn_img = np.concatenate((drawn_img, drawn_feat_img), axis=0)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(mmcv.rgb2bgr(drawn_img), out_file)
        else:
            self.add_image(name, drawn_img, step)
