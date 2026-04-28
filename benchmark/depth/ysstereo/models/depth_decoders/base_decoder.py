from abc import abstractmethod
from typing import Dict, Optional, Sequence, Union

import mmcv
import numpy as np
from mmengine.model import BaseModel


class BaseDecoder(BaseModel):
    """Base class for decoder.

    Args:
        init_cfg (dict, list, optional): Config dict of weights initialization.
            Default: None.
    """

    def __init__(self, init_cfg: Optional[Union[dict, list]] = None) -> None:

        super().__init__(init_cfg=init_cfg)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Placeholder of forward function."""
        pass

    @abstractmethod
    def forward_train(self, *args, **kwargs):
        """Placeholder of forward function when model training."""
        pass

    @abstractmethod
    def forward_test(self, *args, **kwargs):
        """Placeholder of forward function when model testing."""
        pass

    @abstractmethod
    def losses(self):
        """Placeholder for model computing losses."""
        pass

    def transform_shape_list(self, input_list):
        num_elements = len(input_list[0])
        results = [[] for _ in range(num_elements)]
        for i in range(num_elements):
            for sublist in input_list:
                results[i].append(sublist[i])
        return [tuple(result) for result in results]

    def transform_pad_list(self, input_list):
        n = len(input_list[0][0])
        result = [[] for _ in range(n)]
        for sublist in input_list:
            for i in range(n):
                result[i].append(tuple(s[i] for s in sublist))
        return result

    def collect_meta_data(self, meta_data):
        if meta_data.get('ori_shape') is not None:
            meta_data['ori_shape'] = self.transform_shape_list(meta_data['ori_shape'])
        if meta_data.get('img_shape') is not None:
            meta_data['img_shape'] = self.transform_shape_list(meta_data.get('img_shape'))
        if meta_data.get('pad_shape') is not None:
            meta_data['pad_shape'] = self.transform_shape_list(meta_data.get('pad_shape'))
        if meta_data.get('pad') is not None:
            meta_data['pad'] = self.transform_pad_list(meta_data.get('pad'))
        keys = meta_data.keys()
        values = zip(*meta_data.values())
        return [dict(zip(keys, value)) for value in values]

    def get_disp(
            self,
            disp_result: Sequence[Dict[str, np.ndarray]],
            img_metas: Union[Sequence[dict], dict] = None
    ) -> Sequence[Dict[str, np.ndarray]]:
        """Reverted disparity as original size of ground truth.

        Args:
            disp_result (Sequence[Dict[str, ndarray]]): predicted results of
                disparity.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the disparity to original ground truth size. Defaults to None.

        Returns:
            Sequence[Dict[str, ndarray]]: the reverted predicted disparity.
        """
        if img_metas is None:
            return disp_result
        if not isinstance(img_metas, Sequence):
            img_metas = self.collect_meta_data(img_metas)
        if img_metas is not None:
            ori_shapes = [img_meta['ori_shape'] for img_meta in img_metas]
            img_shapes = [img_meta['img_shape'] for img_meta in img_metas]
            pad_shapes = [img_meta['pad_shape'] for img_meta in img_metas]

        if (img_metas is None
                or ori_shapes[0] == img_shapes[0] == pad_shapes[0]):
            return disp_result
        for i in range(len(disp_result)):

            pad = img_metas[i].get('pad', None)
            w_scale, h_scale = img_metas[i].get('scale_factor', (None, None))
            ori_shape = img_metas[i]['ori_shape']

            for key, f in disp_result[i].items():
                H, W = pad_shapes[i][:2]                   # fix a bug when the pad_shapes in a batch are not the same.
                if pad is not None:
                    f = f[pad[0][0]:(H - pad[0][1]), pad[1][0]:(W - pad[1][1])]

                elif (w_scale is not None and h_scale is not None):
                    f = mmcv.imresize(
                        f,
                        (ori_shape[1], ori_shape[0]),  # size(w, h)
                        interpolation='bilinear',
                        return_scale=False)
                    # f[:, :, 0] = f[:, :, 0] / w_scale
                    # f[:, :, 1] = f[:, :, 1] / h_scale
                    f[:, :] = f[:, :] / w_scale     # disparity only has 1 dimension, not optical flow
                disp_result[i][key] = f

        return disp_result
