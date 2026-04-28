from typing import Optional, Union, Sequence

import numpy as np
import math
import torch
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor

from ysstereo.registry import MODELS
from ysstereo.structures import StereoDataSample
from mmengine.model.utils import stack_batch
from mmengine.utils import is_seq_of

@MODELS.register_module()
class StereoDataPreProcessor(BaseDataPreprocessor):
    """Image pre-processor for depth estimation tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the input size with defined ``pad_val``, and pad seg map
        with defined ``seg_pad_val``.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).

    Args:
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        non_blocking (bool): Whether block current process
            when transferring data to device.
            New in version v0.3.0.
    """

    def __init__(
        self,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        non_blocking: Optional[bool] = False,
        use_mixed_data = False):
        super().__init__(non_blocking)
        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')

        self._channel_conversion = rgb_to_bgr or bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.use_mixed_data = use_mixed_data

    def forward(self, data, training: bool = False, predict: bool = False):
        if training and self.use_mixed_data:
            data2 = data.pop("data_B")
            data1 = self.forward_impl(data, training, predict)
            data2 = self.forward_impl(data2, training, predict)
            data1['data_samples']['data_B'] = data2
            return data1
        else:
            return self.forward_impl(data, training, predict)

    def forward_impl(self, data: Sequence[StereoDataSample], training: bool = False, predict: bool = False) -> Union[dict, list]:
        """Performs normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataset. If the collate
                function of DataLoader is :obj:`pseudo_collate`, data will be a
                list of dict. If collate function is :obj:`default_collate`,
                data will be a tuple with batch input tensor and list of data
                samples.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.

        Returns:
            dict or list: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        batch_data_samples = dict()
        if predict:
            data_samples = [data_samples]
            meta_info_list = [data_sample.metainfo for data_sample in data_samples]
            batch_data_samples['img_metas'] = meta_info_list
            return dict(inputs=inputs, data_samples=batch_data_samples)

        inputs = self.stack_data(inputs, True)
        meta_info_list = [data_sample.metainfo for data_sample in data_samples]
        batch_data_samples['img_metas'] = meta_info_list
        if training:
            batch_disps = [data_sample.gt_disp.data for data_sample in data_samples]
            batch_valid = [data_sample.valid.data for data_sample in data_samples]
            batch_data_samples['disp_gt'] = self.stack_data(batch_disps, False)
            batch_data_samples['valid'] = self.stack_array_data(batch_valid)

            batch_segs  = [data_sample.seg_gt.data for data_sample in data_samples]
            betch_segs_valid = [data_sample.seg_valid.data for data_sample in data_samples]
            batch_data_samples['seg_gt'] = self.stack_data(batch_segs, False)
            batch_data_samples['seg_valid'] = self.stack_array_data(betch_segs_valid)

            # for fake gt from dav2
            if 'pseudo_gt' in data_samples[0]:
                batch_pseudo_gt = [data_sample.pseudo_gt.data for data_sample in data_samples]
                batch_pseudo_valid = [data_sample.pseudo_valid.data for data_sample in data_samples]
                batch_data_samples['pseudo_gt'] = self.stack_data(batch_pseudo_gt, False)
                batch_data_samples['pseudo_valid'] = self.stack_array_data(batch_pseudo_valid)

            if 'right_pseudo_gt' in data_samples[0]:
                batch_right_pseudo_gt = [data_sample.right_pseudo_gt.data for data_sample in data_samples]
                batch_right_pseudo_valid = [data_sample.right_pseudo_valid.data for data_sample in data_samples]
                batch_data_samples['right_pseudo_gt'] = self.stack_data(batch_right_pseudo_gt, False)
                batch_data_samples['right_pseudo_valid'] = self.stack_array_data(batch_right_pseudo_valid)

            if 'keysets' in data_samples[0]:
                batch_keysets = [data_sample.keysets.data for data_sample in data_samples]
                batch_data_samples['keysets'] = torch.stack(batch_keysets)
            if 'lambda_sets' in data_samples[0]:
                batch_lambda_sets = [data_sample.lambda_sets.data for data_sample in data_samples]
                batch_data_samples['lambda_sets'] = torch.stack(batch_lambda_sets)
            if 'imgs_ori' in data_samples[0]:
                batch_imgs_ori = [data_sample.imgs_ori.data for data_sample in data_samples]
                batch_data_samples['imgs_ori'] = self.stack_data(batch_imgs_ori, True)

        return dict(inputs=inputs, data_samples=batch_data_samples)

    def stack_data(self, data, is_img):
        """Stack data to batch format.
        Args:
            data (list): List of data samples.
            is_img (bool): Whether the data is an image.
        """
        # Process data with `pseudo_collate`.
        if is_seq_of(data, torch.Tensor):
            statck_data = []
            for _batch_img in data:
                # channel transform
                if is_img and self._channel_conversion:
                    _batch_img = _batch_img[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_img = _batch_img.float()
                statck_data.append(_batch_img)
            # Pad and stack Tensor.
            statck_data = stack_batch(statck_data, self.pad_size_divisor,
                                       self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(data, torch.Tensor):
            if is_img:
                assert data.dim() == 4, (
                    'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                    'or a list of tensor, but got a tensor with shape: '
                    f'{data.shape}')
                if self._channel_conversion:
                    data = data[:, [2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            data = data.float()
            h, w = data.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            statck_data = F.pad(data, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')
        return statck_data
    
    def stack_array_data(self, data):
        statck_array = []
        for _batch_data in data:
            # Convert to float after channel conversion to ensure
            # efficiency
            _batch_data = torch.from_numpy(_batch_data).cuda()
            statck_array.append(_batch_data)
        # Pad and stack Tensor.
        statck_array = stack_batch(statck_array, self.pad_size_divisor,
                                    self.pad_value)
        return statck_array