import copy
from collections.abc import Sequence
from typing import Union

import mmcv
import numpy as np
import torch
from mmengine.structures import PixelData
from mmengine.structures import BaseDataElement
from ysstereo.structures import StereoDataSample
from mmcv.transforms import BaseTransform, TRANSFORMS

def to_tensor(
    data: Union[np.ndarray, torch.Tensor, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

@TRANSFORMS.register_module()
class Transpose(BaseTransform):
    """Transpose some results by given keys.

    Args:
        keys (Sequence[str]): Keys of results to be transposed.
        order (Sequence[int]): Order of transpose.
    """

    def __init__(self, keys: Sequence, order: Sequence) -> None:
        self.keys = keys
        self.order = order

    def transform(self, results: dict) -> dict:
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """

        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
               f'(keys={self.keys}, order={self.order})'

@TRANSFORMS.register_module()
class DefaultFormatBundle(BaseTransform):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "disp_gt". These fields are formatted as follows.

    - img1: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - img2: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - disp_gt: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    """

    def transform(self, results: dict) -> dict:
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'imgl' in results:
            img1 = results['imgl']
            img1 = np.expand_dims(img1, -1) if len(img1.shape) < 3 else img1
            img1 = img1.transpose(2, 0, 1)

        if 'imgr' in results:
            img2 = results['imgr']
            img2 = np.expand_dims(img2, -1) if len(img2.shape) < 3 else img2
            img2 = img2.transpose(2, 0, 1)

        # results['imgs'] = BaseDataElement(
        #     img=to_tensor(np.concatenate((img1, img2), axis=0)))
        results['imgs'] = to_tensor(np.concatenate((img1, img2), axis=0))

        if 'imgl_ori' in results:
            img1_ori = results['imgl_ori']
            img1_ori = np.expand_dims(img1_ori, -1) if len(img1_ori.shape) < 3 else img1_ori
            img1_ori = img1_ori.transpose(2, 0, 1)

        if 'imgr_ori' in results:
            img2_ori = results['imgr_ori']
            img2_ori = np.expand_dims(img2_ori, -1) if len(img2_ori.shape) < 3 else img2_ori
            img2_ori = img2_ori.transpose(2, 0, 1)
        
        if 'imgl_ori' in results and 'imgr_ori' in results:
            results['imgs_ori'] = to_tensor(np.concatenate((img1_ori, img2_ori), axis=0))

        if 'ann_fields' in results:
            ann_fields = copy.deepcopy(results['ann_fields'])
            for ann_key in ann_fields:
                if ann_key in results:
                    gt = results[ann_key]
                    gt = np.expand_dims(gt, -1) if len(gt.shape) < 3 else gt
                    gt = np.ascontiguousarray(gt.transpose(2, 0, 1))
                    results[ann_key] = to_tensor(gt.astype(np.float32))
                    # results[ann_key] = BaseDataElement(
                    #     img=to_tensor(gt.astype(np.float32)), stack=True)
        return results

    def __repr__(self) -> str:
        return self.__class__.__name__


@TRANSFORMS.register_module()
class TestFormatBundle(BaseTransform):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img1"
    and "img2". These fields are formatted as follows.

    - img1: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - img2: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    """

    def transform(self, results: dict) -> dict:
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'imgl' in results:
            img1 = results['imgl']
            img1 = np.expand_dims(img1, -1) if len(img1.shape) < 3 else img1
            img1 = img1.transpose(2, 0, 1)

        if 'imgr' in results:
            img2 = results['imgr']
            img2 = np.expand_dims(img2, -1) if len(img2.shape) < 3 else img2
            img2 = img2.transpose(2, 0, 1)

        # results['imgs'] = BaseDataElement(
        #                 img=to_tensor(np.concatenate((img1, img2), axis=0)), stack=True)
        results['imgs'] = to_tensor(np.concatenate((img1, img2), axis=0))

        return results

    def __repr__(self) -> str:
        return self.__class__.__name__


@TRANSFORMS.register_module()
class Collect(BaseTransform):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "disp_gt".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the bottom/right
            if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename1": path to the image1 file

        - "filename2": path to the image2 file

        - "ori_filename1": image1 file name

        - "ori_filename2": image2 file name

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename1', 'filename2', 'ori_filename1',
            'ori_filename2', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg')``
    """

    def __init__(
        self,
        keys: Sequence,
        meta_keys: Sequence = ('filename1', 'filename2', 'ori_filename1',
                               'ori_filename2', 'filename_disp', "filename_seg",
                               'ori_filename_disp', 'ori_shape', 'img_shape',
                               'pad_shape', 'scale_factor', 'flip',
                               'flip_direction', 'img_norm_cfg'),
        extra_meta_info: dict = {},
    ) -> None:
        self.keys = keys
        self.meta_keys = meta_keys
        for k in extra_meta_info:
            assert k not in meta_keys, f'{k} in extra_meta_info is already in meta_keys!!'
        self.extra_meta_info = extra_meta_info

    def check_and_convert_shape(self, data:np.ndarray):
        if data.ndim==2:
            return data[None]
        else:
            return data

    def transform(self, results: dict) -> dict:
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = copy.deepcopy(self.extra_meta_info)

        for key in self.meta_keys:
            img_meta[key] = results[key]
        for key in self.keys:
            data[key] = results[key]
        
        results = data

        # convert to stereo data sample        
        packed_results = dict()
        if 'imgs' in results:
            packed_results['inputs'] = results['imgs']
        data_sample = StereoDataSample()
        if 'imgs_ori' in results:
            imgs_ori_data = dict(data=self.check_and_convert_shape(results['imgs_ori']))
            data_sample.imgs_ori = PixelData(**imgs_ori_data)
        if 'disp_gt' in results:
            disp_gt_data = dict(data=self.check_and_convert_shape(results['disp_gt']))
            data_sample.gt_disp = PixelData(**disp_gt_data)
        if 'valid' in results:
            valid_data = dict(data=self.check_and_convert_shape(results['valid']))
            data_sample.set_data(dict(valid=PixelData(**valid_data)))
        if 'keysets' in results:
            keysets_data = dict(data=results['keysets'])
            data_sample.set_data(dict(keysets=PixelData(**keysets_data)))
        if 'lambda_sets' in results:
            lambda_sets_data = dict(data=results['lambda_sets'])
            data_sample.set_data(dict(lambda_sets=PixelData(**lambda_sets_data)))
        if 'pseudo_gt' in results:
            pseudo_gt_data = dict(data=self.check_and_convert_shape(results['pseudo_gt']))
            data_sample.set_data(dict(pseudo_gt=PixelData(**pseudo_gt_data)))
            pseudo_valid_data = dict(data=self.check_and_convert_shape(results['pseudo_valid']))
            data_sample.set_data(dict(pseudo_valid=PixelData(**pseudo_valid_data)))
        if 'right_pseudo_gt' in results:
            right_pseudo_gt_data = dict(data=self.check_and_convert_shape(results['right_pseudo_gt']))
            data_sample.set_data(dict(right_pseudo_gt=PixelData(**right_pseudo_gt_data)))
            right_pseudo_valid_data = dict(data=self.check_and_convert_shape(results['right_pseudo_valid']))
            data_sample.set_data(dict(right_pseudo_valid=PixelData(**right_pseudo_valid_data)))
        if 'seg_gt' in results:
            seg_gt_data = dict(data=self.check_and_convert_shape(results['seg_gt']))
            data_sample.set_data(dict(seg_gt=PixelData(**seg_gt_data)))
        if 'seg_valid' in results:
            seg_valid_data = dict(data=self.check_and_convert_shape(results['seg_valid']))
            data_sample.set_data(dict(seg_valid=PixelData(**seg_valid_data)))

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'
