from mmengine.structures import BaseDataElement, PixelData


class StereoDataSample(BaseDataElement):
    """A data structure interface of ysstereo. They are used as
    interfaces between different components.

    The attributes in ``StereoDataSample`` are divided into several parts:

        - ``gt_disp``(PixelData): Ground truth of disparity.
        - ``pred_disp``(PixelData): Prediction of disparity.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.structures import PixelData
         >>> from ysstereo.structures import StereoDataSample

         >>> data_sample = StereoDataSample()
         >>> img_meta = dict(img_shape=(4, 4, 3),
         ...                 pad_shape=(4, 4, 3))
         >>> gt_disp = PixelData(metainfo=img_meta)
         >>> gt_disp.data = torch.randint(0, 2, (1, 4, 4))
         >>> data_sample.gt_disp = gt_disp
         >>> assert 'img_shape' in data_sample.gt_disp.metainfo_keys()
         >>> data_sample.gt_disp.shape
         (4, 4)
         >>> print(data_sample)
        <StereoDataSample(

            META INFORMATION

            DATA FIELDS
            gt_disp: <PixelData(

                    META INFORMATION
                    img_shape: (4, 4, 3)
                    pad_shape: (4, 4, 3)

                    DATA FIELDS
                    data: tensor([[[1, 1, 1, 0],
                                 [1, 0, 1, 1],
                                 [1, 1, 1, 1],
                                 [0, 1, 0, 1]]])
                ) at 0x1c2b4156460>
        ) at 0x1c2aae44d60>

        >>> data_sample = StereoDataSample()
        >>> gt_disp_data = dict(sem_seg=torch.rand(1, 4, 4))
        >>> gt_disp = PixelData(**gt_disp_data)
        >>> data_sample.gt_disp = gt_disp
        >>> assert 'gt_disp' in data_sample
    """

    @property
    def gt_disp(self) -> PixelData:
        return self._gt_disp

    @gt_disp.setter
    def gt_disp(self, value: PixelData) -> None:
        self.set_field(value, '_gt_disp', dtype=PixelData)

    @gt_disp.deleter
    def gt_disp(self) -> None:
        del self._gt_disp

    @property
    def pred_disp(self) -> PixelData:
        return self._pred_disp

    @pred_disp.setter
    def pred_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_pred_disp', dtype=PixelData)

    @pred_disp.deleter
    def pred_disp(self) -> None:
        del self._pred_disp
