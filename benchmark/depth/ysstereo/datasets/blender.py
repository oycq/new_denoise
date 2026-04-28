import os.path as osp
from glob import glob
from typing import Sequence, Union

from ysstereo.datasets.basedataset import BaseStereoDataset
from ysstereo.registry import DATASETS

@DATASETS.register_module()
class Blender(BaseStereoDataset):
    """Blender subset dataset.
    """

    def __init__(self,
                 *args,
                 **kwargs) -> None:
        setattr(self, self.UNIQUE_NAME_KEY, '')
        super().__init__(*args, **kwargs)

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        disparity."""
        self._get_data_dir()

        self.img1_dir.sort()
        self.img2_dir.sort()
        self.disp_dir.sort()

        self.load_img_info(self.data_infos, self.img1_dir, self.img2_dir)

        self.load_ann_info(self.data_infos, self.disp_dir, 'filename_disp')

    def _get_data_dir(self) -> None:
        """Get the paths for stereo images and disparity."""

        img1s_dirs = []
        img2s_dirs = []
        disp_dirs = []

        img1s_dirs_ = glob(
            osp.join(self.data_root, '*/*_left.jpg'))
        img1s_dirs += img1s_dirs_

        img2s_dirs_ = glob(
            osp.join(self.data_root, '*/*_right.jpg'))
        img2s_dirs += img2s_dirs_

        disp_dirs_ = glob(
            osp.join(self.data_root, '*/*_left.disp.png'))
        disp_dirs += disp_dirs_

        self.img1_dir = img1s_dirs
        self.img2_dir = img2s_dirs
        self.disp_dir = disp_dirs
