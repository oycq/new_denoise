import os.path as osp
from glob import glob
from typing import Sequence, Union

from ysstereo.datasets.basedataset import BaseStereoDataset
from ysstereo.registry import DATASETS

@DATASETS.register_module()
class NewFisheyeSim2TransEqui(BaseStereoDataset):
    """NewFisheyeSim2TransEqui subset dataset.
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

        img1_filenames = []
        img2_filenames = []
        disp_filenames = []

        for idir1, idir2, ddir in zip(self.img1_dir, self.img2_dir,
                                        self.disp_dir):

            img1_filenames += self.get_data_filename(idir1, self.img1_suffix)
            img1_filenames.sort()

            img2_filenames += self.get_data_filename(idir2, self.img2_suffix)
            img2_filenames.sort()

            disp_filenames += self.get_data_filename(ddir, self.disp_suffix)
            disp_filenames.sort()

        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)

        self.load_ann_info(self.data_infos, disp_filenames, 'filename_disp')

    def _get_data_dir(self) -> None:
        """Get the paths for stereo images and disparity."""

        self.disp_suffix = '.exr'
        self.img1_suffix = '.jpg'
        self.img2_suffix = '.jpg'

        img1s_dirs = []
        img2s_dirs = []
        disp_dirs = []

        img1s_dirs_ = glob(
            osp.join(self.data_root, '*/*/left/'))
        img1s_dirs += img1s_dirs_
        img1s_dirs_ = glob(
            osp.join(self.data_root, '*/*/*/left/'))
        img1s_dirs += img1s_dirs_

        img2s_dirs_ = glob(
            osp.join(self.data_root, '*/*/right/'))
        img2s_dirs += img2s_dirs_
        img2s_dirs_ = glob(
            osp.join(self.data_root, '*/*/*/right/'))
        img2s_dirs += img2s_dirs_

        disp_dirs_ = glob(
            osp.join(self.data_root, '*/*/depth_left/'))
        disp_dirs += disp_dirs_
        disp_dirs_ = glob(
            osp.join(self.data_root, '*/*/*/depth_left/'))
        disp_dirs += disp_dirs_

        self.img1_dir = img1s_dirs
        self.img2_dir = img2s_dirs
        self.disp_dir = disp_dirs
