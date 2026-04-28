import os.path as osp
from glob import glob
from typing import Sequence, Union

from ysstereo.datasets.basedataset import BaseStereoDataset
from ysstereo.registry import DATASETS
@DATASETS.register_module()
class FlyingThings3D(BaseStereoDataset):
    """FlyingThings3D subset dataset.

    Args:
        pass_style (str): Pass style for FlyingThing3D dataset, and it has 2
            options ['clean', 'final']. Default: 'final'.
    """

    def __init__(self,
                 *args,
                 pass_style: str = 'final',
                 **kwargs) -> None:

        assert pass_style in ['clean', 'final']
        self.pass_style = pass_style
        setattr(self, self.UNIQUE_NAME_KEY, pass_style)
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

        self.disp_suffix = '.pfm'
        self.img1_suffix = '.png'
        self.img2_suffix = '.png'

        self.subset_dir = 'TEST' if self.test_mode else 'TRAIN'

        pass_dir = 'frames_' + self.pass_style + 'pass'
        img1s_dirs = []
        img2s_dirs = []
        disp_dirs = []

        img1s_dirs_ = glob(
            osp.join(self.data_root, pass_dir, self.subset_dir + '/*/*/left/'))
        img1s_dirs += img1s_dirs_

        img2s_dirs_ = glob(
            osp.join(self.data_root, pass_dir, self.subset_dir + '/*/*/right/'))
        img2s_dirs += img2s_dirs_

        disp_dirs_ = glob(
            osp.join(self.data_root, 'disparity', self.subset_dir + '/*/*/left/'))
        disp_dirs += disp_dirs_

        self.img1_dir = img1s_dirs
        self.img2_dir = img2s_dirs
        self.disp_dir = disp_dirs
