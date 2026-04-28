import os.path as osp
from glob import glob
from typing import Sequence, Union

from ysstereo.datasets.basedataset import BaseStereoDataset
from ysstereo.registry import DATASETS

@DATASETS.register_module()
class Booster(BaseStereoDataset):
    """Booster dataset for training. 
    
    Args:
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
            image1_list = sorted(glob(osp.join(idir1, self.img1_suffix)))
            image2_list = sorted(glob(osp.join(idir1, self.img2_suffix)))
            for img1 in image1_list:
                for img2 in image2_list:
                    img1_filenames.append(img1)
                    img2_filenames.append(img2)
                    disp_filenames.append(osp.join(idir1, self.disp_suffix))

            img1_filenames.sort()
            img2_filenames.sort()
            disp_filenames.sort()

        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)

        self.load_ann_info(self.data_infos, disp_filenames, 'filename_disp')

    def _get_data_dir(self) -> None:
        """Get the paths for stereo images and disparity."""

        self.disp_suffix = 'disp_00.npy'
        self.img1_suffix = 'camera_00/im*.png'
        self.img2_suffix = 'camera_02/im*.png'

        img1s_dirs = []
        img2s_dirs = []
        disp_dirs = []

        img1s_dirs_ = glob(
            osp.join(self.data_root, 'train/balanced/*'))
        img1s_dirs += img1s_dirs_

        img2s_dirs_ = glob(
            osp.join(self.data_root, 'train/balanced/*'))
        img2s_dirs += img2s_dirs_

        disp_dirs_ = glob(
            osp.join(self.data_root, 'train/balanced/*'))
        disp_dirs += disp_dirs_

        self.img1_dir = img1s_dirs
        self.img2_dir = img2s_dirs
        self.disp_dir = disp_dirs
