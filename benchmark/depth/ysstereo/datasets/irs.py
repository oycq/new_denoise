import os.path as osp
from glob import glob
from typing import Sequence, Union

from ysstereo.datasets.basedataset import BaseStereoDataset, compute_md5
from ysstereo.registry import DATASETS
@DATASETS.register_module()
class IRSDataset(BaseStereoDataset):
    """IRS dataset.
    """

    def __init__(self,
                 *args,
                 imglist: str,
                 **kwargs) -> None:

        self.imglist = imglist
        setattr(self, self.UNIQUE_NAME_KEY, compute_md5(imglist, '')[:6])
        super().__init__(*args, **kwargs)

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        disparity."""
        self._get_data_dir()

        self.load_img_info(self.data_infos, self.img1_dir, self.img2_dir)

        self.load_ann_info(self.data_infos, self.disp_dir, 'filename_disp')

    def _get_data_dir(self) -> None:
        """Get the paths for stereo images and disparity."""

        img1s_dirs = []
        img2s_dirs = []
        disp_dirs = []

        with open(self.imglist, "r") as f:
            imgpairs = f.readlines()
        for i in range(len(imgpairs)):
            names = imgpairs[i].rstrip().split()
            img1s_dirs_ = osp.join(self.data_root, names[0])
            img1s_dirs.append(img1s_dirs_)
            img2s_dirs_ = osp.join(self.data_root, names[1])
            img2s_dirs.append(img2s_dirs_)
            disp_dirs_ = osp.join(self.data_root, names[2])
            disp_dirs.append(disp_dirs_)

        self.img1_dir = img1s_dirs
        self.img2_dir = img2s_dirs
        self.disp_dir = disp_dirs
