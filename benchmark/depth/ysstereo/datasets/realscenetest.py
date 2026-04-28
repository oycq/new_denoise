import os.path as osp
from glob import glob
from typing import Sequence, Union

from ysstereo.datasets.basedataset import BaseStereoDataset
from ysstereo.registry import DATASETS

@DATASETS.register_module()
class RealSceneTest(BaseStereoDataset):
    """RealScene dataset for test. 

    Args:
    """

    def __init__(self,
                 img_suffix=None,
                 unique_name='',
                 *args,
                 **kwargs) -> None:
        self.img_suffix = img_suffix
        self.unique_name = unique_name
        setattr(self, self.UNIQUE_NAME_KEY, self.unique_name)
        super().__init__(*args, **kwargs)

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        disparity."""

        img1_filenames, img2_filenames, disp_filenames, seg_filenames = self._get_data_dir()
        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)
        self.load_ann_info(self.data_infos, disp_filenames, 'filename_disp')
        self.load_ann_info(self.data_infos, seg_filenames, 'filename_seg')

    def _get_data_dir(self) -> None:
        """Get the paths for stereo images and disparity."""

        self.disp_suffix = '.exr'
        self.seg_suffix  = '.png'
        if self.img_suffix is None:
            self.img1_suffix = '.jpg'
            self.img2_suffix = '.jpg'
        else:
            self.img1_suffix = self.img_suffix
            self.img2_suffix = self.img_suffix

        img1s_dirs = []
        img2s_dirs = []
        disp_dirs = []
        seg_dirs  = []

        img1_filenames = []
        img2_filenames = []
        disp_filenames = []
        seg_filenames  = []

        img1s_left_dirs_name = 'left/'
        seg_dirs_name = 'seg_op_da_pgt/'

        img1s_dirs_ = sorted(glob(
            osp.join(self.data_root, img1s_left_dirs_name)))
        img1s_dirs += img1s_dirs_

        img2s_dirs_ = sorted(glob(
            osp.join(self.data_root, 'right/')))
        img2s_dirs += img2s_dirs_

        disp_dirs_ = sorted(glob(
            osp.join(self.data_root, 'pseudo_gt/')))
        disp_dirs += disp_dirs_

        seg_dirs_ = sorted(glob(
            osp.join(self.data_root, seg_dirs_name)))
        seg_dirs += seg_dirs_

        for idir1 in img1s_dirs:
            img1_filenames += sorted(self.get_data_filename(idir1, self.img1_suffix))
        for idir2 in img2s_dirs:
            img2_filenames += sorted(self.get_data_filename(idir2, self.img2_suffix))
        for ddir in disp_dirs:
            disp_filenames += sorted(self.get_data_filename(ddir, self.disp_suffix))
        for sdir in seg_dirs:
            seg_filenames  += sorted(self.get_data_filename(sdir, self.seg_suffix))

        if len(seg_filenames) != len(img1_filenames):
            for img_path in img1_filenames:
                seg_expected_path = img_path.replace(img1s_left_dirs_name, seg_dirs_name)
                seg_expected_path = seg_expected_path.replace(self.img1_suffix, self.seg_suffix)
                seg_filenames.append(seg_expected_path)

        self.img1_dir = img1s_dirs
        self.img2_dir = img2s_dirs
        self.disp_dir = disp_dirs
        self.seg_dir = seg_dirs
        return img1_filenames, img2_filenames, disp_filenames, seg_filenames
            
