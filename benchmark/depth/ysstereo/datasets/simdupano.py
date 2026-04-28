import os.path as osp
from glob import glob
from typing import Sequence, Union

from ysstereo.datasets.basedataset import BaseStereoDataset
from ysstereo.registry import DATASETS

@DATASETS.register_module()
class SimDuPano(BaseStereoDataset):
    """SimDuPano subset dataset.
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

        for idir, idir1, idir2, ddir, ddir1, ddir2 in zip(self.img_dir, self.img1_dir, self.img2_dir, 
                                                        self.disp_dir, self.disp1_dir, self.disp2_dir):

            img_filename = self.get_data_filename(idir, self.img_suffix)
            img_filename.sort()
            img1_filename = self.get_data_filename(idir1, self.img1_suffix)
            img1_filename.sort()
            img2_filename = self.get_data_filename(idir2, self.img2_suffix)
            img2_filename.sort()

            disp_filename = self.get_data_filename(ddir, self.disp_suffix)
            disp_filename.sort()
            disp1_filename = self.get_data_filename(ddir1, self.disp1_suffix)
            disp1_filename.sort()
            disp2_filename = self.get_data_filename(ddir2, self.disp2_suffix)
            disp2_filename.sort()

            # 0-1 pair, baseline = 0.2m
            img1_filenames += img_filename
            img2_filenames += img1_filename
            disp_filenames += disp_filename

            # 1-0 pair, baseline = 0.2m
            img1_filenames += img1_filename
            img2_filenames += img_filename
            disp_filenames += disp1_filename

            # 1-2 pair, baseline = 0.2m
            img1_filenames += img1_filename
            img2_filenames += img2_filename
            disp_filenames += disp1_filename

            # 2-1 pair, baseline = 0.2m
            img1_filenames += img2_filename
            img2_filenames += img1_filename
            disp_filenames += disp2_filename

            # 0-2 pair, baseline = 0.4m
            img1_filenames += img_filename
            img2_filenames += img2_filename
            disp_filenames += disp_filename

            # 2-0 pair, baseline = 0.4m
            img1_filenames += img2_filename
            img2_filenames += img_filename
            disp_filenames += disp2_filename

        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)

        self.load_ann_info(self.data_infos, disp_filenames, 'filename_disp')

    def _get_data_dir(self) -> None:
        """Get the paths for stereo images and disparity."""

        self.disp_suffix = '.exr'
        self.disp1_suffix = '.exr'
        self.disp2_suffix = '.exr'
        self.img_suffix = '.jpg'
        self.img1_suffix = '.jpg'
        self.img2_suffix = '.jpg'

        imgs_dirs = []
        img1s_dirs = []
        img2s_dirs = []
        disp_dirs = []
        disp1_dirs = []
        disp2_dirs = []

        imgs_dirs_ = glob(
            osp.join(self.data_root, '*/*/cam_0/scene'))
        imgs_dirs += imgs_dirs_
        imgs_dirs_ = glob(
            osp.join(self.data_root, '*/*/*/cam_0/scene'))
        imgs_dirs += imgs_dirs_
        imgs_dirs_ = glob(
            osp.join(self.data_root, '*/*/*/*/cam_0/scene'))
        imgs_dirs += imgs_dirs_

        img1s_dirs_ = glob(
            osp.join(self.data_root, '*/*/cam_1/scene'))
        img1s_dirs += img1s_dirs_
        img1s_dirs_ = glob(
            osp.join(self.data_root, '*/*/*/cam_1/scene'))
        img1s_dirs += img1s_dirs_
        img1s_dirs_ = glob(
            osp.join(self.data_root, '*/*/*/*/cam_1/scene'))
        img1s_dirs += img1s_dirs_

        img2s_dirs_ = glob(
            osp.join(self.data_root, '*/*/cam_2/scene'))
        img2s_dirs += img2s_dirs_
        img2s_dirs_ = glob(
            osp.join(self.data_root, '*/*/*/cam_2/scene'))
        img2s_dirs += img2s_dirs_
        img2s_dirs_ = glob(
            osp.join(self.data_root, '*/*/*/*/cam_2/scene'))
        img2s_dirs += img2s_dirs_

        disp_dirs_ = glob(
            osp.join(self.data_root, '*/*/cam_0/depth'))
        disp_dirs += disp_dirs_
        disp_dirs_ = glob(
            osp.join(self.data_root, '*/*/*/cam_0/depth'))
        disp_dirs += disp_dirs_
        disp_dirs_ = glob(
            osp.join(self.data_root, '*/*/*/*/cam_0/depth'))
        disp_dirs += disp_dirs_

        disp1_dirs_ = glob(
            osp.join(self.data_root, '*/*/cam_1/depth'))
        disp1_dirs += disp1_dirs_
        disp1_dirs_ = glob(
            osp.join(self.data_root, '*/*/*/cam_1/depth'))
        disp1_dirs += disp1_dirs_
        disp1_dirs_ = glob(
            osp.join(self.data_root, '*/*/*/*/cam_1/depth'))
        disp1_dirs += disp1_dirs_

        disp2_dirs_ = glob(
            osp.join(self.data_root, '*/*/cam_2/depth'))
        disp2_dirs += disp2_dirs_
        disp2_dirs_ = glob(
            osp.join(self.data_root, '*/*/*/cam_2/depth'))
        disp2_dirs += disp2_dirs_
        disp2_dirs_ = glob(
            osp.join(self.data_root, '*/*/*/*/cam_2/depth'))
        disp2_dirs += disp2_dirs_

        self.img_dir = imgs_dirs
        self.img1_dir = img1s_dirs
        self.img2_dir = img2s_dirs
        self.disp_dir = disp_dirs
        self.disp1_dir = disp1_dirs
        self.disp2_dir = disp2_dirs