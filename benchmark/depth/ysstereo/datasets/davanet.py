import os.path as osp
from glob import glob

from ysstereo.datasets.basedataset import BaseStereoDataset
from ysstereo.registry import DATASETS

@DATASETS.register_module()
class DAVANet(BaseStereoDataset):
    """DAVANet dataset for training. 

    Args:
    """

    def __init__(self,
                 *args,
                 sharp_times: int = 5,
                 **kwargs) -> None:
        self.sharp_times = sharp_times
        setattr(self, self.UNIQUE_NAME_KEY, 's'+str(self.sharp_times))
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

            img1_filenames += sorted(self.get_data_filename(idir1, self.img1_suffix))

            img2_filenames += sorted(self.get_data_filename(idir2, self.img2_suffix))

            disp_filenames += sorted(self.get_data_filename(ddir, self.disp_suffix))

        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)

        self.load_ann_info(self.data_infos, disp_filenames, 'filename_disp')

    def _get_data_dir(self) -> None:
        """Get the paths for stereo images and disparity."""

        self.disp_suffix = '.exr'
        self.img1_suffix = '.png'
        self.img2_suffix = '.png'

        img1s_dirs = []
        img2s_dirs = []
        disp_dirs = []

        img1s_dirs_ = sorted(glob(
            osp.join(self.data_root, '*/image_left/')))
        img1s_dirs += self.sharp_times*img1s_dirs_

        img1s_dirs_ = sorted(glob(
            osp.join(self.data_root, '*/image_left_blur_ga/')))
        img1s_dirs += img1s_dirs_

        img2s_dirs_ = sorted(glob(
            osp.join(self.data_root, '*/image_right/')))
        img2s_dirs += self.sharp_times*img2s_dirs_

        img2s_dirs_ = sorted(glob(
            osp.join(self.data_root, '*/image_right_blur_ga/')))
        img2s_dirs += img2s_dirs_

        disp_dirs_ = sorted(glob(
            osp.join(self.data_root, '*/disparity_left/')))
        disp_dirs += (self.sharp_times+1)*disp_dirs_

        self.img1_dir = img1s_dirs
        self.img2_dir = img2s_dirs
        self.disp_dir = disp_dirs
