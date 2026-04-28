import os.path as osp
from glob import glob
from typing import Sequence, Union
import os
from ysstereo.datasets.basedataset import BaseStereoDataset
from ysstereo.registry import DATASETS

@DATASETS.register_module()
class SnowSyntheticDataset(BaseStereoDataset):
    """SnowSynthetic dataset.

    Args:
        pass_style (str): Pass style for SnowSynthetic dataset, and it has 3
            options ['bottom', 'front', 'all']. Default: 'all'.
    """

    def __init__(self,
                 *args,
                 subset: str = 'all',
                 **kwargs) -> None:

        assert subset in ['bottom', 'front', 'all']
        self.subset = subset
        setattr(self, self.UNIQUE_NAME_KEY, subset)
        super().__init__(*args, **kwargs)

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        disparity."""
        img1_filenames, img2_filenames, disp_filenames = self._get_data_dir()

        assert len(img1_filenames) == len(img2_filenames) == len(disp_filenames)
        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)

        self.load_ann_info(self.data_infos, disp_filenames, 'filename_disp')

    def _get_data_dir(self) -> None:
        """Get the paths for stereo images and disparity."""

        self.disp_suffix = '.exr'
        self.img1_suffix = '.png'
        self.img2_suffix = '.png'

        if self.subset == 'bottom':
            self.subset_dirs = ['bottom']
        elif self.subset == 'front':
            self.subset_dirs = ['front']
        else:
            self.subset_dirs = ['bottom', 'front']

        img1s_dirs = []
        img2s_dirs = []
        disp_dirs = []

        img1_filenames = []
        img2_filenames = []
        disp_filenames = []

        left_folder = "cam0"
        right_folder = "cam1"
        gt_folder = "depth"

        for subset in self.subset_dirs:
            scenes = os.listdir(f"{self.data_root}/{subset}")
            for scene in scenes:
                cur_scene_folder = f"{self.data_root}/{subset}/{scene}"
                img1s_dirs.append(f"{cur_scene_folder}/{left_folder}")
                img2s_dirs.append(f"{cur_scene_folder}/{right_folder}")
                disp_dirs.append(f"{cur_scene_folder}/{gt_folder}")
                # read key frames split
                filelist = os.listdir(f"{cur_scene_folder}/{left_folder}")
                for line in filelist:
                    line = line.strip()
                    if line == "" or not line.endswith(self.img1_suffix):
                        continue
                    img1_filenames.append(f"{cur_scene_folder}/{left_folder}/{line}")
                    img2_filenames.append(f"{cur_scene_folder}/{right_folder}/{line}")
                    disp_filename = line.replace(self.img1_suffix, "_left" + self.disp_suffix)
                    disp_filenames.append(f"{cur_scene_folder}/{gt_folder}/{disp_filename}")

        self.img1_dir = img1s_dirs
        self.img2_dir = img2s_dirs
        self.disp_dir = disp_dirs
        assert len(img1_filenames) == len(img2_filenames) == len(disp_filenames)

        return img1_filenames, img2_filenames, disp_filenames
