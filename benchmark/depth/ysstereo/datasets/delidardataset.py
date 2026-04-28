import os.path as osp
from glob import glob
from typing import Sequence, Union
import os
from ysstereo.datasets.basedataset import BaseStereoDataset
from ysstereo.registry import DATASETS

@DATASETS.register_module()
class DeV3LidarDataset(BaseStereoDataset):
    """DeV3 dataset for training. 
    """
    def __init__(self,
                 gt_folder:str, # name of the folder containing ground truth
                 seg_gt_folder:str, # name of the folder containing seg ground truth
                 gt_suffix:str = '.pfm', # suffix of the ground truth files
                 seg_suffix:str = ".png", # suffix of the seg ground truth files
                 keyframes:bool=True,
                 lr_folder: str = 'image_0', # left or image_0
                 *args,
                 **kwargs) -> None:
        self.gt_folder = gt_folder
        self.gt_suffix = gt_suffix
        self.lr_folder = lr_folder
        self.seg_gt_folder = seg_gt_folder
        self.seg_suffix = seg_suffix
        suffix = gt_suffix.replace(".", "-")
        seg_suffix = seg_suffix.replace(".", "-")
        setattr(self, self.UNIQUE_NAME_KEY, ("kf" if keyframes else "") + self.gt_folder + self.lr_folder + suffix + "_" + self.seg_gt_folder + seg_suffix)
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

        self.disp_suffix = self.gt_suffix
        self.img1_suffix = '.png'
        self.img2_suffix = '.png'

        gt_folder = self.gt_folder
        seg_gt_folder = self.seg_gt_folder

        img1s_dirs = []
        img2s_dirs = []
        disp_dirs = []
        seg_dirs = []

        img1_filenames = []
        img2_filenames = []
        disp_filenames = []
        seg_filenames = []

        scenes = os.listdir(self.data_root)
        left_folder = "left" if self.lr_folder == "left" else "image_0"
        right_folder = "right" if self.lr_folder == "left" else "image_1"
        for scene in scenes:
            cur_scene_folder = f"{self.data_root}/{scene}"
            img1s_dirs.append(f"{cur_scene_folder}/{left_folder}")
            img2s_dirs.append(f"{cur_scene_folder}/{right_folder}")
            disp_dirs.append(f"{cur_scene_folder}/{gt_folder}")
            seg_dirs.append(f"{cur_scene_folder}/{seg_gt_folder}")
            # read key frames split
            filelist = os.listdir(f"{cur_scene_folder}/{left_folder}")
            for line in filelist:
                line = line.strip()
                if line == "" or not line.endswith(self.img1_suffix):
                    continue
                img1_filenames.append(f"{cur_scene_folder}/{left_folder}/{line}")
                img2_filenames.append(f"{cur_scene_folder}/{right_folder}/{line}")
                disp_name = line.replace(self.img1_suffix, self.disp_suffix)
                disp_filenames.append(f"{cur_scene_folder}/{gt_folder}/{disp_name}")
                seg_name = line.replace(self.img1_suffix, self.seg_suffix)
                seg_filenames.append(f"{cur_scene_folder}/{seg_gt_folder}/{seg_name}")
        self.img1_dir = img1s_dirs
        self.img2_dir = img2s_dirs
        self.disp_dir = disp_dirs
        self.seg_dirs = seg_dirs

        return img1_filenames, img2_filenames, disp_filenames, seg_filenames
