import os.path as osp
from glob import glob
from typing import Sequence, Union
import os
from ysstereo.datasets.basedataset import BaseStereoDataset
from ysstereo.registry import DATASETS

@DATASETS.register_module()
class HandLidarDataset(BaseStereoDataset):
    """DeV3 dataset for training. 
    """

    SPLIT_VERTICAL = "vertical"
    SPLIT_HORIZON = "horizon"
    def __init__(self,
                 gt_folder:str, # name of the folder containing ground truth
                 seg_gt_folder:str, # name of the folder containing seg ground truth
                 split = "vertical",
                 gt_suffix:str = '.pfm', # suffix of the ground truth files
                 seg_gt_suffix:str = ".png", # suffix of the seg ground truth files
                 lr_folder: str = 'image_0', # left or image_0
                 *args,
                 **kwargs) -> None:
        self.gt_folder = gt_folder
        self.gt_suffix = gt_suffix
        self.split = split
        self.lr_folder = lr_folder
        self.seg_gt_folder = seg_gt_folder
        self.seg_gt_suffix = seg_gt_suffix
        suffix = gt_suffix.replace(".", "-")
        seg_suffix = seg_gt_suffix.replace(".", "-")
        assert split in [self.SPLIT_VERTICAL, self.SPLIT_HORIZON]
        setattr(self, self.UNIQUE_NAME_KEY, self.split + self.gt_folder + self.lr_folder + suffix + "_" + self.seg_gt_folder + seg_suffix)
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
        self.seg_suffix = self.seg_gt_suffix
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

        #scenes = os.listdir(self.data_root)
        scenes =  [entry for entry in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, entry))]
        with open(self.data_root+"/bad_data.txt", "r", encoding="utf-8") as bad_file:
            bad_data = [line.rstrip('\n') for line in bad_file]
        left_folder = "left" if self.lr_folder == "left" else "image_0"
        right_folder = "right" if self.lr_folder == "left" else "image_1"
        for scene in scenes:
            for date in os.listdir(f"{self.data_root}/{scene}"):
                seq_dir = f"{self.data_root}/{scene}/{date}/{self.split}"
                tgt_path = f"{seq_dir}/{gt_folder}"
                if not osp.exists(tgt_path):
                    continue
                files = os.listdir(tgt_path)
                for file in files:
                    if not file.endswith(self.gt_suffix):
                        continue
                    if f"{scene}/{date}/{self.split}/{left_folder}/{file.replace(self.gt_suffix, '.png')}" not in bad_data:
                        img1_filenames.append(f"{seq_dir}/{left_folder}/{file.replace(self.gt_suffix, '.png')}")
                        img2_filenames.append(f"{seq_dir}/{right_folder}/{file.replace(self.gt_suffix, '.png')}")
                        disp_filenames.append(f"{seq_dir}/{gt_folder}/{file}")
                        seg_filenames.append(f"{seq_dir}/{seg_gt_folder}/{file.replace(self.gt_suffix, '.png')}")

        self.img1_dir = img1s_dirs
        self.img2_dir = img2s_dirs
        self.disp_dir = disp_dirs
        self.seg_dir  = seg_dirs

        return img1_filenames, img2_filenames, disp_filenames, seg_filenames
