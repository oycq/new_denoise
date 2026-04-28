import os.path as osp
import os
from glob import glob
from typing import Sequence, Union

from ysstereo.datasets.basedataset import BaseStereoDataset
from ysstereo.registry import DATASETS

@DATASETS.register_module()
class DeV3Dataset(BaseStereoDataset):
    """DeV3 dataset for training. 
    """
    def __init__(self,
                 gt_folder:str, # name of the folder containing ground truth
                 seg_gt_folder:str, # name of the folder containing seg ground truth
                 gt_suffix:str = '.pfm', # suffix of the ground truth files
                 seg_gt_suffix:str = ".png", # suffix of the seg ground truth files
                 keyframes:bool=True,
                 altitude:str = 'None',
                 *args,
                 **kwargs) -> None:
        self.gt_suffix = gt_suffix
        self.use_keyframe_split = keyframes
        self.altitude = altitude
        self.gt_folder = gt_folder
        self.seg_gt_folder = seg_gt_folder
        self.seg_gt_suffix = seg_gt_suffix
        suffix = gt_suffix.replace(".", "-")
        seg_suffix = seg_gt_suffix.replace(".", "-")
        setattr(self, self.UNIQUE_NAME_KEY, ("kf" if keyframes else "") + self.gt_folder + suffix + "_" + self.seg_gt_folder + seg_suffix)
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

        scenes = os.listdir(self.data_root)
        for scene in scenes:
            cur_scene_folder = f"{self.data_root}/{scene}"
            if os.path.exists(f"{cur_scene_folder}/left"):
                lfolder, rfolder = "left", "right"
            elif os.path.exists(f"{cur_scene_folder}/cam0"):
                lfolder, rfolder = "cam0", "cam1"
            else:
                raise ValueError("No left or cam0 folder found")
            img1s_dirs.append(f"{cur_scene_folder}/{lfolder}")
            img2s_dirs.append(f"{cur_scene_folder}/{rfolder}")
            disp_dirs.append(f"{cur_scene_folder}/{gt_folder}")
            seg_dirs.append(f"{cur_scene_folder}/{seg_gt_folder}")
            # read key frames split
            if self.use_keyframe_split:
                filelist_path = f"{cur_scene_folder}/keyframes_fly.txt"
                if self.altitude == 'high':
                    filelist_path = f"{cur_scene_folder}/keyframes_high.txt"
                    filelist_path_superhigh = f"{cur_scene_folder}/keyframes_superhigh.txt"
                elif self.altitude == 'low':
                    filelist_path = f"{cur_scene_folder}/keyframes_low.txt"
                elif self.altitude == 'all':
                    filelist_path= f"{cur_scene_folder}/keyframes_high.txt"
                    filelist_path_low = f"{cur_scene_folder}/keyframes_low.txt"
                    filelist_path_superhigh = f"{cur_scene_folder}/keyframes_superhigh.txt"
            else:
                filelist_path = f"{cur_scene_folder}/frames_fly.txt"
                # for none keyframes, currently there is not pesudo disp
                raise NotImplementedError
            with open(filelist_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    img1_filenames.append(f"{cur_scene_folder}/{lfolder}/{line}")
                    img2_filenames.append(f"{cur_scene_folder}/{rfolder}/{line}")
                    disp_name = line.replace(self.img1_suffix, self.disp_suffix)
                    disp_filenames.append(f"{cur_scene_folder}/{gt_folder}/{disp_name}")
                    seg_name = line.replace(self.img1_suffix, self.seg_suffix)
                    seg_filenames.append(f"{cur_scene_folder}/{seg_gt_folder}/{seg_name}")
            if self.altitude in ['high','all']:
                with open(filelist_path_superhigh, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line == "":
                            continue
                        img1_filenames.append(f"{cur_scene_folder}/{lfolder}/{line}")
                        img2_filenames.append(f"{cur_scene_folder}/{rfolder}/{line}")
                        disp_filenames.append("/media/nfs/datasets/Tracking_OA_team/Insta360Depth/Dev3DatasetExtra/defom_uint16/zero.png")
                        seg_name = line.replace(self.img1_suffix, self.seg_suffix)
                        seg_filenames.append(f"{cur_scene_folder}/{seg_gt_folder}/{seg_name}") 
                if self.altitude == 'all':
                    with open(filelist_path_low, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line == "":
                                continue
                            img1_filenames.append(f"{cur_scene_folder}/{lfolder}/{line}")
                            img2_filenames.append(f"{cur_scene_folder}/{rfolder}/{line}")
                            disp_name = line.replace(self.img1_suffix, self.disp_suffix)
                            disp_filenames.append(f"{cur_scene_folder}/{gt_folder}/{disp_name}")
                            seg_name = line.replace(self.img1_suffix, self.seg_suffix)
                            seg_filenames.append(f"{cur_scene_folder}/{seg_gt_folder}/{seg_name}")

        self.img1_dir = img1s_dirs
        self.img2_dir = img2s_dirs
        self.disp_dir = disp_dirs
        self.seg_dirs = seg_dirs
        return img1_filenames, img2_filenames, disp_filenames, seg_filenames
