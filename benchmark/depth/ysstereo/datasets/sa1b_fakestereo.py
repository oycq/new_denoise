import os
from ysstereo.datasets.basedataset import BaseStereoDataset, compute_md5
from ysstereo.registry import DATASETS

@DATASETS.register_module()
class Sa1BFakeStereo(BaseStereoDataset):
    """Sa1B Fake Stereo dataset for training. 

    Args:
    """

    def __init__(self,
                 scenes_file: str,
                 *args,
                 **kwargs) -> None:
        self.splite_file = scenes_file
        sfile_md5 = compute_md5(scenes_file.split(".")[0], 'txt')
        setattr(self, self.UNIQUE_NAME_KEY, sfile_md5)
        with open(scenes_file, 'r') as f:
            scenes = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    scenes.append(line.strip())
        self.scenes = scenes
        super().__init__(*args, **kwargs)

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        disparity."""
        img1_filenames, img2_filenames, disp_filenames = self._get_data_dir()

        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)

        self.load_ann_info(self.data_infos, disp_filenames, 'filename_disp')

    def _get_data_dir(self) -> None:
        """Get the paths for stereo images and disparity."""

        self.disp_suffix = '.exr'
        self.img1_suffix = '.jpg'
        self.img2_suffix = '.jpg'

        img1s_dirs = []
        img2s_dirs = []
        disp_dirs = []

        img1_filenames = []
        img2_filenames = []
        disp_filenames = []

        for scene in self.scenes:
            rgb_folder = f"{self.data_root}/rgb/{scene}"
            depth_folder = f"{self.data_root}/dav2/{scene}"
            depth_files = os.listdir(depth_folder)
            img1s_dirs.append(rgb_folder)
            img2s_dirs.append(rgb_folder)
            disp_dirs.append(depth_folder)
            for depth_file in depth_files:
                img1_filenames.append(f"{rgb_folder}/{depth_file.replace('exr', 'jpg')}")
                img2_filenames.append(f"{rgb_folder}/{depth_file.replace('exr', 'jpg')}")
                disp_filenames.append(f"{depth_folder}/{depth_file}")

        self.img1_dir = img1s_dirs
        self.img2_dir = img2s_dirs
        self.disp_dir = disp_dirs

        return img1_filenames, img2_filenames, disp_filenames