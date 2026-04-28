import os.path as osp
from glob import glob

from ysstereo.datasets.basedataset import BaseStereoDataset
from ysstereo.registry import DATASETS

@DATASETS.register_module()
class ETH3D(BaseStereoDataset):
    """ETH3D stereo dataset."""

    def __init__(self, *args, **kwargs) -> None:
        setattr(self, self.UNIQUE_NAME_KEY, f'')
        super().__init__(*args, **kwargs)

    def _get_data_dir(self) -> None:
        """Get the paths for images and disparity."""
        # only provide ground truth for training
        #self.subset_dir = 'training'

        #self.data_root = osp.join(self.data_root, self.subset_dir)
        # In KITTI 2015, data in `image_2` is left image data
        self.img1_dir = glob(osp.join(self.data_root, 'two_view_training/*/'))
        self.img2_dir = glob(osp.join(self.data_root, 'two_view_training/*/'))
        self.disp_dir = glob(osp.join(self.data_root, 'two_view_training_gt/*/'))

        self.img1_suffix = 'im0.png'
        self.img2_suffix = 'im1.png'
        self.disp_suffix = 'disp0GT.pfm'

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        disparity."""
        self._get_data_dir()
        img1_filenames = self.get_data_filename(self.img1_dir,
                                                self.img1_suffix)
        img2_filenames = self.get_data_filename(self.img2_dir,
                                                self.img2_suffix)
        disp_filenames = self.get_data_filename(self.disp_dir,
                                                self.disp_suffix)

        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)
        self.load_ann_info(self.data_infos, disp_filenames, 'filename_disp')
