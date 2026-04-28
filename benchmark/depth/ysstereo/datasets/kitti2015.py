import os.path as osp

from ysstereo.datasets.basedataset import BaseStereoDataset
from ysstereo.registry import DATASETS

@DATASETS.register_module()
class KITTI2015(BaseStereoDataset):
    """KITTI disp 2015 dataset."""

    def __init__(self, *args, **kwargs) -> None:
        setattr(self, self.UNIQUE_NAME_KEY, '')
        super().__init__(*args, **kwargs)

    def _get_data_dir(self) -> None:
        """Get the paths for images and disparity."""
        # only provide ground truth for training
        #self.subset_dir = 'training'

        #self.data_root = osp.join(self.data_root, self.subset_dir)
        # In KITTI 2015, data in `image_2` is left image data
        self.img1_dir = osp.join(self.data_root, 'image_2')
        self.img2_dir = osp.join(self.data_root, 'image_3')
        self.disp_dir = osp.join(self.data_root, 'disp_occ_0')

        self.img1_suffix = '_10.png'
        self.img2_suffix = '_10.png'
        self.disp_suffix = '_10.png'

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
