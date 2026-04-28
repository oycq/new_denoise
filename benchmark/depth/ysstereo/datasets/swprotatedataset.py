import os.path as osp
from glob import glob

from ysstereo.datasets.basedataset import BaseStereoDataset
from ysstereo.registry import DATASETS

@DATASETS.register_module()
class SwpRotateDataset(BaseStereoDataset):
    """RealScene dataset for test. 

    Args:
    """
    def __init__(self,
                 *args,
                 img_data_dirs: list = None,
                 ann_dirs: list = None,
                 unique_name: str = '',
                 **kwargs) -> None:
        self.img_data_dirs = img_data_dirs
        self.ann_dirs = ann_dirs
        self.unique_name = unique_name
        setattr(self, self.UNIQUE_NAME_KEY, self.unique_name)
        super().__init__(*args, **kwargs)

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2,
        disparity and segmentation."""
        img1_filenames, img2_filenames, disp_filenames, seg_filenames = self._get_data_dir()
        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)
        self.load_ann_info(self.data_infos, disp_filenames, 'filename_disp')
        self.load_ann_info(self.data_infos, seg_filenames, 'filename_seg')

    def _get_data_dir(self) -> None:
        """Get the paths for stereo images, disparity and segmentation."""

        self.disp_suffix = '.pfm'
        self.img1_suffix = '.jpg'
        self.img2_suffix = '.jpg'
        self.seg_suffix  = '.png'

        img_left_dirs_name = 'img_dir'
        img_right_dirs_name = 'img_dir'
        seg_dirs_name  = 'anno_data'
        disp_dirs_name = 'pseudo_gt'

        img1_filenames = []
        img2_filenames = []
        disp_filenames = []
        seg_filenames  = []

        img1s_dirs = []
        img2s_dirs = []
        disp_dirs = []
        seg_dirs  = []

        for img_dir in self.img_data_dirs:
            img1s_dirs.append(osp.join(self.data_root, img_dir))
        for seg_dir in self.ann_dirs:
            seg_dirs.append(osp.join(self.data_root, seg_dir))

        assert len(img1s_dirs) == len(seg_dirs), "length of img_dirs is not equal to seg_dirs"

        for idx in range(len(img1s_dirs)):
            idir1 = img1s_dirs[idx]
            sdir  = seg_dirs[idx]

            img1_filename = sorted(self.get_data_filename(idir1, self.img1_suffix))
            seg_filename  = sorted(self.get_data_filename(sdir, self.seg_suffix))

            if len(img1_filename) != len(seg_filename):
                img1_filename_tmp = img1_filename
                seg_filename = []
                img1_filename = []

                for img1_path in img1_filename_tmp:
                    seg_expected_path = img1_path.replace(idir1, sdir)
                    seg_expected_path = seg_expected_path.replace(self.img1_suffix, self.seg_suffix)
                    if osp.exists(seg_expected_path):
                        img1_filename.append(img1_path)
                        seg_filename.append(seg_expected_path)

            img1_filenames += img1_filename
            seg_filenames  += seg_filename

            assert len(img1_filenames) == len(seg_filenames), f"data length is not equal: {idir1} != {sdir}"

        for img_path in img1_filenames:
            img2_expected_path = img_path.replace(img_left_dirs_name, img_right_dirs_name)
            img2_expected_path = img2_expected_path.replace(self.img1_suffix, self.img2_suffix)
            img2_filenames.append(img2_expected_path)

        for img_path in seg_filenames:
            disp_expected_path = img_path.replace(seg_dirs_name, disp_dirs_name)
            disp_expected_path = disp_expected_path.replace(self.seg_suffix, self.disp_suffix)
            disp_filenames.append(disp_expected_path)

        self.img1_dir = img1s_dirs
        self.img2_dir = img2s_dirs
        self.disp_dir = disp_dirs
        self.seg_dir  = seg_dirs
        return img1_filenames, img2_filenames, disp_filenames, seg_filenames





