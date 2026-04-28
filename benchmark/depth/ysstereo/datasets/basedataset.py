import copy, hashlib, os
import os.path as osp
import pickle as pkl
from abc import abstractmethod
from collections.abc import Mapping
from mmengine.config import Config
from mmengine.logging import MMLogger, print_log
from typing import Callable, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import mmengine
import mmengine.fileio as fileio
from mmengine.dataset import BaseDataset, Compose, force_full_init
from prettytable import PrettyTable

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from ysstereo.registry import DATASETS

FILELIST_CACHE_DIR = "./.yss_cache/"

SEG_CLASSES = {'background': 0, 'water': 1, 'sky' : 2}
DATA_STATISTIC = {'total_data_count': 0, 'valid_disp_count' : 0,
                  'valid_seg_count': 0,  'background': 0, 'water': 0, 'sky' : 0}

def compute_md5(dir:str, suffix:str):
    md5_hash = hashlib.md5((dir+"."+suffix).encode()).hexdigest()
    return md5_hash

@DATASETS.register_module()
class BaseStereoDataset(BaseDataset):
    """Custom dataset for stereo disparity estimation.
    Args:
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        downsample_rate: downsample times of dataset
        use_filelist_cache: use filelist cache instead of mmcv scandir
    """
    METAINFO: dict = dict()
    UNIQUE_NAME_KEY = 'unique_name'
    def __init__(self,
                 metainfo: Union[Mapping, Config, None] = None,
                 data_root: Optional[str] = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 use_filelist_cache: bool = True,
                 downsample_rate: int = 1,
                 max_refetch: int = 1000,
                 statistical_data: bool = True) -> None:
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        self.data_root = data_root
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self._indices = indices
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray
        self.downsample_rate = downsample_rate
        self.use_filelist_cache = use_filelist_cache
        self.statistical_data = statistical_data
        os.makedirs(FILELIST_CACHE_DIR, exist_ok=True)
        # Build pipeline.
        self.pipeline = Compose(pipeline)
        self.dataset_name = self.__class__.__name__
        """
        data_infos is the list of data_info containing img_info and ann_info
        data_info
          - img_info
              - filename1
              - filename2
          - ann_info
              - filename_key
        key might be disp, disp_fw, disp_bw, occ, occ_fw, occ_bw, valid
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.data_infos = []
        unique_name = getattr(self, self.UNIQUE_NAME_KEY, None)
        if self.use_filelist_cache and  unique_name is not None and isinstance(unique_name, str):
            split_name = 'train' if not test_mode else 'test'
            if len(unique_name)>0:
                unique_name = f"_{unique_name}_"
            hash_str = hashlib.md5((self.dataset_name+data_root+unique_name).encode()).hexdigest()[:8]
            cache_name = self.dataset_name+f"_{split_name}_"+unique_name+hash_str+".pkl"
            cache_path = os.path.join(FILELIST_CACHE_DIR, cache_name)
            print_log("catche_path: " + cache_path, logger)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    file_data = pkl.load(f)
                    self.data_infos = file_data['data_info']
                    self.img1_dir = file_data['img1_dir']
                    self.img2_dir = file_data['img2_dir']
            else:
                self.load_data_info()
                with open(cache_path, 'wb') as f:
                    pkl.dump(dict(data_info=self.data_infos, img1_dir=self.img1_dir, img2_dir=self.img2_dir), f)
        else:
            self.load_data_info()

        if self.statistical_data:
            global DATA_STATISTIC
            num_worker = 16
            chunk_size = int(len(self.data_infos) / num_worker)
            tasks = [self.data_infos[i: i+chunk_size] for i in range(0, len(self.data_infos), chunk_size)]

            with Pool(num_worker) as pool:
                statistic_resuls = list(tqdm(pool.imap_unordered(self.data_statistic, tasks), total=len(tasks)))
                for result in statistic_resuls:
                    for key, val in result.items():
                        DATA_STATISTIC[key] += result[key]

            class_table_data = PrettyTable()
            for key, val in DATA_STATISTIC.items():
                class_table_data.add_column(key, [val])
            print_log(self.dataset_name)
            print_log('\n' + class_table_data.get_string(), logger = logger)

    def data_statistic(self, params):
        global DATA_STATISTIC
        global SEG_CLASSES
        data_statistic = dict()
        for key, val in DATA_STATISTIC.items():
            data_statistic[key] = 0

        data_infos = params
        for data_results in data_infos:
            data_statistic['total_data_count'] += 1
            disp_data_path = data_results['ann_info']['filename_disp']
            seg_data_path = data_results['ann_info']['filename_seg']
            if os.path.exists(disp_data_path):
                data_statistic['valid_disp_count'] += 1
            if os.path.exists(seg_data_path):
                data_statistic['valid_seg_count'] += 1
                seg_data = cv2.imread(seg_data_path, cv2.IMREAD_UNCHANGED)
                for key, val in SEG_CLASSES.items():
                    if np.count_nonzero(seg_data == val) > 0:
                        data_statistic[key] += 1

        return data_statistic

    @abstractmethod
    def load_data_info(self):
        """Placeholder for load data information."""
        pass

    @force_full_init
    def __len__(self) -> int:
        """return downsampled dataset length
        """
        if self.downsample_rate > 0:
            return super().__len__()//self.downsample_rate
        else:
            return -super().__len__()*self.downsample_rate

    def __getitem__(self, idx: int) -> dict:
        """map downsampled idx to full dataset idx
        """
        if self.downsample_rate > 0:
            return super().__getitem__(idx * self.downsample_rate)
        else:
            roll_idx = idx % super().__len__()
            return super().__getitem__(roll_idx)

    def load_data_list(self) -> List[dict]:
        data_list = self.data_infos
        return data_list

    def pre_pipeline(self, results: dict) -> None:
        """Prepare results dict for pipeline."""

        results['img_fields'] = ['imgl', 'imgr']
        results['ann_fields'] = []
        results['img1_dir'] = self.img1_dir
        results['img2_dir'] = self.img2_dir

    def prepare_data(self, idx: int) -> dict:
        """Get data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        results = copy.deepcopy(self.data_list[idx])
        self.pre_pipeline(results)
        return self.pipeline(results)

    @staticmethod
    def load_img_info(data_infos: Sequence[dict], img1_filename: Sequence[str],
                      img2_filename: Sequence[str]) -> None:
        """Load information of images.

        Args:
            data_infos (list): data information.
            img1_filename (list): ordered list of abstract file path of img1.
            img2_filename (list): ordered list of abstract file path of img2.
        """

        num_file = len(img1_filename)
        for i in range(num_file):
            data_info = dict(
                img_info=dict(
                    filename1=img1_filename[i], filename2=img2_filename[i]),
                ann_info=dict())
            data_infos.append(data_info)

    @staticmethod
    def load_ann_info(data_infos: Sequence[dict], filename: Sequence[str],
                      filename_key: str) -> None:
        """Load information of annotation.

        Args:
            data_infos (list): data information.
            filename (list): ordered list of abstract file path of annotation.
            filename_key (str): the annotation key e.g. 'disp'.
        """
        assert len(filename) == len(data_infos)
        num_files = len(filename)
        for i in range(num_files):
            data_infos[i]['ann_info'][filename_key] = filename[i]

    def get_data_filename(
            self, data_dirs: Union[Sequence[str], str],
            suffix: Optional[str] = None,
            exclude: Optional[Sequence[str]] = None) -> Sequence[str]:
        """Get file name from data directory.

        Args:
            data_dirs (list, str): the directory of data
            suffix (str, optional): the suffix for data file. Defaults to None.
            exclude (list, optional): list of files will be excluded.

        Returns:
            list: the list of data file.
        """

        if data_dirs is None:
            return None
        data_dirs = data_dirs \
            if isinstance(data_dirs, (list, tuple)) else [data_dirs]

        suffix = '' if suffix is None else suffix

        if exclude is None:
            exclude = []
        else:
            assert isinstance(exclude, (list, tuple))

        files = []
        for data_dir in data_dirs:
            for f in mmengine.scandir(data_dir, suffix=suffix):
                if f not in exclude:
                    files.append(osp.join(data_dir, f))
        files.sort()
        return files
