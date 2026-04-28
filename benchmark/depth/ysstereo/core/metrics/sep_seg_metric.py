from typing import Dict, List, Optional, Sequence
import numpy as np

import mmengine
import mmcv
from mmengine.evaluator.metric import BaseMetric, _to_cpu
from mmengine.utils import mkdir_or_exist
from mmengine.dist import is_main_process
from mmengine.logging import MMLogger, print_log

from collections import OrderedDict
from prettytable import PrettyTable
from ysstereo.registry import METRICS
from ysstereo.structures import StereoDataSample
from .seg_metric import eval_metrics, intersect_and_union, pre_eval_to_metrics

@METRICS.register_module()
class SepSegMetric(BaseMetric):
    """Sep evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    CLASSES = None
    PALETTE = None

    def __init__(self,
                ignore_index: int = 255,
                iou_metrics: str = 'mIoU',
                classes: Optional[List[str]] = ['Sky'],
                palette: Optional[List[List[int]]] = [[0, 0, 0]],
                nan_to_num: Optional[int] = None,
                beta: int = 1,
                collect_device: str = 'cpu',
                output_dir: Optional[str] = None,
                format_only: bool = False,
                prefix: Optional[str] = None,
                **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

        self.metrics = [iou_metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(self.metrics).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(self.metrics))
    
    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmengine.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(class_names).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = 255
                else:
                    self.label_map[i] = class_names.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette
    
    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != 255:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                # Get random state before set seed, and restore
                # random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE

        return palette

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        if len(self.results) == 0:
            self.results.append({})

        img_metas:List[StereoDataSample] = data_batch['data_samples']
        for i in range(len(data_samples)):
            pred_seg = data_samples[i]['seg'].squeeze()
            img_meta = img_metas[i].metainfo
            gt_seg   = img_meta['seg_gt'].squeeze()
            gt_seg_valid = img_meta['seg_valid'].squeeze()
            dataset_name = img_meta['dataset_name']
            pred_seg = _to_cpu(pred_seg)
            gt_seg = _to_cpu(gt_seg)
            gt_seg_valid = _to_cpu(gt_seg_valid)
            if not (np.count_nonzero(gt_seg_valid) > 0):
                continue

            if dataset_name not in self.results[0]:
                self.results[0][dataset_name] = {}
                self.results[0][dataset_name]['pred_seg'] = []
                self.results[0][dataset_name]['gt_seg'] = []
            # gt_seg = mmcv.imresize(gt_seg, size=(pred_seg.shape[1], pred_seg.shape[0]),)
            gt_seg = mmcv.imresize(gt_seg, size=(pred_seg.shape[1], pred_seg.shape[0]), interpolation='nearest')

            self.results[0][dataset_name]['pred_seg'].append(pred_seg)
            self.results[0][dataset_name]['gt_seg'].append(gt_seg)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        eval_results = {}
        data_dict = results[0]
        num_classes = len(self.CLASSES)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        for dataset_name in data_dict:
            pred_seg = data_dict[dataset_name]['pred_seg']
            gt_seg   = data_dict[dataset_name]['gt_seg']
            ret_metrics = eval_metrics(pred_seg,
                                       gt_seg,
                                       num_classes,
                                       self.ignore_index,
                                       self.metrics,
                                       label_map=dict(),
                                       reduce_zero_label=False)

            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })

            # each class table
            ret_metrics.pop('aAcc', None)
            ret_metrics_class = OrderedDict({
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })
            ret_metrics_class.update({'Class': class_names})
            ret_metrics_class.move_to_end('Class', last=False)

            class_table_data = PrettyTable()
            class_table_data.title = 'Seg per class ' + dataset_name + "(" + f'{len(pred_seg)}' + ")"
            for key, val in ret_metrics_class.items():
                class_table_data.add_column(key, val)

            summary_table_data = PrettyTable()
            summary_table_data.title = 'Seg summary ' + dataset_name + "(" + f'{len(pred_seg)}' + ")"
            for key, val in ret_metrics_summary.items():
                if key == 'aAcc':
                    summary_table_data.add_column(key, [val])
                else:
                    summary_table_data.add_column('m' + key, [val])

            print_log('\n' + class_table_data.get_string(), logger=logger)
            print_log('\n' + summary_table_data.get_string(), logger=logger)

            # each metric dict
            for key, value in ret_metrics_summary.items():
                if key == 'aAcc':
                    eval_results[f'seg_metric_{dataset_name}/{key}'] = value / 100.0
                else:
                    eval_results[f'seg_metric_{dataset_name}/m' + key] = value / 100.0

            ret_metrics_class.pop('Class', None)
            for key, value in ret_metrics_class.items():
                eval_results.update({
                    f'seg_metric_{dataset_name}/{key}' + '.' + str(name): value[idx] / 100.0
                    for idx, name in enumerate(class_names)
                })

        return eval_results
        