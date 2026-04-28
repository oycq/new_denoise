from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from mmengine.dist import is_main_process
from mmengine.evaluator.metric import BaseMetric, _to_cpu
from mmengine.logging import MMLogger, print_log
from mmengine.structures import BaseDataElement
from mmengine.utils import mkdir_or_exist
from prettytable import PrettyTable
from ysstereo.registry import METRICS
from ysstereo.structures import StereoDataSample
from .sep_disp_metric import SepDispMetric
from .sep_seg_metric import SepSegMetric

@METRICS.register_module()
class DispSegJointMetric(BaseMetric):
    """Disp Seg joint evaluation metric.
    """

    def __init__(self,
                 disp_metrics: Optional[List[str]] = None,
                 min_disp_eval: float = 0.0,
                 max_disp_eval: float = float('inf'),
                 disp_scale_factor: float = 1.0,
                 iou_metrics: str = 'mIOU',
                 classes: Optional[List[str]] = None,
                 palette: Optional[List[List[int]]] = None,
                 ignore_index: int = 255,
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 rop_type: Optional[str] = None,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        assert prefix is None, 'prefix can not be defined in this metric'
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.disp_metrics = SepDispMetric(disp_metrics,
                                          min_disp_eval,
                                          max_disp_eval,
                                          rop_type,
                                          disp_scale_factor,
                                          collect_device,
                                          output_dir,
                                          format_only,
                                          prefix,
                                          **kwargs)
        self.seg_metric = SepSegMetric(ignore_index,
                                       iou_metrics,
                                       classes,
                                       palette,
                                       nan_to_num,
                                       beta,
                                       collect_device,
                                       output_dir,
                                       format_only,
                                       prefix,
                                       **kwargs)
        
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
            self.results.append({})

        self.disp_metrics.process(data_batch, data_samples)
        self.seg_metric.process(data_batch, data_samples)
        self.results[0]['disp'] = self.disp_metrics.results
        self.results[1]['seg'] = self.seg_metric.results

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
        out_metrics = {}
        disp_out_metrics = self.disp_metrics.compute_metrics(results[0]['disp'])
        seg_out_metrics  = self.seg_metric.compute_metrics(results[1]['seg'])

        for disp_out_metric in disp_out_metrics:
            out_metrics[disp_out_metric] = disp_out_metrics[disp_out_metric]
        for seg_out_metric in seg_out_metrics:
            out_metrics[seg_out_metric] = seg_out_metrics[seg_out_metric]

        self.disp_metrics.results.clear()
        self.seg_metric.results.clear()

        return out_metrics