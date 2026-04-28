from collections import defaultdict
from typing import Dict, List, Optional, Sequence
import numpy as np

from mmengine.dist import is_main_process
from mmengine.evaluator.metric import BaseMetric, _to_cpu
from mmengine.logging import MMLogger, print_log
from mmengine.structures import BaseDataElement
from mmengine.utils import mkdir_or_exist
from prettytable import PrettyTable
from ysstereo.registry import METRICS
from ysstereo.structures import StereoDataSample

@METRICS.register_module()
class SepDispMetric(BaseMetric):
    """Disparity estimation evaluation metric.

    Args:
        disp_metrics (List[str], optional): List of metrics to compute. If
            not specified, defaults to all metrics in self.METRICS.
        min_disp_eval (float): Minimum disparity value for evaluation.
            Defaults to 0.0.
        max_disp_eval (float): Maximum disparity value for evaluation.
            Defaults to infinity.
        crop_type (str, optional): Specifies the type of cropping to be used
            during evaluation. This option can affect how the evaluation mask
            is generated. Currently, 'nyu_crop' is supported, but other
            types can be added in future. Defaults to None if no cropping
            should be applied.
        disparity_scale_factor (float): Factor to scale the disparity values.
            Defaults to 1.0.
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
    METRICS = ('EPE', '1PX', '2PX', '3PX', '5PX')

    def __init__(self,
                 disp_metrics: Optional[List[str]] = None,
                 min_disp_eval: float = 0.0,
                 max_disp_eval: float = float('inf'),
                 crop_type: Optional[str] = None,
                 disp_scale_factor: float = 1.0,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        assert prefix is None, 'prefix can not be defined in this metric'
        super().__init__(collect_device=collect_device, prefix=prefix)

        if disp_metrics is None:
            self.metrics = self.METRICS
        elif isinstance(disp_metrics, (tuple, list)):
            for metric in disp_metrics:
                assert metric in self.METRICS, f'the metric {metric} is not ' \
                    f'supported. Please use metrics in {self.METRICS}'
            self.metrics = disp_metrics

        # Validate crop_type, if provided
        assert crop_type in [
            None, 'nyu_crop'
        ], (f'Invalid value for crop_type: {crop_type}. Supported values are '
            'None or \'nyu_crop\'.')
        self.crop_type = crop_type
        self.min_disp_eval = min_disp_eval
        self.max_disp_eval = max_disp_eval
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        self.disp_scale_factor = disp_scale_factor

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
            pred_disp = data_samples[i]['disp'].squeeze()
            img_meta = img_metas[i].metainfo
            gt_disp = img_meta['disp_gt'].squeeze()
            gt_valid = img_meta['valid'].squeeze()
            dataset_name = img_meta['dataset_name']
            gt_valid = _to_cpu(gt_valid)
            if not (np.count_nonzero(gt_valid) > 0):
                continue
            if dataset_name not in self.results[0]:
                self.results[0][dataset_name] = {k:[] for k in self.metrics}
            pred_disp = _to_cpu(pred_disp)
            gt_disp = _to_cpu(gt_disp)
            ret_metrics = self.eval_metrics([pred_disp], [gt_disp], [gt_valid])
            for k in self.metrics:
                self.results[0][dataset_name][k].append(ret_metrics[k])

    def end_point_error_map(self,
                            disp_pred: np.ndarray,
                            disp_gt: np.ndarray) -> np.ndarray:
        """Calculate end point error map.

        Args:
            disp_pred (ndarray): The predicted disparity with the
                shape (H, W).
            disp_gt (ndarray): The ground truth of disparity with the shape
                (H, W).

        Returns:
            ndarray: End point error map with the shape (H , W).
        """
        return np.sqrt((disp_pred - disp_gt)**2)

    def end_point_error(self,
                        disp_pred: Sequence[np.ndarray],
                        disp_gt: Sequence[np.ndarray],
                        valid_gt: Sequence[np.ndarray]) -> float:
        """Calculate end point errors between prediction and ground truth.

        Args:
            disp_pred (list): output list of disp map from disp_estimator
                shape(H, W, 2).
            disp_gt (list): ground truth list of disp map shape(H, W, 2).
            valid_gt (list): the list of valid mask for ground truth with the
                shape (H, W).

        Returns:
            float: end point error for output.
        """
        epe_list = []
        # assert len(disp_pred) == len(disp_gt)
        for _disp_pred, _disp_gt, _valid_gt in zip(disp_pred, disp_gt, valid_gt):
            _disp_pred = _disp_pred.squeeze()
            epe_map = self.end_point_error_map(_disp_pred, _disp_gt)
            val = _valid_gt.reshape(-1) >= 0.5
            epe_list.append(epe_map.reshape(-1)[val])

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)

        return epe

    def npx_error(self,
                err_px: int,
                disp_pred: Sequence[np.ndarray],
                disp_gt: Sequence[np.ndarray],
                valid_gt: Sequence[np.ndarray]) -> float:
        """Calculate percentage of error end points number between prediction and ground truth.

        Args:
            err_px (int): number of pixel error
            disp_pred (list): output list of disp map from disp_estimator
                shape(H, W, 2).
            disp_gt (list): ground truth list of disp map shape(H, W, 2).
            valid_gt (list): the list of valid mask for ground truth with the
                shape (H, W).

        Returns:
            float: percentage of error end points number for output.
        """
        err_npx_list = []
        num_valid = 0
        assert len(disp_pred) == len(disp_gt)
        for _disp_pred, _disp_gt, _valid_gt in zip(disp_pred, disp_gt, valid_gt):
            _disp_pred = _disp_pred.squeeze()
            epe_map = self.end_point_error_map(_disp_pred, _disp_gt)
            val = _valid_gt.reshape(-1) >= 0.5
            out = epe_map > err_px
            num_valid += np.size(_disp_gt.reshape(-1)[val])
            err_npx_list.append(out.reshape(-1)[val])

        err_npx_all = np.concatenate(err_npx_list)
        err_npx = np.sum(err_npx_all).item() / max(num_valid,1)  * 100

        return err_npx

    def eval_metrics(self,
            disp_preds: Sequence[np.ndarray],
            disp_gt: Sequence[np.ndarray],
            valid_gt: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate evaluation metrics.

        Args:
            disp_preds (list): list of predicted disp maps.
            disp_gt (list): list of ground truth disp maps
            metrics (list, str): metrics to be evaluated.
                Defaults to ['EPE'], end-point error.

        Returns:
            dict: metrics and their values.
        """
        # if isinstance(metrics, str):
        #     metrics = [metrics]
        # allowed_metrics = ['EPE', '1PX', '2PX', '3PX', '5PX']
        # if not set(metrics).issubset(set(allowed_metrics)):
        #     raise KeyError('metrics {} is not supported'.format(metrics))
        # ret_metrics = dict()
        # if 'EPE' in metrics:
        #     ret_metrics['EPE'] = self.end_point_error(disp_preds, disp_gt, valid_gt)
        # if '1PX' in metrics:
        #     ret_metrics['1PX'] = self.npx_error(1, disp_preds, disp_gt, valid_gt)
        # if '2PX' in metrics:
        #     ret_metrics['2PX'] = self.npx_error(2, disp_preds, disp_gt, valid_gt)
        # if '3PX' in metrics:
        #     ret_metrics['3PX'] = self.npx_error(3, disp_preds, disp_gt, valid_gt)
        # if '5PX' in metrics:
        #     ret_metrics['5PX'] = self.npx_error(5, disp_preds, disp_gt, valid_gt)

        ret_metrics = dict()
        if 'EPE' in self.metrics:
            ret_metrics['EPE'] = self.end_point_error(disp_preds, disp_gt, valid_gt)
        if '1PX' in self.metrics:
            ret_metrics['1PX'] = self.npx_error(1, disp_preds, disp_gt, valid_gt)
        if '2PX' in self.metrics:
            ret_metrics['2PX'] = self.npx_error(2, disp_preds, disp_gt, valid_gt)
        if '3PX' in self.metrics:
            ret_metrics['3PX'] = self.npx_error(3, disp_preds, disp_gt, valid_gt)
        if '5PX' in self.metrics:
            ret_metrics['5PX'] = self.npx_error(5, disp_preds, disp_gt, valid_gt)

        return ret_metrics

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The keys
                are identical with self.metrics.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        metric_dict = results[0]
        out_metrics = {}
        table_data = PrettyTable(field_names=['EvalSet']+list(self.metrics))
        table_data.title = 'Disp Eval Metrics'
        for dataset_name in metric_dict:
            row_data = [dataset_name]
            for k in self.metrics:
                v = metric_dict[dataset_name].get(k, None)
                if v is None:
                    row_data.append('NONE')
                else:
                    row_data.append(np.mean(v))
                    out_metrics[f'disp_metric_{dataset_name}/{k}'] = row_data[-1]
            table_data.add_row(row_data)

        print_log('results:', logger)
        print_log('\n' + table_data.get_string(), logger=logger)

        return out_metrics
