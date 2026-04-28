import torch
import torch.nn as nn
from mmengine.optim import OptimWrapper
from mmengine.registry import OPTIM_WRAPPERS
from typing import Dict, Optional

@OPTIM_WRAPPERS.register_module()
class PruneOptimWrapper(OptimWrapper):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer=optimizer, **kwargs)

    def update_params_prune(  # type: ignore
            self,
            model,
            loss: torch.Tensor,
            sr: float,
            sr_coe: float,
            resrep_mask: list,
            step_kwargs: Optional[Dict] = None,
            zero_kwargs: Optional[Dict] = None) -> None:
        """Update parameters in :attr:`optimizer`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
            step_kwargs (dict): Arguments for optimizer.step.
                Defaults to None.
                New in version v0.4.0.
            zero_kwargs (dict): Arguments for optimizer.zero_grad.
                Defaults to None.
                New in version v0.4.0.
        """
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        loss = super().scale_loss(loss)
        super().backward(loss)
        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`

        #bn_list = [0,3,6,9,12,15,18,21,24,27]
        #coe_list = [pow(sr_coe,2), sr_coe, 1.0, 1/sr_coe, 1/sr_coe, 1/sr_coe, 1/sr_coe, 1/pow(sr_coe,2), 1/pow(sr_coe,2), 1/pow(sr_coe,2)]
        bn_list = [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45]
        coe_list = [pow(sr_coe,2), sr_coe, 1.0, 1.0, 1.0, 1.0, 1/sr_coe, 1/sr_coe, 1/sr_coe, 1/sr_coe, 1/sr_coe, 1/sr_coe, 1/pow(sr_coe,2), 1/pow(sr_coe,2), 1/pow(sr_coe,2), 1/pow(sr_coe,2)]
        sum_bn = 0
        index = 0
        bn_index = 0
        weight_bn = torch.Tensor([]).cuda()
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                if index in bn_list:
                    if resrep_mask:
                        m.weight.grad.data.add_(sr * coe_list[bn_index] *  torch.sign(m.weight.data) * resrep_mask[bn_index])  # L1
                    else:
                        m.weight.grad.data.add_(sr * coe_list[bn_index] *  torch.sign(m.weight.data))  # L1
                    sum_bn += m.weight.data.sum()
                    weight_bn = torch.cat((weight_bn, m.weight.data),0)
                    bn_index += 1
                index += 1

        if super().should_update():
            super().step(**step_kwargs)
            super().zero_grad(**zero_kwargs)
            
        return sum_bn, weight_bn
