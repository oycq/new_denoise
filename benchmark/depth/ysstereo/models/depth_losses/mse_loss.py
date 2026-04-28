from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ysstereo.models.builder import LOSSES

@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSELoss Loss for RAFT.
    
    Args:
        weight (float): The weight for the loss. 
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self,
                sources: Sequence[Tensor],
                targets: Sequence[Tensor]
                ) -> Tensor:
        """Forward function for MSELoss.

        Args:
            sources Sequence[Tensor]: The list of source tensors
            targets (Tensor): The list of target tensors

        Returns:
            Tensor: value of pixel-wise end point error loss.
        """

        if self.weight <= 0:
            return 0
        
        loss = 0 
        for src, tgt in zip(sources, targets):
            loss += torch.pow((src-tgt), 2).mean()
        return self.weight * loss

