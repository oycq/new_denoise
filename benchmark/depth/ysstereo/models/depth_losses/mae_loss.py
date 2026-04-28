from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ysstereo.models.builder import LOSSES

@LOSSES.register_module()
class MAELoss(nn.Module):
    """MAELoss Loss for RAFT.
    
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
        """Forward function for MAELoss.

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
            loss += (src-tgt).abs().mean()

        return self.weight * loss

