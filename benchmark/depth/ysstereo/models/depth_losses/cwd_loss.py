from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ysstereo.models.builder import LOSSES

class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n, c, h, w = featmap.shape
        featmap = featmap.reshape((n,c,-1))
        featmap = featmap.softmax(dim=-1)
        return featmap

@LOSSES.register_module()
class CWDLoss(nn.Module):
    """CWDLoss Loss for RAFT.
    
    Args:
        weight (float): The weight for the loss. 
    """

    def __init__(self, weight: float = 1.0, 
                 norm_type = 'channel', 
                 divergence = 'kl',
                 temperature=1.0) -> None:
        super().__init__()
        self.weight = weight
        self.norm_type = norm_type
        self.divergence = divergence

        if norm_type == 'channel':
            self.normalize = ChannelNorm()
            self.temperature = temperature
        elif norm_type =='spatial':
            self.normalize = nn.Softmax(dim=1)
            self.temperature = temperature
        else:
            self.normalize = None
            self.temperature = 1.0
        
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self,
                sources: Sequence[Tensor],
                targets: Sequence[Tensor]
                ) -> Tensor:
        """Forward function for CWDLoss.

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
            n, c, h, w = src.shape
            if self.normalize is not None:
                norm_s = self.normalize(src/self.temperature)
                norm_t = self.normalize(tgt.detach()/self.temperature)
            else:
                norm_s = src
                norm_t = tgt.detach()
            
            if self.divergence == 'kl':
                norm_s = norm_s.log()
            
            loss_i = self.criterion(norm_s,norm_t)

            if self.norm_type == 'channel':
                loss_i /= n * c
            elif self.norm_type =='spatial':
                loss_i /= n * h * w
            else:
                loss_i /= n * c * h * w
            loss += loss_i
        return self.weight * loss_i * (self.temperature**2)

