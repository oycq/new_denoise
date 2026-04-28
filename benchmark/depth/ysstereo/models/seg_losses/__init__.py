from .vbce_loss import VideoBootstrappedCELoss
from .l1_loss import VideoL1Loss
from .l2_loss import VideoL2Loss
from .smoothl1_loss import VideoSmoothL1Loss
from .irt_l1_loss import VideoIrtL1Loss
from .irt_l2_loss import VideoIrtL2Loss
from .time_consistent_loss import VideoTimeConsistentLoss

__all__ = [
    'VideoBootstrappedCELoss', 'VideoL1Loss', 'VideoL2Loss',
    'VideoSmoothL1Loss', 'VideoIrtL1Loss', 'VideoTimeConsistentLoss'
]
