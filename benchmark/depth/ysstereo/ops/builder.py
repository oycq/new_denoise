from ysstereo.registry import MODELS
from torch.nn import Module

OPERATORS = MODELS
def build_operators(cfg: dict) -> Module:
    """build opterator with config dict.

    Args:
        cfg (dict): The config dict of operator.

    Returns:
        Module: The built operator.
    """
    return OPERATORS.build(cfg)
