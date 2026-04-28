from ysstereo.registry import MODELS
from torch.nn import Module

ENCODERS = MODELS
DECODERS = MODELS
STEREO_ESTIMATORS = MODELS
LOSSES = MODELS
COMPONENTS = MODELS
ATTENTIONS = MODELS

def build_encoder(cfg: dict) -> Module:
    """Build encoder for stereo estimator.

    Args:
        cfg (dict): Config for encoder.

    Returns:
        Module: Encoder module.
    """
    return ENCODERS.build(cfg)


def build_decoder(cfg: dict) -> Module:
    """Build decoder for stereo estimator.

    Args:
        cfg (dict): Config for decoder.

    Returns:
        Module: Decoder module.
    """
    return DECODERS.build(cfg)


def build_components(cfg: dict) -> Module:
    """Build encoder for model component.

    Args:
        cfg (dict): Config for component of model.

    Returns:
        Module: Component of model.
    """
    return COMPONENTS.build(cfg)


def build_loss(cfg: dict) -> Module:
    """Build loss function.

    Args:
        cfg (dict): Config for loss function.

    Returns:
        Module: Loss function.
    """
    return LOSSES.build(cfg)


def build_stereo_estimator(cfg: dict) -> Module:
    """Build stereo estimator.

    Args:
        cfg (dict): Config for stereo estimator.

    Returns:
        Module: stereo estimator.
    """
    return STEREO_ESTIMATORS.build(cfg)


def build_attentions(cfg: dict) -> Module:
    """Build encoder for model attention.

    Args:
        cfg (dict): Config for attention of model.

    Returns:
        Module: Attention of model.
    """
    return ATTENTIONS.build(cfg)