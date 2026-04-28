import copy
from typing import List, Optional, Union

import numpy as np
import torch
from mmengine.registry import init_default_scope
from mmengine import Config
from mmengine.runner import load_checkpoint

from mmengine.dataset import Compose
# from ysstereo.models import build_stereo_estimator
from ysstereo.registry import MODELS

def init_model(config: Union[str, Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[dict] = None) -> torch.nn.Module:
    """Initialize a stereo estimator from config file.

    Args:
        config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights. Default to: None.
        device (str): Represent the device. Default to: 'cuda:0'.
        cfg_options (dict, optional): Options to override some settings in the
            used config. Default to: None.
    Returns:
        nn.Module: The constructed stereo estimator.
    """

    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    config.model.train_cfg = None
    # model = build_stereo_estimator(config.model)
    init_default_scope(config.get('default_scope', 'ysstereo'))
    model = MODELS.build(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    return model


def inference_model(
        model: torch.nn.Module, imgls: Union[str, np.ndarray],
        imgrs: Union[str, np.ndarray]) -> Union[List[np.ndarray], np.ndarray]:
    """Inference images pairs with the stereo estimator.

    Args:
        model (nn.Module): The loaded stereo estimator.
        imgls (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.
        imgrs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.
    Returns:
        If img-pairs is a list or tuple, the same length list type results
        will be returned, otherwise return the stereo map from left image to right image
        directly.
    """
    if isinstance(imgls, (list, tuple)):
        is_batch = True
    else:
        imgls = [imgls]
        imgrs = [imgrs]
        is_batch = False
    cfg = model.cfg

    if isinstance(imgls[0], np.ndarray):
        # set loading pipeline type
        cfg.test_pipeline[0].type = 'LoadImageFromWebcam'

    # as load annotation is for online evaluation
    # there is no need to load annotation.
    #if dict(type='LoadAnnotations') in cfg.pipeline:
    #    cfg.pipeline.remove(dict(type='LoadAnnotations'))
    for i in range(len(cfg.test_pipeline)):
        if cfg.test_pipeline[i].type == 'LoadDispAnnotations':
            del cfg.test_pipeline[i]
            break
    
    for i in range(len(cfg.test_pipeline)):
        if cfg.test_pipeline[i].type == 'Fisheye2TransEqui':
            cfg.test_pipeline[i].test_mode=True
            break
    
    # remove old test crop in config for avoiding cropping the original shape of new test images
    cfg.test_pipeline = list(filter(lambda i: i['type'] != 'RandomCrop', cfg.test_pipeline))

    if 'disp_gt' in cfg.test_pipeline[-1]['meta_keys']:
        cfg.test_pipeline[-1]['meta_keys'].remove('disp_gt')
    if 'disp_fw_gt' in cfg.test_pipeline[-1]['meta_keys']:
        cfg.test_pipeline[-1]['meta_keys'].remove('disp_fw_gt')
    if 'disp_bw_gt' in cfg.test_pipeline[-1]['meta_keys']:
        cfg.test_pipeline[-1]['meta_keys'].remove('disp_bw_gt')
    if 'valid' in cfg.test_pipeline[-1]['meta_keys']:
        cfg.test_pipeline[-1]['meta_keys'].remove('valid')
    if 'distance' in cfg.test_pipeline[-1]['meta_keys']:
        cfg.test_pipeline[-1]['meta_keys'].remove('distance')

    test_pipeline = Compose(cfg.test_pipeline)
    for imgl, imgr in zip(imgls, imgrs):
        # prepare data
        if isinstance(imgl, np.ndarray) and isinstance(imgr, np.ndarray):
            # directly add img
            data = dict(imgl=imgl, imgr=imgr)
        else:
            # add information into dict
            data = dict(
                img_info=dict(filename1=imgl, filename2=imgr),
                img1_prefix=None,
                img2_prefix=None)
        data['img_fields'] = ['imgl', 'imgr']
        # build the data pipeline
        data = test_pipeline(data)

    # forward the model
    with torch.no_grad():
        results = model.predict(data)
        return results
