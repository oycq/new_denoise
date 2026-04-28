from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Optional, Union

import torch
import torch.distributed as dist
from mmengine.model import BaseModel
from ysstereo.registry import MODELS
from mmengine.logging import print_log
from mmengine.model import ImgDataPreprocessor
from mmengine.model.wrappers.distributed import (
    MODEL_WRAPPERS, MMDistributedDataParallel,
    detect_anomalous_params,
)

from ysstereo.utils.logger import get_root_logger

@MODELS.register_module()
class StereoEstimator(BaseModel, metaclass=ABCMeta):
    """Base class for stereo estimator.

    Args:
        freeze_net (bool): Whether freeze the weights of model. If set True,
            the model will not update the weights.
        init_cfg (dict, list, optional): Config dict of weights initialization.
            Defaults to None.
    """

    def __init__(self,
                 sr: float = 0.0,
                 freeze_net: bool = False,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[list, dict]] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.sr = sr # default prune params
        self.status = dict(iter=0, max_iters=0)
        self.freeze_net = freeze_net
        # if set freeze_net True, the weights in this model
        # will be not updated and predict the disparity maps.
        if self.freeze_net:
            logger = get_root_logger()
            print_log(
                f'Freeze the parameters in {self.__class__.__name__}',
                logger=logger)
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @abstractmethod
    def forward_train(self, *args, **kwargs):
        """Placeholder for forward function of stereo estimator when training."""
        pass

    @abstractmethod
    def forward_test(self, *args, **kwargs):
        """Placeholder for forward function of stereo estimator when testing."""
        pass

    @abstractmethod
    def forward_onnx(self, *args, **kwargs):
        """Placeholder for forward function of stereo estimator when converting onnx."""
        pass

    def forward(self, inputs, data_samples, mode='test'):
        if mode == 'train':
            return self.forward_train(imgs=inputs, **data_samples)
        elif mode == 'test' or 'predict':
            return self.forward_test(imgs=inputs, **data_samples)

    def train_step(self, data, optim_wrapper):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self(**data, mode='train')
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, num_samples=data['inputs'].shape[0])
        
        if self.sr <= 0.0:
            optim_wrapper.update_params(loss)
        else:
            sum_bn, _ = optim_wrapper.update_params_prune(self, loss, self.sr, self.sr_coe, self.resrep_mask)
            log_vars['sum_bn'] = sum_bn

        outputs.update(log_vars)

        return outputs

    def val_step(self, data):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        data = self.data_preprocessor(data, False)
        outputs = self(**data, mode='test')

        return outputs

    def test_step(self, data):
        """The test step.

        This method shares the same signature as :func:`train_step`, but used
        during test epochs.
        """
        data = self.data_preprocessor(data, False)
        outputs = self(**data, mode='test')

        return outputs

    def predict(self, data):
        """The predict.

        This method is used during prediction
        """
        data = self.data_preprocessor(data, False, True)
        outputs = self(**data, mode='predict')

        return outputs

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        new_log_vars = OrderedDict()
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            prefix = 'loss/' if 'loss' in loss_name else 'train_logs/'
            new_log_vars[prefix+loss_name] = loss_value.item()

        return loss, new_log_vars

@MODEL_WRAPPERS.register_module()
class StereoDistributedDataParallel(MMDistributedDataParallel):
    def __init__(self,
                 module,
                 detect_anomalous_params: bool = False,
                 **kwargs):
        super().__init__(module=module, **kwargs)
        # only support stereoestimtor
        assert isinstance(module, StereoEstimator)
        self.detect_anomalous_params = detect_anomalous_params

    def train_step(self, data, optim_wrapper):
        with optim_wrapper.optim_context(self):
            data = self.module.data_preprocessor(data, True)
            losses = self(**data, mode='train')
        loss, log_vars = self.module._parse_losses(losses)

        outputs = dict(
            loss=loss, num_samples=data['inputs'].shape[0])
        if self.module.sr <= 0.0:
            optim_wrapper.update_params(loss)
        else:
            sum_bn, _ = optim_wrapper.update_params_prune(self, loss, self.module.sr, self.module.sr_coe, self.module.resrep_mask)
            log_vars['sum_bn'] = sum_bn
        #print(self.module.sr)
        outputs.update(log_vars)
        #optim_wrapper.update_params(loss)
        if self.detect_anomalous_params:
            detect_anomalous_params(loss, model=self)
        return outputs

    def val_step(self, data):
        return self.module.val_step(data)