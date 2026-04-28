from mmengine.registry import RUNNERS
from mmengine.runner import Runner, IterBasedTrainLoop
from mmengine.runner.loops import _InfiniteDataloaderIterator, print_log, BaseLoop
from torch.utils.data import DataLoader
import logging, copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class MixedIterTrainLoop(IterBasedTrainLoop):
    def __init__(
        self,
        runner,
        dataloader_A: Union[DataLoader, Dict],
        dataloader_B: Union[DataLoader, Dict],
        max_iters: int,
        val_begin: int = 1,
        val_interval: int = 1000,
        dynamic_intervals: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__(
            runner=runner,
            dataloader=dataloader_A,
            max_iters=max_iters,
            val_begin=val_begin,
            val_interval=val_interval,
            dynamic_intervals=dynamic_intervals,
        )
        if isinstance(dataloader_B, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.dataloader_B = runner.build_dataloader(
                dataloader_B, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader_B = dataloader_B
        self.dataloader_B_iterator = _InfiniteDataloaderIterator(self.dataloader_B)
    
    def run(self) -> None:
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        if self._iter > 0:
            print_log(
                f'Advance dataloader {self._iter} steps to skip data '
                'that has already been trained',
                logger='current',
                level=logging.WARNING)
            self.dataloader_iterator.skip_iter(self._iter)
            self.dataloader_B_iterator.skip_iter(self._iter)
        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
            data_batch_B = next(self.dataloader_B_iterator)
            data_batch["data_B"] = data_batch_B
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and (self._iter % self.val_interval == 0
                         or self._iter == self._max_iters)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model

class MixedRunner(Runner):
    def set_dataloader_B(self, train_dataloader_B):
        self._train_dataloader_B = train_dataloader_B

    def build_train_loop(self, loop):
        if isinstance(loop, MixedIterTrainLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'train_loop should be a Loop object or dict, but got {loop}')
        loop_cfg = copy.deepcopy(loop)
        loop_type = loop_cfg.get("type", None)
        assert loop_type == "MixedIterTrainLoop"
        loop_cfg.pop("type")
        loop = MixedIterTrainLoop(
            **loop_cfg, runner=self, dataloader_A = self._train_dataloader,
            dataloader_B = self._train_dataloader_B,
        )
        return loop