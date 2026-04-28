from torch.utils.data.sampler import BatchSampler
from mmengine.dataset import DefaultSampler, ConcatDataset
from mmengine.registry import DATA_SAMPLERS

@DATA_SAMPLERS.register_module()
class MultiDataBatchSampler(BatchSampler):
    """batch sampler for multi-dataset evaluation via just one validate dataloader
    """
    def __init__(self, sampler:DefaultSampler, batch_size:int, drop_last:bool=False):
        super().__init__(sampler, batch_size, drop_last)
        assert isinstance(sampler, DefaultSampler), 'only default sampler is supported'
        assert isinstance(sampler.dataset, ConcatDataset) and not sampler.shuffle, 'dataset should be a non-shuffle ConcatDataset'
        assert drop_last is False, 'drop_last can only be False for MultiDataBatchSampler'
        self.sub_data_lengths = [len(d) for d in sampler.dataset.datasets]
        self.batch_nums = sum([(l-1)//self.batch_size + 1 for l in self.sub_data_lengths])
        self.total_samples = 0
        self.cur_sub_data_idx = 0

    def __iter__(self):
        self.total_samples = 0
        self.cur_sub_data_idx = 0
        batch = [0] * self.batch_size
        idx_in_batch = 0
        for idx in self.sampler:
            batch[idx_in_batch] = idx
            idx_in_batch += 1
            self.total_samples += 1
            if self.total_samples == self.sub_data_lengths[self.cur_sub_data_idx]:
                self.cur_sub_data_idx += 1
                self.total_samples = 0
                yield batch[:idx_in_batch]
                idx_in_batch = 0
                batch = [0] * self.batch_size
            elif idx_in_batch == self.batch_size:
                yield batch
                idx_in_batch = 0
                batch = [0] * self.batch_size
        if idx_in_batch > 0:
            yield batch[:idx_in_batch]

    def __len__(self):
        return self.batch_nums
