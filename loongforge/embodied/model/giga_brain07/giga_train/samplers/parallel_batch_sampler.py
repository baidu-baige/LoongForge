import torch


class ParallelBatchSampler(torch.utils.data.Sampler):
    """Replicate each batch for data-parallel consumption within a single
    process.

    This wrapper expands each batch yielded by ``batch_sampler`` into
    ``data_parallel_size`` identical batches, so downstream code can split them
    across multiple model replicas inside one process.

    Args:
        batch_sampler: The underlying batch sampler yielding lists of indices.
        data_parallel_size: Number of times each batch is replicated.
    """

    def __init__(self, batch_sampler, data_parallel_size: int):
        self.batch_sampler = batch_sampler
        self.data_parallel_size = data_parallel_size
        self.batch_size = getattr(batch_sampler, 'batch_size', None)
        self.drop_last = getattr(batch_sampler, 'drop_last', False)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.batch_sampler, 'set_epoch'):
            self.batch_sampler.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.batch_sampler) * self.data_parallel_size

    def __iter__(self):
        for batch in self.batch_sampler:
            for i in range(self.data_parallel_size):
                yield batch
