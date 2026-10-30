import math
from typing import Iterator

import numpy as np
import torch

from ..datasets import ConcatDataset


class SpecialDatasetSampler(torch.utils.data.Sampler):
    """Sampler that preserves dataset boundaries inside a ``ConcatDataset``.

    Args:
        dataset: A ``ConcatDataset`` instance.
        batch_size (int | None): If set, pad each child dataset to a multiple of
            this value to avoid underfull last batches.
        shuffle (bool): Shuffle within each child dataset.
        infinite (bool): Yield forever (True) or once-through (False).
        seed (int): Base random seed.
    """

    def __init__(self, dataset: ConcatDataset, batch_size: int | None = None, shuffle: bool = True, infinite: bool = True, seed: int = 6666):
        assert isinstance(dataset, ConcatDataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed
        self.epoch = 0
        self.data_size_list = []
        self.total_size_list = []
        for dataset in dataset.datasets:
            data_size = len(dataset)
            if batch_size is not None:
                total_size = int(math.ceil(data_size / batch_size)) * batch_size
            else:
                total_size = data_size
            self.data_size_list.append(data_size)
            self.total_size_list.append(total_size)

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch to control deterministic shuffling.

        Args:
            epoch (int): Epoch number used to seed the RNG.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Number of indices yielded per epoch (sum over sub-datasets)."""
        return sum(self.total_size_list)

    def __iter__(self) -> Iterator[int]:
        while True:
            np.random.seed(self.seed + self.epoch)
            self.epoch += 1
            start_idx = 0
            indices_list = []
            for i in range(len(self.data_size_list)):
                # Build indices within each child dataset range
                indices = np.zeros((0,), dtype=np.int64)
                data_size = self.data_size_list[i]
                total_size = self.total_size_list[i]
                while len(indices) < total_size:
                    indices_i = np.arange(data_size) + start_idx
                    if self.shuffle:
                        # Shuffle within child dataset to avoid ordering bias
                        indices_i = np.random.permutation(indices_i)
                    num_data = min(len(indices_i), total_size - len(indices))
                    indices = np.hstack((indices, indices_i[:num_data]))
                if self.batch_size is not None:
                    indices = indices.reshape((-1, self.batch_size))
                indices_list.append(indices)
                start_idx += data_size
            indices = np.concatenate(indices_list, axis=0)
            if self.shuffle:
                # Shuffle across child datasets for global mixing
                indices = np.random.permutation(indices)
            indices = indices.reshape(-1)
            yield from indices
            if not self.infinite:
                break
