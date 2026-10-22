import math
from typing import Iterator

import numpy as np
import torch

from ..datasets import ConcatDataset, Dataset


class BucketSampler(torch.utils.data.Sampler):
    """Sample within precomputed buckets indicated by ``bucket_index``.

    Args:
        dataset: Dataset/ConcatDataset whose samples include ``bucket_index``.
        batch_size (int | None): Pad each bucket to a multiple of this value,
            avoiding last-batch underflow per bucket.
        shuffle (bool): Shuffle order within each bucket.
        infinite (bool): Yield forever (True) or once-through (False).
        seed (int): Base random seed.
    """

    def __init__(self, dataset, batch_size: int | None = None, shuffle: bool = True, infinite: bool = True, seed: int = 6666):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed
        self.epoch = 0
        self.indices_list = _get_indices_list(dataset=dataset)
        self.total_size_list = []
        for indices in self.indices_list:
            data_size = len(indices)
            if batch_size is not None:
                total_size = int(math.ceil(data_size / batch_size)) * batch_size
            else:
                total_size = data_size
            self.total_size_list.append(total_size)

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch to control deterministic shuffling.

        Args:
            epoch (int): Epoch number used to seed the RNG.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Number of indices yielded per epoch (sum over buckets)."""
        return sum(self.total_size_list)

    def __iter__(self) -> Iterator[int]:
        while True:
            np.random.seed(self.seed + self.epoch)
            self.epoch += 1
            indices_list = []
            for i in range(len(self.indices_list)):
                # Build indices for each bucket, padding to bucket total_size
                indices = np.zeros((0,), dtype=np.int64)
                total_size = self.total_size_list[i]
                while len(indices) < total_size:
                    indices_i = self.indices_list[i]
                    if self.shuffle:
                        # Shuffle within-bucket order
                        indices_i = np.random.permutation(indices_i)
                    num_data = min(len(indices_i), total_size - len(indices))
                    indices = np.hstack((indices, indices_i[:num_data]))
                if self.batch_size is not None:
                    indices = indices.reshape((-1, self.batch_size))
                indices_list.append(indices)
            indices = np.concatenate(indices_list, axis=0)
            if self.shuffle:
                # Shuffle bucket order for global mixing
                indices = np.random.permutation(indices)
            indices = indices.reshape(-1)
            yield from indices
            if not self.infinite:
                break


def _process_dataset(dataset: ConcatDataset | Dataset):
    if isinstance(dataset, ConcatDataset):
        return [d.datasets[0] for d in dataset.datasets]
    elif isinstance(dataset, Dataset):
        return [dataset.datasets[0]]
    else:
        assert False


def _get_indices_list(dataset: ConcatDataset | Dataset):
    dataset_list = _process_dataset(dataset)
    indices_dict = dict()
    count = 0
    for dataset in dataset_list:
        for i in range(len(dataset)):
            data_dict = dataset[i]
            bucket_index = data_dict['bucket_index']
            if bucket_index not in indices_dict:
                indices_dict[bucket_index] = []
            # Use running global counter so subsequent samplers can concatenate
            indices_dict[bucket_index].append(count)
            count += 1
    indices_list = []
    for indices in indices_dict.values():
        indices = np.array(indices, dtype=np.int64)
        indices_list.append(indices)
    return indices_list
