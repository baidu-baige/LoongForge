import math
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler

from ..datasets import ConcatDataset


class WeightedSampler(Sampler):
    """A weighted sampler for `ConcatDataset` that rebalances data from
    different sources.

    This sampler adjusts the number of samples from each sub-dataset based on a weight calculated as `size ** exponent`. The total number of samples
    remains the same as the original dataset. Sub-datasets can be over-sampled (with replacement) or under-sampled (without replacement) depending on
    their calculated weight.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        batch_size: int | None = None,
        shuffle: bool = True,
        infinite: bool = True,
        seed: int = 6666,
        weighting_exponent: float = 0.43,
    ):
        """Initialize the sampler.

        Args:
            dataset (ConcatDataset): The dataset to sample from.
            batch_size (int | None): If set, pad each child dataset to a multiple of
                this value to avoid underfull last batches.
            shuffle (bool): If ``True``, sampler will shuffle the indices.
            infinite (bool): If ``True``, the sampler will yield data indefinitely.
            seed (int): Random seed used to shuffle the sampler if shuffle is ``True``.
            weighting_exponent (float): The exponent for re-weighting. Defaults to 0.43.
        """
        if not isinstance(dataset, ConcatDataset):
            raise TypeError('dataset should be a ConcatDataset')

        self.dataset = dataset
        self.batch_size = batch_size
        self.weighting_exponent = weighting_exponent
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed
        self.epoch = 0

        self.sub_dataset_lengths = [len(d) for d in self.dataset.datasets]
        weights = [n**self.weighting_exponent for n in self.sub_dataset_lengths]
        total_weight = sum(weights)
        if total_weight > 0:
            probabilities = [w / total_weight for w in weights]
            original_total_size = sum(self.sub_dataset_lengths)
            if self.batch_size is not None:
                original_total_size = int(math.ceil(original_total_size / self.batch_size)) * self.batch_size
            self.num_samples_per_sub_dataset = [int(round(p * original_total_size)) for p in probabilities]
            self.total_size = sum(self.num_samples_per_sub_dataset)
        else:
            self.num_samples_per_sub_dataset = [0] * len(self.sub_dataset_lengths)
            self.total_size = 0

        cumulative_sizes = [0] * len(self.dataset.datasets)
        for i in range(1, len(self.dataset.datasets)):
            cumulative_sizes[i] = cumulative_sizes[i - 1] + len(self.dataset.datasets[i - 1])
        self.offsets = cumulative_sizes

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler. When :attr:`shuffle=True`, this
        ensures all replicas use a different random ordering for each epoch.
        Otherwise, the next iteration of this sampler will yield the same
        ordering of indices.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Total number of samples in the down-sampled dataset."""
        return self.total_size

    def __iter__(self) -> Iterator[int]:
        while True:
            np.random.seed(self.seed + self.epoch)
            self.epoch += 1

            all_indices = []
            for i in range(len(self.dataset.datasets)):
                n = self.sub_dataset_lengths[i]
                k = self.num_samples_per_sub_dataset[i]
                offset = self.offsets[i]

                if k == 0:
                    continue

                if self.shuffle:
                    replace = k > n
                    local_indices = np.random.choice(n, k, replace=replace)
                else:
                    if k > n:
                        base_indices = np.arange(n)
                        num_repeats = int(np.ceil(k / n))
                        local_indices = np.tile(base_indices, num_repeats)[:k]
                    else:
                        local_indices = np.arange(k)

                global_indices = local_indices + offset
                all_indices.append(global_indices)

            if not all_indices:
                indices = np.array([], dtype=np.int64)
            else:
                indices = np.concatenate(all_indices)

            if self.shuffle and len(indices) > 0:
                np.random.shuffle(indices)

            yield from indices.tolist()

            if not self.infinite:
                break
