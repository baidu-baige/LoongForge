import math
from typing import Iterator

import numpy as np
import torch


class DefaultSampler(torch.utils.data.Sampler):
    """Infinite or finite sampler with optional shuffling and padding."""

    def __init__(
        self, dataset: torch.utils.data.Dataset, batch_size: int | None = None, shuffle: bool = True, infinite: bool = True, seed: int = 6666
    ) -> None:
        """Initialize the sampler.

        Args:
            dataset (torch.utils.data.Dataset): Dataset used to determine length.
            batch_size (int | None): If set, pad total size up to a multiple of this
                value. This ensures the number of yielded indices is divisible by
                ``batch_size`` to avoid last-batch underflow.
            shuffle (bool): Whether to shuffle indices at each epoch.
            infinite (bool): If True, iterate forever; otherwise, stop after one pass.
            seed (int): Base random seed for reproducibility across epochs.
        """
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed
        self.epoch = 0
        self.data_size = len(dataset)
        if batch_size is not None:
            self.total_size = int(math.ceil(self.data_size / batch_size)) * batch_size
        else:
            self.total_size = self.data_size

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch to control deterministic shuffling.

        Args:
            epoch (int): Epoch number used to seed the RNG.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Number of indices yielded per epoch (after padding if any)."""
        return self.total_size

    def __iter__(self) -> Iterator[int]:
        """Yield indices, resetting RNG per epoch for deterministic shuffles.

        Yields:
            int: Next sample index (possibly repeated due to padding).
        """
        while True:
            np.random.seed(self.seed + self.epoch)
            self.epoch += 1
            # Build one epoch worth of indices, padding to total_size if batch_size provided
            indices = np.zeros((0,), dtype=np.int64)
            while len(indices) < self.total_size:
                indices_i = np.arange(self.data_size)
                if self.shuffle:
                    # Shuffle per small block for cache-friendliness and stability
                    indices_i = np.random.permutation(indices_i)
                num_data = min(len(indices_i), self.total_size - len(indices))
                indices = np.hstack((indices, indices_i[:num_data]))  # pad up to total_size if needed
                indices = indices.tolist()
            yield from indices
            if not self.infinite:
                break
