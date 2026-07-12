# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Latent dataset used by diffusion training."""

from pathlib import Path

import numpy as np
import torch


class TensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        steps_per_epoch=0,
        seed=0,
        keep_keys=None,
        data_parallel_size=1,
    ):
        self.data_paths = []
        self.load_data(data_path)
        self.steps_per_epoch = steps_per_epoch
        self.data_parallel_size = data_parallel_size
        if not 0 < self.data_parallel_size <= len(self.data_paths):
            raise ValueError("Need at least one physical file per DP rank")
        self.samples_per_rank = len(self.data_paths) // self.data_parallel_size

        print(
            f"self.steps_per_epoch: {self.steps_per_epoch}, "
            f"total_samples: {len(self.data_paths)}, "
            f"dropped_samples: {len(self.data_paths) % self.data_parallel_size}"
        )
        self.manual_seed = seed
        self._shuffle_epoch = None
        self._shuffle_order = None
        # Optional whitelist of keys to keep from each loaded sample. Caller decides
        # the policy (e.g. wan2-1-i2v) so this dataset stays decoupled from global state.
        self.keep_keys = set(keep_keys) if keep_keys else None

    def load_data(self, data_path):
        """load data files, collect all file absolute paths from data_path directory"""
        base_path = Path(data_path).resolve()
        assert base_path.is_dir(), f"data_path must be a directory: {data_path}"
        self.data_paths = sorted([str(p) for p in base_path.rglob("*") if p.is_file()])

    def __getitem__(self, index):
        index = int(index)
        logical_rank = index % self.data_parallel_size
        shuffle_epoch, shard_offset = divmod(
            index // self.data_parallel_size, self.samples_per_rank
        )

        if shuffle_epoch != self._shuffle_epoch:
            rng = np.random.RandomState((self.manual_seed + shuffle_epoch) % 2**32)
            self._shuffle_order = rng.permutation(self.samples_per_rank)
            self._shuffle_epoch = shuffle_epoch

        data_id = logical_rank * self.samples_per_rank + int(
            self._shuffle_order[shard_offset]
        )
        path = self.data_paths[data_id]
        data = torch.load(path, weights_only=False, map_location="cpu")
        if self.keep_keys is not None:
            data = {k: v for k, v in data.items() if k in self.keep_keys}
        data = {k: v for k, v in data.items() if v is not None}
        # used for generate timestep
        seed = (self.manual_seed + index) % 2**32
        data["seed"] = seed
        return data

    def __len__(self):
        return self.steps_per_epoch
