# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""latent dataset"""

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from megatron.training import get_args


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, steps_per_epoch=0):
        args = get_args()
        self.data_paths = []
        self.load_data(data_path)
        self.steps_per_epoch = steps_per_epoch
        print(
            f"self.steps_per_epoch: {self.steps_per_epoch}, total_samples: {len(self.data_paths)}"
        )
        assert len(self.data_paths) > 0
        self.manual_seed = args.seed

    def load_data(self, data_path):
        """load data files, collect all file absolute paths from data_path directory"""
        base_path = Path(data_path).resolve()
        assert base_path.is_dir(), f"data_path must be a directory: {data_path}"
        self.data_paths = sorted([str(p) for p in base_path.rglob("*") if p.is_file()])

    def __getitem__(self, index):
        seed = (self.manual_seed + index) % 2**32
        numpy_random_state = np.random.RandomState(seed=seed)
        data_id = numpy_random_state.randint(0, self.steps_per_epoch)
        data_id = data_id % len(self.data_paths)
        # data_id = 0
        path = self.data_paths[data_id]
        data = torch.load(path, weights_only=False, map_location="cpu")
        # used for generate timestep
        data["seed"] = seed
        return data

    def __len__(self):
        return self.steps_per_epoch
