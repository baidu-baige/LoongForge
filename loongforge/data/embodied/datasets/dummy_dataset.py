# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Dummy Dataset - Synthetic data for debugging and CI testing

Generates random image/action/state/lang samples without real data files.

Usage (CLI):
    --dataset-format dummy_datasets model.action_horizon=7 data.image_size=224
"""

import logging
from typing import Any, Dict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DUMMY_TASKS = [
    "pick up the red block and place it on the green plate",
    "open the drawer and take out the cup",
    "push the blue cube to the right side",
    "close the laptop lid carefully",
    "stack the blocks in order from largest to smallest",
]


class DummyVLADataset(Dataset):
    """Synthetic VLA dataset for debugging."""

    def __init__(
        self,
        num_samples: int = 100,
        action_dim: int = 7,
        state_dim: int = 7,
        action_horizon: int = 7,
        image_size: int = 224,
        num_cameras: int = 2,
    ):
        self.num_samples = num_samples
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.num_cameras = num_cameras

        logger.info(
            f"DummyVLADataset: {num_samples} samples, "
            f"action_dim={action_dim}, horizon={action_horizon}"
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = np.random.RandomState(idx)

        images = []
        for _ in range(self.num_cameras):
            img_array = rng.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
            images.append(Image.fromarray(img_array))

        action = rng.randn(self.action_horizon, self.action_dim).astype(np.float32) * 0.1
        state = rng.randn(1, self.state_dim).astype(np.float32) * 0.1
        lang = DUMMY_TASKS[idx % len(DUMMY_TASKS)]

        return {
            "image": images,
            "lang": lang,
            "action": action,
            "state": state,
        }


# ═══════════════════════════════════════════════════════════════
# Builder (called by data/__init__.py)
# ═══════════════════════════════════════════════════════════════

def build_dummy_dataset(model_cfg, data_cfg, training_args) -> Dataset:
    """Build dummy dataset from typed configs + CLI training_args."""
    num_samples = training_args.num_samples
    action_dim = model_cfg.action_dim
    state_dim = model_cfg.state_dim
    action_horizon = model_cfg.action_horizon
    image_size = data_cfg.image_size

    return DummyVLADataset(
        num_samples=num_samples,
        action_dim=action_dim,
        state_dim=state_dim,
        action_horizon=action_horizon,
        image_size=image_size,
    )
