# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
ActionTransform - Generic action preprocessing transform for VLA models.

Supports configurable normalization, dimension padding, and horizon padding strategies.

Core processing:
  1. Convert to tensor [T, D]
  2. Normalize using Normalizer (q99 / min_max / mean_std / scale / identity)
  3. Pad/truncate action dimension to max_action_dim
  4. Pad/truncate action horizon with configurable strategy (zero / repeat_last / none)

Output: torch.Tensor [action_horizon, max_action_dim]
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from loongforge.embodied.data.datasets.transforms.base import BaseTransform
from loongforge.embodied.data.datasets.transforms.utils.normalizer import Normalizer


class ActionTransform(BaseTransform):
    """Action normalization + dim padding + horizon padding transform.

    Supports configurable padding strategies for horizon:
        "zero" — Zero-pad shorter sequences (default)
        "repeat_last" — Repeat last action frame to fill horizon
        "none" — No horizon padding/truncation (pass through)
    """

    PADDING_STRATEGIES = ["zero", "repeat_last", "none"]

    def __init__(
        self,
        apply_to: List[str],
        action_horizon: Optional[int] = None,
        max_action_dim: Optional[int] = None,
        normalization_mode: str = "q99",
        statistics: Optional[Dict[str, Any]] = None,
        padding_strategy: str = "zero",
        training: bool = True,
    ):
        """
        Args:
            apply_to: Keys in data dict to transform
            action_horizon: Target action sequence length (None to skip horizon padding)
            max_action_dim: Target action dimension (None to skip dim padding)
            normalization_mode: Normalizer mode (q99, min_max, mean_std, scale, binary, identity)
            statistics: Dataset statistics for normalization (None to skip)
            padding_strategy: Horizon padding strategy ("zero", "repeat_last", "none")
            training: Whether in training mode
        """
        super().__init__(apply_to=apply_to, training=training)
        self.action_horizon = action_horizon
        self.max_action_dim = max_action_dim
        self.padding_strategy = padding_strategy

        assert padding_strategy in self.PADDING_STRATEGIES, (
            f"Invalid padding_strategy: {padding_strategy}. Valid: {self.PADDING_STRATEGIES}"
        )

        # Build normalizer
        self.normalizer = None
        if statistics is not None and normalization_mode != "identity":
            self.normalizer = Normalizer(
                mode=normalization_mode,
                statistics=statistics,
            )

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.apply_to:
            if key not in data or data[key] is None:
                continue
            value = data[key]

            # Convert to torch tensor
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).float()
            elif isinstance(value, (list, tuple)):
                value = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, torch.Tensor):
                value = value.float()

            # Ensure 2D: [T, D]
            if value.ndim == 1:
                value = value.unsqueeze(0)

            # Normalize
            if self.normalizer is not None:
                value = self.normalizer.forward(value)

            # Pad/truncate action dimension
            if self.max_action_dim is not None:
                D = value.shape[-1]
                if D < self.max_action_dim:
                    value = F.pad(value, (0, self.max_action_dim - D))
                elif D > self.max_action_dim:
                    value = value[..., :self.max_action_dim]

            # Pad/truncate action horizon
            if self.action_horizon is not None and self.padding_strategy != "none":
                T = value.shape[0]
                if T > self.action_horizon:
                    value = value[:self.action_horizon]
                elif T < self.action_horizon:
                    if self.padding_strategy == "zero":
                        pad = torch.zeros(
                            self.action_horizon - T, value.shape[-1], dtype=value.dtype
                        )
                    elif self.padding_strategy == "repeat_last":
                        pad = value[-1:].expand(self.action_horizon - T, -1)
                    value = torch.cat([value, pad], dim=0)

            data[key] = value
        return data

    def unapply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Denormalization (used during inference)."""
        for key in self.apply_to:
            if key not in data or data[key] is None:
                continue
            if self.normalizer is not None:
                value = data[key]
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value).float()
                data[key] = self.normalizer.inverse(value)
        return data
