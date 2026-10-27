# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Normalizer - Multi-mode normalizer for state/action data.

Supports normalization modes:
  - q99:     2*(x - q01) / (q99 - q01) - 1  → [-1, 1] (NO clamp, values can exceed)
  - min_max: 2*(x - min) / (max - min) - 1  → [-1, 1]
  - mean_std: (x - mean) / std              → unbounded
  - scale:   x / max(|min|, |max|)          → [-1, 1]
  - binary:  x > threshold                  → {0, 1}
  - identity: no normalization

Statistics dict format:
    {"mean": [...], "std": [...], "min": [...], "max": [...], "q01": [...], "q99": [...]}

Degenerate case handling: eps-based (zero-range denominators replaced with eps).
"""

from typing import Dict

import numpy as np
import torch


class Normalizer:
    """General normalizer, supports forward/inverse."""

    VALID_MODES = ["q99", "min_max", "mean_std", "binary", "scale", "identity"]

    def __init__(self, mode: str, statistics: Dict[str, np.ndarray], binary_threshold: float = 0.5, eps: float = 1e-8):
        """
        Args:
            mode: Normalization mode (q99, min_max, mean_std, binary, scale, identity)
            statistics: Dataset statistics dictionary
            binary_threshold: Threshold for binary mode
            eps: Epsilon for zero-range denominator handling (matches base framework)
        """
        assert mode in self.VALID_MODES, f"Invalid mode: {mode}. Valid: {self.VALID_MODES}"
        self.mode = mode
        self.binary_threshold = binary_threshold
        self.eps = eps

        # Convert to tensors
        self.statistics = {}
        for key, value in statistics.items():
            if isinstance(value, np.ndarray):
                self.statistics[key] = torch.from_numpy(value).float()
            elif isinstance(value, (list, tuple)):
                self.statistics[key] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, torch.Tensor):
                self.statistics[key] = value.float()
            else:
                self.statistics[key] = torch.tensor(value, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if self.mode == "identity":
            return x

        if self.mode == "q99":
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)
            denom = q99 - q01
            denom = torch.where(denom == 0, torch.tensor(self.eps, dtype=x.dtype), denom)
            return 2.0 * (x - q01) / denom - 1.0

        elif self.mode == "min_max":
            mn = self.statistics["min"].to(x.dtype)
            mx = self.statistics["max"].to(x.dtype)
            denom = mx - mn
            denom = torch.where(denom == 0, torch.tensor(self.eps, dtype=x.dtype), denom)
            return 2.0 * (x - mn) / denom - 1.0

        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)
            std = torch.where(std == 0, torch.tensor(self.eps, dtype=x.dtype), std)
            return (x - mean) / std

        elif self.mode == "binary":
            return (x > self.binary_threshold).to(x.dtype)

        elif self.mode == "scale":
            mn = self.statistics["min"].to(x.dtype)
            mx = self.statistics["max"].to(x.dtype)
            abs_max = torch.max(torch.abs(mn), torch.abs(mx))
            denom = torch.where(abs_max == 0, torch.tensor(self.eps, dtype=x.dtype), abs_max)
            return x / denom

        raise ValueError(f"Invalid mode: {self.mode}")

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize (used during inference)."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if self.mode == "identity":
            return x

        if self.mode == "q99":
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)
            return (x + 1) / 2 * (q99 - q01) + q01

        elif self.mode == "min_max":
            mn = self.statistics["min"].to(x.dtype)
            mx = self.statistics["max"].to(x.dtype)
            return (x + 1) / 2 * (mx - mn) + mn

        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)
            return x * std + mean

        elif self.mode == "binary":
            return (x > self.binary_threshold).to(x.dtype)

        elif self.mode == "scale":
            mn = self.statistics["min"].to(x.dtype)
            mx = self.statistics["max"].to(x.dtype)
            abs_max = torch.max(torch.abs(mn), torch.abs(mx))
            return x * abs_max

        raise ValueError(f"Invalid mode: {self.mode}")

    def __call__(self, x, inverse=False):
        return self.inverse(x) if inverse else self.forward(x)
