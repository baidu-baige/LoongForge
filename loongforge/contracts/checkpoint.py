# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Metadata published after a Native checkpoint save completes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NativeCheckpointMetadata:
    completed_steps: int
    epoch: int
    ckpt_format: str
    world_size: int
    use_lora: bool
