# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""InternW0-Delta's typed config: the family base plus three provider arguments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..families import DualStreamWanConfig

__all__ = ["InternW0DeltaConfig"]


@dataclass(frozen=True)
class InternW0DeltaConfig(DualStreamWanConfig):
    """``--model-name internw0_delta``.

    Attributes:
        condition_frames: Clean history frames prepended to the noised video
            sequence. Enters the frame count the provider hands to the geometry,
            so it cannot be a plain geometry override.
        inference_steps: Denoising steps for the action stream at eval time.
        proprio_dim: Width of the raw proprioception vector.
    """

    model_type: str = "internw0_delta"
    condition_frames: Optional[int] = None
    inference_steps: Optional[int] = None
    proprio_dim: Optional[int] = None
