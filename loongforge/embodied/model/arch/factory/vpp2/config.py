# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VPP2's typed config: the family base plus three provider arguments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..families import DualStreamWanConfig

__all__ = ["Vpp2Config"]


@dataclass(frozen=True)
class Vpp2Config(DualStreamWanConfig):
    """``--model-name vpp2``.

    Attributes:
        video_prefix_frames: How many leading video frames the action stream
            may attend to. Shapes a visibility rule, hence a provider argument
            and not a section key. The name matches the provider signature
            because ``provider_kwargs()`` forwards field names verbatim.
        inference_steps: Denoising steps for the action stream at eval time.
        proprio_dim: Width of the raw proprioception vector, i.e. the input
            width of the token encoder whose output is appended to the text
            slot. Unrelated to any stream width.
    """

    model_type: str = "vpp2"
    video_prefix_frames: Optional[int] = None
    inference_steps: Optional[int] = None
    proprio_dim: Optional[int] = None
