# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Unified data and condition interface where we save the tokenized states and/or
noised latent states for diffusion/flow-matching training.
Used for the VFM generation model.
"""

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class GenerationDataClean:
    """
    Container for tokenized states and conditioning info (clean states)
    for the multi-modal (vision, sound, action) MoT training.
    Used for the VFM generation model.
    """

    batch_size: int
    # Vision (list of per-sample tensors)
    is_image_batch: bool
    raw_state_vision: list[torch.Tensor] | None = None  # raw state in pixel space
    x0_tokens_vision: list[torch.Tensor] | None = None  # tokenized latent state
    fps_vision: torch.Tensor | None = None

    # Image editing: number of vision items per sample.
    # When set, x0_tokens_vision is a flat list of individually-encoded image latents
    # (e.g. [src1, tgt1, src2, tgt2, ...]) and this field records how many items belong
    # to each sample (e.g. [2, 2, ...]).  None for standard T2I/T2V (one item per sample).
    num_vision_items_per_sample: list[int] | None = None

    # Audio (Sound)
    raw_state_sound: torch.Tensor | None = None
    x0_tokens_sound: torch.Tensor | None = None
    fps_sound: torch.Tensor | None = None

    # Action (dense list of per-sample tensors, only action-having samples)
    raw_state_action: list[torch.Tensor] | None = None
    x0_tokens_action: list[torch.Tensor] | None = None
    fps_action: torch.Tensor | None = None
    action_domain_id: list[torch.Tensor] | None = None  # per-sample domain IDs, None when no action samples
    raw_action_dim: list[torch.Tensor] | None = None  # raw action dimension, used adding masks to loss calculation
