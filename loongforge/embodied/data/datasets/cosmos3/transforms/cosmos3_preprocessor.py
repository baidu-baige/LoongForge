# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3 preprocessor: collate function producing Cosmos3Batch."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from loongforge.embodied.data.datasets.transforms.collator import BasePreprocessor, register_preprocessor
from loongforge.embodied.model.cosmos3.sequence_packing import SequencePlan


@dataclass
class Cosmos3Batch:
    """Batch for Cosmos3 training with raw video (online VAE encode in forward).

    Action fields are populated only when the per-sample transform produces
    them (e.g. for the DROID action-policy SFT recipe). They stay ``None`` /
    empty for vision-only video SFT, so existing call sites continue to work.
    """

    videos: List[torch.Tensor]
    text_token_ids: List[List[int]]
    sequence_plans: List[SequencePlan]
    fps_values: List[float]

    # Action SFT extensions (populated by Cosmos3ActionTransform).
    actions: Optional[List[torch.Tensor]] = None              # per-sample [T, max_action_dim]
    raw_action_dims: Optional[List[torch.Tensor]] = None      # per-sample scalar long tensor
    action_domain_ids: Optional[List[torch.Tensor]] = None    # per-sample scalar long tensor
    idle_frames: Optional[List[torch.Tensor]] = None          # per-sample scalar long tensor
    dataset_indices: Optional[List[torch.Tensor]] = None
    episode_indices: Optional[List[torch.Tensor]] = None
    start_frames: Optional[List[torch.Tensor]] = None
    task_indices: Optional[List[torch.Tensor]] = None
    image_sizes: Optional[List[torch.Tensor]] = None  # per-sample [target_h, target_w, orig_h, orig_w]

    def to(self, device):
        """Move videos (and actions if present) to device."""
        self.videos = [v.to(device) for v in self.videos]
        if self.actions is not None:
            self.actions = [a.to(device) for a in self.actions]
        if self.raw_action_dims is not None:
            self.raw_action_dims = [d.to(device) for d in self.raw_action_dims]
        if self.action_domain_ids is not None:
            self.action_domain_ids = [d.to(device) for d in self.action_domain_ids]
        return self


@register_preprocessor("cosmos3")
class Cosmos3Preprocessor(BasePreprocessor):
    """Cosmos3 collate_fn: applies per-sample transform and assembles Cosmos3Batch."""

    @classmethod
    def from_config(
        cls,
        model_cfg,
        data_cfg,
        training_args=None,
        dataset_stats=None,
        dataset=None,
    ):
        """from_config."""
        return cls()

    def __call__(self, examples: List[Dict[str, Any]]) -> Cosmos3Batch:
        """Apply transform to each sample and collate into batch."""
        has_action = all("action" in s and s.get("sequence_plan") is not None
                         and getattr(s["sequence_plan"], "has_action", False) for s in examples)
        actions = [s["action"] for s in examples] if has_action else None
        raw_action_dims = [s["raw_action_dim"] for s in examples] if has_action else None
        action_domain_ids = [s["domain_id"] for s in examples] if has_action else None
        idle_frames = [s.get("idle_frames") for s in examples] if has_action else None
        dataset_indices = [s.get("dataset_index") for s in examples]
        episode_indices = [s.get("episode_index") for s in examples]
        start_frames = [s.get("start_frame") for s in examples]
        task_indices = [s.get("task_index") for s in examples]

        return Cosmos3Batch(
            videos=[s["video"] for s in examples],
            text_token_ids=[s["text_token_ids"] for s in examples],
            sequence_plans=[s["sequence_plan"] for s in examples],
            fps_values=[s["fps"] for s in examples],
            actions=actions,
            raw_action_dims=raw_action_dims,
            action_domain_ids=action_domain_ids,
            idle_frames=idle_frames,
            dataset_indices=dataset_indices,
            episode_indices=episode_indices,
            start_frames=start_frames,
            task_indices=task_indices,
            image_sizes=[s.get("image_size") for s in examples],
        )
