# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Data objects shared by dataset and model layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    import torch
    from loongforge.models.world.cosmos3.sequence_packing import SequencePlan


@dataclass
class Cosmos3Batch:
    """Batch passed from the Cosmos3 preprocessor to the model."""

    videos: Optional[List[torch.Tensor]]
    text_token_ids: List[List[int]]
    sequence_plans: List[SequencePlan]
    fps_values: List[float]
    actions: Optional[List[torch.Tensor]] = None
    raw_action_dims: Optional[List[torch.Tensor]] = None
    action_domain_ids: Optional[List[torch.Tensor]] = None
    idle_frames: Optional[List[torch.Tensor]] = None
    dataset_indices: Optional[List[torch.Tensor]] = None
    episode_indices: Optional[List[torch.Tensor]] = None
    start_frames: Optional[List[torch.Tensor]] = None
    task_indices: Optional[List[torch.Tensor]] = None
    image_sizes: Optional[List[torch.Tensor]] = None
    video_stacks: Optional[List[torch.Tensor]] = None
    view_splits: Optional[List[Any]] = None
    video_ops: Optional[List[Any]] = None
    video_rng_states: Optional[List[torch.Tensor]] = None
    video_pipeline: Optional[Any] = None

    def _materialize_videos(self) -> List[torch.Tensor]:
        return [
            self.video_pipeline(
                stack,
                int(self.view_splits[i][0]),
                int(self.view_splits[i][1]),
                self.video_ops[i],
                self.video_rng_states[i],
            )
            for i, stack in enumerate(self.video_stacks)
        ]

    def to(self, device):
        """Move video and action tensors to ``device``; keep index metadata on CPU."""
        if self.video_stacks is not None:
            self.video_stacks = [s.to(device) for s in self.video_stacks]
            if self.videos is None:
                self.videos = self._materialize_videos()
        else:
            self.videos = [v.to(device) for v in self.videos]
        for name in ("actions", "raw_action_dims", "action_domain_ids"):
            values = getattr(self, name)
            if values is not None:
                setattr(self, name, [value.to(device) for value in values])
        return self
