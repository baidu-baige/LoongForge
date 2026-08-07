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

    videos: Optional[List[torch.Tensor]]
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

    # Deferred video tail. In this mode the dataset stops after crop/resize and
    # ``videos`` starts out ``None``: ``video_stacks`` holds the pre-ColorJitter
    # 3-view stacks (uint8) and ``video_pipeline`` runs ColorJitter + the whole
    # tail on the target device in :meth:`to`. Keeping this in the data layer is
    # what lets the model be built from ``model_cfg`` alone.
    video_stacks: Optional[List[torch.Tensor]] = None
    view_splits: Optional[List[Any]] = None
    video_ops: Optional[List[Any]] = None
    video_rng_states: Optional[List[torch.Tensor]] = None
    video_pipeline: Optional[Any] = None

    def _materialize_videos(self) -> List[torch.Tensor]:
        """Run the deferred tail per sample, on whatever device the stacks are on."""
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
        """Move videos (and actions if present) to device."""
        if self.video_stacks is not None:
            self.video_stacks = [s.to(device) for s in self.video_stacks]
            if self.videos is None:
                self.videos = self._materialize_videos()
        else:
            self.videos = [v.to(device) for v in self.videos]
        if self.actions is not None:
            self.actions = [a.to(device) for a in self.actions]
        if self.raw_action_dims is not None:
            self.raw_action_dims = [d.to(device) for d in self.raw_action_dims]
        if self.action_domain_ids is not None:
            self.action_domain_ids = [d.to(device) for d in self.action_domain_ids]
        return self


class DeferredVideoTail:
    """ColorJitter + everything after it, moved out of the dataloader worker.

    ColorJitter on a full-resolution 3-view stack is the worker bottleneck, so
    ``colorjitter_on_gpu`` defers it. Deferring only the jitter would change the
    augmentation's pipeline position (it would land after downsample/concat), so
    this object owns the whole tail and reproduces the CPU order exactly:

        ColorJitter -> split views -> half-res L|R -> concat_view
                    -> uint8 [C,T,H,W] -> temporal truncate -> resize + pad

    Only the four jitter magnitudes are pickled (the transform is rebuilt lazily
    per process), so this stays cheap to ship to spawn workers.
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.4,
        saturation: float = 0.5,
        hue: float = 0.08,
    ) -> None:
        """Store jitter magnitudes; the transform itself is built on first use."""
        self.params = (float(brightness), float(contrast), float(saturation), float(hue))
        self._augmentor = None
        self._augmentor_hw = None

    def __getstate__(self):
        """Ship only the parameters across the worker boundary."""
        return {"params": self.params}

    def __setstate__(self, state):
        """Rebuild from the shipped parameters (the transform is not pickled)."""
        self.__init__(*state["params"])

    # NOTE: params are kept for provenance; build_image_augmentor owns the values
    # actually used, so the in-worker and device-side transforms cannot drift.

    def _augment(self, stack: torch.Tensor, rng_state: torch.Tensor) -> torch.Tensor:
        """Run crop -> resize -> ColorJitter from the worker's RNG position.

        torchvision samples transform parameters from the CPU generator (the input's
        device is irrelevant), so restoring that stream reproduces the in-worker draws
        exactly. fork_rng keeps the main process stream untouched.
        """
        from loongforge.embodied.data.datasets.cosmos3.droid_lerobot_dataset import (
            build_image_augmentor,
        )

        _, _, height, width = stack.shape
        if self._augmentor_hw != (height, width):
            self._augmentor = build_image_augmentor(height, width)
            self._augmentor_hw = (height, width)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(rng_state)
            return self._augmentor(stack)

    def __call__(
        self,
        stack: torch.Tensor,
        n: int,
        m: int,
        ops,
        rng_state: torch.Tensor,
    ) -> torch.Tensor:
        """Turn the raw uint8 3-view stack into the final ``[C,T,H,W]`` uint8 video.

        ``ops`` is ``(target_t, target_w, target_h)`` as decided by the transform from
        shapes alone.
        """
        from loongforge.embodied.data.datasets.cosmos3.droid_lerobot_dataset import (
            compose_concat_view,
            format_video_uint8,
        )

        from .cosmos3_action_transform import _reflection_pad_to_target

        target_t, target_w, target_h = ops
        # uint8 -> float on the k/255 grid: exactly what the worker decoded.
        stack = self._augment(stack.to(torch.float32).div_(255.0), rng_state)
        video = format_video_uint8(compose_concat_view(stack, n, m))
        video = video[:, :target_t]
        video, _, _ = _reflection_pad_to_target(
            video, target_w=int(target_w), target_h=int(target_h), keep_aspect_ratio=True
        )
        return video


@register_preprocessor("cosmos3")
class Cosmos3Preprocessor(BasePreprocessor):
    """Cosmos3 collate_fn: applies per-sample transform and assembles Cosmos3Batch."""

    def __init__(self, video_pipeline: Optional[DeferredVideoTail] = None) -> None:
        """Hold the deferred video tail (if any) to attach to each batch."""
        self._video_pipeline = video_pipeline

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
        # The dataset always does decode + crop/resize. When ColorJitter is
        # deferred, everything from the jitter onwards runs in DeferredVideoTail.
        video_pipeline = None
        if data_cfg is not None and (
            getattr(data_cfg, "use_image_augmentation", False)
            and getattr(data_cfg, "colorjitter_on_gpu", False)
        ):
            video_pipeline = DeferredVideoTail()
        return cls(video_pipeline=video_pipeline)

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

        deferred = self._video_pipeline is not None and "video_stack" in examples[0]

        return Cosmos3Batch(
            videos=None if deferred else [s["video"] for s in examples],
            video_stacks=[s["video_stack"] for s in examples] if deferred else None,
            view_splits=[s["view_split"] for s in examples] if deferred else None,
            video_ops=[s["video_ops"] for s in examples] if deferred else None,
            video_rng_states=[s["video_rng_state"] for s in examples] if deferred else None,
            video_pipeline=self._video_pipeline if deferred else None,
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
