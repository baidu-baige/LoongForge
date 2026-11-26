# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VPP2 batch collator preserving BASE dtypes and mask semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import default_collate

from loongforge.embodied.data.datasets.transforms.collator import (
    BasePreprocessor,
    PreparedBatch,
    register_preprocessor,
)


@dataclass
class VPP2PreparedBatch(PreparedBatch):
    video: torch.Tensor | None = None
    action: torch.Tensor | None = None
    proprio: torch.Tensor | None = None
    context: torch.Tensor | None = None
    context_mask: torch.Tensor | None = None
    image_is_pad: torch.Tensor | None = None
    action_is_pad: torch.Tensor | None = None
    proprio_is_pad: torch.Tensor | None = None
    video_num_frames: torch.Tensor | None = None

    def to_sample(self) -> dict[str, torch.Tensor]:
        return {
            "video": self.video,
            "action": self.action,
            "proprio": self.proprio,
            "context": self.context,
            "context_mask": self.context_mask,
            "image_is_pad": self.image_is_pad,
            "action_is_pad": self.action_is_pad,
            "proprio_is_pad": self.proprio_is_pad,
            "video_num_frames": self.video_num_frames,
        }


@register_preprocessor("vpp2")
class VPP2Preprocessor(BasePreprocessor):
    """Stack already-normalized VPP2 samples without changing context dtype."""

    @classmethod
    def from_config(cls, model_cfg, data_cfg, training_args=None, dataset_stats=None, dataset=None):
        if data_cfg.pipeline:
            return RoboDojoPreprocessor()
        return cls()

    def __call__(self, examples: list[dict[str, Any]]) -> VPP2PreparedBatch:
        if not examples:
            raise ValueError("VPP2 collator received an empty batch")
        return VPP2PreparedBatch(
            video=torch.stack([item["video"] for item in examples]),
            action=torch.stack([item["action"] for item in examples]),
            proprio=torch.stack([item["proprio"] for item in examples]),
            context=torch.stack([item["context"] for item in examples]),
            context_mask=torch.stack([item["context_mask"] for item in examples]),
            image_is_pad=torch.stack([item["image_is_pad"] for item in examples]),
            action_is_pad=torch.stack([item["action_is_pad"] for item in examples]),
            proprio_is_pad=torch.stack([item["proprio_is_pad"] for item in examples]),
            video_num_frames=torch.stack([item["video_num_frames"] for item in examples]),
        )


@dataclass
class RoboDojoBatch(PreparedBatch):
    sample: dict

    @property
    def action(self):
        return self.sample["action"]

    def to_sample(self):
        return self.sample

class RoboDojoPreprocessor(BasePreprocessor):
    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls()

    def __call__(self, examples):
        return RoboDojoBatch(default_collate(examples))
