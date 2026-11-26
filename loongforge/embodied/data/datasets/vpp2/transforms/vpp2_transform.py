# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VPP2 per-sample image, action, proprio and text-cache transforms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as TF

from loongforge.embodied.data.datasets.transforms.base import BaseTransform
from loongforge.embodied.data.datasets.transforms.registry import (
    TransformBuilderContext,
    register_transform_builder,
)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def _linear_params(stats: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    input_min = torch.as_tensor(stats["global_min"], dtype=torch.float32)
    input_max = torch.as_tensor(stats["global_max"], dtype=torch.float32)
    input_range = input_max - input_min
    ignore_dim = input_range < 1e-4
    input_range = input_range.clone()
    input_range[ignore_dim] = 2.0
    scale = 2.0 / input_range
    offset = -1.0 - scale * input_min
    offset[ignore_dim] = -input_min[ignore_dim]
    return scale, offset


class VPP2LinearNormalizeTransform(BaseTransform):
    """Apply BASE's global min/max normalizer and [-5, 5] clamp."""

    def __init__(self, stats_path: str, delta_action_dim_mask: list[bool], training: bool = True):
        super().__init__(apply_to=["action", "observation.state"], training=training)
        with open(stats_path, encoding="utf-8") as handle:
            stats = json.load(handle)
        self.params = {
            "action": _linear_params(stats["action"]["default"]),
            "observation.state": _linear_params(stats["state"]["default"]),
        }
        self.delta_action_dim_mask = torch.as_tensor(delta_action_dim_mask, dtype=torch.bool)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            value = data.get(key)
            if value is None:
                continue
            value = torch.as_tensor(value, dtype=torch.float32).clone()
            if key == "action" and data.get("action_is_pad") is not None:
                pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
                value[pad.unsqueeze(1) & self.delta_action_dim_mask.unsqueeze(0)] = 0.0
            scale, offset = self.params[key]
            data[key] = torch.clamp(value * scale + offset, -5.0, 5.0)
        return data


class VPP2KeyMappingTransform(BaseTransform):
    """Map generic LeRobot v2.1 samples to the VPP2 model batch contract."""

    def __init__(
        self,
        image_size: int = 224,
        context_len: int = 128,
        text_embedding_cache_dir: str | None = None,
        text_embedding_cache_tag: str = "wan21i2v14b480p",
        training: bool = True,
    ):
        super().__init__(apply_to=[], training=training)
        self.image_size = int(image_size)
        self.context_len = int(context_len)
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.text_embedding_cache_tag = text_embedding_cache_tag

    def _load_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_embedding_cache_dir is None:
            raise ValueError("VPP2 requires text_embedding_cache_dir")
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = Path(self.text_embedding_cache_dir) / (
            f"{digest}.t5_len{self.context_len}.{self.text_embedding_cache_tag}.pt"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing VPP2 text cache: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        context = payload["context"]
        mask = payload["mask"].bool()
        if tuple(context.shape[:1]) != (self.context_len,) or tuple(mask.shape) != (self.context_len,):
            raise ValueError(f"Invalid VPP2 cache shapes in {path}: {tuple(context.shape)}, {tuple(mask.shape)}")
        context = context.clone()
        context[~mask] = 0.0
        return context, torch.ones_like(mask)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        image = data["observation.images.image"].float()
        wrist = data["observation.images.wrist_image"].float()
        resized = [
            TF.resize(
                camera,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            for camera in (image, wrist)
        ]
        video = torch.cat(resized, dim=-1)
        video = TF.normalize(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        video = video.permute(1, 0, 2, 3).contiguous()

        task = str(data.get("task", ""))
        prompt = task if task.startswith("A video recorded") else DEFAULT_PROMPT.format(task=task)
        context, context_mask = self._load_context(prompt)
        proprio = torch.as_tensor(data["observation.state"], dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        return {
            "video": video,
            "video_num_frames": torch.tensor(video.shape[1], dtype=torch.int64),
            "action": torch.as_tensor(data["action"], dtype=torch.float32),
            "proprio": proprio[:1],
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": torch.as_tensor(data["image_is_pad"], dtype=torch.bool),
            "action_is_pad": torch.as_tensor(data["action_is_pad"], dtype=torch.bool),
            "proprio_is_pad": torch.zeros((1,), dtype=torch.bool),
            "sample_index": torch.as_tensor(data["index"], dtype=torch.int64),
        }


@register_transform_builder("vpp2")
def build_vpp2_transforms(ctx: TransformBuilderContext):
    """Build VPP2 transforms from typed config fields."""
    if ctx.data_cfg.pipeline:
        # Native pipelines already apply their configured sample transforms.
        return []
    if ctx.data_cfg.norm_stats_path is None:
        raise ValueError("VPP2 requires data_cfg.norm_stats_path")
    return [
        VPP2LinearNormalizeTransform(
            stats_path=ctx.data_cfg.norm_stats_path,
            delta_action_dim_mask=ctx.data_cfg.delta_action_dim_mask,
        ),
        VPP2KeyMappingTransform(
            image_size=ctx.data_cfg.image_size,
            context_len=ctx.data_cfg.context_len,
            text_embedding_cache_dir=ctx.data_cfg.text_embedding_cache_dir,
            text_embedding_cache_tag=ctx.data_cfg.text_embedding_cache_tag,
        ),
    ]
