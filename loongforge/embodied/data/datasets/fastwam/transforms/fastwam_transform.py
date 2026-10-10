"""FastWAM per-sample transform for LoongForge datasets."""

import json
from typing import Any, Dict, List, Optional

import torch
import torchvision.transforms.functional as TF

from loongforge.embodied.data.datasets.transforms.base import BaseTransform
from loongforge.embodied.data.datasets.transforms.registry import (
    TransformBuilderContext,
    register_transform_builder,
)


class FastWAMKeyMappingTransform(BaseTransform):
    """Map standard VLA samples to FastWAM collator-friendly fields.

    Handles both single-frame [C,H,W] and multi-frame [T,C,H,W] image inputs.
    Multi-frame inputs (from LeRobotDataset with observation_delta_indices) are
    assembled directly into a video tensor [C,T,H,W] in [-1,1].
    Single-frame inputs fall back to the images list for the collator to handle.
    """

    DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

    def __init__(self, image_size: int = 224, training: bool = True):
        super().__init__(apply_to=[], training=training)
        self.image_size = image_size

    def _build_prompt(self, data: Dict[str, Any]) -> str:
        """Build the text prompt string from sample data."""
        task = str(data.get("prompt", data.get("task", "")))
        return task if task.startswith("A video recorded") else self.DEFAULT_PROMPT.format(task=task)

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply key mapping and preprocessing to a single sample."""
        image_keys = sorted(
            k for k in data
            if k.startswith("observation.images.") and not k.endswith("_is_pad")
        )
        images = [data[key].float() for key in image_keys]

        action = data.get("action")
        if action is not None and not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)

        proprio = data.get("observation.state", data.get("proprio", None))
        if proprio is not None and not isinstance(proprio, torch.Tensor):
            proprio = torch.as_tensor(proprio, dtype=torch.float32)

        prompt = self._build_prompt(data)

        if images and images[0].ndim == 4:
            # Multi-frame path: each image is [T, C, H, W], values in [0, 1].
            # Match FastWAM reference (robot_video_dataset.py + libero_2cam.yaml):
            #   1. per-camera bilinear resize to [image_size, image_size]
            #   2. horizontal concat → [T, C, image_size, image_size*n_cam]
            #   3. normalize(0.5, 0.5) → [-1, 1]
            resized = [
                TF.resize(img, [self.image_size, self.image_size],
                          interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
                for img in images
            ]  # each [T, C, image_size, image_size]
            video = torch.cat(resized, dim=-1)  # [T, C, image_size, image_size*n_cam]
            video = TF.normalize(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            video = video.permute(1, 0, 2, 3)  # [C, T, H, W]
            out = {"video": video, "action": action, "proprio": proprio, "prompt": prompt}
            for key in ("action_is_pad", "image_is_pad"):
                if key in data and data[key] is not None:
                    out[key] = data[key]
            return out

        # Preserve mask fields from input for collator use
        out = {"images": images, "action": action, "proprio": proprio, "prompt": prompt}
        for key in ("action_is_pad", "image_is_pad"):
            if key in data and data[key] is not None:
                out[key] = data[key]
        return out


def _fastwam_linear_params(stats: Dict[str, Any], mode: str):
    """Scale/offset of FastWAM's ``SingleFieldLinearNormalizer`` from ``global_*`` stats."""
    stats = {k.removeprefix("global_"): torch.tensor(v, dtype=torch.float64).to(torch.float32)
             for k, v in stats.items() if k.startswith("global_")}
    if mode == "z-score":
        return 1.0 / (stats["std"] + 1e-8), -stats["mean"] / (stats["std"] + 1e-8)
    if mode == "min/max":
        input_min, input_max = stats["min"], stats["max"]
    elif mode == "q01/q99":
        input_min, input_max = stats["q01"], stats["q99"]
    else:
        raise ValueError(f"Unsupported fastwam_norm_mode: {mode!r} (min/max | q01/q99 | z-score)")
    input_range = input_max - input_min
    ignore_dim = input_range < 1e-4
    input_range[ignore_dim] = 2.0
    scale = 2.0 / input_range
    offset = -1.0 - scale * input_min
    offset[ignore_dim] = -input_min[ignore_dim]
    return scale, offset


class FastWAMLinearNormalizeTransform(BaseTransform):
    """Normalize ``action`` / ``observation.state`` exactly like FastWAM's processor.

    Padded action steps have their delta dims zeroed first, then both fields go
    through ``x * scale + offset`` clamped to [-5, 5].
    """

    def __init__(
        self,
        stats_path: str,
        mode: str = "min/max",
        delta_action_dim_mask: Optional[List[bool]] = None,
        training: bool = True,
    ):
        super().__init__(apply_to=["action", "observation.state"], training=training)
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.params = {
            "action": _fastwam_linear_params(stats["action"]["default"], mode),
            "observation.state": _fastwam_linear_params(stats["state"]["default"], mode),
        }
        self.delta_action_dim_mask = (
            None if delta_action_dim_mask is None
            else torch.as_tensor(list(delta_action_dim_mask), dtype=torch.bool)
        )

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Zero padded delta dims, then normalize and clamp."""
        for key in self.apply_to:
            if data.get(key) is None:
                continue
            value = torch.as_tensor(data[key], dtype=torch.float32).clone()
            if key == "action" and self.delta_action_dim_mask is not None and data.get("action_is_pad") is not None:
                is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
                value[is_pad.unsqueeze(1) & self.delta_action_dim_mask.unsqueeze(0)] = 0.0
            scale, offset = self.params[key]
            data[key] = torch.clamp(value * scale + offset, -5.0, 5.0)
        return data

    def unapply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Inverse normalization (clamping is not undone)."""
        for key in self.apply_to:
            if data.get(key) is None:
                continue
            scale, offset = self.params[key]
            data[key] = (torch.as_tensor(data[key], dtype=torch.float32) - offset) / scale
        return data


@register_transform_builder("fastwam")
def build_fastwam_transforms(ctx: TransformBuilderContext):
    """Build FastWAM-specific per-sample transforms."""
    from loongforge.embodied.data.datasets.transforms.utils.action_transform import ActionTransform
    from loongforge.embodied.data.datasets.transforms.utils.builders import convert_stats

    transforms = []

    fastwam_norm = ctx.data_cfg.norm_stats_path is not None
    if fastwam_norm:
        transforms.append(FastWAMLinearNormalizeTransform(
            stats_path=ctx.data_cfg.norm_stats_path,
            mode=ctx.data_cfg.fastwam_norm_mode,
            delta_action_dim_mask=ctx.data_cfg.delta_action_dim_mask,
        ))

    normalization_mode = ctx.data_cfg.normalization_mode

    # Action normalization: matches bak pipeline.py step 2.
    # ActionTransform(apply_to=["action"], action_horizon=32, normalization_mode=q99)
    action_stats = (
        convert_stats(ctx.dataset_stats.get("action"))
        if ctx.dataset_stats and not fastwam_norm
        else None
    )
    action_horizon = getattr(ctx.model_cfg, "action_horizon", None)
    max_action_dim = getattr(ctx.model_cfg, "max_action_dim", None)
    transforms.append(ActionTransform(
        apply_to=["action"],
        action_horizon=action_horizon,
        max_action_dim=max_action_dim,
        normalization_mode=normalization_mode,
        statistics=action_stats,
        padding_strategy=ctx.data_cfg.action_padding_strategy,
    ))

    # Proprio normalization: normalize observation.state to match BCTrainer pipeline.
    # bak pipeline.py applies ActionTransform(apply_to=["observation.state"], normalization_mode=q99)
    # before FastWAMKeyMappingTransform reads it as `proprio`.
    proprio_stats = (
        convert_stats(ctx.dataset_stats.get("observation.state"))
        if ctx.dataset_stats and not fastwam_norm
        else None
    )
    transforms.append(ActionTransform(
        apply_to=["observation.state"],
        action_horizon=None,
        max_action_dim=None,
        normalization_mode=normalization_mode,
        statistics=proprio_stats,
        padding_strategy="none",
    ))

    transforms.append(FastWAMKeyMappingTransform(image_size=ctx.data_cfg.image_size))
    return transforms
