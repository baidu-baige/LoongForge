"""FastWAM per-sample transform for LoongForge datasets."""

from typing import Any, Dict

import math
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


@register_transform_builder("fastwam")
def build_fastwam_transforms(ctx: TransformBuilderContext):
    """Build FastWAM-specific per-sample transforms."""
    from loongforge.embodied.data.datasets.transforms.utils.action_transform import ActionTransform
    from loongforge.embodied.data.datasets.transforms.utils.builders import convert_stats

    transforms = []

    normalization_mode = ctx.data_cfg.normalization_mode

    # Action normalization: matches bak pipeline.py step 2.
    # ActionTransform(apply_to=["action"], action_horizon=32, normalization_mode=q99)
    action_stats = (
        convert_stats(ctx.dataset_stats.get("action"))
        if ctx.dataset_stats
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
        if ctx.dataset_stats
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
