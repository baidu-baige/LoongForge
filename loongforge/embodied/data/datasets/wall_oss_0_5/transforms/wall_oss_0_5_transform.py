# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Wall-X under the Apache-2.0 License.

"""Per-sample transform for Wall-OSS-0.5 LeRobot samples."""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
from PIL import Image
from qwen_vl_utils.vision_process import IMAGE_FACTOR, MAX_PIXELS, MIN_PIXELS, smart_resize

from loongforge.embodied.data.datasets.transforms.base import BaseTransform
from loongforge.embodied.data.datasets.transforms.registry import (
    TransformBuilderContext,
    register_transform_builder,
)
from loongforge.embodied.model.wall_oss_0_5.constant import is_action_dataset_name

logger = logging.getLogger(__name__)


class WallOss05LeRobotTransform(BaseTransform):
    """Convert generic LeRobot samples into Wall-X-style per-sample dicts."""

    def __init__(self, model_cfg, data_cfg):
        """Initialize the instance."""
        super().__init__(apply_to=[], training=True)
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.key_mappings = data_cfg.key_mappings
        self.cam_key_mapping = self.key_mappings["camera"]
        self.state_key = self.key_mappings["state"]
        self.action_key = self.key_mappings["action"]
        self.image_factor = IMAGE_FACTOR
        self.min_pixels = MIN_PIXELS
        self.max_pixels = MAX_PIXELS

    def _to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """To pil."""
        current_obs = tensor.clone()
        if current_obs.ndim == 3 and current_obs.shape[0] in (1, 3):
            current_obs = current_obs.permute(1, 2, 0)
        if current_obs.dtype != torch.uint8:
            current_obs = (current_obs.clamp(0, 1) * 255).to(torch.uint8)
        return Image.fromarray(current_obs.cpu().numpy())

    def _vision_preprocess(self, data: Dict[str, Any]):
        """Vision preprocess."""
        processed_frames = []
        orig_height = orig_width = resized_height = resized_width = None
        for raw_key, view_name in self.cam_key_mapping.items():
            img_pil = self._to_pil(data[raw_key])
            orig_width, orig_height = img_pil.size
            target_size = self.data_cfg.resolution.get(view_name, -1)
            if target_size != -1:
                if orig_width > orig_height:
                    new_width = target_size
                    new_height = int(target_size * orig_height / orig_width)
                else:
                    new_height = target_size
                    new_width = int(target_size * orig_width / orig_height)
                img_pil = img_pil.resize((new_width, new_height))

            current_width, current_height = img_pil.size
            resized_height, resized_width = smart_resize(
                current_height,
                current_width,
                factor=self.image_factor,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            processed_frames.append(img_pil.resize((resized_width, resized_height)))
        return processed_frames, orig_height, orig_width, resized_height, resized_width

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply."""
        image_inputs, h, w, resize_h, resize_w = self._vision_preprocess(data)
        frame_index = data.get("frame_index", torch.tensor(0))
        instruction_info = {"instruction": data.get("task", "")}

        from loongforge.embodied.data.datasets.wall_oss_0_5.transforms.wall_oss_0_5_utils import (
            get_wallx_normal_text,
            process_grounding_points,
        )

        complete_text, _ = get_wallx_normal_text(
            instruction_info,
            self.model_cfg.action_horizon,
            int(frame_index.item()) if isinstance(frame_index, torch.Tensor) else int(frame_index),
            self.data_cfg.priority_order,
            self.cam_key_mapping,
            generate_subtask_ratio=self.data_cfg.generate_subtask_ratio,
            camera_name_mapping=self.data_cfg.camera_name_mapping,
        )
        text = process_grounding_points(complete_text, h, w, resize_h, resize_w, "qwen2_5")
        return {
            "image_inputs": image_inputs,
            "text": text,
            "action": data[self.action_key],
            "agent_pos": data[self.state_key],
            "frame_index": frame_index,
        }


@register_transform_builder("wall_oss_0_5")
def build_wall_oss_0_5_transforms(ctx: TransformBuilderContext):
    """Build wall oss 0 5 transforms."""
    model_type = ctx.model_cfg.model_type
    if model_type != "wall_oss_0_5":
        return []
    return [WallOss05LeRobotTransform(ctx.model_cfg, ctx.data_cfg)]
