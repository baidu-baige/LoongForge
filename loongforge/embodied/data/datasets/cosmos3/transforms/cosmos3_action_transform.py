# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3 action transform: video resize + caption tokenize + SequencePlan + action padding.

Per-sample transform tailored for the DROID action-policy SFT recipe. Mirrors
cosmos-framework's ``ActionTransformPipeline`` bit-for-bit on the pieces that
feed ``Cosmos3._forward_raw_video_with_action``:

1. Aspect-preserving bicubic resize + bottom/right reflection-pad (edge-pad
   when padding exceeds spatial dim) to the closest predefined target in
   ``VIDEO_RES_SIZE_INFO[resolution]``. Writes ``image_size`` =
   ``[target_h, target_w, orig_h_resized, orig_w_resized]``.
2. Caption augmentor chain (``". "`` separator, period-aware) appending
   ``ViewpointTextInfo`` -> ``DurationFPSTextTimeStamps`` ->
   ``ResolutionTextInfo`` (with ``duration = int(num_frames / fps)`` per cosmos).
3. Qwen3-VL chat template tokenization with ``add_vision_id=False``.
4. ``build_sequence_plan_from_mode("policy", video_length, action_length)``
   (returns ``action_start_frame_offset = 0`` and ``condition_frame_indexes_action = [0]``
   when ``action_length == video_length`` — the ``use_state=True`` shape).
5. Pad ``action`` to ``max_action_dim`` and store ``raw_action_dim`` for
   downstream channel masking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as transforms_F
from transformers import AutoTokenizer

from loongforge.embodied.data.datasets.transforms.base import BaseTransform
from loongforge.embodied.model.cosmos3.sequence_packing import SequencePlan
from loongforge.embodied.data.datasets.transforms.registry import (
    TransformBuilderContext,
    register_transform_builder,
)

# ---------- VIDEO_RES_SIZE_INFO (verbatim from cosmos_framework/data/vfm/utils.py) ----------
VIDEO_RES_SIZE_INFO: Dict[str, Dict[str, Tuple[int, int]]] = {
    "256": {"1,1": (256, 256), "4,3": (320, 256), "3,4": (256, 320), "16,9": (320, 192), "9,16": (192, 320)},
    "480": {"1,1": (640, 640), "4,3": (736, 544), "3,4": (544, 736), "16,9": (832, 480), "9,16": (480, 832)},
    "704": {"1,1": (960, 960), "4,3": (1088, 832), "3,4": (832, 1088), "16,9": (1280, 704), "9,16": (704, 1280)},
    "720": {"1,1": (960, 960), "4,3": (1104, 832), "3,4": (832, 1104), "16,9": (1280, 720), "9,16": (720, 1280)},
    "768": {"1,1": (1024, 1024), "4,3": (1184, 880), "3,4": (880, 1184), "16,9": (1360, 768), "9,16": (768, 1360)},
    "1080": {"1,1": (1440, 1440), "4,3": (1664, 1248), "3,4": (1248, 1664), "16,9": (1920, 1080), "9,16": (1080, 1920)},
    "1280": {"1,1": (1712, 1712), "4,3": (1968, 1472), "3,4": (1472, 1968), "16,9": (2272, 1280), "9,16": (1280, 2272)},
    "2048": {"1,1": (2728, 2728), "4,3": (3160, 2368), "3,4": (2368, 3160), "16,9": (3640, 2048), "9,16": (2048, 3640)},
    "gt_2048": {
        "1,1": (5464, 5464),
        "4,3": (6304, 4728),
        "3,4": (4728, 6304),
        "16,9": (7280, 4096),
        "9,16": (4096, 7280),
    },
}


# ---------- viewpoint templates (verbatim from cosmos viewpoint_utils.py) ----------
_VIEWPOINT_TEMPLATES: Dict[str, str] = {
    "ego_view": "This video is captured from a first-person perspective looking at the scene.",
    "third_person_view": (
        "This video is captured from a third-person perspective "
        "looking towards the agent from the front."
    ),
    "wrist_view": "This video is captured from a wrist-mounted camera.",
    "concat_view": "This video contains concatenated views from multiple camera perspectives.",
}

_DURATION_FPS_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
_RESOLUTION_VIDEO_TEMPLATE = "This video is of {height}x{width} resolution."


def _find_closest_target_size(h: int, w: int, resolution: str) -> Tuple[int, int]:
    """Return ``(target_w, target_h)`` from ``VIDEO_RES_SIZE_INFO[resolution]``.

    Mirrors ``cosmos_framework.data.vfm.action.transforms.find_closest_target_size``:
    selects the aspect bucket whose ``H/W`` ratio is closest to the input's.
    """
    if resolution not in VIDEO_RES_SIZE_INFO:
        raise ValueError(
            f"resolution={resolution!r} not in VIDEO_RES_SIZE_INFO; "
            f"available: {list(VIDEO_RES_SIZE_INFO)}"
        )
    candidates = VIDEO_RES_SIZE_INFO[resolution]
    input_ratio = h / w
    best_key = None
    best_diff = float("inf")
    for aspect_key, (cand_w, cand_h) in candidates.items():
        cand_ratio = cand_h / cand_w
        diff = abs(input_ratio - cand_ratio)
        if diff < best_diff:
            best_diff = diff
            best_key = aspect_key
    target_w, target_h = candidates[best_key]
    return target_w, target_h


def _reflection_pad_to_target(
    video: torch.Tensor,
    target_w: int,
    target_h: int,
    keep_aspect_ratio: bool = True,
) -> Tuple[torch.Tensor, int, int]:
    """Aspect-preserving bicubic resize + bottom/right reflection-pad.

    Returns ``(padded_video, orig_h_resized, orig_w_resized)`` matching
    cosmos's ``reflection_pad_to_target`` semantics:

    * Cap scaling factor at 1.0 (no upscaling).
    * Pad on bottom + right only.
    * Use edge-pad when padding exceeds spatial dim, else reflection-pad.
    """
    orig_h, orig_w = video.shape[-2:]
    if keep_aspect_ratio:
        scaling_ratio = min(target_w / orig_w, target_h / orig_h, 1.0)
        orig_h_resized = int(scaling_ratio * orig_h + 0.5)
        orig_w_resized = int(scaling_ratio * orig_w + 0.5)
    else:
        orig_h_resized = target_h
        orig_w_resized = target_w

    if orig_h_resized != orig_h or orig_w_resized != orig_w:
        video = transforms_F.resize(
            video,
            size=[orig_h_resized, orig_w_resized],
            interpolation=transforms_F.InterpolationMode.BICUBIC,
            antialias=True,
        )

    if orig_w_resized != target_w or orig_h_resized != target_h:
        padding_right = target_w - orig_w_resized
        padding_bottom = target_h - orig_h_resized
        padding = [0, 0, padding_right, padding_bottom]
        if padding_right >= orig_w_resized or padding_bottom >= orig_h_resized:
            video = transforms_F.pad(video, padding, padding_mode="edge")
        else:
            video = transforms_F.pad(video, padding, padding_mode="reflect")

    return video, orig_h_resized, orig_w_resized


def _append_with_period_separator(caption: str, addition: str) -> str:
    """Append ``addition`` to ``caption`` using ``" "`` if caption ends with
    a period, else ``". "`` — exactly the augmentor convention.
    """
    if not isinstance(caption, str) or caption == "":
        return caption
    caption = caption.rstrip()
    sep = " " if caption.endswith(".") else ". "
    return caption + sep + addition


def _build_sequence_plan_policy(
    video_length: int,
    action_length: int,
    has_text: bool,
) -> SequencePlan:
    """Inline copy of cosmos's ``build_sequence_plan_from_mode`` for ``mode="policy"``.

    For ``policy`` with ``num_history_actions = 0``:

    * ``condition_frame_indexes_vision = [0]``
    * If ``action_length == video_length - 1``:
        ``condition_frame_indexes_action = []``, ``action_start_frame_offset = 1``
    * If ``action_length == video_length`` (use_state=True path):
        ``condition_frame_indexes_action = [0]``, ``action_start_frame_offset = 0``
    """
    condition_frame_indexes_vision: List[int] = [0]
    if action_length == video_length - 1:
        condition_frame_indexes_action: List[int] = []
        action_start_frame_offset = 1
    elif action_length == video_length:
        condition_frame_indexes_action = [0]
        action_start_frame_offset = 0
    else:
        # Cosmos asserts/raises in this branch; mirror with an explicit error.
        raise ValueError(
            f"Unsupported (video_length={video_length}, action_length={action_length}) "
            "for policy mode: must satisfy action_length in {video_length-1, video_length}."
        )
    return SequencePlan(
        has_text=has_text,
        has_vision=True,
        has_action=True,
        condition_frame_indexes_vision=condition_frame_indexes_vision,
        condition_frame_indexes_action=condition_frame_indexes_action,
        action_start_frame_offset=action_start_frame_offset,
    )


def _pad_action_to_max_dim(action: torch.Tensor, max_action_dim: int) -> torch.Tensor:
    """Pad ``[T, D]`` along D up to ``max_action_dim``."""
    cur_dim = action.shape[-1]
    if cur_dim > max_action_dim:
        raise ValueError(f"action_dim {cur_dim} exceeds max_action_dim {max_action_dim}")
    if cur_dim == max_action_dim:
        return action
    pad = torch.zeros(*action.shape[:-1], max_action_dim - cur_dim, dtype=action.dtype, device=action.device)
    return torch.cat([action, pad], dim=-1)


@register_transform_builder("cosmos3")
def build_cosmos3_transforms(ctx: TransformBuilderContext):
    """Build Cosmos3 action-policy per-sample transforms."""
    model_cfg = ctx.model_cfg
    data_cfg = ctx.data_cfg
    training_args = ctx.training_args
    return [Cosmos3ActionTransform(
        tokenizer_path=training_args.tokenizer_path,
        max_text_tokens=data_cfg.max_text_tokens,
        max_action_dim=model_cfg.max_action_dim,
        action_chunk_length=data_cfg.action_chunk_length,
        temporal_compression=model_cfg.vae_temporal_compression,
        cfg_dropout_rate=data_cfg.cfg_dropout_rate,
        target_h=data_cfg.target_h,
        target_w=data_cfg.target_w,
    )]


class Cosmos3ActionTransform(BaseTransform):
    """Per-sample transform for Cosmos3 action-policy SFT.

    Args:
        tokenizer_path: HF identifier or local path for the Qwen3-VL tokenizer.
        max_text_tokens: Hard cap on caption token length.
        max_action_dim: Padding target for the action tensor's last dim.
        resolution: Resolution tier key into ``VIDEO_RES_SIZE_INFO`` (e.g.
            ``"480"`` for the DROID nano recipe).
        action_chunk_length: Number of supervised action steps. Combined with
            ``use_state=True`` the dataset yields ``chunk + 1`` action frames.
        temporal_compression: VAE temporal compression factor (4 for Wan2.2).
            Used to align the video frame count to ``(T - 1) %
            temporal_compression == 0``.
        cfg_dropout_rate: Probability of replacing the caption with the empty
            string (classifier-free guidance). Set to 0.0 for deterministic
            parity tests.
    """

    def __init__(
        self,
        tokenizer_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_text_tokens: int = 1024,
        max_action_dim: int = 64,
        resolution: str = "480",
        action_chunk_length: int = 32,
        temporal_compression: int = 4,
        cfg_dropout_rate: float = 0.0,
        # Backwards-compat aliases used by older configs/tests:
        target_h: Optional[int] = None,
        target_w: Optional[int] = None,
    ) -> None:
        super().__init__(apply_to=["video", "action", "ai_caption"], training=True)
        self.max_text_tokens = max_text_tokens
        self.max_action_dim = max_action_dim
        self.resolution = str(resolution)
        if target_h is not None or target_w is not None:
            # Old single-square (480, 480) callers: keep map to "480" tier.
            # The actual closest aspect bucket is selected per-sample via
            # find_closest_target_size; target_h/target_w are no longer used.
            pass
        self.action_chunk_length = int(action_chunk_length)
        self.temporal_compression = int(temporal_compression)
        self.cfg_dropout_rate = float(cfg_dropout_rate)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the cosmos3 action transform to a single sample."""
        video: torch.Tensor = data["video"]               # [3, T, H, W] uint8
        action: torch.Tensor = data["action"]             # [chunk+1, 8] float
        caption: str = data.get("ai_caption", "")
        viewpoint = data.get("viewpoint", "concat_view")
        fps = data.get("conditioning_fps", torch.tensor(15.0))
        domain_id = data.get("domain_id", torch.tensor(0, dtype=torch.long))
        idle_frames = data.get("idle_frames", torch.tensor(0, dtype=torch.long))
        mode = data.get("mode", "policy")
        description = data.get("additional_view_description")

        # 1. Frame alignment: (T - 1) % temporal_compression == 0
        c, t, h, w = video.shape
        target_t = max(1, (t - 1) // self.temporal_compression * self.temporal_compression + 1)
        target_t = min(target_t, t)
        video = video[:, :target_t]

        # 2. Spatial resize + reflection-pad to closest VIDEO_RES_SIZE_INFO bucket.
        target_w, target_h = _find_closest_target_size(video.shape[-2], video.shape[-1], self.resolution)
        video, orig_h_resized, orig_w_resized = _reflection_pad_to_target(
            video, target_w=target_w, target_h=target_h, keep_aspect_ratio=True
        )
        image_size = torch.tensor(
            [target_h, target_w, orig_h_resized, orig_w_resized], dtype=torch.float
        )

        # 3. Caption augmentor chain (cosmos ". " separator semantics).
        if self.training and self.cfg_dropout_rate > 0.0 and torch.rand(1).item() < self.cfg_dropout_rate:
            full_caption = ""
        else:
            full_caption = caption
            # 3a. Viewpoint
            if isinstance(full_caption, str) and full_caption != "":
                tmpl = _VIEWPOINT_TEMPLATES.get(str(viewpoint))
                if tmpl is not None:
                    if description is not None:
                        tmpl = _append_with_period_separator(tmpl, description)
                    full_caption = _append_with_period_separator(full_caption, tmpl)
            # 3b. Duration / FPS — duration uses int(num_frames / fps).
            fps_val = float(fps.item() if torch.is_tensor(fps) else fps)
            num_frames = video.shape[1]
            if isinstance(full_caption, str) and full_caption != "" and fps_val > 0:
                duration = int(num_frames / fps_val)
                full_caption = _append_with_period_separator(
                    full_caption,
                    _DURATION_FPS_TEMPLATE.format(duration=duration, fps=fps_val),
                )
            # 3c. Resolution — uses image_size[0]/image_size[1] (target_h/target_w).
            if isinstance(full_caption, str) and full_caption != "":
                full_caption = _append_with_period_separator(
                    full_caption,
                    _RESOLUTION_VIDEO_TEMPLATE.format(height=int(image_size[0]), width=int(image_size[1])),
                )

        # 4. Tokenize caption via Qwen3-VL chat template — must match cosmos
        # tokenize_caption(use_system_prompt=False, add_vision_id=False).
        conversations = [{"role": "user", "content": full_caption}]
        text_ids = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=False,
            return_dict=False,
        )
        text_ids = text_ids[: self.max_text_tokens]

        # 5. SequencePlan (cosmos build_sequence_plan_from_mode for "policy").
        video_length = video.shape[1]
        action_length = action.shape[0]
        plan = _build_sequence_plan_policy(
            video_length=video_length,
            action_length=action_length,
            has_text=True,
        )

        raw_action_dim = torch.tensor(action.shape[-1], dtype=torch.long)
        action_padded = _pad_action_to_max_dim(action, self.max_action_dim)

        return {
            "video": video,
            "action": action_padded,
            "raw_action_dim": raw_action_dim,
            "domain_id": domain_id.long() if torch.is_tensor(domain_id) else torch.tensor(int(domain_id)),
            "fps": float(fps.item() if torch.is_tensor(fps) else fps),
            "idle_frames": idle_frames,
            "image_size": image_size,
            "ai_caption": full_caption,
            "text_token_ids": text_ids,
            "sequence_plan": plan,
            "dataset_index": data.get("dataset_index"),
            "episode_index": data.get("episode_index"),
            "start_frame": data.get("start_frame"),
            "task_index": data.get("task_index"),
        }
