# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from https://github.com/thu-ml/Motus under the Apache-2.0 License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Motus camera-view resolution + per-frame image assembly.

Single source of truth (shared by the dataset and the transform) for:

- Which video keys must be multi-frame decoded (``resolve_view_mode``): the
  dataset uses this to restrict ``delta_timestamps`` to exactly the keys the
  transform will consume, so lerobot never decodes redundant views.
- How a decoded lerobot sample is turned into ``first_frame`` / ``video_frames``
  (``assemble_first_and_video``): verbatim numerics from the source Motus
  ``LeRobotMotusDataset`` (``_resize_frame_chw`` + ``load_concatenated_view``).

View priority (matches source):
  1. ``observation.images.cam_concatenated`` — use directly.
  2. ``cam_high`` + ``cam_left_wrist`` + ``cam_right_wrist`` — stitch back into a
     concatenated view (top=high, bottom-left=left, bottom-right=right).
  3. single-view fallback: ``observation.images.main`` / ``observation.image`` /
     ``image`` (first present), else the first visual key deterministically.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

_CONCAT_KEY = "observation.images.cam_concatenated"
_THREE_CAM_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
_SINGLE_VIEW_CANDIDATES = ("observation.images.main", "observation.image", "image")


def resolve_view_mode(feature_keys: Sequence[str]) -> Tuple[str, List[str]]:
    """Resolve the view mode and the video keys that must be decoded.

    Args:
        feature_keys: All feature keys of the dataset (``dataset.features``).

    Returns:
        ``(mode, decode_keys)`` where ``mode`` is ``"concat"`` / ``"three_cam"``
        / ``"single"`` and ``decode_keys`` is the minimal set of keys to attach
        ``delta_timestamps`` to.
    """
    keys = set(feature_keys)
    if _CONCAT_KEY in keys:
        return "concat", [_CONCAT_KEY]
    if all(k in keys for k in _THREE_CAM_KEYS):
        return "three_cam", list(_THREE_CAM_KEYS)

    for cand in _SINGLE_VIEW_CANDIDATES:
        if cand in keys:
            return "single", [cand]

    # Last resort: first visual (video/image) key, chosen deterministically.
    raise ValueError(
        "No usable image keys found in dataset features "
        f"(looked for {_CONCAT_KEY}, {_THREE_CAM_KEYS}, {_SINGLE_VIEW_CANDIDATES}); "
        f"available: {sorted(keys)}"
    )


def _to_chw_float(img: torch.Tensor) -> torch.Tensor:
    """Normalize a decoded frame to float ``[C, H, W]`` (verbatim from source)."""
    img = img.float()
    if img.ndim == 3 and img.shape[0] != 3 and img.shape[-1] == 3:
        img = img.permute(2, 0, 1)
    return img


def resize_frame_chw(frame_chw: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """Resize+pad a ``[C,H,W]`` float frame to ``target_size=(H,W)``, keeping [0,1].

    Verbatim from source ``LeRobotMotusDataset._resize_frame_chw``.
    """
    from loongforge.embodied.data.datasets.motus.transforms.image_utils import resize_with_padding

    if frame_chw.dim() != 3:
        raise ValueError(f"Expected frame [C,H,W], got {tuple(frame_chw.shape)}")
    c, h, w = frame_chw.shape
    th, tw = target_size
    if (h, w) == (th, tw):
        return frame_chw
    frame_hwc = frame_chw.permute(1, 2, 0).cpu().numpy()  # float32 [H,W,C] in [0,1]
    frame_uint8 = np.clip(frame_hwc * 255.0, 0, 255).astype(np.uint8)
    resized = resize_with_padding(frame_uint8, target_size)  # uint8 [th,tw,3]
    return torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0


def _stitch_three_cam(
    cam_high: torch.Tensor,
    cam_left: torch.Tensor,
    cam_right: torch.Tensor,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """Reconstruct a concatenated view from three cameras (verbatim from source)."""
    cam_high = _to_chw_float(cam_high)
    cam_left = _to_chw_float(cam_left)
    cam_right = _to_chw_float(cam_right)

    c = cam_high.shape[0]
    top_h = int(cam_high.shape[1])
    target_w = int(cam_high.shape[2])

    bottom_h = int(max(cam_left.shape[1], cam_right.shape[1]))
    split_w = target_w // 2
    right_w = target_w - split_w

    cam_high_r = resize_frame_chw(cam_high, (top_h, target_w))
    cam_left_r = resize_frame_chw(cam_left, (bottom_h, split_w))
    cam_right_r = resize_frame_chw(cam_right, (bottom_h, right_w))

    out = torch.zeros((c, top_h + bottom_h, target_w), dtype=cam_high_r.dtype)
    out[:, :top_h, :target_w] = cam_high_r
    out[:, top_h:, :split_w] = cam_left_r
    out[:, top_h:, split_w:] = cam_right_r

    return resize_frame_chw(out, target_size)


def assemble_first_and_video(
    item: Dict[str, Any],
    mode: str,
    decode_keys: List[str],
    target_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a decoded multi-frame lerobot sample into (first_frame, video_frames).

    Each decode key in ``item`` is a ``[T, C, H, W]`` stack where ``T = 1 +
    num_video_frames``: index 0 is the condition (anchor) frame, ``1:`` are the
    future video-prediction frames.

    Returns:
        ``first_frame`` ``[C, H, W]`` and ``video_frames`` ``[num_video_frames,
        C, H, W]``, both resized/padded to ``target_size`` and in [0, 1].
    """
    if mode == "three_cam":
        high = item[_THREE_CAM_KEYS[0]]
        left = item[_THREE_CAM_KEYS[1]]
        right = item[_THREE_CAM_KEYS[2]]
        t = high.shape[0]
        stitched = [
            _stitch_three_cam(high[i], left[i], right[i], target_size) for i in range(t)
        ]
        first_frame = stitched[0]
        video_frames = torch.stack(stitched[1:], dim=0)
        return first_frame, video_frames

    # concat / single: one key, [T, C, H, W]
    frames = item[decode_keys[0]]
    t = frames.shape[0]
    first_frame = resize_frame_chw(_to_chw_float(frames[0]), target_size)
    video_frames = torch.stack(
        [resize_frame_chw(_to_chw_float(frames[i]), target_size) for i in range(1, t)],
        dim=0,
    )
    return first_frame, video_frames
