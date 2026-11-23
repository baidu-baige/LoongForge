# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3 action-policy PayloadBuilder.

Cosmos3-Nano-Policy-DROID was trained on a single image that mosaics the three
RoboLab views (540 x 640 for the default 720p cameras):

    +---------------------------+
    |          wrist            |  360 x 640
    +-------------+-------------+
    |    left     |    right    |  180 x 320 each
    +-------------+-------------+

plus 7 arm joint positions and one gripper value (DROID 8D layout). This packing
mirrors ``RoboLab/policies/cosmos3/client.py`` so the model sees exactly the
frames the official client sends (the Cosmos3 server logs ``image=540x640``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from loongforge.embodied.eval.payload_builders.base import PayloadBuilder, resize_with_pad as _resize_with_pad
from loongforge.embodied.eval.payload_builders.registry import register_payload_builder


def _half(image: np.ndarray) -> np.ndarray:
    """Bilinear downsample by 2 (same operator the official client uses)."""
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    size = (image.shape[0] // 2, image.shape[1] // 2)
    resized = F.interpolate(tensor, size=size, mode="bilinear")
    return resized.squeeze(0).permute(1, 2, 0).numpy().astype(image.dtype)


def compose_cosmos3_image(
    images: Dict[str, Optional[np.ndarray]], height: int = 360, width: int = 640
) -> np.ndarray:
    """Build Cosmos3's wrist-over-(left|right) mosaic from canonical views."""
    wrist = images.get("wrist")
    if wrist is None:
        wrist = images.get("primary")
    left = images.get("left")
    right = images.get("right")
    if wrist is None or left is None or right is None:
        raise ValueError(
            "cosmos3 needs the wrist, left and right views; "
            f"got {sorted(name for name, value in images.items() if value is not None)}"
        )
    top = _resize_with_pad(np.asarray(wrist), height, width)
    bottom = np.concatenate(
        [_half(_resize_with_pad(np.asarray(left), height, width)),
         _half(_resize_with_pad(np.asarray(right), height, width))],
        axis=1,
    )
    return np.concatenate([top, bottom], axis=0)


@register_payload_builder("cosmos3")
class Cosmos3PayloadBuilder(PayloadBuilder):
    """Cosmos3 client-side payload assembly."""

    # DROID 8D proprio: 7 arm joints + 1 gripper (see adapters/robolab.py).
    state_encoding: str = "joint_gripper"
    action_encoding: str = "joint_pos"
    action_dim: int = 8
    action_horizon: int = 32
    # Target size of the wrist view; the mosaic adds a half-height row of
    # left|right below it (360x640 wrist -> 540x640 image).
    image_height: int = 360
    image_width: int = 640

    def build(self, canonical: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Return the kwargs consumed by ``Cosmos3.predict_action``."""
        if self.state_encoding != "joint_gripper":
            raise ValueError(f"Unsupported cosmos3 state_encoding: {self.state_encoding!r}")
        joint = (canonical.get("state_raw") or {}).get("joint")
        if joint is None:
            raise ValueError("cosmos3 requires canonical.state_raw['joint'] (7 arm joints + gripper)")
        state = np.asarray(joint, dtype=np.float32).reshape(-1)
        if state.size != self.action_dim:
            raise ValueError(f"cosmos3 state must be {self.action_dim}D, got {state.size}D")

        return {
            "images": [compose_cosmos3_image(canonical["images"], self.image_height, self.image_width)],
            "instructions": [str(canonical["instruction"])],
            "state": state,
        }
