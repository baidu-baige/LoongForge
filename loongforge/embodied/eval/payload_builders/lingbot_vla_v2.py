# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VLA v2 PayloadBuilder (RoboTwin).

Sends what the community RoboTwin client sends: the three raw RGB views in
``head, left, right`` order (HWC uint8, no resize), the 14D
``joint_action.vector`` as float64, and the environment instruction. All
preprocessing (resize, normalization, chat template) runs inside the model.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from loongforge.embodied.eval.payload_builders.base import PayloadBuilder
from loongforge.embodied.eval.payload_builders.registry import register_payload_builder

_VIEWS = ("head", "left", "right")


@register_payload_builder("lingbot_vla_v2")
class LingBotVLAV2PayloadBuilder(PayloadBuilder):
    """LingBot-VLA v2 client-side payload assembly."""

    state_encoding: str = "robotwin_joint14"
    action_encoding: str = "joint_abs"
    action_dim: int = 14
    action_horizon: int = 50
    # Open-loop chunk execution: the server replans once the 50-step chunk is used up.
    disable_action_cache: bool = False

    def build(self, canonical: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Return the kwargs consumed by ``LingBotVLAV2PredictActionModel.predict_action``."""
        if self.state_encoding != "robotwin_joint14":
            raise ValueError(f"Unsupported lingbot_vla_v2 state_encoding: {self.state_encoding!r}")
        images_by_cam = canonical["images"]
        views = []
        for cam in _VIEWS:
            image = images_by_cam.get(cam)
            if image is None:
                raise ValueError(f"lingbot_vla_v2 needs the RoboTwin {cam!r} camera")
            views.append(np.asarray(image, dtype=np.uint8))
        state_raw = canonical.get("state_raw") or {}
        # The chunk-execution bridge carries the unrounded float64 vector in ctx (the community
        # client sends float64); fall back to the canonical float32 joint when it is absent.
        joint = ctx.get("joint_f64")
        if joint is None:
            joint = state_raw.get("joint")
        if joint is None:
            raise ValueError("robotwin state_raw must contain 'joint' (joint_action.vector)")
        state = np.asarray(joint, dtype=np.float64).reshape(-1)
        if state.size != 14:
            raise ValueError(f"robotwin_joint14 state expects 14D, got {state.size}D")
        return {
            "images": views,
            "instructions": [str(canonical["instruction"])],
            "state": state,
        }
