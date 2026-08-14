# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VA PayloadBuilder.

LingBot-VA conditions on camera images plus the task instruction only — it consumes
no proprio state, so ``state_encoding`` stays empty. Its action is already LIBERO's
native 7-D end-effector command (pos + axis-angle + gripper), so ``action_encoding``
matches the LIBERO adapter's ``action_space`` and the composed ActionDecoder key is
identity (no decoding).

The one non-default capability is ``disable_action_cache``: LingBot-VA is an
autoregressive world model whose next chunk is conditioned on the observations
produced *while* the current chunk executes. Being called once per chunk would hide
those observations and turn the rollout open-loop, so the model must see every env
step and owns the action queue itself.
"""

from __future__ import annotations

from typing import Any, Dict

from loongforge.embodied.eval.payload_builders.base import PayloadBuilder
from loongforge.embodied.eval.payload_builders.pi05 import _pack_images
from loongforge.embodied.eval.payload_builders.registry import register_payload_builder


@register_payload_builder("lingbot_va")
class LingBotVAPayloadBuilder(PayloadBuilder):
    """LingBot-VA client-side payload assembly."""

    # Capability declarations (YAML-overridable via type annotations).
    state_encoding: str = ""  # no proprio: images + instruction only
    action_encoding: str = "axis_angle"  # pos(3) + axis_angle(3) + grip(1), LIBERO-native
    action_dim: int = 7
    action_horizon: int = 1  # one env step per call; the model holds the chunk
    disable_action_cache: bool = True  # closed loop within a chunk — see module docstring

    def build(self, canonical: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Return the kwargs consumed by ``LingBotVAPredictActionModel.predict_action``."""
        return {
            "images": _pack_images(canonical["images"]),
            "instructions": [str(canonical["instruction"])],
            "state": None,
        }
