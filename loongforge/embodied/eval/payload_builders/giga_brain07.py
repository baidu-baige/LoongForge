# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain-0.7 PayloadBuilder.

GigaBrain-0.7 is a flow-matching VLA whose eval action chunk is ee6d-style
(``pos(3) + rot6d_interleaved(6) + gripper(1)`` per end-effector, zero-padded to
``max_action_dim=32``), so it reuses the same ``ee6d_to_*`` ActionDecoders as
X-VLA. The state-encoding helpers are shared verbatim with the X-VLA builder
(same canonical ``state_raw`` -> ee6d proprio layout); the only model-specific
difference on the payload side is the embodiment selector: GigaBrain consumes an
``emb_id`` kwarg (``GigaBrain07Policy.predict_action(..., emb_id=...)``) rather
than X-VLA's ``domain_id``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from loongforge.embodied.eval.payload_builders.base import PayloadBuilder
from loongforge.embodied.eval.payload_builders.pi05 import _pack_images
from loongforge.embodied.eval.payload_builders.registry import register_payload_builder
from loongforge.embodied.eval.payload_builders.xvla import (
    _encode_axis_angle,
    _encode_ee6d,
    _encode_ee6d_calvin,
    _encode_ee6d_widowx_initial,
    build_ee6d_dual_proprio,
)

# Benchmark-name -> GigaBrain embodiment id (``EmbodimentId`` in the giga
# transform). GigaBrain's released embodiments do not include the sim benchmarks
# 1:1, so every benchmark defaults to 0 (AGILEX) and is overridable per run via
# the YAML ``model.emb_id`` field or ``model.emb_id_map``.
DEFAULT_EMB_ID_MAP: Dict[str, int] = {
    "libero": 0,
    "robotwin": 6,  # ROBOCOIN_AGILEX_COBOT_MAGIC
    "calvin": 0,
    "simplerenv": 0,
    "maniskill": 0,
}


@register_payload_builder("giga_brain07")
class GigaBrain07PayloadBuilder(PayloadBuilder):
    """GigaBrain-0.7 client-side payload assembly."""

    # Capability declarations (YAML-overridable via type annotations).
    #
    # Supported ``state_encoding`` values (shared with X-VLA):
    #   ``ee6d``          — LIBERO, RoboTwin single-arm (concat layout)
    #   ``axis_angle``    — LIBERO fallback (pos + axis-angle + gripper)
    #   ``ee6d_calvin``   — CALVIN (tcp_pos + euler->rot6d_interleaved + grip)
    #   ``ee6d_widowx``   — SimplerEnv WidowX (stateful, closed-loop backfill)
    #   ``ee6d_dual``     — RoboTwin dual-arm (stateful, closed-loop backfill)
    #   ``passthrough``   — send ``canonical.state_raw['joint']`` as-is
    #   ``""``            — no state kwarg emitted
    state_encoding: str = "ee6d"
    action_encoding: str = "ee6d"
    action_dim: int = 32
    action_horizon: int = 50
    state_dim: int = 32
    # Embodiment id forwarded to the model. -1 -> auto from DEFAULT_EMB_ID_MAP
    # (keyed by benchmark name), else this fixed value.
    emb_id: int = -1

    # Non-annotated attributes: internal (not YAML-overridable).
    _STATEFUL_ENCODINGS = frozenset({"ee6d_widowx", "ee6d_calvin"})

    def __init__(
        self,
        yaml_model=None,
        yaml_server=None,
        yaml_benchmark=None,
    ) -> None:
        """Read giga-specific instance fields on top of annotated defaults."""
        super().__init__(yaml_model=yaml_model, yaml_server=yaml_server, yaml_benchmark=yaml_benchmark)
        # Optional per-run benchmark -> emb_id override map from YAML.
        raw_map = self._yaml_model.get("emb_id_map") or {}
        self._emb_id_map: Dict[str, int] = {str(k).lower(): int(v) for k, v in dict(raw_map).items()}
        # Stateful closed-loop buffers (same semantics as the X-VLA builder).
        self._state_buffer: Optional[np.ndarray] = None
        self._robotwin_last_ee: Optional[np.ndarray] = None

    def reset(self, episode_id: str) -> None:
        """Clear the closed-loop backfill buffers at episode / subtask boundaries."""
        self._state_buffer = None
        self._robotwin_last_ee = None

    def note_env_action(self, env_action) -> None:
        """RoboTwin ``ee6d_dual`` closed-loop: remember the last decoded ee action."""
        if self.state_encoding == "ee6d_dual" and env_action is not None:
            self._robotwin_last_ee = np.asarray(env_action, dtype=np.float32).reshape(-1).copy()

    def update_from_response(self, response: Any) -> None:
        """Closed-loop backfill: ``proprio[:10] = last consumed action[:10]``.

        Only active for stateful encodings (``ee6d_widowx`` / ``ee6d_calvin``),
        matching the X-VLA WidowX / CALVIN client update rule. No-op otherwise.
        """
        if self.state_encoding not in self._STATEFUL_ENCODINGS or self._state_buffer is None:
            return
        try:
            actions = np.asarray(response["actions"], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            return
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.size == 0 or actions.shape[-1] < 10:
            return
        self._state_buffer[:10] = actions[0, :10]

    def _resolve_emb_id(self, ctx: Dict[str, Any]) -> Optional[int]:
        """Pick ``emb_id`` from a fixed YAML value, a per-run map, or the default map."""
        if int(self.emb_id) >= 0:
            return int(self.emb_id)
        benchmark_name = ctx.get("benchmark_name") if isinstance(ctx, dict) else None
        if benchmark_name:
            key = str(benchmark_name).lower()
            if key in self._emb_id_map:
                return self._emb_id_map[key]
            if key in DEFAULT_EMB_ID_MAP:
                return DEFAULT_EMB_ID_MAP[key]
        return None

    def _initial_stateful_proprio(
        self, canonical: Dict[str, Any], state_raw: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Build the first-step proprio for ``ee6d_widowx`` / ``ee6d_calvin``."""
        if self.state_encoding == "ee6d_widowx":
            return _encode_ee6d_widowx_initial(state_raw, target_dim=self.state_dim)
        if self.state_encoding == "ee6d_calvin":
            return _encode_ee6d_calvin(state_raw, target_dim=self.state_dim)
        return None

    def _encode_state(self, canonical: Dict[str, Any]) -> Optional[np.ndarray]:
        """Encode ``canonical.state_raw`` per ``self.state_encoding``."""
        if not self.state_encoding:
            return None
        state_raw = canonical.get("state_raw") or {}

        if self.state_encoding == "passthrough":
            joint = state_raw.get("joint")
            return np.asarray(joint, dtype=np.float32) if joint is not None else None

        if self.state_encoding == "ee6d_dual":
            endpose = state_raw.get("endpose")
            if endpose is None:
                return None
            return build_ee6d_dual_proprio(endpose, self._robotwin_last_ee)

        if self.state_encoding in self._STATEFUL_ENCODINGS:
            if self._state_buffer is None:
                self._state_buffer = self._initial_stateful_proprio(canonical, state_raw)
            return self._state_buffer

        if self.state_encoding == "ee6d":
            return _encode_ee6d(state_raw, target_dim=self.state_dim)
        if self.state_encoding == "axis_angle":
            return _encode_axis_angle(state_raw, target_dim=self.state_dim)
        raise ValueError(f"Unsupported giga_brain07 state_encoding: {self.state_encoding!r}")

    def build(self, canonical: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Return the kwargs consumed by ``GigaBrain07Policy.predict_action``."""
        images: List[np.ndarray] = _pack_images(canonical["images"])
        kwargs: Dict[str, Any] = {
            "images": images,
            "instructions": [str(canonical["instruction"])],
            "state": self._encode_state(canonical),
        }
        emb_id = self._resolve_emb_id(ctx if isinstance(ctx, dict) else {})
        if emb_id is not None:
            kwargs["emb_id"] = int(emb_id)
        return kwargs
