# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""X-VLA PayloadBuilder.

X-VLA is a multi-embodiment model that requires:

- Per-benchmark ``domain_id`` (LIBERO=3, RoboTwin=6, CALVIN=2, SimplerEnv=4)
- Encoded proprio state (``ee6d`` for LIBERO/RoboTwin, no state for others)
- Image list ordered ``[primary(+wrist|left|right)]``

The single class collects every X-VLA-specific transform previously scattered
across the LIBERO adapter (``_build_model_state``), the runner
(``_canonical_to_policy_payload``), and the server policy
(``_build_image_input``). It emits a plain-python payload; ``domain_id`` is
still tensorized server-side by the factory wrapper (torch cannot cross the
RPC boundary).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from loongforge.evaluation.payload_builders.base import PayloadBuilder
from loongforge.evaluation.payload_builders.pi05 import _pack_images
from loongforge.evaluation.payload_builders.registry import register_payload_builder


def quat_to_rot6d_interleaved(quat: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to interleaved 6D rotation (official X-VLA layout)."""
    from scipy.spatial.transform import Rotation

    mat = Rotation.from_quat(np.asarray(quat, dtype=np.float64)).as_matrix()
    return mat[:, :2].reshape(6).astype(np.float32)


def build_ee6d_dual_proprio(
    endpose: Dict[str, Any], last_env_action: Optional[np.ndarray] = None
) -> np.ndarray:
    """Build the 20D dual-arm proprio used by the official X-VLA RoboTwin client.

    Layout: [left pos(3) + quat->rot6d(6) + (1 - 2*gripper)(1),
             right pos(3) + quat->rot6d(6) + (1 - 2*gripper)(1)].

    Official closed-loop detail: after the first chunk, the client overwrites
    the endpose with the last commanded ee action (pos+quat), while grippers
    stay measured. ``last_env_action`` is the 16D env action from the previous
    step; when provided, its pose parts replace the measured endpose.
    """
    if last_env_action is not None:
        left_ee = np.asarray(last_env_action[:7], dtype=np.float32)
        right_ee = np.asarray(last_env_action[8:15], dtype=np.float32)
    else:
        left_ee = np.asarray(endpose["left_endpose"], dtype=np.float32).reshape(-1)
        right_ee = np.asarray(endpose["right_endpose"], dtype=np.float32).reshape(-1)
    left_grip = 1.0 - 2.0 * float(endpose["left_gripper"])
    right_grip = 1.0 - 2.0 * float(endpose["right_gripper"])
    return np.concatenate(
        [
            left_ee[:3],
            quat_to_rot6d_interleaved(left_ee[3:7]),
            np.asarray([left_grip], dtype=np.float32),
            right_ee[:3],
            quat_to_rot6d_interleaved(right_ee[3:7]),
            np.asarray([right_grip], dtype=np.float32),
        ]
    ).astype(np.float32)


def _quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, w] quaternion to X-VLA training-side 6D rotation.

    Column-major layout ``[R00, R10, R20, R01, R11, R21]`` (concat first two
    columns), matching the LIBERO client's ``Mat_to_Rotate6D`` order.
    """
    from scipy.spatial.transform import Rotation

    mat = Rotation.from_quat(quat).as_matrix()  # 3x3
    return np.concatenate([mat[:, 0], mat[:, 1]]).astype(np.float32)


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, w] quaternion to axis-angle."""
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def _euler_to_rot6d_interleaved(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert intrinsic-xyz Euler angles to interleaved 6D rotation.

    Matches the official X-VLA calvin client:
    ``R.from_euler("xyz", q).as_matrix()[..., :, :2].reshape(6)`` — the first
    two columns of the rotation matrix are flattened with column-major
    interleave (not concatenation, unlike LIBERO's layout).
    """
    from scipy.spatial.transform import Rotation

    mat = Rotation.from_euler("xyz", np.asarray(euler_xyz, dtype=np.float64)).as_matrix()
    return mat[:, :2].reshape(6).astype(np.float32)


def _encode_ee6d(state_raw: Dict[str, Any], target_dim: int = 20) -> np.ndarray:
    """Build the 20D proprio X-VLA expects in ee6d state format for LIBERO.

    Layout: ``[eef_pos(3), rot6d(6), gripper(1)]`` padded to ``target_dim``
    with zeros. Prefer ``ee_ori_mat`` (from the controller) when available;
    fall back to converting ``eef_quat``. Gripper is forced to 0.0 to match
    the original X-VLA LIBERO client.
    """
    eef_pos = np.asarray(state_raw["eef_pos"], dtype=np.float32).reshape(-1)[:3]
    ee_ori_mat = state_raw.get("ee_ori_mat")
    if ee_ori_mat is not None:
        ee_ori_mat = np.asarray(ee_ori_mat, dtype=np.float32).reshape(3, 3)
        rot = np.concatenate([ee_ori_mat[:, 0], ee_ori_mat[:, 1]]).astype(np.float32)
    else:
        rot = _quat_to_rot6d(np.asarray(state_raw["eef_quat"], dtype=np.float32).reshape(-1)[:4])
    grip = np.array([0.0], dtype=np.float32)
    proprio = np.concatenate([eef_pos, rot, grip])  # 10D
    state = np.zeros(target_dim, dtype=np.float32)
    state[: proprio.size] = proprio
    return state


def _encode_axis_angle(state_raw: Dict[str, Any], target_dim: int = 20) -> np.ndarray:
    """Build the 20D proprio in axis-angle format (fallback for X-VLA)."""
    eef_pos = np.asarray(state_raw["eef_pos"], dtype=np.float32).reshape(-1)[:3]
    rot = _quat_to_axis_angle(np.asarray(state_raw["eef_quat"], dtype=np.float32).reshape(-1)[:4])
    grip = np.array([0.0], dtype=np.float32)
    proprio = np.concatenate([eef_pos, rot, grip])  # 7D
    state = np.zeros(target_dim, dtype=np.float32)
    state[: proprio.size] = proprio
    return state


def _encode_ee6d_calvin(state_raw: Dict[str, Any], target_dim: int = 20) -> Optional[np.ndarray]:
    """Build the 20D proprio X-VLA expects on CALVIN.

    Official X-VLA CALVIN client layout:
    ``[tcp_pos(3), euler->rot6d_interleaved(6), gripper_action > 0 (1)]``
    padded to ``target_dim`` with zeros. Requires ``robot_obs[:15]`` from the
    CALVIN env (pos(3) + euler(3) + ... + gripper_action(1) at index 14).
    Returns ``None`` when the raw obs is too short.
    """
    robot_obs = np.asarray(state_raw.get("robot_obs"), dtype=np.float32).reshape(-1)
    if robot_obs.size < 15:
        return None
    pos = robot_obs[:3].astype(np.float32)
    rot6d = _euler_to_rot6d_interleaved(robot_obs[3:6])
    grip = np.asarray([1.0 if float(robot_obs[-1]) > 0.0 else 0.0], dtype=np.float32)
    proprio = np.concatenate([pos, rot6d, grip])
    state = np.zeros(target_dim, dtype=np.float32)
    state[: proprio.size] = proprio
    return state


def _encode_ee6d_widowx_initial(state_raw: Dict[str, Any], target_dim: int = 20) -> Optional[np.ndarray]:
    """Build SimplerEnv WidowX's initial 20D proprio (identity rot, TCP pos).

    Official X-VLA WidowX client:
    ``[ee_pos_wrt_base(3), 1, 0, 0, 1, 0, 0 (identity rot6d, interleaved),
    0 (gripper)]`` padded to ``target_dim``. Subsequent steps overwrite the
    first 10 dims with the last consumed action (closed-loop backfill, see
    ``XVLAPayloadBuilder.update_from_response``).
    """
    base_pose = state_raw.get("base_pose")
    tcp_pose = state_raw.get("tcp_pose")
    if base_pose is None or tcp_pose is None:
        return None
    try:
        from sapien.core import Pose

        base_arr = np.asarray(base_pose, dtype=np.float64).reshape(-1)
        tcp_arr = np.asarray(tcp_pose, dtype=np.float64).reshape(-1)
        ee_pose_wrt_base = (
            Pose(p=base_arr[:3], q=base_arr[3:]).inv() * Pose(p=tcp_arr[:3], q=tcp_arr[3:])
        )
        pos = np.asarray(ee_pose_wrt_base.p, dtype=np.float32)
    except Exception:
        return None
    proprio = np.concatenate([pos, np.array([1, 0, 0, 1, 0, 0, 0], dtype=np.float32)])
    state = np.zeros(target_dim, dtype=np.float32)
    state[: proprio.size] = proprio
    return state


DEFAULT_DOMAIN_ID_MAP = {
    "libero": 3,
    "robotwin": 6,
    "calvin": 2,
    "simplerenv": 0,
    "maniskill": 5,
}


@register_payload_builder("xvla")
class XVLAPayloadBuilder(PayloadBuilder):
    """X-VLA client-side payload assembly."""

    # Capability declarations (YAML-overridable via type annotations).
    #
    # Supported ``state_encoding`` values:
    #   ``ee6d``          — LIBERO, RoboTwin (single-arm ee6d, concat layout)
    #   ``axis_angle``    — LIBERO fallback (pos + axis-angle + gripper)
    #   ``ee6d_calvin``   — CALVIN (tcp_pos + euler→rot6d_interleaved + grip)
    #   ``ee6d_widowx``   — SimplerEnv WidowX (stateful, closed-loop backfill)
    #   ``ee6d_dual``     — RoboTwin dual-arm (stateful, closed-loop endpose backfill)
    #   ``passthrough``   — send ``canonical["model_state"]`` as-is (ManiSkill qpos)
    #   ``""``            — no state kwarg emitted
    state_encoding: str = "ee6d"
    action_encoding: str = "ee6d"
    action_dim: int = 20
    action_horizon: int = 30
    state_dim: int = 20
    domain_id: int = -1  # -1 → auto from DOMAIN_ID_MAP[ctx.benchmark_name]

    # Non-annotated attributes: internal (not YAML-overridable).
    unnorm_key = ""

    def __init__(
        self,
        yaml_model=None,
        yaml_server=None,
        yaml_benchmark=None,
    ) -> None:
        """Read xvla-specific instance fields on top of annotated defaults."""
        super().__init__(yaml_model=yaml_model, yaml_server=yaml_server, yaml_benchmark=yaml_benchmark)
        self.unnorm_key = str(self._yaml_model.get("unnorm_key", "") or "")
        # Stateful backfill buffer used by ``ee6d_widowx`` (SimplerEnv) and
        # ``ee6d_calvin`` (CALVIN): the initial 20D proprio is captured from
        # env state on the first step, then ``update_from_response`` overwrites
        # the first 10 dims with the last consumed action each step.
        self._state_buffer: Optional[np.ndarray] = None
        # RoboTwin ``ee6d_dual`` closed-loop buffer: the last decoded 16D env
        # ee action, fed back by the bridge via ``note_env_action``.
        self._robotwin_last_ee: Optional[np.ndarray] = None

    # Encodings that keep proprio state across steps via closed-loop backfill.
    _STATEFUL_ENCODINGS = frozenset({"ee6d_widowx", "ee6d_calvin"})

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

        Only active for stateful encodings (``ee6d_widowx``, ``ee6d_calvin``).
        Matches the official X-VLA WidowX / CALVIN client update rule. On
        other encodings this is a no-op.
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

    def _resolve_domain_id(self, ctx: Dict[str, Any]) -> Optional[int]:
        """Pick ``domain_id`` from YAML override or benchmark-name map."""
        if int(self.domain_id) >= 0:
            return int(self.domain_id)
        benchmark_name = ctx.get("benchmark_name") if isinstance(ctx, dict) else None
        if benchmark_name and benchmark_name in DEFAULT_DOMAIN_ID_MAP:
            return DEFAULT_DOMAIN_ID_MAP[benchmark_name]
        return None

    def _initial_stateful_proprio(
        self, canonical: Dict[str, Any], state_raw: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Build the first-step proprio for ``ee6d_widowx`` / ``ee6d_calvin``.

        ``ee6d_calvin`` reads ``state_raw['robot_obs']`` from the CALVIN
        adapter; ``ee6d_widowx`` reads ``state_raw['base_pose']`` +
        ``['tcp_pose']`` from SimplerEnv.
        """
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
            # RoboTwin dual-arm: pose from last decoded ee action (closed-loop)
            # once available, else measured endpose; grippers always measured.
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
        raise ValueError(f"Unsupported xvla state_encoding: {self.state_encoding!r}")

    def build(self, canonical: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Return the kwargs consumed by ``XVLA.predict_action``."""
        images: List[np.ndarray] = _pack_images(canonical["images"])
        state = self._encode_state(canonical)
        kwargs: Dict[str, Any] = {
            "images": images,
            "instructions": [str(canonical["instruction"])],
            "state": state,
        }
        domain_id = self._resolve_domain_id(ctx if isinstance(ctx, dict) else {})
        if domain_id is not None:
            kwargs["domain_id"] = int(domain_id)
        if self.unnorm_key:
            kwargs["unnorm_key"] = self.unnorm_key
        return kwargs
