# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Canonical protocol types for VLA offline evaluation.

The protocol is framework-agnostic: LoongForge-VLA, OpenVLA, Octo, pi0, or any custom
policy only needs to implement a compatible policy server.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

import numpy as np

PROTOCOL_VERSION = "1.0"

ImageArray = np.ndarray


class CanonicalState(TypedDict, total=False):
    """Provide CanonicalState behavior."""

    eef_pos: Optional[List[float]]  # EEF = end-effector; pos = 3D position [x, y, z].
    eef_rot_axis_angle: Optional[List[float]]  # rot = rotation as axis-angle [ax, ay, az].
    gripper: Optional[float]
    joint: Optional[List[float]]
    frame: Optional[Literal["base", "world"]]
    units: Dict[str, str]


class CanonicalMeta(TypedDict, total=False):
    """Provide CanonicalMeta behavior."""

    benchmark: Literal["libero", "simplerenv", "robotwin", "vla_arena", "real_robot"]
    robot_setup: str
    control_hz: int
    episode_id: str
    episode_step: int
    runtime: Literal["sim", "real_robot"]
    bimanual: bool
    chunk_policy: Literal["full", "preempt"]
    realtime_deadline_ms: Optional[int]


class CanonicalObservation(TypedDict):
    """Provide CanonicalObservation behavior."""

    instruction: str
    images: Dict[str, Optional[ImageArray]]
    state: CanonicalState
    meta: CanonicalMeta


class SingleArmAction(TypedDict):
    """Provide SingleArmAction behavior."""

    world_vector: List[float]
    rotation_delta: List[float]
    gripper: float


class BimanualArmAction(TypedDict):
    """Provide BimanualArmAction behavior."""

    world_vector: List[float]
    rotation_delta: List[float]
    gripper: float


class BimanualAction(TypedDict):
    """Provide BimanualAction behavior."""

    left: BimanualArmAction
    right: BimanualArmAction
    terminate: bool


class CanonicalAction(SingleArmAction, total=False):
    """Provide CanonicalAction behavior."""

    terminate: bool


class ActionScale(TypedDict, total=False):
    """Provide ActionScale behavior."""

    pos_scale: Union[float, List[float]]
    rot_scale: Union[float, List[float]]
    gripper_scale: float
    gripper_bias: float
    left: Optional["ActionScale"]
    right: Optional["ActionScale"]


class EvalContext(TypedDict):
    """Provide EvalContext behavior."""

    benchmark: str
    robot_setup: str
    control_hz: int
    max_steps: Dict[str, int]
    action_scale: ActionScale
    bimanual: bool
    has_state_fields: List[str]
    episodes_per_task: int
    runtime: Literal["sim", "real_robot"]
    success_oracle_type: Literal["info_flag", "human", "vlm_judge", "script"]


class ServerMetadata(TypedDict, total=False):
    """Provide ServerMetadata behavior."""

    protocol_version: str
    env: str
    ckpt_path: str
    action_chunk_size: int
    supports_preempt: bool
    available_unnorm_keys: List[str]
    default_unnorm_key: str
    action_keys: List[str]
    state_keys: List[str]


class PolicyRequest(TypedDict, total=False):
    """Provide PolicyRequest behavior."""

    type: Literal["ping", "reset", "infer", "predict_action"]
    request_id: str
    protocol_version: str
    payload: Dict[str, Any]


class PolicyResponse(TypedDict, total=False):
    """Provide PolicyResponse behavior."""

    status: Literal["ok", "error"]
    ok: bool
    type: str
    request_id: str
    data: Dict[str, Any]
    error: Dict[str, Any]
