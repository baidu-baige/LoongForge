# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Shared predict_action interface helpers for embodied eval model adapters.

A model that uses the generic eval policy should implement
``predict_action(images, instructions, state=None, dataset_stats=None)`` with
these responsibilities:

- Accept batched eval images and instructions.
- Accept optional model-ready state from the benchmark adapter.
- Accept optional dataset statistics passed through by eval.
- Apply any model-specific preprocessing needed before inference.
- Apply model-specific state normalization when the model consumes state.
- Apply model-specific action unnormalization when the model emits normalized actions.
- Return an action chunk in the action convention expected by the benchmark adapter.

This module only validates the interface and normalizes output shape. It does
not apply q01/q99, mean/std, min/max, or other model-specific normalization
rules; those belong inside the model's ``predict_action`` implementation.

Action post-processing
----------------------
After ``predict_action`` returns, benchmark runners may need to convert the raw
action chunk into a format the benchmark environment accepts.  For example,
xvla emits ``[H, 20]`` actions in ee6d format (pos(3)+rot6d(6)+grip(1) per arm)
while LIBERO expects ``[H, 7]`` in pos(3)+axis-angle(3)+grip(1) format.

Use ``ACTION_POSTPROCESS_REGISTRY`` to register named post-processors, and
set ``benchmark.action_postprocess`` in the eval YAML to select one.  Runners
call ``postprocess_actions(raw_actions, action_postprocess_key)`` after
receiving the raw action chunk from the policy server.  The default (empty
key) is a simple ``[:, :action_dim]`` truncation.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PredictActionModel(Protocol):
    """Protocol for models that can be reused by the generic eval policy path."""

    def predict_action(
        self,
        images: Any,
        instructions: Any,
        state: Optional[Any] = None,
        dataset_stats: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Return an adapter-ready action chunk for batched model inputs.

        Implementations may use ``dataset_stats`` for their own normalization or
        unnormalization logic; the generic caller only passes it through.
        """


def validate_predict_action_model(model: Any) -> None:
    """Validate that a model exposes the eval-compatible predict_action method."""
    predict_action = getattr(model, "predict_action", None)
    if not callable(predict_action):
        raise TypeError("model must expose a callable predict_action(images, instructions, state, dataset_stats)")

    signature = inspect.signature(predict_action)
    parameters = signature.parameters
    has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    required = ("images", "instructions")
    optional = ("state", "dataset_stats")
    missing = [name for name in required if name not in parameters]
    if missing:
        raise TypeError(f"model.predict_action is missing required parameters: {missing}")
    unsupported = [name for name in optional if name not in parameters and not has_kwargs]
    if unsupported:
        raise TypeError(f"model.predict_action cannot accept eval keyword parameters: {unsupported}")


def _filter_supported_kwargs(func: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs the model's predict_action signature cannot accept."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def call_predict_action(
    model: PredictActionModel,
    images: Any,
    instructions: Any,
    state: Optional[Any],
    dataset_stats: Optional[Dict[str, Any]],
    action_dim: int,
    **kwargs: Any,
) -> np.ndarray:
    """Call a compatible predict_action implementation and normalize its output shape."""
    validate_predict_action_model(model)
    kwargs = _filter_supported_kwargs(model.predict_action, kwargs)
    result = model.predict_action(
        images=images,
        instructions=instructions,
        state=state,
        dataset_stats=dataset_stats,
        **kwargs,
    )
    actions = np.asarray(result, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    elif actions.ndim == 3:
        actions = actions.reshape(-1, actions.shape[-1])
    elif actions.ndim != 2:
        raise ValueError(f"model.predict_action returned unsupported action shape: {actions.shape}")

    if actions.shape[-1] < action_dim:
        raise ValueError(
            f"model.predict_action returned action dim {actions.shape[-1]}, expected at least {action_dim}"
        )
    return actions[:, :action_dim]


# ---------------------------------------------------------------------------
# Action post-processing registry
# ---------------------------------------------------------------------------
# Registered functions have signature: (actions: np.ndarray) -> np.ndarray
# where actions is [H, D] (raw model output, already shape-normalised by
# call_predict_action before reaching the runner).
# Runners call postprocess_actions(raw_actions, key) after receiving the
# action chunk from the policy server.

def _postprocess_ee6d_to_axis_angle(actions: np.ndarray) -> np.ndarray:
    """Convert ee6d action chunk [H, >=10] to axis-angle 7D [H, 7].

    Input layout (per step): pos(3) + rot6d(6) + grip(1) + optional padding
    Output layout:           pos(3) + axis_angle(3) + grip(1)

    Gripper is binarized: > 0.5 -> +1.0, else -1.0  (LIBERO convention).
    """
    pos = actions[:, :3]
    rot6d = actions[:, 3:9]
    grip_raw = actions[:, 9:10]

    axis_angle = _rot6d_to_axis_angle(rot6d)  # [H, 3]
    grip = np.where(grip_raw > 0.5, 1.0, -1.0)

    return np.concatenate([pos, axis_angle, grip], axis=-1)  # [H, 7]


def _rot6d_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to axis-angle. Input: [N, 6], Output: [N, 3]."""
    from scipy.spatial.transform import Rotation

    a1 = rot6d[:, 0:3]
    a2 = rot6d[:, 3:6]

    # Gram-Schmidt orthonormalization
    eps = 1e-8
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + eps)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2_orth = a2 - dot * b1
    b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + eps)
    b3 = np.cross(b1, b2)

    rot_mat = np.stack([b1, b2, b3], axis=-1)  # [N, 3, 3]
    axis_angle = Rotation.from_matrix(rot_mat).as_rotvec()  # [N, 3]
    return axis_angle.astype(np.float32)


def _postprocess_ee6d_to_euler(actions: np.ndarray) -> np.ndarray:
    """Convert ee6d action chunk [H, >=10] to Euler 7D [H, 7].

    Input layout (per step): pos(3) + rot6d(6) + grip(1) + optional padding
    Output layout:           pos(3) + euler_xyz(3) + grip(1)

    Gripper is binarized: > 0.25 -> +1.0, else -1.0  (SimplerEnv convention).
    """
    from scipy.spatial.transform import Rotation
    pos = actions[:, :3]
    rot6d = actions[:, 3:9]
    grip_raw = actions[:, 9:10]
    axis_angle = _rot6d_to_axis_angle(rot6d)
    euler = Rotation.from_rotvec(axis_angle).as_euler("xyz")
    grip = np.where(grip_raw > 0.25, 1.0, -1.0)
    return np.concatenate([pos, euler, grip], axis=-1).astype(np.float32)


def _postprocess_ee6d_to_quat(actions: np.ndarray) -> np.ndarray:
    """Convert ee6d action chunk [H, >=10] to quaternion 8D [H, 8].

    Input layout (per step): pos(3) + rot6d(6) + grip(1) + optional padding
    Output layout:           pos(3) + quat(4) + grip(1)

    Gripper is binarized: > 0.5 -> +1.0, else -1.0.
    """
    from scipy.spatial.transform import Rotation
    pos = actions[:, :3]
    rot6d = actions[:, 3:9]
    grip_raw = actions[:, 9:10]
    axis_angle = _rot6d_to_axis_angle(rot6d)
    quat = Rotation.from_rotvec(axis_angle).as_quat()
    grip = np.where(grip_raw > 0.5, 1.0, -1.0)
    return np.concatenate([pos, quat, grip], axis=-1).astype(np.float32)


def _rot6d_interleaved_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert interleaved 6D rotation to matrices. Input: [N, 6], Output: [N, 3, 3].

    Interleaved layout (official X-VLA calvin/simpler/robotwin clients):
    ``mat[:, :2].reshape(6)`` = [R00, R01, R10, R11, R20, R21], i.e. even
    indices form the first column and odd indices the second column. This
    differs from the concatenated layout ([R[:,0], R[:,1]]) used by the
    LIBERO client.
    """
    a1 = rot6d[:, 0:5:2]
    a2 = rot6d[:, 1:6:2]

    eps = 1e-8
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + eps)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2_orth = a2 - dot * b1
    b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + eps)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # [N, 3, 3]


def rot6d_interleaved_to_quat(rot6d: np.ndarray) -> np.ndarray:
    """Convert interleaved 6D rotation to xyzw quaternions. Input: [N, 6], Output: [N, 4]."""
    from scipy.spatial.transform import Rotation

    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    return Rotation.from_matrix(_rot6d_interleaved_to_matrix(rot6d)).as_quat().astype(np.float32)


def _postprocess_ee6d_to_calvin_abs(actions: np.ndarray) -> np.ndarray:
    """Convert ee6d chunk [H, >=10] to CALVIN absolute pose 8D [H, 8].

    Matches the official calvin_client.py step():
      pos(3) + rotate6D_to_quat(rot6d[interleaved])(4) + (grip < 0.8 -> +1 else -1)
    """
    pos = actions[:, :3]
    quat = rot6d_interleaved_to_quat(actions[:, 3:9])
    grip = np.where(actions[:, 9:10] < 0.8, 1.0, -1.0)
    return np.concatenate([pos, quat, grip], axis=-1).astype(np.float32)


def _postprocess_ee6d_to_simpler_abs_euler(actions: np.ndarray) -> np.ndarray:
    """Convert ee6d chunk [H, >=10] to SimplerEnv WidowX 7D [H, 7].

    Matches the official simpler WidowX client step():
      pos(3) + (rotate6D_to_euler_xyz(rot6d[interleaved]) + [0, pi/2, 0])(3)
      + (grip < 0.91 -> +1 else -1)
    """
    import math

    from scipy.spatial.transform import Rotation

    pos = actions[:, :3]
    rot6d = np.asarray(actions[:, 3:9], dtype=np.float64)
    euler = Rotation.from_matrix(_rot6d_interleaved_to_matrix(rot6d)).as_euler("xyz")
    euler = euler + np.array([0.0, math.pi / 2.0, 0.0])
    grip = np.where(actions[:, 9:10] < 0.91, 1.0, -1.0)
    return np.concatenate([pos, euler, grip], axis=-1).astype(np.float32)


ACTION_POSTPROCESS_REGISTRY: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "ee6d_to_axis_angle": _postprocess_ee6d_to_axis_angle,
    "ee6d_to_euler": _postprocess_ee6d_to_euler,
    "ee6d_to_quat": _postprocess_ee6d_to_quat,
    "ee6d_to_calvin_abs": _postprocess_ee6d_to_calvin_abs,
    "ee6d_to_simpler_abs_euler": _postprocess_ee6d_to_simpler_abs_euler,
}


def postprocess_actions(actions: np.ndarray, key: str = "") -> np.ndarray:
    """Apply a named post-processor from ACTION_POSTPROCESS_REGISTRY.

    Args:
        actions: Raw action chunk [H, D] from the policy server.
        key: Registry key from ``benchmark.action_postprocess`` in the eval YAML.
             Empty string or None uses the identity (no-op) path.

    Returns:
        Post-processed action chunk.
    """
    if not key:
        return actions
    fn = ACTION_POSTPROCESS_REGISTRY.get(key)
    if fn is None:
        raise KeyError(
            f"Unknown action_postprocess key: {key!r}. "
            f"Registered: {sorted(ACTION_POSTPROCESS_REGISTRY.keys())}"
        )
    return fn(actions)
