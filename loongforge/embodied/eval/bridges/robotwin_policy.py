# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""RoboTwin official-evaluator policy bridge for the standalone eval module.

RoboTwin is special because the eval module reuses RoboTwin's official
evaluator, which imports a policy plugin by `policy_name`; LIBERO and
SimplerEnv are controlled directly by local runners.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict, List, Optional

import numpy as np

from loongforge.embodied.eval.adapters.robotwin import ROBOTWIN_ACTION_DIM, ROBOTWIN_DEFAULT_MAX_STEPS, RoboTwinAdapter

ROBOTWIN_EE6D_ACTION_DIM = 20

# openpi Aloha/RoboTwin helpers (adapt_to_pi + delta joints). Only used by
# action_bridge="pi05_aloha_14d"; other bridges are unchanged.
_PI05_JOINT_FLIP_MASK = np.asarray([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1], dtype=np.float32)
_PI05_DELTA_JOINT_MASK = np.asarray(
    [True, True, True, True, True, True, False, True, True, True, True, True, True, False],
    dtype=bool,
)


def _normalize_range(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    return (x - min_val) / (max_val - min_val)


def _unnormalize_range(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value: np.ndarray) -> np.ndarray:
    """Aloha linear gripper -> pi angular space (openpi aloha_policy)."""
    value = _unnormalize_range(value, min_val=0.01844, max_val=0.05800)

    def linear_to_radian(linear_position: np.ndarray, arm_length: float, horn_radius: float) -> np.ndarray:
        ratio = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(ratio, -1.0, 1.0))

    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return _normalize_range(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value: np.ndarray) -> np.ndarray:
    """pi angular gripper -> Aloha/RoboTwin env gripper (openpi aloha_policy)."""
    value = value + 0.5476
    return _normalize_range(value, min_val=-0.6213, max_val=1.4910)


def _adapt_to_pi_decode_state(state: np.ndarray) -> np.ndarray:
    """Env joint state -> pi internal space (input side of adapt_to_pi)."""
    out = np.asarray(state, dtype=np.float32).reshape(-1).copy()
    if out.size < ROBOTWIN_ACTION_DIM:
        raise ValueError(f"pi05_aloha_14d state must be {ROBOTWIN_ACTION_DIM}D, got {out.size}D")
    out = out[:ROBOTWIN_ACTION_DIM]
    out = _PI05_JOINT_FLIP_MASK * out
    out[[6, 13]] = _gripper_to_angular(out[[6, 13]])
    return out.astype(np.float32)


def _adapt_to_pi_encode_actions(actions: np.ndarray) -> np.ndarray:
    """pi internal absolute actions -> env joint commands (output side of adapt_to_pi)."""
    out = np.asarray(actions, dtype=np.float32).reshape(-1).copy()
    if out.size < ROBOTWIN_ACTION_DIM:
        raise ValueError(f"pi05_aloha_14d action must be {ROBOTWIN_ACTION_DIM}D, got {out.size}D")
    out = out[:ROBOTWIN_ACTION_DIM]
    out = _PI05_JOINT_FLIP_MASK * out
    out[[6, 13]] = _gripper_from_angular(out[[6, 13]])
    return out.astype(np.float32)


def _delta_to_absolute_actions(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Delta joints -> absolute using state at inference (openpi AbsoluteActions).

    Mask make_bool_mask(6, -1, 6, -1): joints relative to state; grippers absolute.
    """
    out = np.asarray(actions, dtype=np.float32).reshape(-1).copy()
    st = np.asarray(state, dtype=np.float32).reshape(-1)
    if out.size < ROBOTWIN_ACTION_DIM or st.size < ROBOTWIN_ACTION_DIM:
        raise ValueError(
            f"pi05_aloha_14d delta->abs requires {ROBOTWIN_ACTION_DIM}D action/state, "
            f"got action={out.size}D state={st.size}D"
        )
    out = out[:ROBOTWIN_ACTION_DIM]
    st = st[:ROBOTWIN_ACTION_DIM]
    out = np.where(_PI05_DELTA_JOINT_MASK, out + st, out)
    return out.astype(np.float32)


def _quat_to_rot6d_interleaved(quat: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to interleaved 6D rotation (official X-VLA layout)."""
    from scipy.spatial.transform import Rotation

    mat = Rotation.from_quat(np.asarray(quat, dtype=np.float64)).as_matrix()
    return mat[:, :2].reshape(6).astype(np.float32)


def _build_ee6d_proprio(
    observation: Dict[str, Any], last_env_action: Optional[np.ndarray] = None
) -> np.ndarray:
    """Build the 20D dual-arm proprio used by the official X-VLA RoboTwin client.

    Layout: [left pos(3) + quat->rot6d(6) + (1 - 2*gripper)(1),
             right pos(3) + quat->rot6d(6) + (1 - 2*gripper)(1)].

    Official closed-loop detail: after the first chunk, the client overwrites
    the endpose with the last commanded ee action (pos+quat), while grippers
    stay measured. ``last_env_action`` is the 16D env action from the previous
    step; when provided, its pose parts replace the measured endpose.
    """
    endpose = observation["endpose"]
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
            _quat_to_rot6d_interleaved(left_ee[3:7]),
            np.asarray([left_grip], dtype=np.float32),
            right_ee[:3],
            _quat_to_rot6d_interleaved(right_ee[3:7]),
            np.asarray([right_grip], dtype=np.float32),
        ]
    ).astype(np.float32)


def _ee6d_action_to_env(raw_action: np.ndarray) -> np.ndarray:
    """Convert a 20D ee6d model action to the 16D RoboTwin ee-control action.

    Official client: per arm xyz(3) + rot6d->quat(4) + (1 - 2*(grip > 0.7))(1),
    executed via ``TASK_ENV.take_action(action, action_type='ee')``.
    """
    from loongforge.embodied.eval.servers.predict_action_interface import rot6d_interleaved_to_quat

    raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)

    def arm(offset: int) -> np.ndarray:
        pos = raw[offset : offset + 3]
        quat = rot6d_interleaved_to_quat(raw[offset + 3 : offset + 9]).reshape(-1)
        grip = np.asarray([1.0 - 2.0 * float(raw[offset + 9] > 0.7)], dtype=np.float32)
        return np.concatenate([pos, quat, grip])

    return np.concatenate([arm(0), arm(10)]).astype(np.float32)


class ModelClient:
    """Provide ModelClient behavior."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 10093,
        unnorm_key: Optional[str] = None,
        task_name: str = "robotwin_task",
        robot_setup: str = "bimanual_dual_arm",
        control_hz: int = 10,
        max_steps: int = ROBOTWIN_DEFAULT_MAX_STEPS,
        action_mode: str = "abs",
        reorder_action: bool = True,
        timeout: float = 300,
        disable_action_cache: bool = False,
        return_action_chunk: bool = False,
        action_bridge: str = "strict_14d",
        trace_path: Optional[str] = None,
        domain_id: Optional[int] = None,
    ) -> None:
        """Run __init__."""
        from loongforge.embodied.eval.transport import PolicyClient

        self.client = PolicyClient(host=host, port=port, timeout=timeout)
        if action_bridge == "ee6d_dual":
            # The adapter only formats observations in this mode; joint-space
            # action modes (and 'ee6d' from model config) do not apply.
            action_mode = "abs"
        if action_bridge == "pi05_aloha_14d":
            # openpi RoboTwin pi05: absolute joint after AbsoluteActions; no
            # RoboTwin-specific joint reorder (layout already left7+right7).
            action_mode = "abs"
            reorder_action = False
        self.adapter = RoboTwinAdapter(
            task_name=task_name,
            robot_setup=robot_setup,
            control_hz=control_hz,
            max_steps=max_steps,
            action_mode=action_mode,
            reorder_action=reorder_action,
        )
        self.unnorm_key = unnorm_key
        self.disable_action_cache = disable_action_cache
        self.return_action_chunk = return_action_chunk
        self.action_bridge = action_bridge
        self.domain_id = domain_id
        self.ee6d_proprio: Optional[np.ndarray] = None
        self.last_ee_env_action: Optional[np.ndarray] = None
        self.chunk_pi_state: Optional[np.ndarray] = None
        self.task_description: Optional[str] = None
        self.episode_id: Optional[str] = None
        self.prev_action: Optional[np.ndarray] = None
        self.initial_joint: Optional[np.ndarray] = None
        self.trace_path = pathlib.Path(trace_path) if trace_path else None
        self.trace_records: List[Dict[str, Any]] = []

    def reset(self, task_description: str = "", episode_id: Optional[str] = None) -> None:
        """Run reset."""
        self.task_description = task_description
        self.episode_id = episode_id or f"robotwin/{self.adapter.task_name}/{task_description or 'default'}"
        self.prev_action = None
        self.initial_joint = None
        self.ee6d_proprio = None
        self.last_ee_env_action = None
        self.chunk_pi_state = None
        self.trace_records = []
        self.client.reset(self.episode_id)
        self._flush_trace()

    def step(self, observation: Dict[str, Any], instruction: str, step: int = 0) -> np.ndarray:
        """Run step."""
        if instruction != self.task_description or self.episode_id is None:
            self.reset(task_description=instruction)

        joint = np.asarray(observation["joint_action"]["vector"], dtype=np.float32).reshape(-1)
        if self.initial_joint is None:
            self.initial_joint = joint.copy()

        canonical_obs = self.adapter.obs_to_canonical(
            observation,
            {
                "instruction": instruction,
                "episode_id": self.episode_id,
                "episode_step": step,
            },
        )
        pi_state: Optional[np.ndarray] = None
        if self.action_bridge == "ee6d_dual":
            # Official X-VLA robotwin protocol: pose part of the proprio uses
            # the last commanded ee action (closed-loop), grippers stay measured.
            self.ee6d_proprio = _build_ee6d_proprio(observation, self.last_ee_env_action)
            state = self.ee6d_proprio
        elif self.action_bridge == "pi05_aloha_14d":
            # openpi AlohaInputs(adapt_to_pi): env joint -> pi space before model.
            pi_state = _adapt_to_pi_decode_state(joint)
            state = pi_state
        else:
            state = canonical_obs.get("model_state")
        extra_kwargs: Dict[str, Any] = {}
        if self.domain_id is not None:
            extra_kwargs["domain_id"] = int(self.domain_id)
        response = self.client.predict_action(
            images=canonical_obs["images"],
            instruction=canonical_obs["instruction"],
            episode_id=canonical_obs["meta"]["episode_id"],
            episode_step=canonical_obs["meta"]["episode_step"],
            state=state,
            meta=canonical_obs["meta"],
            unnorm_key=self.unnorm_key,
            disable_action_cache=self.disable_action_cache,
            return_action_chunk=self.return_action_chunk,
            **extra_kwargs,
        )
        if not response.get("ok", False):
            raise RuntimeError(f"Policy error: {response}")

        if self.action_bridge == "ee6d_dual":
            raw_flat = np.asarray(response["data"]["actions"], dtype=np.float32).reshape(-1)
            if raw_flat.size < ROBOTWIN_EE6D_ACTION_DIM:
                raise ValueError(
                    f"ee6d_dual bridge requires {ROBOTWIN_EE6D_ACTION_DIM}D actions, got {raw_flat.size}D"
                )
            raw_action = raw_flat[:ROBOTWIN_EE6D_ACTION_DIM]
            env_action = _ee6d_action_to_env(raw_action)
            self.last_ee_env_action = env_action.copy()
            self._record_trace(step, instruction, joint, raw_action, raw_action, env_action, response)
            return env_action

        if self.action_bridge == "pi05_aloha_14d":
            # Model returns unnormalized delta joints (q99 done in predict_action).
            # AbsoluteActions uses state at chunk inference time; cached steps
            # reuse that state (openpi applies AbsoluteActions once per chunk).
            raw_action = self._extract_robotwin_action(response["data"]["actions"])
            if response["data"].get("inference_latency_ms") is not None or self.chunk_pi_state is None:
                if pi_state is None:
                    pi_state = _adapt_to_pi_decode_state(joint)
                self.chunk_pi_state = pi_state.copy()
            abs_action = _delta_to_absolute_actions(raw_action, self.chunk_pi_state)
            env_action = _adapt_to_pi_encode_actions(abs_action)
            self._record_trace(step, instruction, joint, raw_action, abs_action, env_action, response)
            return env_action

        raw_action = self._extract_robotwin_action(response["data"]["actions"])
        action_mode = self.adapter.action_mode
        if action_mode == "delta":
            base = self.prev_action if self.prev_action is not None else joint
            output_action = np.asarray(base, dtype=np.float32).reshape(-1) + raw_action
            self.prev_action = output_action.copy()
            env_action = self.adapter.action_from_canonical({"actions": output_action}, {"action_mode": "abs"})
        elif action_mode == "rel":
            output_action = np.asarray(self.initial_joint, dtype=np.float32).reshape(-1) + raw_action
            env_action = self.adapter.action_from_canonical({"actions": output_action}, {"action_mode": "abs"})
        else:
            output_action = raw_action
            env_action = self.adapter.action_from_canonical({"actions": raw_action})
        self._record_trace(step, instruction, joint, raw_action, output_action, env_action, response)
        return env_action

    def _extract_robotwin_action(self, actions: Any) -> np.ndarray:
        """Run _extract_robotwin_action."""
        flat = np.asarray(actions, dtype=np.float32).reshape(-1)
        if flat.size >= ROBOTWIN_ACTION_DIM:
            return flat[:ROBOTWIN_ACTION_DIM]
        if flat.size == 7 and self.action_bridge == "duplicate_7d":
            return np.concatenate([flat, flat], axis=0).astype(np.float32)
        raise ValueError(
            f"RoboTwin requires {ROBOTWIN_ACTION_DIM}D bimanual actions, got {flat.size}D. "
            "Use a 14D RoboTwin pi05 checkpoint, or set action_bridge='duplicate_7d' only for smoke testing."
        )

    def _record_trace(
        self,
        step: int,
        instruction: str,
        joint: np.ndarray,
        raw_action: np.ndarray,
        output_action: np.ndarray,
        env_action: np.ndarray,
        response: Dict[str, Any],
    ) -> None:
        """Append and persist a RoboTwin step trace record."""
        data = response.get("data", {})
        self.trace_records.append(
            {
                "step": int(step),
                "episode_id": self.episode_id,
                "instruction": instruction,
                "state": np.asarray(joint).tolist(),
                "raw_action": np.asarray(raw_action).tolist(),
                "output_action": np.asarray(output_action).tolist(),
                "env_action": np.asarray(env_action).tolist(),
                "action_mode": self.adapter.action_mode,
                "inference_latency_ms": data.get("inference_latency_ms"),
                "timestamp_sec": time.time(),
            }
        )
        self._flush_trace()

    def _flush_trace(self) -> None:
        """Write trace records when a trace path is configured."""
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "robotwin",
            "task_name": self.adapter.task_name,
            "episode_id": self.episode_id,
            "steps": self.trace_records,
        }
        self.trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self) -> None:
        """Run close."""
        self._flush_trace()
        self.client.close()


def get_model(usr_args: Dict[str, Any]) -> ModelClient:
    """Run get_model."""
    return ModelClient(
        host=usr_args.get("host", "127.0.0.1"),
        port=int(usr_args.get("port", 10093)),
        unnorm_key=usr_args.get("unnorm_key"),
        task_name=usr_args.get("task_name") or "robotwin_task",
        robot_setup=usr_args.get("robot_setup", "bimanual_dual_arm"),
        control_hz=int(usr_args.get("control_hz", 10)),
        max_steps=int(usr_args.get("max_steps", ROBOTWIN_DEFAULT_MAX_STEPS)),
        action_mode=usr_args.get("action_mode", "abs"),
        reorder_action=bool(usr_args.get("reorder_action", True)),
        timeout=float(usr_args.get("timeout", 300)),
        disable_action_cache=bool(usr_args.get("disable_action_cache", False)),
        return_action_chunk=bool(usr_args.get("return_action_chunk", False)),
        action_bridge=usr_args.get("action_bridge", "strict_14d"),
        trace_path=usr_args.get("trace_path"),
        domain_id=usr_args.get("domain_id"),
    )


def reset_model(model: ModelClient) -> None:
    """Run reset_model."""
    model.reset(task_description="")


def eval(TASK_ENV: Any, model: ModelClient, observation: Dict[str, Any]) -> None:
    """Run eval."""
    if model.action_bridge == "ee6d_dual":
        # Official X-VLA robotwin client: instruction is the plain task name
        # with underscores replaced (e.g. "adjust bottle"), NOT the
        # env-generated natural-language instruction.
        instruction = model.adapter.task_name.replace("_", " ")
        action = model.step(observation, instruction=instruction, step=TASK_ENV.take_action_cnt)
        TASK_ENV.take_action(action, action_type="ee")
    else:
        instruction = str(TASK_ENV.get_instruction())
        action = model.step(observation, instruction=instruction, step=TASK_ENV.take_action_cnt)
        TASK_ENV.take_action(action)
