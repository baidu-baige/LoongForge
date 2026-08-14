# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""RoboTwin official-evaluator policy bridge for the standalone eval module.

RoboTwin is a "form B" benchmark: the eval module reuses RoboTwin's official
evaluator, which imports a policy plugin by ``policy_name`` and reverse-calls
it. This bridge is the thin shell that hosts the shared 4-component chain
(adapter -> PayloadBuilder -> PolicyClient -> ActionDecoder) inside the
RoboTwin subprocess.

Per-model protocol wiring:
- ``ee6d_dual``       -> XVLAPayloadBuilder(state_encoding="ee6d_dual")
                         + RoboTwinEe6dDualDecoder
- ``pi05_aloha_14d``  -> Pi05PayloadBuilder(state_encoding="aloha_pi")
                         + RoboTwinPi05AlohaDecoder
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict, List, Optional

import numpy as np

from loongforge.embodied.eval.action_decoders import build_action_decoder
from loongforge.embodied.eval.adapters.robotwin import ROBOTWIN_DEFAULT_MAX_STEPS, RoboTwinAdapter
from loongforge.embodied.eval.payload_builders import build_payload_builder

# action_bridge -> (model_type, payload-builder state_encoding, decoder key).
_BRIDGE_WIRING = {
    "ee6d_dual": ("xvla", "ee6d_dual", "ee6d_robotwin_ee_dual"),
    "pi05_aloha_14d": ("pi05", "aloha_pi", "pi05_aloha_robotwin"),
    # LingBot-VA consumes no proprio (images + instruction only) and emits a
    # dual-arm ee pose relative to the episode's initial endpose.
    "lingbot_va_ee_quat_16d": ("lingbot_va", "", "lingbot_va_robotwin_ee_dual"),
}


class ModelClient:
    """Thin RoboTwin plugin shell hosting the shared 4-component eval chain."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 10093,
        unnorm_key: Optional[str] = None,
        task_name: str = "robotwin_task",
        robot_setup: str = "bimanual_dual_arm",
        control_hz: int = 10,
        max_steps: int = ROBOTWIN_DEFAULT_MAX_STEPS,
        timeout: float = 300,
        disable_action_cache: bool = False,
        return_action_chunk: bool = False,
        action_bridge: str = "ee6d_dual",
        trace_path: Optional[str] = None,
        domain_id: Optional[int] = None,
    ) -> None:
        """Run __init__."""
        from loongforge.embodied.eval.transport import PolicyClient

        if action_bridge not in _BRIDGE_WIRING:
            raise ValueError(
                f"Unsupported RoboTwin action_bridge {action_bridge!r}. "
                f"Supported: {sorted(_BRIDGE_WIRING)}"
            )
        self.client = PolicyClient(host=host, port=port, timeout=timeout)
        # ee6d_dual / pi05_aloha_14d own the action protocol via the paired
        # PayloadBuilder + ActionDecoder; the adapter only formats observations.
        self.adapter = RoboTwinAdapter(
            task_name=task_name,
            robot_setup=robot_setup,
            control_hz=control_hz,
            max_steps=max_steps,
        )
        self.unnorm_key = unnorm_key
        self.disable_action_cache = disable_action_cache
        self.return_action_chunk = return_action_chunk
        self.action_bridge = action_bridge
        self.domain_id = domain_id

        model_type, state_encoding, decoder_key = _BRIDGE_WIRING[action_bridge]
        yaml_model: Dict[str, Any] = {"state_encoding": state_encoding}
        if unnorm_key:
            yaml_model["unnorm_key"] = unnorm_key
        if domain_id is not None:
            yaml_model["domain_id"] = int(domain_id)
        self.payload_builder = build_payload_builder(model_type, yaml_model=yaml_model)
        # Closed-loop-within-chunk models (PayloadBuilder capability) must be
        # called every env step or their KV cache never sees the intermediate
        # frames. The YAML flag can only turn the cache off, never back on.
        if getattr(self.payload_builder, "disable_action_cache", False):
            self.disable_action_cache = True
        self.decoder = build_action_decoder(decoder_key)

        self.task_description: Optional[str] = None
        self.episode_id: Optional[str] = None
        self.trace_path = pathlib.Path(trace_path) if trace_path else None
        self.trace_records: List[Dict[str, Any]] = []

    def reset(self, task_description: str = "", episode_id: Optional[str] = None) -> None:
        """Run reset."""
        self.task_description = task_description
        self.episode_id = episode_id or f"robotwin/{self.adapter.task_name}/{task_description or 'default'}"
        self.payload_builder.reset(self.episode_id)
        self.decoder.reset()
        self.trace_records = []
        self.client.reset(self.episode_id)
        self._flush_trace()

    def step(self, observation: Dict[str, Any], instruction: str, step: int = 0) -> np.ndarray:
        """Run one plugin step: adapter -> PayloadBuilder -> client -> decoder."""
        if instruction != self.task_description or self.episode_id is None:
            self.reset(task_description=instruction)

        joint = np.asarray(observation["joint_action"]["vector"], dtype=np.float32).reshape(-1)
        canonical_obs = self.adapter.obs_to_canonical(
            observation,
            {
                "instruction": instruction,
                "episode_id": self.episode_id,
                "episode_step": step,
            },
        )
        ctx = {
            "benchmark_name": "robotwin",
            "episode_id": self.episode_id,
            "episode_step": step,
            "instruction": instruction,
        }
        model_kwargs = self.payload_builder.build(canonical_obs, ctx)
        response = self.client.predict_action(
            episode_id=canonical_obs["meta"]["episode_id"],
            episode_step=canonical_obs["meta"]["episode_step"],
            disable_action_cache=self.disable_action_cache,
            return_action_chunk=self.return_action_chunk,
            **model_kwargs,
        )
        if not response.get("ok", False):
            raise RuntimeError(f"Policy error: {response}")

        data = response["data"]
        raw_chunk = np.asarray(data["actions"], dtype=np.float32).reshape(1, -1)
        decode_ctx = {
            "current_joint": joint,
            # aloha_pi decoder anchors delta->abs on the pi-space proprio the
            # Pi05PayloadBuilder produced (== adapt_to_pi_decode_state(joint)).
            # For ee6d_dual the decoder ignores ctx, so this is harmless.
            "pi_state": model_kwargs.get("state"),
            # lingbot_va anchors its relative dual-arm ee pose on the endpose of
            # the episode's first step; the other decoders ignore this key.
            "endpose": canonical_obs["state_raw"].get("endpose"),
            "is_fresh_chunk": data.get("inference_latency_ms") is not None,
        }
        env_action = np.asarray(self.decoder(raw_chunk, decode_ctx), dtype=np.float32)[0]
        # Closed-loop endpose backfill for ee6d_dual (no-op for aloha_pi).
        self.payload_builder.note_env_action(env_action)
        self._record_trace(step, instruction, joint, raw_chunk[0], env_action, response)
        return env_action

    def _record_trace(
        self,
        step: int,
        instruction: str,
        joint: np.ndarray,
        raw_action: np.ndarray,
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
                "env_action": np.asarray(env_action).tolist(),
                "action_bridge": self.action_bridge,
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
        timeout=float(usr_args.get("timeout", 300)),
        disable_action_cache=bool(usr_args.get("disable_action_cache", False)),
        return_action_chunk=bool(usr_args.get("return_action_chunk", False)),
        action_bridge=usr_args.get("action_bridge", "ee6d_dual"),
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
    elif model.action_bridge == "lingbot_va_ee_quat_16d":
        # Official LingBot-VA robotwin client uses the env instruction and
        # commands absolute ee poses (its 16D output is composed onto the
        # episode's initial endpose by the decoder).
        instruction = str(TASK_ENV.get_instruction())
        action = model.step(observation, instruction=instruction, step=TASK_ENV.take_action_cnt)
        TASK_ENV.take_action(action, action_type="ee")
    else:
        instruction = str(TASK_ENV.get_instruction())
        action = model.step(observation, instruction=instruction, step=TASK_ENV.take_action_cnt)
        TASK_ENV.take_action(action)

