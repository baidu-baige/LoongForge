# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""RoboDojo XPolicyLab policy bridge for the standalone eval module.

RoboDojo is a "form B" benchmark like RoboTwin: its official Isaac Sim client
drives the episode and calls an XPolicyLab policy server, which instantiates
``XPolicyLab.policy.<name>.model.Model(deploy_cfg)``. The LoongForge plugin
directory (``bridges/robodojo_xpolicylab``) exposes :class:`RoboDojoPolicyModel`
as that ``Model``; it hosts the shared 4-component chain
(adapter -> PayloadBuilder -> PolicyClient -> ActionDecoder) and forwards every
env step to the LoongForge policy server.

Per-model protocol wiring (``action_bridge`` -> model_type, decoder key,
raw-action channels, RoboDojo action space, PayloadBuilder defaults):

- ``lingbot_va_joint30``: LingBot-VA's 30-D action layout (left/right ee 7+7,
  left/right joints 7+7, left/right gripper 1+1). The official RoboDojo
  LingBot_VA client executes channels [14..19, 28, 21..26, 29] as joint control.
  The model sees every env step (its own action queue), so the server-side
  chunk cache is off.
- ``pi05_joint14``: pi0.5 as the official RoboDojo Pi_05 adapter (openpi aloha protocol,
  ``adapt_to_pi=False``): three cameras resized the openpi way (antialiased, uint8) to 224,
  the 14-D joint state passed through, delta arm
  joints made absolute against the chunk's state, 14-D joint control.
- ``xvla_ee20``: X-VLA's 20-D ee6d action executed as ee control, as the official
  RoboDojo X_VLA adapter does: head camera only, 20-D proprio from the measured
  ee poses and raw gripper joints, ``domain_id`` 0. One inference per chunk; the
  server-side chunk cache serves the rest of the chunk step by step.
- ``fastwam_joint14``: FastWAM as the official RoboDojo FastWAM adapter (its RoboTwin
  ``deploy_policy`` reused): three cameras INTER_AREA-resized to 320x240, then packed to the
  384x320 T layout model-side; the raw 14-D joint state, z-score normalized; 24 of the 32
  absolute joint targets executed per chunk.

Chunk boundaries: the official ``deploy.py`` of both policies executes one predicted
chunk per ``get_action`` and calls ``is_episode_end()`` once more at every chunk
end. The plugin's ``deploy.py`` replays that call order, so the bridge reports
``chunk_end`` from the model's fixed chunk schedule (first chunk
``first_chunk_len`` actions, then ``chunk_len``).

Episode-end checks: ``is_episode_end()`` also latches success, so when it is called matters.
The official LingBot_VA / Pi_05 / X_VLA ``deploy.py`` call it after every action; the official
FastWAM ``deploy.py`` only at chunk ends (``_EPISODE_END_CHECK``). The bridge reports the mode
with every action and the plugin ``deploy.py`` follows it.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from loongforge.embodied.eval.action_decoders import build_action_decoder
from loongforge.embodied.eval.adapters.robodojo import RoboDojoAdapter
from loongforge.embodied.eval.payload_builders import build_payload_builder

LINGBOT_VA_JOINT_CHANNELS = [14, 15, 16, 17, 18, 19, 28, 21, 22, 23, 24, 25, 26, 29]

# action_bridge -> (model_type, decoder key, raw channels, RoboDojo action space, PayloadBuilder defaults).
_BRIDGE_WIRING = {
    "lingbot_va_joint30": ("lingbot_va", "", LINGBOT_VA_JOINT_CHANNELS, "joint14", {}),
    "pi05_joint14": ("pi05", "pi05_aloha_robodojo", None, "joint14", {"state_encoding": "passthrough", "action_dim": 14, "openpi_image_size": 224}),
    "xvla_ee20": (
        "xvla",
        "",
        None,
        "ee20",
        {"state_encoding": "ee6d_robodojo", "max_image_views": 1, "domain_id": 0},
    ),
    "fastwam_joint14": (
        "fastwam",
        "",
        None,
        "joint14",
        {"state_encoding": "robotwin_joint14", "action_dim": 14, "pre_resize_width": 320, "pre_resize_height": 240},
    ),
}

# action_bridge -> when the plugin deploy.py calls is_episode_end(): "action" (after every
# action, the default) or "chunk" (only at chunk ends).
_EPISODE_END_CHECK = {"fastwam_joint14": "chunk"}


class ChunkSchedule:
    """Fixed chunk lengths: ``first`` actions for the first chunk, ``rest`` afterwards."""

    def __init__(self, first: int, rest: int) -> None:
        """Run __init__."""
        if first <= 0 or rest <= 0:
            raise ValueError(f"chunk lengths must be positive, got first={first} rest={rest}")
        self.first = int(first)
        self.rest = int(rest)

    def is_chunk_end(self, step: int) -> bool:
        """Return True when action number ``step`` (0-based) is the last of its chunk."""
        if step < self.first:
            return step == self.first - 1
        return (step - self.first) % self.rest == self.rest - 1


class RoboDojoPolicyModel:
    """XPolicyLab ``Model`` that forwards RoboDojo env steps to a LoongForge policy server."""

    def __init__(self, deploy_cfg: Dict[str, Any]) -> None:
        """Build the chain from the XPolicyLab deploy config (deploy.yml + overrides)."""
        from loongforge.embodied.eval.transport import PolicyClient

        cfg = dict(deploy_cfg or {})
        self.action_bridge = str(cfg.get("action_bridge", "lingbot_va_joint30"))
        if self.action_bridge not in _BRIDGE_WIRING:
            raise ValueError(
                f"Unsupported RoboDojo action_bridge {self.action_bridge!r}. Supported: {sorted(_BRIDGE_WIRING)}"
            )
        model_type, decoder_key, default_channels, action_space, builder_defaults = _BRIDGE_WIRING[self.action_bridge]
        channels = cfg.get("action_channel_ids") or default_channels
        self.task_name = str(cfg.get("task_name") or "robodojo_task")
        self.seed = int(cfg.get("seed") or 0)
        self.adapter = RoboDojoAdapter(
            task_name=self.task_name,
            env_cfg_type=str(cfg.get("env_cfg_type") or "arx_x5"),
            seed=self.seed,
            action_channel_ids=channels,
            action_space=action_space,
        )
        builder_cfg = {**builder_defaults, **dict(cfg.get("payload_builder") or {})}
        self.payload_builder = build_payload_builder(model_type, yaml_model=builder_cfg)
        self.decoder = build_action_decoder(decoder_key)
        # LingBot-VA must see every env step (closed loop within a chunk); chunked models keep it on.
        self.disable_action_cache = bool(
            cfg.get("disable_action_cache", False) or getattr(self.payload_builder, "disable_action_cache", False)
        )
        self.schedule = ChunkSchedule(int(cfg["first_chunk_len"]), int(cfg["chunk_len"]))
        self.episode_end_check = _EPISODE_END_CHECK.get(self.action_bridge, "action")
        self.client = PolicyClient(
            host=str(cfg.get("lf_host", "127.0.0.1")),
            port=int(cfg["lf_port"]),
            timeout=float(cfg.get("lf_timeout_sec", 600)),
            # A chunk-boundary call can exceed the 20 s keepalive (official recipe: VAE / T5 on
            # CPU); the official LingBot_VA client also runs with keepalive off.
            ping_interval=None,
            ping_timeout=None,
        )
        self.trace_dir = pathlib.Path(cfg["trace_dir"]) if cfg.get("trace_dir") else None

        self.episode_index = -1
        self.episode_id: Optional[str] = None
        self.episode_step = 0
        self._new_episode_pending = True
        self._obs: Optional[Dict[str, Any]] = None
        self._trace: List[Dict[str, Any]] = []

    # ── XPolicyLab Model interface ───────────────────────────────────
    def reset(self) -> None:
        """Mark that the next step starts a new episode (RoboDojo calls reset twice per episode)."""
        self._flush_trace()
        self._new_episode_pending = True
        self._obs = None

    def update_obs(self, obs: Dict[str, Any]) -> None:
        """Keep the latest observation (generic XPolicyLab call order)."""
        self._obs = obs

    def get_action(self, obs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return ``{"actions": [one action dict], "chunk_end": bool, "end_check": str}`` for the current observation."""
        if obs is not None:
            self._obs = obs
        if self._obs is None:
            raise RuntimeError("get_action called before any observation")
        if self._new_episode_pending:
            self._start_episode()
        step = self.episode_step
        instruction = str(self._obs.get("instruction") or "")
        canonical = self.adapter.obs_to_canonical(
            self._obs, {"instruction": instruction, "episode_id": self.episode_id, "episode_step": step}
        )
        ctx = {
            "benchmark_name": "robodojo",
            "episode_id": self.episode_id,
            "episode_step": step,
            "instruction": instruction,
        }
        model_kwargs = self.payload_builder.build(canonical, ctx)
        start = time.perf_counter()
        response = self.client.predict_action(
            episode_id=self.episode_id,
            episode_step=step,
            disable_action_cache=self.disable_action_cache,
            **model_kwargs,
        )
        if not response.get("ok", False):
            raise RuntimeError(f"Policy error: {response}")
        data = response["data"]
        raw = np.asarray(data["actions"], dtype=np.float32).reshape(1, -1)
        decode_ctx = {
            "current_joint": canonical["state_raw"]["joint"],
            "pi_state": model_kwargs.get("state"),
            # A model forward (not a server chunk-cache hit) starts a new chunk.
            "is_fresh_chunk": data.get("inference_latency_ms") is not None,
        }
        decoded = np.asarray(self.decoder(raw, decode_ctx), dtype=np.float32)
        action = self.adapter.action_from_canonical(decoded[0])
        self.payload_builder.note_env_action(decoded[0])
        chunk_end = self.schedule.is_chunk_end(step)
        self._record(step, instruction, raw[0], action, chunk_end, data, time.perf_counter() - start)
        if chunk_end:
            # Flush per chunk: the policy server is killed after the last episode, so a
            # flush on the next reset alone would drop the final episode's trace.
            self._flush_trace()
        self.episode_step += 1
        self._obs = None
        return {"actions": [action], "chunk_end": chunk_end, "end_check": self.episode_end_check}

    def prepare_case(self, case_meta=None) -> None:
        """Run prepare_case (no-op)."""

    def on_trial_end(self, result=None) -> None:
        """Run on_trial_end."""
        self._flush_trace()

    # ── internals ────────────────────────────────────────────────────
    def _start_episode(self) -> None:
        self.episode_index += 1
        self.episode_id = f"robodojo/{self.task_name}/seed{self.seed}/ep{self.episode_index:04d}"
        self.episode_step = 0
        self._new_episode_pending = False
        self._trace = []
        self.payload_builder.reset(self.episode_id)
        self.decoder.reset()
        self.client.reset(self.episode_id)

    def _record(self, step, instruction, raw, action, chunk_end, data, rtt_sec) -> None:
        if self.trace_dir is None:
            return
        self._trace.append(
            {
                "step": int(step),
                "instruction": instruction if step == 0 else None,
                "raw_action": np.asarray(raw).tolist(),
                "env_action": action,
                "chunk_end": bool(chunk_end),
                "inference_latency_ms": data.get("inference_latency_ms"),
                "rtt_ms": round(rtt_sec * 1000.0, 3),
            }
        )

    def _flush_trace(self) -> None:
        # Appends the buffered steps (one JSON line each) to ep<NNNN>.jsonl at every
        # chunk end and at reset / trial end.
        if self.trace_dir is None or not self._trace or self.episode_id is None:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"ep{self.episode_index:04d}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for record in self._trace:
                handle.write(json.dumps({"episode_id": self.episode_id, **record}, ensure_ascii=False) + "\n")
        self._trace = []


def chunk_lengths_from_model(model_cfg: Dict[str, Any]) -> Sequence[int]:
    """LingBot-VA chunk lengths: the first chunk drops frame 0's actions."""
    frame_chunk_size = int(model_cfg["frame_chunk_size"])
    action_per_frame = int(model_cfg["action_per_frame"])
    return action_per_frame * (frame_chunk_size - 1), action_per_frame * frame_chunk_size
