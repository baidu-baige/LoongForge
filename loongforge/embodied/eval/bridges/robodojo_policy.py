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
- ``pi05_sf_joint14``: Spatial_Forcing as the official RoboDojo Spatial_Forcing adapter (open_sf
  ``pi05sf_jax_robodojo_v21_offcache``): the pi05_joint14 inputs, the 14 joints executed as
  predicted (absolute, ``use_delta_joint_actions=False``), whole 50-step chunk.
- ``xvla_ee20``: X-VLA's 20-D ee6d action executed as ee control, as the official
  RoboDojo X_VLA adapter does: head camera only, 20-D proprio from the measured
  ee poses and raw gripper joints, ``domain_id`` 0. One inference per chunk; the
  server-side chunk cache serves the rest of the chunk step by step.
- ``fastwam_joint14``: FastWAM as the official RoboDojo FastWAM adapter (its RoboTwin
  ``deploy_policy`` reused): three cameras INTER_AREA-resized to 320x240, then packed to the
  384x320 T layout model-side; the raw 14-D joint state, z-score normalized; 24 of the 32
  absolute joint targets executed per chunk.
- ``gr00t_n17_joint14``: GR00T-N1.7 as the official RoboDojo GR00T_N17 adapter: the three
  env-native views letterboxed to squares (as its vendored GR00T eval transform does) and the
  raw 14-D joint state (embodiment ``robodojo_arx_x5``, the
  checkpoint's NEW_EMBODIMENT projector); the 16-step chunk of relative joints is made absolute
  inside the model and executed in full.
- ``gr00t_n16_joint14``: GR00T-N1.6 with the same RoboDojo conventions as ``gr00t_n17_joint14``
  (there is no official N1.6 RoboDojo plugin or checkpoint); the N1.6 model applies the
  letterbox in its own eval image transform, so the builder sends the env-native views.
- ``dreamzero_joint14``: DreamZero-AgiBot as the official RoboDojo DreamZero adapter: called on
  every env step (the server keeps the last 4 frames of [head, left, right] and the 24-step
  chunk), AgiBot state / action mapping and four-view prompt server-side (eval_protocol
  ``robodojo_agibot``); 24 absolute 14-D joint targets executed per chunk.

Chunk boundaries: the official ``deploy.py`` of both policies executes one predicted
chunk per ``get_action`` and calls ``is_episode_end()`` once more at every chunk
end. The plugin's ``deploy.py`` replays that call order, so the bridge reports
``chunk_end`` from the model's fixed chunk schedule (first chunk
``first_chunk_len`` actions, then ``chunk_len``).

Episode-end checks: ``is_episode_end()`` also latches success, so when it is called matters.
The official LingBot_VA / Pi_05 / X_VLA ``deploy.py`` call it after every action; the official
FastWAM ``deploy.py`` only at chunk ends (``_EPISODE_END_CHECK``). The bridge reports the mode
with every action and the plugin ``deploy.py`` follows it.

``eval_batch`` (several envs in one Isaac Sim process, plugin ``robodojo_xpolicylab_batch``):
RoboDojo's client then calls ``update_obs_batch`` / ``get_action_batch`` instead, and expects a
whole chunk per env (the official ``eval_one_episode_batch`` executes it step by step). Each env
gets its own episode id, PayloadBuilder, ActionDecoder and trace file; one request per env per
chunk, with the chunk returned by the server instead of cached there. One env per process
(``eval_batch: false``) keeps the per-step protocol above unchanged.
"""

from __future__ import annotations

import dataclasses
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
    # Spatial Forcing (pi0.5 + a training-only VGGT alignment loss): same model and inputs as
    # pi05_joint14, but trained with use_delta_joint_actions=False, so the 14 joints come out absolute.
    "pi05_sf_joint14": ("pi05", "", list(range(14)), "joint14", {"state_encoding": "passthrough", "action_dim": 14, "openpi_image_size": 224}),
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
    "gr00t_n17_joint14": (
        "gr00tn1d7",
        "",
        None,
        "joint14",
        {"state_encoding": "robodojo_joint14", "action_dim": 14, "action_horizon": 16, "letterbox_pad": True},
    ),
    "gr00t_n16_joint14": (
        "gr00tn1d6",
        "",
        None,
        "joint14",
        {"state_encoding": "robodojo_joint14", "action_dim": 14, "action_horizon": 16},
    ),
    "dreamzero_joint14": (
        "dreamzero",
        "",
        None,
        "joint14",
        {"state_encoding": "robodojo_joint14", "action_dim": 14, "action_horizon": 24},
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


@dataclasses.dataclass
class _EnvState:
    """Per-env chain state for ``eval_batch``: episode id, step, its own builder / decoder / trace."""

    idx: int
    batch: int
    episode_id: str
    builder: Any
    decoder: Any
    step: int = 0
    trace: List[Dict[str, Any]] = dataclasses.field(default_factory=list)


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
        # eval_batch builds one chain per env from the same spec.
        self._chain_spec = (model_type, builder_cfg, decoder_key)
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

        # eval_batch: one chain per env, keyed by the env index the client reports.
        self._envs: Dict[int, _EnvState] = {}
        self._batch_obs: Dict[int, Dict[str, Any]] = {}
        self._batch_index = -1

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
        for env in self._envs.values():
            self._flush_env_trace(env)

    # ── eval_batch (several envs in one Isaac Sim process) ──────────────
    def update_obs_batch(self, obs: List[Dict[str, Any]]) -> None:
        """Keep the latest observation of every env in the batch (keyed by ``env_idx``)."""
        if self._new_episode_pending:
            self._start_batch()
        for item in obs:
            self._batch_obs[int(item["env_idx"])] = item

    def get_action_batch(self, obs: Sequence[int]) -> List[List[Dict[str, Any]]]:
        """Per env in ``obs`` (env indices, the official call convention): the rest of its chunk.

        One request per env per chunk: the official ``eval_one_episode_batch`` asks once at a
        chunk start and executes the returned actions step by step. Every env of a batch starts
        its episode together, so they all sit at the same step of the same chunk schedule.
        """
        env_idx_list = [int(i) for i in obs]
        if self._new_episode_pending:
            self._start_batch()
        for idx in env_idx_list:
            if idx not in self._envs or self._envs[idx].batch != self._batch_index:
                self._envs[idx] = self._new_env(idx)
        return [self._env_chunk(idx) for idx in env_idx_list]

    def _start_batch(self) -> None:
        for env in self._envs.values():
            self._flush_env_trace(env)
        self._batch_index += 1
        self._batch_obs = {}
        self._new_episode_pending = False

    def _new_env(self, idx: int) -> _EnvState:
        model_type, builder_cfg, decoder_key = self._chain_spec
        env = _EnvState(
            idx=idx,
            batch=self._batch_index,
            episode_id=f"robodojo/{self.task_name}/seed{self.seed}/b{self._batch_index:04d}e{idx:02d}",
            builder=build_payload_builder(model_type, yaml_model=builder_cfg),
            decoder=build_action_decoder(decoder_key),
        )
        env.builder.reset(env.episode_id)
        env.decoder.reset()
        self.client.reset(env.episode_id)
        return env

    # ── internals ────────────────────────────────────────────────────
    def _env_chunk(self, idx: int) -> List[Dict[str, Any]]:
        """One request for env ``idx``: the server returns the chunk, decoded here action by action."""
        env = self._envs[idx]
        obs = self._batch_obs.get(idx)
        if obs is None:
            raise RuntimeError(f"get_action_batch: no observation for env {idx}")
        step = env.step
        instruction = str(obs.get("instruction") or "")
        canonical = self.adapter.obs_to_canonical(
            obs, {"instruction": instruction, "episode_id": env.episode_id, "episode_step": step}
        )
        ctx = {
            "benchmark_name": "robodojo",
            "episode_id": env.episode_id,
            "episode_step": step,
            "instruction": instruction,
        }
        model_kwargs = env.builder.build(canonical, ctx)
        start = time.perf_counter()
        response = self.client.predict_action(
            episode_id=env.episode_id,
            episode_step=step,
            disable_action_cache=False,
            return_action_chunk=True,
            **model_kwargs,
        )
        if not response.get("ok", False):
            raise RuntimeError(f"Policy error: {response}")
        data = response["data"]
        rows = np.asarray(data["actions"], dtype=np.float32)
        rows = rows.reshape(-1, rows.shape[-1]) if rows.ndim > 1 else rows.reshape(1, -1)
        n_exec = 1
        while not self.schedule.is_chunk_end(step + n_exec - 1):
            n_exec += 1
        if rows.shape[0] < n_exec:
            raise RuntimeError(f"server returned {rows.shape[0]} actions, chunk needs {n_exec}")
        rtt = time.perf_counter() - start
        actions = []
        for k in range(n_exec):
            raw = rows[k : k + 1]
            decode_ctx = {
                "current_joint": canonical["state_raw"]["joint"],
                "pi_state": model_kwargs.get("state"),
                "is_fresh_chunk": k == 0,
            }
            decoded = np.asarray(env.decoder(raw, decode_ctx), dtype=np.float32)
            action = self.adapter.action_from_canonical(decoded[0])
            env.builder.note_env_action(decoded[0])
            self._record_env(env, step + k, instruction, raw[0], action, data if k == 0 else {}, rtt if k == 0 else 0.0)
            actions.append(action)
        env.step += n_exec
        self._flush_env_trace(env)
        return actions

    def _record_env(self, env, step, instruction, raw, action, data, rtt_sec) -> None:
        if self.trace_dir is None:
            return
        env.trace.append(
            {
                "step": int(step),
                "instruction": instruction if step == 0 else None,
                "raw_action": np.asarray(raw).tolist(),
                "env_action": action,
                "chunk_end": bool(self.schedule.is_chunk_end(step)),
                "inference_latency_ms": data.get("inference_latency_ms"),
                "rtt_ms": round(rtt_sec * 1000.0, 3),
            }
        )

    def _flush_env_trace(self, env: _EnvState) -> None:
        if self.trace_dir is None or not env.trace:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"b{env.batch:04d}e{env.idx:02d}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for record in env.trace:
                handle.write(json.dumps({"episode_id": env.episode_id, **record}, ensure_ascii=False) + "\n")
        env.trace = []

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
