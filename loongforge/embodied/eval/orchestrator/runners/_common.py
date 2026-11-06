# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for orchestrator benchmark runners.

Centralizes logic that was duplicated across the per-benchmark runners:
SAPIEN/Vulkan runtime bootstrap, numeric/JSON utilities, signal-based
timeouts, and artifact (GIF replay / JSON trace) IO.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import signal
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from loongforge.embodied.eval.action_decoders import build_action_decoder
from loongforge.embodied.eval.orchestrator.config import (
    build_payload_builder_from_config,
    resolve_action_decoder_key,
)


# --------------------------------------------------------------------------- #
# Policy stack assembly (shared by all in-process runners)
# --------------------------------------------------------------------------- #
def build_policy_stack(args: argparse.Namespace, adapter: Any):
    """Build the (payload_builder, decoder_key, action_decoder) triple.

    Depends only on ``args`` (for ``raw_config`` / ``model_type``) and the
    benchmark ``adapter``; the caller owns adapter/env/client construction.
    """
    payload_builder = build_payload_builder_from_config(
        getattr(args, "raw_config", {}) or {},
        model_type_override=getattr(args, "model_type", None),
    )
    action_decoder_key = resolve_action_decoder_key(payload_builder, adapter)
    action_decoder = build_action_decoder(action_decoder_key)
    return payload_builder, action_decoder_key, action_decoder



# --------------------------------------------------------------------------- #
# SAPIEN / Vulkan runtime bootstrap (ManiSkill, SimplerEnv)
# --------------------------------------------------------------------------- #
def vulkan_runtime_env(args: argparse.Namespace) -> Dict[str, str]:
    """Build environment variables required before importing SAPIEN."""
    env = os.environ.copy()
    library_paths = env.get("LD_LIBRARY_PATH", "").split(":") if env.get("LD_LIBRARY_PATH") else []
    for path in reversed([args.nvidia_lib_dir or "/path/to/nvidia_lib", "/usr/lib64"]):
        if path and path not in library_paths:
            library_paths.insert(0, path)
    env["LD_LIBRARY_PATH"] = ":".join(library_paths)
    env["VK_ICD_FILENAMES"] = args.nvidia_icd_json or env.get("VK_ICD_FILENAMES", "/path/to/nvidia_lib/10_nvidia.json")
    runtime_dir = pathlib.Path(env.get("XDG_RUNTIME_DIR") or f"/tmp/runtime-{os.getuid()}")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.chmod(stat.S_IRWXU)
    env["XDG_RUNTIME_DIR"] = str(runtime_dir)
    return env


def ensure_vulkan_runtime(args: argparse.Namespace, marker: str) -> None:
    """Re-exec the process once so Vulkan library paths load before SAPIEN.

    ``marker`` is a per-benchmark env flag that prevents infinite re-exec.
    """
    if os.environ.get(marker) == "1":
        return
    env = vulkan_runtime_env(args)
    env[marker] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


# --------------------------------------------------------------------------- #
# Numeric / JSON utilities
# --------------------------------------------------------------------------- #
def avg(values: List[Optional[float]]) -> Optional[float]:
    """Return the mean of the non-None values, or None when empty."""
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def json_safe(value: Any) -> Any:
    """Convert tensors and arrays in nested values to JSON-safe objects."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# --------------------------------------------------------------------------- #
# Output-directory guards (shared by LIBERO / CALVIN)
# --------------------------------------------------------------------------- #
def guard_output_dir(args: argparse.Namespace) -> pathlib.Path:
    """Refuse to append to a results directory unless resume or restart is set.

    Summaries do not de-duplicate ``(task, episode)`` keys, so silently
    appending to an existing ``results.jsonl`` inflates or corrupts the
    success rate. Only an explicit ``resume`` or ``restart`` is allowed.
    """
    output_dir = pathlib.Path(args.output_dir)
    results_path = output_dir / "results.jsonl"
    if results_path.exists() and not getattr(args, "restart", False) and not getattr(args, "resume", False):
        raise SystemExit(
            f"{results_path} already exists; set run.resume=true to continue it, "
            "run.restart=true to replace it, or choose a new output_dir"
        )
    return output_dir


def resolve_min_episodes(args: argparse.Namespace) -> int:
    """Return the per-task completeness floor (0 means 'use episodes_per_task')."""
    configured = int(getattr(args, "min_episodes_per_task", 0) or 0)
    if configured > 0:
        return configured
    return int(getattr(args, "episodes_per_task", 0) or 0)
# --------------------------------------------------------------------------- #
# Client-side action chunks (T-14: 10 requests -> 1)
# --------------------------------------------------------------------------- #
class ChunkConsumer:
    """Hold one action chunk client-side and serve its actions step by step.

    The server returns a whole chunk when ``return_action_chunk=True``. Keeping
    it in the runner removes the per-step RPC (and its full observation
    payload); the model still sees exactly the same requests as before, so the
    executed actions are unchanged.

    ``disable_action_cache`` models are closed-loop within a chunk: the runner
    asks for a fresh request every env step and consumes only the first action.
    """

    def __init__(self) -> None:
        self.requests = 0
        self.local_steps = 0
        self._chunk: Optional[np.ndarray] = None
        self._index = 0

    def reset(self) -> None:
        """Drop the current chunk (call at episode start / after client.reset)."""
        self._chunk = None
        self._index = 0
        self.requests = 0
        self.local_steps = 0

    def needs_request(self, disable_action_cache: bool = False) -> bool:
        """Return True when the runner must issue a new inference request."""
        return (
            disable_action_cache
            or self._chunk is None
            or self._index >= self._chunk.shape[0]
        )

    def load(self, chunk: Any, limit: Optional[int] = None) -> None:
        """Install a freshly returned chunk and reset the cursor."""
        actions = np.asarray(chunk, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if limit is not None and int(limit) > 0:
            actions = actions[: int(limit)]
        self._chunk = actions
        self._index = 0
        self.requests += 1

    def next_action(self, disable_action_cache: bool = False) -> np.ndarray:
        """Return the next action as ``[1, action_dim]``.

        ``disable_action_cache`` requests are single-step by construction: the
        cursor is invalidated so the next call triggers a new request.
        """
        if self._chunk is None or self._index >= self._chunk.shape[0]:
            raise RuntimeError("ChunkConsumer.next_action() called without a loaded chunk")
        action = self._chunk[self._index : self._index + 1]
        if disable_action_cache:
            self._index = self._chunk.shape[0]
        else:
            self._index += 1
            self.local_steps += 1
        return action


@dataclass(frozen=True)
class PolicyStep:
    """One env step's action plus the inference request that produced it."""

    action: np.ndarray
    issued_request: bool
    request_step: Optional[int]
    request_id: Optional[str]
    inference_latency_ms: Optional[float]
    e2e_latency_ms: Optional[float]
    #: The full array returned by the server on this request (``None`` on cache
    #: hits). Action-ensemble loops need the whole chunk, not just the slice.
    raw_actions: Optional[np.ndarray] = None


class EpisodePolicyStepper:
    """Own the client-side request cadence for one episode.

    Runners call :meth:`next_action` once per env step and supply a
    ``request_fn(request_step)`` that issues one RPC and returns
    ``(data, e2e_latency_ms)``, where ``data`` is ``response["data"]`` and the
    latency covers the RPC only (not local payload assembly, matching the
    metric recorded before this helper existed). The stepper decides whether
    that step needs a fresh inference (chunk exhausted, or
    ``disable_action_cache``), buffers the returned action chunk, and hands
    back one action per step.

    This is the single place where "when to request / how to buffer" lives, so
    every benchmark runner gets the T-14 client-owned chunk path instead of
    only LIBERO. With ``return_action_chunk=False`` the server returns one
    action per call, so the chunk is exhausted every step and the cadence is
    identical to a plain per-step request loop.

    Two opt-outs must be respected by callers:

    * ``disable_action_cache`` models are closed-loop within a chunk — a new
      request is issued every env step and only the first action is used.
    * action-ensemble / streaming loops need a fresh prediction per step, so
      they pass ``chunk_steps=1``.
    """

    def __init__(
        self,
        *,
        chunk_steps: int = 0,
        return_action_chunk: bool = False,
        disable_action_cache: bool = False,
        payload_builder: Any = None,
    ) -> None:
        self.chunk_steps = max(0, int(chunk_steps or 0))
        self.return_action_chunk = bool(return_action_chunk)
        self.disable_action_cache = bool(disable_action_cache)
        self.payload_builder = payload_builder
        self._consumer = ChunkConsumer()
        self._request_step = 0
        self._last_request_id: Optional[str] = None

    @property
    def requests(self) -> int:
        """Number of inference requests issued in the current episode."""
        return self._consumer.requests

    @property
    def local_steps(self) -> int:
        """Number of env steps served from a buffered chunk."""
        return self._consumer.local_steps

    @property
    def request_step(self) -> int:
        """Index of the next request (0-based, reset by ``begin_episode``)."""
        return self._request_step

    def begin_episode(self) -> None:
        """Drop the current chunk and per-episode counters."""
        self._consumer.reset()
        self._request_step = 0
        self._last_request_id = None

    def next_action(
        self, request_fn: Callable[[int], tuple[Dict[str, Any], float]]
    ) -> PolicyStep:
        """Return the action for this env step, requesting when required.

        ``request_fn`` receives the 0-based request index (this is what LIBERO's
        payload uses as ``episode_step``; for chunked runs it advances once per
        chunk, not once per env step) and returns ``(data, e2e_latency_ms)``.
        """
        issued_request = self._consumer.needs_request(self.disable_action_cache)
        request_step: Optional[int] = None
        inference_latency_ms: Optional[float] = None
        e2e_latency_ms: Optional[float] = None
        raw_actions: Optional[np.ndarray] = None
        if issued_request:
            data, e2e_latency_ms = request_fn(self._request_step)
            if self.payload_builder is not None:
                self.payload_builder.update_from_response(data)
            inference_latency_ms = data.get("inference_latency_ms")
            self._last_request_id = data.get("request_id")
            raw_actions = np.asarray(data["actions"], dtype=np.float32)
            if raw_actions.ndim == 1:
                raw_actions = raw_actions.reshape(1, -1)
            self._consumer.load(
                raw_actions,
                limit=1 if self.disable_action_cache else self.chunk_steps,
            )
            request_step = self._request_step
            self._request_step += 1

        action = self._consumer.next_action(self.disable_action_cache)
        return PolicyStep(
            action=action,
            issued_request=issued_request,
            request_step=request_step,
            # Keep the owning request id on cache hits: the trace records which
            # request produced the chunk that this action came from.
            request_id=self._last_request_id,
            inference_latency_ms=inference_latency_ms,
            e2e_latency_ms=e2e_latency_ms if issued_request else 0.0,
            raw_actions=raw_actions,
        )


# --------------------------------------------------------------------------- #
# Signal-based timeouts (LIBERO, CALVIN)
# --------------------------------------------------------------------------- #
class StepTimeoutError(TimeoutError):
    """Raised when a single simulator step exceeds its time budget."""


@contextmanager
def alarm_timeout(seconds: float, error_cls):
    """Raise ``error_cls`` if the wrapped block runs longer than ``seconds``.

    A non-positive ``seconds`` disables the alarm. Uses ``SIGALRM`` /
    ``ITIMER_REAL`` and is therefore main-thread only.
    """
    if seconds <= 0:
        yield
        return

    def _handle_timeout(signum, frame):
        raise error_cls(f"Timed out after {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


# --------------------------------------------------------------------------- #
# Artifact IO (replay GIF / action trace JSON)
# --------------------------------------------------------------------------- #
def write_replay_gif(frames: List[np.ndarray], replay_path: pathlib.Path, duration: float) -> str:
    """Write replay frames to a GIF artifact and return its path."""
    import imageio.v2 as imageio

    replay_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(replay_path, frames, duration=duration)
    return str(replay_path)


def write_trace_json(trace: List[Dict[str, Any]], trace_path: pathlib.Path) -> str:
    """Write a per-step action trace to a JSON artifact and return its path."""
    import json

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as file:
        json.dump(trace, file, ensure_ascii=False, indent=2)
    return str(trace_path)
