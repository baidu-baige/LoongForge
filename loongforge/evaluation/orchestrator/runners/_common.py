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
from typing import Any, Dict, List, Optional

import numpy as np

from loongforge.evaluation.action_decoders import build_action_decoder
from loongforge.evaluation.orchestrator.config import (
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
