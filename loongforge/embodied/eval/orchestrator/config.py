# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the LoongForge VLA evaluation module."""

from __future__ import annotations

import copy
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from loongforge.embodied.eval.payload_builders import PayloadBuilder, build_payload_builder


def build_payload_builder_from_config(
    raw_config: Dict[str, Any], model_type_override: Optional[str] = None
) -> PayloadBuilder:
    """Instantiate the per-model PayloadBuilder from a raw eval-YAML dict.

    Reads ``model:`` / ``server:`` / ``benchmark:`` sections and passes them
    to the per-model builder's ``__init__``. New-schema YAMLs put all
    capability fields (``state_encoding`` / ``action_encoding`` / ``domain_id``
    / ``unnorm_key``) directly under ``model:``.
    """
    yaml_model = dict(raw_config.get("model") or {})
    yaml_server = dict(raw_config.get("server") or {})
    yaml_benchmark = dict(raw_config.get("benchmark") or {})

    model_type = str(
        model_type_override or yaml_model.get("model_type") or "pi05"
    ).lower()
    return build_payload_builder(
        model_type,
        yaml_model=yaml_model,
        yaml_server=yaml_server,
        yaml_benchmark=yaml_benchmark,
    )


def resolve_action_decoder_key(
    payload_builder: PayloadBuilder,
    adapter: Any,
) -> str:
    """Compose the ActionDecoder key for this (model, benchmark).

    Auto-composed from ``payload_builder.action_encoding × adapter.action_space``.
    Returns ``""`` (no decoder — the runner uses a no-op passthrough / the
    benchmark adapter's native action conversion) when:

    - source == target (identity), or
    - the composed key has no registered decoder. This is the case for models
      whose action already matches the benchmark's native (relative) action
      that ``adapter.action_from_canonical`` consumes directly — e.g. pi05's
      7D delta on CALVIN/SimplerEnv, where the adapter converts rather than an
      absolute-EE decoder. A warning is logged so a genuinely mis-declared
      encoding is still visible.
    """
    from loongforge.embodied.eval.action_decoders import is_action_decoder_registered

    src = str(getattr(payload_builder, "action_encoding", "") or "").strip()
    dst = str(getattr(adapter, "action_space", "") or "").strip()
    if not src or not dst or src == dst:
        return ""
    key = f"{src}_to_{dst}"
    if not is_action_decoder_registered(key):
        logging.warning(
            "No ActionDecoder registered for %r (model action_encoding=%r × "
            "adapter action_space=%r); falling back to the benchmark adapter's "
            "native action conversion.",
            key, src, dst,
        )
        return ""
    return key


def build_rpc_payload(
    payload_builder: PayloadBuilder,
    canonical_obs: Dict[str, Any],
    ctx: Dict[str, Any],
    disable_action_cache: bool = False,
    return_action_chunk: bool = False,
    extra_control: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the transport-level RPC dict from a PayloadBuilder's kwargs.

    RPC-control fields (``episode_id`` / ``episode_step`` / cache flags plus
    any ``extra_control``) are added around the model kwargs so the server
    side policy can consume them without leaking into
    ``model.predict_action``.
    """
    model_kwargs = payload_builder.build(canonical_obs, ctx)
    rpc: Dict[str, Any] = {
        "episode_id": ctx.get("episode_id", canonical_obs.get("meta", {}).get("episode_id", "default")),
        "episode_step": int(
            ctx.get("episode_step", canonical_obs.get("meta", {}).get("episode_step", 0))
        ),
        "disable_action_cache": bool(disable_action_cache),
        "return_action_chunk": bool(return_action_chunk),
    }
    if extra_control:
        rpc.update(extra_control)
    rpc.update(model_kwargs)
    return rpc


def load_config(path: str) -> Dict[str, Any]:
    """Run load_config."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML config requires PyYAML. Install pyyaml or use a .json config file.") from exc
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


CONFIG_MAPPING = {
    "model.server_kind": "server_kind",
    "model.name": "model_name",
    "model.backend": "server_kind",
    "model.action_dim": "action_dim",
    "model.action_horizon": "action_horizon",
    "model.action_chunk_size": "action_horizon",
    "model.state_dim": "state_dim",
    "model.max_action_dim": "max_action_dim",
    "model.max_state_dim": "max_state_dim",
    "model.compile_model": "compile_model",
    "model.compile_mode": "compile_mode",
    # server runtime fields live ONLY in the server: section
    "server.ckpt_path": "ckpt_path",
    "server.tokenizer_path": "tokenizer_path",
    "server.tokenizer_name": "tokenizer_name",
    "server.dataset_statistics_path": "dataset_statistics_path",
    "server.use_bf16": "use_bf16",
    "server.random_init": "random_init",
    "server.host": "host",
    "server.port": "port",
    "server.health_port": "health_port",
    "server.start_timeout_sec": "server_start_timeout_sec",
    "server.log": "server_log",
    "server.pid_file": "server_pid_file",
    "server.kill_after_healthcheck": "kill_server_after_healthcheck",
    "server.stop_after_healthcheck": "stop_server_after_healthcheck",
    "server.python": "server_python",
    "env.eval_root": "eval_root",
    "env.loongforge_root": "loongforge_root",
    "env.dataset_statistics_path": "dataset_statistics_path",
    "env.libero_config_path": "libero_config_path",
    "env.mujoco_gl": "mujoco_gl",
    "env.pyopengl_platform": "pyopengl_platform",
    "env.ld_library_path": "ld_library_path",
    "benchmark.name": "task_suite_name",
    "benchmark.suite": "task_suite_name",
    "benchmark.max_tasks": "max_tasks",
    "benchmark.task_ids": "task_ids",
    "benchmark.episodes_per_task": "episodes_per_task",
    "benchmark.num_steps_wait": "num_steps_wait",
    "benchmark.max_steps": "max_steps",
    "benchmark.continuous_gripper": "continuous_gripper",
    # Camera render resolution passed to robosuite. Omit to keep the shared
    # default (LIBERO_ENV_RESOLUTION); set it when the official eval client of
    # a model renders natively at another size (e.g. LingBot-VA renders 128).
    "benchmark.render_resolution": "render_resolution",
    # LIBERO OSC: auto | absolute | delta (auto ≈ absolute when the composed
    # action decoder key is non-identity — capability matching drives this).
    "benchmark.control_mode": "control_mode",
    "server.chunk_execute_steps": "chunk_execute_steps",
    "benchmark.dataset_path": "dataset_path",
    "benchmark.calvin_config_path": "calvin_config_path",
    "benchmark.eval_sequences_path": "eval_sequences_path",
    "benchmark.num_sequences": "num_sequences",
    "benchmark.max_steps_per_subtask": "max_steps_per_subtask",
    "benchmark.control_hz": "control_hz",
    "run.min_episodes_per_task": "min_episodes_per_task",
    "run.seed": "seed",
    "run.output_dir": "output_dir",
    "run.save_replay": "save_replay",
    "run.save_trace": "save_trace",
    "run.restart": "restart",
    "run.max_retries": "max_retries",
    "timeouts.policy_call_ms": "policy_call_timeout_ms",
    "timeouts.per_step_sec": "per_step_timeout_sec",
    "timeouts.per_episode_sec": "per_episode_timeout_sec",
}


def apply_config(args: Any, config: Dict[str, Any]) -> Any:
    """Run apply_config."""
    for source, target in CONFIG_MAPPING.items():
        value = _get_nested(config, source)
        if value is not None:
            setattr(args, target, value)
    return args


def expand_sweep(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run expand_sweep."""
    sweep = config.get("sweep") or {}
    if not sweep:
        return [config]
    if isinstance(sweep, list):
        return [_expand_item(config, item, index) for index, item in enumerate(sweep)]
    if not isinstance(sweep, dict):
        raise ValueError("sweep must be a mapping or a list of mappings")
    keys = list(sweep.keys())
    values = [value if isinstance(value, list) else [value] for value in sweep.values()]
    return [
        _expand_item(config, dict(zip(keys, combination)), index)
        for index, combination in enumerate(itertools.product(*values))
    ]


def _expand_item(config: Dict[str, Any], overrides: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Run _expand_item."""
    item = copy.deepcopy(config)
    item.pop("sweep", None)
    for dotted_key, value in overrides.items():
        _set_nested(item, dotted_key, value)
    run_id = _get_nested(item, "run.id") or f"sweep_{index:03d}"
    output_dir = _get_nested(item, "run.output_dir")
    if output_dir:
        _set_nested(item, "run.output_dir", str(Path(output_dir) / str(run_id)))
    return item


def _get_nested(data: Dict[str, Any], dotted_key: str) -> Any:
    """Run _get_nested."""
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _set_nested(data: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Run _set_nested."""
    current = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value
