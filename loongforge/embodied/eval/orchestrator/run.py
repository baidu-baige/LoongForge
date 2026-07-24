# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unified YAML dispatcher for supported VLA evaluation benchmarks."""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import pathlib
import sys
from types import SimpleNamespace
from typing import Any, Dict

from loongforge.embodied.eval.orchestrator.runners import libero_runner
from loongforge.embodied.eval.orchestrator.config import apply_config, expand_sweep, load_config
from loongforge.embodied.eval.orchestrator.server_manager import (
    ManagedServer,
    apply_runtime_env,
    build_argparser as build_server_argparser,
    ensure_libero_config,
)


def _benchmark_family(config: Dict[str, Any]) -> str:
    """Run _benchmark_family."""
    benchmark = config.get("benchmark") or {}
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark section must be a mapping")
    name = str(benchmark.get("name") or "").lower()
    if name == "libero" or name.startswith("libero_"):
        return "libero"
    if name in {"calvin", "simplerenv", "robotwin", "maniskill"}:
        return name
    raise ValueError(f"Unsupported benchmark.name: {benchmark.get('name')!r}")


def _timestamped_run_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run _timestamped_run_config."""
    run_config = config.get("run") or {}
    if not isinstance(run_config, dict):
        raise ValueError("run section must be a mapping when set")
    if run_config.get("timestamped_output", True) is False:
        return config

    output_dir = run_config.get("output_dir")
    if not output_dir:
        return config

    output_path = pathlib.Path(str(output_dir))
    run_tag = str(run_config.get("run_name") or output_path.name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output = output_path.parent / f"{timestamp}_{run_tag}"

    item = copy.deepcopy(config)
    item.setdefault("run", {})["output_dir"] = str(timestamped_output)
    server_config = item.setdefault("server", {})
    if isinstance(server_config, dict):
        server_config["log"] = str(timestamped_output / "policy_server.log")
    return item


def _server_args(config: Dict[str, Any], config_path: str = "") -> argparse.Namespace:
    """Run _server_args."""
    args = build_server_argparser().parse_args([])
    args = apply_config(args, config)
    args.config = config_path
    server_config = config.get("server") or {}
    if not isinstance(server_config, dict) or not server_config.get("log"):
        output_dir = (config.get("run") or {}).get("output_dir") or "/tmp/vla_eval"
        args.server_log = str(pathlib.Path(output_dir) / "policy_server.log")
    return args


def _run_libero_once(config: Dict[str, Any], config_path: str = "") -> Dict[str, Any]:
    """Run _run_libero_once."""
    args = libero_runner.build_argparser().parse_args([])
    args = apply_config(args, config)
    args.config = config_path
    server_args = _server_args(config, config_path)
    ensure_libero_config(server_args)
    original = {
        key: os.environ.get(key) for key in ["MUJOCO_GL", "PYOPENGL_PLATFORM", "LD_LIBRARY_PATH", "LIBERO_CONFIG_PATH"]
    }
    apply_runtime_env(server_args, os.environ)
    server = ManagedServer(server_args)
    try:
        server.start()
        batch_args = SimpleNamespace(**vars(args), server_manager=server)
        result = libero_runner.run_batch(batch_args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    finally:
        server.stop()
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_calvin_once(config: Dict[str, Any], config_path: str = "") -> Dict[str, Any]:
    """Run _run_calvin_once."""
    from loongforge.embodied.eval.orchestrator.runners import calvin_runner

    args = calvin_runner.build_argparser().parse_args([])
    args = apply_config(args, config)
    args.config = config_path
    if hasattr(args, "per_episode_timeout_sec"):
        args.per_sequence_timeout_sec = args.per_episode_timeout_sec
    if not args.dataset_path:
        raise ValueError("CALVIN dataset path must be set by benchmark.dataset_path in YAML")
    if not pathlib.Path(args.dataset_path).exists():
        raise FileNotFoundError(f"CALVIN dataset path not found: {args.dataset_path}")
    if not args.calvin_config_path:
        raise ValueError("CALVIN config path must be set by benchmark.calvin_config_path in YAML")
    if not pathlib.Path(args.calvin_config_path).exists():
        raise FileNotFoundError(f"CALVIN config path not found: {args.calvin_config_path}")
    if not args.eval_sequences_path:
        raise ValueError("CALVIN eval sequences path must be set by benchmark.eval_sequences_path in YAML")
    if not pathlib.Path(args.eval_sequences_path).exists():
        raise FileNotFoundError(f"CALVIN eval sequences path not found: {args.eval_sequences_path}")
    calvin_models_root = pathlib.Path(args.calvin_config_path).parent
    calvin_repo_root = calvin_models_root.parent
    calvin_env_root = calvin_repo_root / "calvin_env"
    calvin_paths = [str(calvin_repo_root), str(calvin_models_root), str(calvin_env_root)]
    for path in reversed(calvin_paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ["PYTHONPATH"] = ":".join(calvin_paths + [os.environ.get("PYTHONPATH", "")])

    server = ManagedServer(_server_args(config, config_path))
    try:
        server.start()
        batch_args = SimpleNamespace(**vars(args), server_manager=server)
        result = calvin_runner.run_batch(batch_args)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return result
    finally:
        server.stop()


def _run_simplerenv_once(config: Dict[str, Any], config_path: str = "") -> Dict[str, Any]:
    """Run _run_simplerenv_once."""
    from loongforge.embodied.eval.orchestrator.runners import simplerenv_runner

    args = simplerenv_runner.build_argparser().parse_args([])
    args = simplerenv_runner._apply_config(args, config)
    simplerenv_runner._ensure_vulkan_runtime(args)
    if args.simplerenv_root:
        root_paths = [args.simplerenv_root, str(pathlib.Path(args.simplerenv_root) / "ManiSkill2_real2sim")]
        for path in reversed(root_paths):
            if path not in sys.path:
                sys.path.insert(0, path)
        os.environ["PYTHONPATH"] = ":".join(root_paths + [os.environ.get("PYTHONPATH", "")])

    server = ManagedServer(_server_args(config, config_path))
    try:
        server.start()
        result = simplerenv_runner.run_evaluation(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return result
    finally:
        server.stop()


def _run_robotwin_once(config: Dict[str, Any], config_path: str = "") -> Dict[str, Any]:
    """Run _run_robotwin_once."""
    from loongforge.embodied.eval.orchestrator.runners import robotwin_runner

    args = robotwin_runner.build_argparser().parse_args([])
    args = robotwin_runner._apply_config(args, config)
    model = config.get("model") or {}
    server_cfg = config.get("server") or {}
    random_init = bool(server_cfg.get("random_init", model.get("random_init", False)))
    if not args.policy_ckpt_path and not random_init:
        raise ValueError("policy checkpoint must be set by server.ckpt_path in YAML unless server.random_init is true")
    if not args.task_name:
        raise ValueError("task name must be set by benchmark.task_name in YAML")

    server = ManagedServer(_server_args(config, config_path))
    try:
        server.start()
        returncode = robotwin_runner.run_evaluation(args)
        result = {"benchmark": "robotwin", "returncode": int(returncode), "output_dir": args.output_dir}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if returncode != 0:
            raise SystemExit(returncode)
        return result
    finally:
        server.stop()


def _run_maniskill_once(config: Dict[str, Any], config_path: str = "") -> Dict[str, Any]:
    """Run _run_maniskill_once."""
    from loongforge.embodied.eval.orchestrator.runners import maniskill_runner

    args = maniskill_runner.build_argparser().parse_args([])
    args = maniskill_runner._apply_config(args, config)
    maniskill_runner._ensure_vulkan_runtime(args)
    if not args.task_name:
        raise ValueError("task name must be set by benchmark.task_name in YAML")

    server = ManagedServer(_server_args(config, config_path))
    try:
        server.start()
        result = maniskill_runner.run_evaluation(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return result
    finally:
        server.stop()


def _run_once(config: Dict[str, Any], config_path: str = "") -> Dict[str, Any]:
    """Run _run_once."""
    family = _benchmark_family(config)
    if family == "libero":
        return _run_libero_once(config, config_path)
    if family == "calvin":
        return _run_calvin_once(config, config_path)
    if family == "simplerenv":
        return _run_simplerenv_once(config, config_path)
    if family == "robotwin":
        return _run_robotwin_once(config, config_path)
    if family == "maniskill":
        return _run_maniskill_once(config, config_path)
    raise AssertionError(f"unreachable benchmark family: {family}")


def build_argparser() -> argparse.ArgumentParser:
    """Run build_argparser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    """Run main."""
    args = build_argparser().parse_args()
    config_path = str(pathlib.Path(args.config).resolve())
    config = load_config(config_path)
    results = [_run_once(_timestamped_run_config(sweep_config), config_path) for sweep_config in expand_sweep(config)]
    if len(results) > 1:
        print(json.dumps({"n_runs": len(results), "runs": results}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
