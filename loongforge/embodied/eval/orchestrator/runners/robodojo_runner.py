# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Launch RoboDojo's official Isaac Sim evaluator against a standalone LoongForge server.

RoboDojo is a "form B" benchmark: the official sim client (``scripts/eval_policy.sh``
-> ``src/eval_client/main.py``) drives every episode and talks to an XPolicyLab policy
server. For one ``(task, layout seed)`` this runner

1. links the LoongForge XPolicyLab plugin (``bridges/robodojo_xpolicylab``) as
   ``<RoboDojo>/XPolicyLab/policy/LoongForge``;
2. starts the XPolicyLab policy server with a per-run deploy config, which hosts
   :class:`~loongforge.embodied.eval.bridges.robodojo_policy.RoboDojoPolicyModel`
   and forwards each env step to the already running LoongForge server;
3. runs the official sim client unchanged and waits for it;
4. converts the official ``_result.json`` into ``results.jsonl`` / ``summary.csv`` rows
   (one row per episode, carrying RoboDojo's partial-progress ``score``).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import yaml

from loongforge.embodied.eval.metrics.results import append_jsonl, write_suite_summary_csv, write_summary_csv

PLUGIN_NAME = "LoongForge"
BENCHMARK_DIR = "RoboDojo"
_PLUGIN_SOURCE = pathlib.Path(__file__).resolve().parents[2] / "bridges" / "robodojo_xpolicylab"


def build_argparser() -> argparse.ArgumentParser:
    """Run build_argparser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--loongforge-root", default=os.environ.get("LOONGFORGE_ROOT", ""))
    parser.add_argument("--robodojo-root", default=os.environ.get("ROBODOJO_ROOT", ""))
    parser.add_argument("--robodojo-python", default=os.environ.get("ROBODOJO_PYTHON", sys.executable))
    parser.add_argument("--policy-ckpt-path", default="")
    parser.add_argument("--task-name", default="")
    parser.add_argument("--seed", type=int, default=0, help="RoboDojo layout seed (Eval_Layout/<env>/<seed>)")
    parser.add_argument("--env-cfg-type", default="arx_x5")
    parser.add_argument("--eval-num", type=int, default=None, help="cap on episodes; None = native count")
    parser.add_argument("--action-bridge", default="lingbot_va_joint30")
    parser.add_argument("--action-channel-ids", type=json.loads, default=None)
    parser.add_argument("--first-chunk-len", type=int, default=None)
    parser.add_argument("--chunk-len", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093, help="LoongForge policy server port")
    parser.add_argument("--policy-server-port", type=int, default=0, help="XPolicyLab policy server port")
    parser.add_argument("--device-id", type=int, default=None, help="GPU for the sim client")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--additional-info", default="")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    parser.add_argument("--policy-server-start-timeout-sec", type=float, default=120.0)
    parser.add_argument("--lf-timeout-sec", type=float, default=600.0)
    parser.add_argument("--sim-env", type=json.loads, default=None, help="extra env vars for the sim client")
    return parser


def _apply_config(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    """Map YAML sections onto runner args."""
    benchmark = config.get("benchmark") or {}
    model = config.get("model") or {}
    server = config.get("server") or {}
    env = config.get("env") or {}
    run = config.get("run") or {}
    timeouts = config.get("timeouts") or {}
    if not isinstance(benchmark, dict) or not isinstance(model, dict) or not isinstance(server, dict):
        raise ValueError("benchmark, model, and server sections must be mappings")

    mapping = [
        (benchmark, "task_name", "task_name"),
        (benchmark, "seed", "seed"),
        (benchmark, "env_cfg_type", "env_cfg_type"),
        (benchmark, "eval_num", "eval_num"),
        (benchmark, "action_bridge", "action_bridge"),
        (benchmark, "action_channel_ids", "action_channel_ids"),
        (benchmark, "first_chunk_len", "first_chunk_len"),
        (benchmark, "chunk_len", "chunk_len"),
        (server, "ckpt_path", "policy_ckpt_path"),
        (server, "host", "host"),
        (server, "port", "port"),
        (env, "loongforge_root", "loongforge_root"),
        (env, "robodojo_root", "robodojo_root"),
        (env, "robodojo_python", "robodojo_python"),
        (env, "sim_env", "sim_env"),
        (run, "output_dir", "output_dir"),
        (run, "run_id", "run_id"),
        (run, "additional_info", "additional_info"),
        (run, "policy_server_port", "policy_server_port"),
        (run, "device_id", "device_id"),
        (run, "trace", "trace"),
        (timeouts, "per_task_sec", "timeout_sec"),
        (timeouts, "policy_call_sec", "lf_timeout_sec"),
        (timeouts, "policy_server_start_sec", "policy_server_start_timeout_sec"),
    ]
    for section, key, attr in mapping:
        value = section.get(key) if isinstance(section, dict) else None
        if value is not None:
            setattr(args, attr, value)

    # LingBot-VA chunk schedule follows from the model section unless given explicitly.
    if args.first_chunk_len is None or args.chunk_len is None:
        if "frame_chunk_size" in model and "action_per_frame" in model:
            from loongforge.embodied.eval.bridges.robodojo_policy import chunk_lengths_from_model

            first, rest = chunk_lengths_from_model(model)
            args.first_chunk_len = args.first_chunk_len or first
            args.chunk_len = args.chunk_len or rest
    if args.device_id is None and run.get("cuda_visible_devices") is not None:
        args.device_id = int(str(run["cuda_visible_devices"]).split(",")[0])
    return args
def ensure_plugin_link(robodojo_root: pathlib.Path) -> pathlib.Path:
    """Link the LoongForge plugin into ``XPolicyLab/policy``; never replace a foreign entry."""
    link = robodojo_root / "XPolicyLab" / "policy" / PLUGIN_NAME
    target = _PLUGIN_SOURCE.resolve()
    if link.is_symlink() or link.exists():
        if link.resolve() != target:
            raise FileExistsError(f"{link} exists and does not point to {target}; refusing to replace it")
        return link
    link.symlink_to(target, target_is_directory=True)
    return link


def check_robodojo_root(robodojo_root: pathlib.Path, env_cfg_type: str) -> None:
    """Fail early when the RoboDojo checkout or its assets are incomplete."""
    required = [
        robodojo_root / "src" / "eval_client" / "main.py",
        robodojo_root / "scripts" / "eval_policy.sh",
        robodojo_root / "XPolicyLab" / "setup_policy_server.py",
        robodojo_root / "env_cfg" / f"{env_cfg_type}.yml",
        robodojo_root / "Assets" / "Eval_Layout",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"RoboDojo checkout is incomplete, missing: {missing}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_port(host: str, port: int, timeout: float, process: subprocess.Popen) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"XPolicyLab policy server exited early with code {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"XPolicyLab policy server did not listen on {host}:{port} within {timeout}s")


def _stop_process_group(process: Optional[subprocess.Popen], grace_sec: float = 10.0) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_sec)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def write_policy_deploy_config(args: argparse.Namespace, path: pathlib.Path) -> Dict[str, Any]:
    """Per-run XPolicyLab deploy config consumed by ``RoboDojoPolicyModel``."""
    if args.first_chunk_len is None or args.chunk_len is None:
        raise ValueError("benchmark.first_chunk_len / chunk_len (or model frame_chunk_size / action_per_frame) required")
    cfg = {
        "policy_name": PLUGIN_NAME,
        "protocol": "ws",
        "eval_batch": False,
        "host": "127.0.0.1",
        "port": int(args.policy_server_port),
        "task_name": args.task_name,
        "seed": int(args.seed),
        "env_cfg_type": args.env_cfg_type,
        "action_bridge": args.action_bridge,
        "action_channel_ids": args.action_channel_ids,
        "first_chunk_len": int(args.first_chunk_len),
        "chunk_len": int(args.chunk_len),
        "lf_host": args.host,
        "lf_port": int(args.port),
        "lf_timeout_sec": float(args.lf_timeout_sec),
        "trace_dir": str(pathlib.Path(args.output_dir) / "trace") if args.trace else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg


def _base_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        env.pop(key, None)
    robodojo_root = pathlib.Path(args.robodojo_root).resolve()
    loongforge_root = pathlib.Path(args.loongforge_root).resolve()
    py_paths = [str(robodojo_root), str(robodojo_root / "XPolicyLab"), str(loongforge_root)]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(py_paths)
    python_bin = str(pathlib.Path(args.robodojo_python).parent)
    env["PATH"] = f"{python_bin}:{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = str(pathlib.Path(args.robodojo_python).parent.parent)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def start_policy_server(args: argparse.Namespace, deploy_cfg_path: pathlib.Path, log_path: pathlib.Path) -> subprocess.Popen:
    """Start the XPolicyLab policy server hosting the LoongForge bridge (CPU only)."""
    robodojo_root = pathlib.Path(args.robodojo_root).resolve()
    env = _base_env(args)
    env["CUDA_VISIBLE_DEVICES"] = ""
    command = [
        args.robodojo_python,
        str(robodojo_root / "XPolicyLab" / "setup_policy_server.py"),
        "--config_path",
        str(deploy_cfg_path),
    ]
    log = log_path.open("a", buffering=1)
    log.write("COMMAND: " + " ".join(command) + "\n")
    return subprocess.Popen(
        command,
        cwd=str(robodojo_root),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def build_sim_command(args: argparse.Namespace) -> List[str]:
    """Official sim client entry (unchanged RoboDojo script)."""
    robodojo_root = pathlib.Path(args.robodojo_root).resolve()
    return [
        "bash",
        str(robodojo_root / "scripts" / "eval_policy.sh"),
        "--root_dir", str(robodojo_root),
        "--task_name", args.task_name,
        "--env_cfg_type", args.env_cfg_type,
        "--device_id", str(args.device_id),
        "--policy_name", PLUGIN_NAME,
        "--port", str(args.policy_server_port),
        "--host", "127.0.0.1",
        "--protocol", "ws",
        "--seed", str(args.seed),
        "--additional_info", args.additional_info,
    ]


def result_dir(args: argparse.Namespace) -> pathlib.Path:
    """``eval_result/RoboDojo/<task>/LoongForge/<env>/<seed>_<info>/<run_id>`` under the RoboDojo root."""
    return (
        pathlib.Path(args.robodojo_root).resolve()
        / "eval_result"
        / BENCHMARK_DIR
        / args.task_name
        / PLUGIN_NAME
        / args.env_cfg_type
        / f"{args.seed}_{args.additional_info}"
        / args.run_id
    )
def build_records(args: argparse.Namespace, result: Dict[str, Any], returncode: int, wall_sec: float) -> List[Dict[str, Any]]:
    """One record per evaluated episode from the official ``_result.json``.

    ``score`` is RoboDojo's per-episode progress in [0, 1] (1.0 on success); the
    official task score is its mean × 100. Unstable layouts are skipped by the
    official client and never appear in ``details``.
    """
    details = result.get("details") or {}
    items = sorted(details.items(), key=lambda kv: int(kv[0]))
    n_eps = len(items)
    common = {
        "benchmark": "robodojo",
        "task_suite": args.task_name,
        "task_id": 0,
        "task_description": args.task_name.replace("_", " "),
        "env_cfg_type": args.env_cfg_type,
        "seed": int(args.seed),
        "run_id": args.run_id,
        "returncode": int(returncode),
        "official_success_rate": result.get("success_rate"),
        "official_score": result.get("score"),
        "n_episodes": n_eps,
        "episode_time_sec": float(wall_sec) / n_eps if n_eps else float(wall_sec),
    }
    records = []
    for index, detail in items:
        success = int(bool(detail.get("success")))
        records.append(
            {
                **common,
                "episode_idx": int(index),
                "episode_id": f"robodojo/{args.task_name}/seed{args.seed}/ep{int(index):04d}",
                "layout_id": detail.get("layout_id"),
                "success": success,
                "score": float(detail.get("score", float(success))),
                "failure_reason": None if success else "not_successful",
            }
        )
    return records


def _collect_artifacts(args: argparse.Namespace, source: pathlib.Path) -> Dict[str, Any]:
    target = pathlib.Path(args.output_dir) / "artifacts" / "robodojo" / args.task_name / f"seed{args.seed}"
    target.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Any] = {"official_result_dir": str(source)}
    result_json = source / "_result.json"
    if result_json.is_file():
        shutil.copy2(result_json, target / "_result.json")
        artifacts["result_json"] = str(target / "_result.json")
    # Videos stay where the official client wrote them; only their paths are recorded.
    artifacts["videos"] = sorted(str(p) for p in source.glob("*.mp4"))
    return artifacts


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Evaluate one (task, layout seed) through the official RoboDojo sim client."""
    robodojo_root = pathlib.Path(args.robodojo_root).resolve()
    check_robodojo_root(robodojo_root, args.env_cfg_type)
    if not args.task_name:
        raise ValueError("benchmark.task_name is required")
    if args.device_id is None:
        raise ValueError("run.device_id (or run.cuda_visible_devices) is required for the sim client")
    if not args.loongforge_root:
        args.loongforge_root = str(pathlib.Path(__file__).resolve().parents[5])
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.policy_server_port:
        args.policy_server_port = _free_port()
    if not args.run_id:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_id = f"lf_{args.task_name}_s{args.seed}_{stamp}_{os.getpid()}"
    if not args.additional_info:
        args.additional_info = f"ckpt_name={pathlib.Path(args.policy_ckpt_path).name or 'loongforge'},action_type=joint"

    ensure_plugin_link(robodojo_root)
    deploy_cfg_path = output_dir / "xpolicylab_deploy.yml"
    write_policy_deploy_config(args, deploy_cfg_path)

    policy_log = output_dir / "xpolicylab_policy_server.log"
    sim_log = output_dir / f"{args.task_name}_seed{args.seed}_official_eval.log"
    policy_proc: Optional[subprocess.Popen] = None
    start = time.time()
    returncode = 1
    try:
        policy_proc = start_policy_server(args, deploy_cfg_path, policy_log)
        _wait_port("127.0.0.1", int(args.policy_server_port), float(args.policy_server_start_timeout_sec), policy_proc)

        env = _base_env(args)
        env["ROBODOJO_RUN_ID"] = args.run_id
        if args.eval_num:
            env["EVAL_NUM"] = str(int(args.eval_num))
        for key, value in (args.sim_env or {}).items():
            env[str(key)] = str(value)
        command = build_sim_command(args)
        with sim_log.open("w", encoding="utf-8") as log_file:
            log_file.write("COMMAND: " + " ".join(command) + "\n")
            log_file.write("PYTHONPATH: " + env["PYTHONPATH"] + "\n")
            log_file.flush()
            sim_proc = subprocess.Popen(
                command,
                cwd=str(robodojo_root),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = sim_proc.wait(timeout=args.timeout_sec if args.timeout_sec and args.timeout_sec > 0 else None)
            except subprocess.TimeoutExpired:
                log_file.write(f"\nTIMEOUT after {args.timeout_sec} seconds\n")
                _stop_process_group(sim_proc)
                returncode = 124
    finally:
        _stop_process_group(policy_proc)

    source = result_dir(args)
    artifacts = _collect_artifacts(args, source)
    result: Dict[str, Any] = {}
    if artifacts.get("result_json"):
        result = json.loads(pathlib.Path(artifacts["result_json"]).read_text(encoding="utf-8"))
    records = build_records(args, result, returncode, time.time() - start)
    for record in records:
        append_jsonl(output_dir / "results.jsonl", record)
    if records:
        write_summary_csv(output_dir / "summary.csv", records)
        write_suite_summary_csv(output_dir / "suite_summary.csv", records)
    summary = {
        "benchmark": "robodojo",
        "task_name": args.task_name,
        "seed": int(args.seed),
        "returncode": int(returncode),
        "n_episodes": len(records),
        "success_rate": result.get("success_rate"),
        "score": result.get("score"),
        "output_dir": str(output_dir),
        "artifacts": {k: v for k, v in artifacts.items() if k != "videos"},
        "n_videos": len(artifacts.get("videos", [])),
    }
    (output_dir / "robodojo_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
