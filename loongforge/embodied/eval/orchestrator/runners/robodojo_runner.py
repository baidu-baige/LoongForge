# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Launch RoboDojo's official Isaac Sim evaluator against a standalone LoongForge server.

RoboDojo is a "form B" benchmark: the official sim client (``scripts/eval_policy.sh``
-> ``src/eval_client/main.py``) drives every episode and talks to an XPolicyLab policy
server. For one ``(task, layout seed)`` this runner

1. writes the LoongForge XPolicyLab plugin package into
   ``<RoboDojo>/XPolicyLab/policy/LoongForge`` (``ensure_plugin``: the episode loops come from
   ``bridges/robodojo_deploy.py``, the ``Model`` from ``bridges/robodojo_policy.py``);
2. starts the XPolicyLab policy server with a per-run deploy config, which hosts
   :class:`~loongforge.embodied.eval.bridges.robodojo_policy.RoboDojoPolicyModel`
   and forwards each env step to the already running LoongForge server;
3. runs the official sim client unchanged and waits for it;
4. converts the official ``_result.json`` into ``results.jsonl`` / ``summary.csv`` rows
   (one row per episode, carrying RoboDojo's partial-progress ``score``).

``benchmark.eval_batch: true`` switches to RoboDojo's own batched mode: the sim client runs
``benchmark.num_envs`` envs in one Isaac Sim process and calls ``eval_one_episode_batch``. The
plugin is then generated under the name ``LoongForgeBatch`` with ``eval_batch: true`` in its
deploy.yml (RoboDojo reads it from there), and the runner also writes a copy of
``eval_policy.sh`` with the env count, since the official script takes it from the sim config
and drops unknown flags (see ``eval_policy_script``).
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
BATCH_PLUGIN_NAME = "LoongForgeBatch"  # eval_batch: true (several envs per Isaac Sim process)
BENCHMARK_DIR = "RoboDojo"


def plugin_name(args: argparse.Namespace) -> str:
    """XPolicyLab policy name the sim client runs: the batched plugin when ``eval_batch``."""
    return BATCH_PLUGIN_NAME if getattr(args, "eval_batch", False) else PLUGIN_NAME


def build_argparser() -> argparse.ArgumentParser:
    """Run build_argparser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--loongforge-root", default=os.environ.get("LOONGFORGE_ROOT", ""))
    parser.add_argument("--robodojo-root", default=os.environ.get("ROBODOJO_ROOT", ""))
    parser.add_argument("--robodojo-python", default=os.environ.get("ROBODOJO_PYTHON", sys.executable))
    # XPolicyLab's policy server needs websockets>=14 (websockets.asyncio), while RoboDojo's
    # Isaac Sim env pins websockets 12, so the bridge runs in a separate "policy env"
    # (by default the LoongForge server's Python).
    parser.add_argument("--policy-python", default="")
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
    parser.add_argument("--eval-batch", action="store_true",
                        help="several envs per Isaac Sim process (LoongForgeBatch plugin)")
    parser.add_argument("--num-envs", type=int, default=0,
                        help="envs per Isaac Sim process with --eval-batch (0: the sim config's own count)")
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
        (benchmark, "eval_batch", "eval_batch"),
        (benchmark, "num_envs", "num_envs"),
        (server, "ckpt_path", "policy_ckpt_path"),
        (server, "host", "host"),
        (server, "port", "port"),
        (env, "loongforge_root", "loongforge_root"),
        (env, "robodojo_root", "robodojo_root"),
        (env, "robodojo_python", "robodojo_python"),
        (env, "policy_python", "policy_python"),
        (server, "python", "server_python"),
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
    if not args.policy_python:
        args.policy_python = getattr(args, "server_python", "") or sys.executable
    return args


_GENERATED_BY = "generated by loongforge.embodied.eval.orchestrator.runners.robodojo_runner"
# Copied into the generated plugin as ``deploy.py``; it must stay import-free of loongforge
# because it runs inside RoboDojo's Isaac Sim environment.
_DEPLOY_SOURCE = pathlib.Path(__file__).resolve().parents[2] / "bridges" / "robodojo_deploy.py"


def _plugin_files(name: str, batch: bool) -> Dict[str, str]:
    """The four files of the XPolicyLab plugin package RoboDojo imports by name."""
    return {
        "__init__.py": (
            "# Copyright 2026 The LoongForge Authors.\n"
            "# SPDX-License-Identifier: Apache-2.0\n\n"
            f'"""XPolicyLab policy plugin "{name}" -- {_GENERATED_BY}.\n\n'
            "RoboDojo imports ``XPolicyLab.policy.<name>.deploy`` in its Isaac Sim process and\n"
            "``.model`` in the XPolicyLab policy server, and reads ``eval_batch`` from the static\n"
            "deploy.yml next to them, so the plugin has to exist as a package under this name.\n"
            "Do not edit: the runner rewrites these files whenever LoongForge changes.\n"
            '"""\n'
        ),
        "deploy.py": _DEPLOY_SOURCE.read_text(encoding="utf-8"),
        "model.py": (
            "# Copyright 2026 The LoongForge Authors.\n"
            "# SPDX-License-Identifier: Apache-2.0\n\n"
            f'"""XPolicyLab ``Model`` entry for "{name}" -- {_GENERATED_BY}."""\n\n'
            "from loongforge.embodied.eval.bridges.robodojo_policy import RoboDojoPolicyModel as Model\n\n"
            '__all__ = ["Model"]\n'
        ),
        "deploy.yml": (
            f"# Static plugin metadata read by RoboDojo's sim client -- {_GENERATED_BY}.\n"
            "# The policy server gets its per-run deploy config from the runner instead.\n"
            f"policy_name: {name}\n"
            "protocol: ws\n"
            f"eval_batch: {'true' if batch else 'false'}\n"
        ),
    }


def ensure_plugin(robodojo_root: pathlib.Path, batch: bool = False) -> pathlib.Path:
    """Write the LoongForge plugin package into ``XPolicyLab/policy``; never touch a foreign one.

    Generating it (rather than shipping two plugin directories) keeps the repository free of
    packages that only exist to satisfy RoboDojo's name lookup, and lets one bridge serve both
    env counts: the only real difference between them is ``eval_batch`` in deploy.yml, which
    RoboDojo reads from the static file. The content depends on nothing but the mode, so
    concurrent units converge on the same files; an existing package is rewritten only where it
    differs, and one that LoongForge did not generate is left alone.
    """
    name = BATCH_PLUGIN_NAME if batch else PLUGIN_NAME
    package = robodojo_root / "XPolicyLab" / "policy" / name
    files = _plugin_files(name, batch)
    if package.exists() or package.is_symlink():
        marker = package / "__init__.py"
        if package.is_symlink() or not marker.is_file() or _GENERATED_BY not in marker.read_text(encoding="utf-8"):
            raise FileExistsError(
                f"{package} exists and was not generated by LoongForge; refusing to replace it"
            )
    package.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        path = package / filename
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
    return package


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


# Isaac Sim's crash reporter (Carbonite) keeps trying to open a ``zenity`` dialog after a GPU /
# PhysX crash and the client never exits. Once a zenity attempt shows up, the client counts as
# hung if no step progress (``step: `` lines) follows for this long. Progress, not "the last
# lines are all zenity", is the test: the crash reporter interleaves its own lines (minidump
# upload attempts) and the official eval_policy.sh relaunches the client after a crash, both of
# which would reset a "last lines" check forever.
ZENITY_HANG_SEC = 240.0


def _zenity_hung(log_path: pathlib.Path, since: Dict[str, float]) -> bool:
    """True once a zenity crash-dialog attempt has not been followed by step progress for ``ZENITY_HANG_SEC``."""
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as handle:
            if "t" not in since:
                handle.seek(max(0, size - 4096))
                if b"zenity" not in handle.read():
                    return False
                since["t"], since["pos"] = time.time(), size
                return False
            handle.seek(int(since["pos"]))
            after = handle.read()
    except OSError:
        return False
    if b"step: " in after:
        since.clear()
        return False
    return time.time() - since["t"] >= ZENITY_HANG_SEC


# Under load Isaac Sim sometimes never exits after "Simulation App Shutting Down", with the
# results already written; such a process is stopped after this long.
SHUTDOWN_HANG_SEC = 120.0


def _shutdown_hung(log_path: pathlib.Path, since: Dict[str, float]) -> bool:
    """True once the sim log has ended with the Kit shutdown line for ``SHUTDOWN_HANG_SEC``."""
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 2048))
            tail = handle.read().decode("utf-8", errors="replace").strip().splitlines()[-3:]
    except OSError:
        return False
    if not any("Simulation App Shutting Down" in line for line in tail):
        since.pop("t", None)
        return False
    first = since.setdefault("t", time.time())
    return time.time() - first >= SHUTDOWN_HANG_SEC


def client_completed(sim_log: pathlib.Path) -> bool:
    """True when the official client logged its normal completion."""
    try:
        return "(eval completed)" in sim_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _shutdown_hang_returncode(args: argparse.Namespace, sim_log: pathlib.Path) -> int:
    """0 when every episode finished before Kit hung on shutdown, else 137.

    RoboDojo writes ``_result.json`` before shutting Kit down, so a hang afterwards is not a
    failed run. ``eval_num`` names the expected episode count when it is set; a full sweep
    leaves it unset (per-task native counts), and then the client's own completion line is what
    says the sweep ran to the end.
    """
    try:
        done = int(json.loads((result_dir(args) / "_result.json").read_text(encoding="utf-8")).get("eval_time", 0))
    except (OSError, TypeError, ValueError):
        return 137
    if args.eval_num:
        return 0 if done >= int(args.eval_num) else 137
    return 0 if done > 0 and client_completed(sim_log) else 137


def _stop_process_group(process: Optional[subprocess.Popen], grace_sec: float = 10.0) -> None:
    """SIGTERM the process group, then SIGKILL whatever is left in it.

    The group leader is ``bash eval_policy.sh``; it exits on SIGTERM while its Isaac Sim child
    ignores SIGTERM, so stopping at the leader's exit leaves the sim running (seen: sims alive
    an hour after their run finished). The group is always swept, also when the leader has
    already exited.
    """
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.time() + grace_sec
    while time.time() < deadline:
        try:
            os.killpg(process.pid, 0)  # anyone left in the group?
        except ProcessLookupError:
            return
        time.sleep(0.5)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def write_policy_deploy_config(args: argparse.Namespace, path: pathlib.Path) -> Dict[str, Any]:
    """Per-run XPolicyLab deploy config consumed by ``RoboDojoPolicyModel``."""
    if args.first_chunk_len is None or args.chunk_len is None:
        raise ValueError("benchmark.first_chunk_len / chunk_len (or model frame_chunk_size / action_per_frame) required")
    cfg = {
        "policy_name": plugin_name(args),
        "protocol": "ws",
        "eval_batch": bool(getattr(args, "eval_batch", False)),
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


def _base_env(args: argparse.Namespace, python: str) -> Dict[str, str]:
    """Environment for a RoboDojo-side process run by ``python`` (sim venv or policy env)."""
    env = os.environ.copy()
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        env.pop(key, None)
    robodojo_root = pathlib.Path(args.robodojo_root).resolve()
    loongforge_root = pathlib.Path(args.loongforge_root).resolve()
    py_paths = [str(robodojo_root), str(robodojo_root / "XPolicyLab"), str(loongforge_root)]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(py_paths)
    python_bin = pathlib.Path(python).parent
    env["PATH"] = f"{python_bin}:{env.get('PATH', '')}"
    if (python_bin.parent / "pyvenv.cfg").is_file():
        env["VIRTUAL_ENV"] = str(python_bin.parent)
    else:
        env.pop("VIRTUAL_ENV", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def start_policy_server(args: argparse.Namespace, deploy_cfg_path: pathlib.Path, log_path: pathlib.Path) -> subprocess.Popen:
    """Start the XPolicyLab policy server hosting the LoongForge bridge (CPU only)."""
    robodojo_root = pathlib.Path(args.robodojo_root).resolve()
    env = _base_env(args, args.policy_python)
    env["CUDA_VISIBLE_DEVICES"] = ""
    command = [
        args.policy_python,
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


_EVAL_POLICY_ROOT_LINE = 'PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"'
_EVAL_POLICY_NUM_ENVS_PREFIX = 'num_envs="$(python3 -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1])).get(\'scene\',{}).get(\'num_envs\',1))" "$sim_cfg_file")"'


def eval_policy_script(args: argparse.Namespace) -> pathlib.Path:
    """The official ``scripts/eval_policy.sh``, or a per-run copy that sets the env count.

    The official script takes ``num_envs`` from the sim config (10 for arx_x5) and drops extra
    flags before calling the client, so ``--num_envs`` cannot be passed through. With
    ``eval_batch`` and an explicit ``num_envs`` the runner writes a copy into the run's output
    dir with exactly two lines changed: the project root points at the checkout (the copy lives
    elsewhere) and ``num_envs`` is the requested value. Both lines are matched verbatim; if the
    official script changes, this fails instead of silently running something else.
    """
    official = pathlib.Path(args.robodojo_root).resolve() / "scripts" / "eval_policy.sh"
    if not (getattr(args, "eval_batch", False) and getattr(args, "num_envs", 0)):
        return official
    num_envs = int(args.num_envs)
    if args.eval_num:
        # An env without an episode still simulates and renders (its seed is None, it is only
        # marked done): never create more envs than the run has episodes.
        num_envs = max(1, min(num_envs, int(args.eval_num)))
    text = official.read_text(encoding="utf-8")
    if _EVAL_POLICY_ROOT_LINE not in text or _EVAL_POLICY_NUM_ENVS_PREFIX not in text:
        raise RuntimeError(f"{official} changed; cannot set num_envs (expected lines not found)")
    text = text.replace(_EVAL_POLICY_ROOT_LINE, f'PROJECT_ROOT="{official.parent.parent}"', 1)
    text = text.replace(_EVAL_POLICY_NUM_ENVS_PREFIX, f'num_envs="{num_envs}"', 1)
    copy = pathlib.Path(args.output_dir) / "eval_policy_num_envs.sh"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(text, encoding="utf-8")
    return copy


def build_sim_command(args: argparse.Namespace) -> List[str]:
    """Official sim client entry (the RoboDojo script; see ``eval_policy_script``)."""
    robodojo_root = pathlib.Path(args.robodojo_root).resolve()
    return [
        "bash",
        str(eval_policy_script(args)),
        "--root_dir", str(robodojo_root),
        "--task_name", args.task_name,
        "--env_cfg_type", args.env_cfg_type,
        "--device_id", str(args.device_id),
        "--policy_name", plugin_name(args),
        "--port", str(args.policy_server_port),
        "--host", "127.0.0.1",
        "--protocol", "ws",
        "--seed", str(args.seed),
        "--additional_info", args.additional_info,
    ]


def result_dir(args: argparse.Namespace) -> pathlib.Path:
    """``eval_result/RoboDojo/<task>/<plugin>/<env>/<seed>_<info>/<run_id>`` under the RoboDojo root."""
    return (
        pathlib.Path(args.robodojo_root).resolve()
        / "eval_result"
        / BENCHMARK_DIR
        / args.task_name
        / plugin_name(args)
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

    ensure_plugin(robodojo_root, batch=bool(getattr(args, "eval_batch", False)))
    deploy_cfg_path = output_dir / "xpolicylab_deploy.yml"
    write_policy_deploy_config(args, deploy_cfg_path)

    policy_log = output_dir / "xpolicylab_policy_server.log"
    sim_log = output_dir / f"{args.task_name}_seed{args.seed}_official_eval.log"
    policy_proc: Optional[subprocess.Popen] = None
    sim_proc: Optional[subprocess.Popen] = None
    start = time.time()
    returncode = 1
    try:
        policy_proc = start_policy_server(args, deploy_cfg_path, policy_log)
        _wait_port("127.0.0.1", int(args.policy_server_port), float(args.policy_server_start_timeout_sec), policy_proc)

        env = _base_env(args, args.robodojo_python)
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
            deadline = start + args.timeout_sec if args.timeout_sec and args.timeout_sec > 0 else None
            hang: Dict[str, float] = {}
            shutdown: Dict[str, float] = {}
            while True:
                try:
                    returncode = sim_proc.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if deadline is not None and time.time() > deadline:
                    log_file.write(f"\nTIMEOUT after {args.timeout_sec} seconds\n")
                    log_file.flush()
                    _stop_process_group(sim_proc)
                    returncode = 124
                    break
                if _zenity_hung(sim_log, hang):
                    log_file.write(
                        f"\nHUNG: no step progress for {ZENITY_HANG_SEC:.0f} s after a zenity crash-dialog attempt; stopping\n"
                    )
                    log_file.flush()
                    _stop_process_group(sim_proc)
                    returncode = 137
                    break
                if _shutdown_hung(sim_log, shutdown):
                    log_file.write(
                        f"\nSHUTDOWN HANG: still running {SHUTDOWN_HANG_SEC:.0f} s after Kit shutdown; stopping\n"
                    )
                    log_file.flush()
                    _stop_process_group(sim_proc)
                    returncode = _shutdown_hang_returncode(args, sim_log)
                    break
    finally:
        _stop_process_group(sim_proc, grace_sec=5.0)
        _stop_process_group(policy_proc)

    if int(returncode) == 0 and not client_completed(sim_log):
        # The official client also exits 0 when it fails while building the env, so rc alone
        # does not mean "done": without its completion line the run counts as failed.
        with sim_log.open("a", encoding="utf-8") as handle:
            handle.write("\n[loongforge runner] client exited 0 without '(eval completed)'; counting as failed\n")
        returncode = 1

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
