# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Launch the official RoboTwin evaluator against a standalone vla_eval server."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Set

import yaml

from loongforge.embodied.eval.metrics.results import append_jsonl, write_suite_summary_csv, write_summary_csv
from loongforge.embodied.eval.orchestrator.config import load_config

# Official eval_policy.py prints (ANSI-colored):
#   Success rate: {suc}/{test_num} => {pct}%, current seed: {now_seed}
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")
_SUCCESS_RATE_LINE_RE = re.compile(
    r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>.*?current seed:\s*(\d+)",
    re.IGNORECASE,
)


def _write_deploy_policy(args: argparse.Namespace, path: pathlib.Path) -> None:
    """Run _write_deploy_policy."""
    fields: Dict[str, object] = {
        "policy_name": "loongforge.embodied.eval.bridges.robotwin_policy",
        "policy_ckpt_path": args.policy_ckpt_path,
        "task_name": args.task_name,
        "task_config": args.task_config,
        "ckpt_setting": args.ckpt_setting,
        "seed": args.seed,
        "instruction_type": args.instruction_type,
        "host": args.host,
        "port": args.port,
        "unnorm_key": args.unnorm_key,
        "action_mode": args.action_mode,
        "reorder_action": not args.no_action_reorder,
        "max_steps": args.max_steps,
        "control_hz": args.control_hz,
        "disable_action_cache": args.disable_action_cache,
        "return_action_chunk": args.return_action_chunk,
        "action_bridge": getattr(args, "action_bridge", "strict_14d"),
        "domain_id": getattr(args, "domain_id", None),
        "trace_path": str(_robotwin_artifact_dir(args) / "trace.json"),
    }
    if args.disable_eval_video_log:
        fields["eval_video_log"] = False
    path.write_text(yaml.safe_dump(fields, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_worker_eval_step_limit(args: argparse.Namespace) -> Optional[pathlib.Path]:
    """Run _write_worker_eval_step_limit."""
    if not args.worker_local_files or args.step_limit_override is None:
        return None
    robotwin_path = pathlib.Path(args.robotwin_path)
    source_dir = robotwin_path / "task_config"
    config_dir = pathlib.Path(args.output_dir) / "worker_files" / "task_config"
    if config_dir.exists():
        shutil.rmtree(config_dir)
    shutil.copytree(source_dir, config_dir)
    worker_step_limit = config_dir / "_eval_step_limit.yml"
    data = yaml.safe_load(worker_step_limit.read_text(encoding="utf-8")) or {}
    data[args.task_name] = int(args.step_limit_override)
    worker_step_limit.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return worker_step_limit


def _worker_eval_script(args: argparse.Namespace) -> Optional[pathlib.Path]:
    """Run _worker_eval_script."""
    if not args.worker_local_files:
        return None
    robotwin_path = pathlib.Path(args.robotwin_path)
    source = robotwin_path / "script" / "eval_policy.py"
    text = source.read_text(encoding="utf-8")
    script_source_dir = str(source.parent)
    text = text.replace(
        'sys.path.append("./description/utils")\n',
        'sys.path.append("./description/utils")\n' f"sys.path.insert(0, {script_source_dir!r})\n",
        1,
    )
    worker_step_limit = _write_worker_eval_step_limit(args)
    if worker_step_limit is not None:
        worker_config_dir = str(worker_step_limit.parent)
        text = text.replace(
            "from envs import CONFIGS_PATH\n",
            "from envs import CONFIGS_PATH\n"
            "import envs\n"
            "import envs._GLOBAL_CONFIGS as _vla_eval_global_configs\n"
            f"CONFIGS_PATH = {worker_config_dir + '/'!r}\n"
            "envs.CONFIGS_PATH = CONFIGS_PATH\n"
            "_vla_eval_global_configs.CONFIGS_PATH = CONFIGS_PATH\n",
            1,
        )
        text = text.replace(
            'camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")',
            f'camera_config_path = os.path.join({worker_config_dir!r}, "_camera_config.yml")',
            1,
        )
    if args.start_seed_override is not None:
        text = text.replace(
            "    st_seed = 100000 * (1 + seed)\n", f"    st_seed = {int(args.start_seed_override)}\n", 1
        )
    if args.test_num_override is not None:
        text = text.replace("    test_num = 100\n", f"    test_num = {int(args.test_num_override)}\n", 1)
    if args.disable_expert_check:
        text = text.replace("    expert_check = True\n", "    expert_check = False\n", 1)
        text = text.replace(
            "        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n"
            '        episode_info_list = [episode_info["info"]]\n',
            "        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n"
            "        episode_info = locals().get('episode_info', {'info': {}})\n"
            '        episode_info_list = [episode_info["info"]]\n',
            1,
        )
        text = text.replace(
            "        instruction = np.random.choice(results[0][instruction_type])\n",
            "        if not results or not results[0].get(instruction_type):\n"
            "            instruction = args[\"task_name\"].replace('_', ' ')\n"
            "        else:\n"
            "            instruction = np.random.choice(results[0][instruction_type])\n",
            1,
        )
    if args.disable_eval_video_log:
        text = text.replace(
            '    args["ckpt_setting"] = ckpt_setting\n',
            '    args["ckpt_setting"] = ckpt_setting\n' '    args["eval_video_log"] = False\n',
            1,
        )
    script_dir = pathlib.Path(args.output_dir) / "worker_files"
    script_dir.mkdir(parents=True, exist_ok=True)
    worker_script = script_dir / f"eval_policy_{args.ckpt_setting}.py"
    worker_script.write_text(text, encoding="utf-8")
    return worker_script


def build_command(
    args: argparse.Namespace, config_path: pathlib.Path, eval_script: Optional[pathlib.Path] = None
) -> List[str]:
    """Run build_command."""
    return [
        args.robotwin_python,
        str(eval_script or (pathlib.Path(args.robotwin_path) / "script" / "eval_policy.py")),
        "--config",
        str(config_path),
        "--overrides",
        "--task_name",
        args.task_name,
        "--task_config",
        args.task_config,
        "--ckpt_setting",
        args.ckpt_setting,
        "--seed",
        str(args.seed),
        "--policy_name",
        "loongforge.embodied.eval.bridges.robotwin_policy",
    ]


def _robotwin_result_root(args: argparse.Namespace) -> pathlib.Path:
    """Run _robotwin_result_root."""
    return (
        pathlib.Path(args.robotwin_path)
        / "eval_result"
        / args.task_name
        / "loongforge.embodied.eval.bridges.robotwin_policy"
        / args.task_config
        / args.ckpt_setting
    )


def _robotwin_result_dirs(args: argparse.Namespace) -> Set[pathlib.Path]:
    """Run _robotwin_result_dirs."""
    result_root = _robotwin_result_root(args)
    if not result_root.is_dir():
        return set()
    return {path.resolve() for path in result_root.iterdir() if path.is_dir()}


def _new_robotwin_result_dir(args: argparse.Namespace, before_dirs: Set[pathlib.Path]) -> Optional[pathlib.Path]:
    """Run _new_robotwin_result_dir."""
    candidates = sorted(_robotwin_result_dirs(args) - before_dirs, key=lambda path: path.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def _robotwin_artifact_dir(args: argparse.Namespace) -> pathlib.Path:
    """Run _robotwin_artifact_dir."""
    return pathlib.Path(args.output_dir) / "artifacts" / "robotwin" / args.task_name / args.task_config


def _copy_file_if_exists(source: pathlib.Path, target_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """Run _copy_file_if_exists."""
    if not source.is_file():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def _collect_robotwin_artifacts(
    args: argparse.Namespace,
    log_path: pathlib.Path,
    config_path: Optional[pathlib.Path],
    result_dir: Optional[pathlib.Path],
) -> Dict[str, object]:
    """Run _collect_robotwin_artifacts."""
    artifact_dir = _robotwin_artifact_dir(args)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    copied: Dict[str, object] = {"artifact_dir": str(artifact_dir)}
    copied_log = _copy_file_if_exists(log_path, artifact_dir)
    if copied_log is not None:
        copied["official_eval_log"] = str(copied_log)
    if config_path is not None:
        copied_config = _copy_file_if_exists(config_path, artifact_dir)
        if copied_config is not None:
            copied["deploy_config"] = str(copied_config)

    if result_dir is None:
        return copied
    copied["robotwin_result_dir"] = str(result_dir)
    copied_result = _copy_file_if_exists(result_dir / "_result.txt", artifact_dir)
    if copied_result is not None:
        copied["result_txt"] = str(copied_result)

    trace_path = artifact_dir / "trace.json"
    if trace_path.is_file():
        copied["trace_path"] = str(trace_path)

    video_dir = artifact_dir / "videos"
    videos = []
    for video in sorted(result_dir.rglob("*.mp4")):
        copied_video = _copy_file_if_exists(video, video_dir)
        if copied_video is not None:
            videos.append(str(copied_video))
    if videos:
        copied["videos"] = videos
    return copied


def _cleanup_robotwin_work_files(
    args: argparse.Namespace,
    log_path: pathlib.Path,
    config_path: Optional[pathlib.Path],
) -> None:
    """Run _cleanup_robotwin_work_files."""
    artifact_dir = _robotwin_artifact_dir(args).resolve()
    for path in [log_path, config_path]:
        if path is None:
            continue
        resolved = path.resolve()
        if path.exists() and artifact_dir not in resolved.parents:
            path.unlink()
    worker_dir = pathlib.Path(args.output_dir) / "worker_files"
    if worker_dir.exists():
        shutil.rmtree(worker_dir)


def _override_step_limit(robotwin_path: pathlib.Path, task_name: str, step_limit: Optional[int]) -> Optional[str]:
    """Run _override_step_limit."""
    if step_limit is None:
        return None
    step_limit_path = robotwin_path / "task_config" / "_eval_step_limit.yml"
    original_text = step_limit_path.read_text(encoding="utf-8")
    data = yaml.safe_load(original_text) or {}
    data[task_name] = int(step_limit)
    step_limit_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return original_text


def _override_eval_policy_for_smoke(
    robotwin_path: pathlib.Path,
    test_num: Optional[int],
    disable_expert_check: bool,
    start_seed: Optional[int] = None,
) -> Optional[str]:
    """Run _override_eval_policy_for_smoke."""
    if test_num is None and not disable_expert_check and start_seed is None:
        return None
    eval_policy_path = robotwin_path / "script" / "eval_policy.py"
    original_text = eval_policy_path.read_text(encoding="utf-8")
    patched_text = original_text
    if start_seed is not None:
        patched_text = patched_text.replace(
            "    st_seed = 100000 * (1 + seed)\n", f"    st_seed = {int(start_seed)}\n", 1
        )
    if test_num is not None:
        patched_text = patched_text.replace("    test_num = 100\n", f"    test_num = {int(test_num)}\n", 1)
    if disable_expert_check:
        patched_text = patched_text.replace("    expert_check = True\n", "    expert_check = False\n", 1)
    eval_policy_path.write_text(patched_text, encoding="utf-8")
    return original_text


def _apply_config(args: argparse.Namespace, config: Dict[str, object]) -> argparse.Namespace:
    """Run _apply_config."""
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
        (benchmark, "task_config", "task_config"),
        (benchmark, "max_steps", "max_steps"),
        (benchmark, "start_seed", "start_seed_override"),
        (benchmark, "test_num", "test_num_override"),
        (benchmark, "episodes_per_task", "test_num_override"),
        (benchmark, "disable_expert_check", "disable_expert_check"),
        (benchmark, "domain_id", "domain_id"),
        (benchmark, "action_bridge", "action_bridge"),
        (server, "ckpt_path", "policy_ckpt_path"),
        (model, "unnorm_key", "unnorm_key"),
        (model, "action_mode", "action_mode"),
        (model, "robotwin_action_bridge", "action_bridge"),
        (server, "host", "host"),
        (server, "port", "port"),
        (env, "loongforge_root", "loongforge_root"),
        (env, "eval_root", "eval_root"),
        (env, "robotwin_root", "robotwin_path"),
        (env, "robotwin_python", "robotwin_python"),
        (run, "seed", "seed"),
        (run, "output_dir", "output_dir"),
        (run, "ckpt_setting", "ckpt_setting"),
        (run, "disable_eval_video_log", "disable_eval_video_log"),
        (run, "keep_config", "keep_config"),
        (run, "worker_local_files", "worker_local_files"),
        (run, "cuda_visible_devices", "cuda_visible_devices"),
        (timeouts, "per_episode_sec", "timeout_sec"),
    ]
    for section, key, attr in mapping:
        value = section.get(key) if isinstance(section, dict) else None
        if value is not None:
            setattr(args, attr, value)

    if args.step_limit_override is None:
        args.step_limit_override = args.max_steps
    return args


def _robotwin_rate_from_result(result_txt: Optional[str]) -> Optional[float]:
    """Parse official `_result.txt` last numeric line as success rate in [0, 1]."""
    if not result_txt:
        return None
    for line in reversed(str(result_txt).strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return float(line)
        except ValueError:
            continue
    return None


def _parse_official_episode_outcomes(log_text: str) -> List[Dict[str, Any]]:
    """Parse per-episode outcomes from official eval_policy log.

    Official loop prints after each valid episode:
      Success rate: {suc}/{test_num} => {pct}%, current seed: {now_seed}

    Episode success is inferred from the cumulative counter delta
    (suc increases by 1 on success, stays flat on fail).
    """
    episodes: List[Dict[str, Any]] = []
    prev_suc = 0
    for raw_line in str(log_text or "").splitlines():
        line = _ANSI_RE.sub("", raw_line)
        match = _SUCCESS_RATE_LINE_RE.search(line)
        if not match:
            continue
        suc = int(match.group(1))
        test_num = int(match.group(2))
        seed = int(match.group(3))
        success = 1 if suc > prev_suc else 0
        prev_suc = suc
        episodes.append(
            {
                "episode_idx": len(episodes),
                "seed": seed,
                "success": success,
                "cum_suc": suc,
                "cum_test_num": test_num,
            }
        )
    return episodes


def _build_robotwin_records(
    args: argparse.Namespace,
    returncode: int,
    artifacts: Dict[str, object],
    episode_time_sec: float,
    result_txt: Optional[str],
    log_text: str,
) -> List[Dict[str, Any]]:
    """Build per-episode records aligned with LIBERO/SimplerEnv (1 ep = 1 row).

    Prefer official log episode lines; fall back to a single row from `_result.txt`
    rate when the log cannot be parsed (keeps smoke/partial runs usable).
    """
    official_rate = _robotwin_rate_from_result(result_txt)
    episodes = _parse_official_episode_outcomes(log_text)
    n_eps = len(episodes)
    # Wall-clock is for the whole official subprocess; split evenly when multi-ep.
    per_ep_time = float(episode_time_sec) / n_eps if n_eps > 0 else float(episode_time_sec)
    common = {
        "benchmark": "robotwin",
        "task_suite": args.task_name,
        "task_id": 0,
        "task_description": args.task_name.replace("_", " "),
        "task_config": args.task_config,
        "ckpt_setting": args.ckpt_setting,
        "steps": int(args.max_steps),
        "returncode": int(returncode),
        "result_path": artifacts.get("result_txt"),
        "official_eval_log": artifacts.get("official_eval_log"),
        "trace_path": artifacts.get("trace_path"),
        "videos": artifacts.get("videos", []),
    }

    if episodes:
        successes = sum(int(ep["success"]) for ep in episodes)
        rate = successes / n_eps
        # Prefer official _result.txt rate when present (canonical RoboTwin metric).
        if official_rate is not None:
            rate = float(official_rate)
        records: List[Dict[str, Any]] = []
        for ep in episodes:
            success = int(ep["success"])
            seed = int(ep["seed"])
            records.append(
                {
                    **common,
                    "episode_idx": int(ep["episode_idx"]),
                    "episode_id": (
                        f"robotwin/{args.task_name}/{args.task_config}/seed={seed}"
                    ),
                    "seed": seed,
                    "success": success,
                    "success_rate": rate,
                    "n_episodes": n_eps,
                    "episode_time_sec": per_ep_time,
                    "failure_reason": None if success else "not_successful",
                }
            )
        return records

    # Fallback: no parseable episodes — one synthetic row (legacy-compatible shape).
    if official_rate is not None:
        success = int(official_rate > 0.0)
        rate = float(official_rate)
    else:
        success = 0
        rate = 0.0
    seed = int(getattr(args, "start_seed_override", None) or args.seed)
    return [
        {
            **common,
            "episode_idx": 0,
            "episode_id": f"robotwin/{args.task_name}/{args.task_config}/seed={seed}",
            "seed": seed,
            "success": success,
            "success_rate": rate,
            "n_episodes": 1,
            "episode_time_sec": float(episode_time_sec),
            "failure_reason": None if success else "not_successful",
        }
    ]


def _write_standard_outputs(
    args: argparse.Namespace,
    returncode: int,
    artifacts: Dict[str, object],
    episode_time_sec: float,
) -> None:
    """Write per-episode results.jsonl + summary CSVs (LIBERO-style aggregation)."""
    result_txt = None
    result_path = artifacts.get("result_txt")
    if result_path:
        path = pathlib.Path(str(result_path))
        if path.is_file():
            result_txt = path.read_text(encoding="utf-8")

    log_text = ""
    log_path = artifacts.get("official_eval_log")
    if log_path:
        lp = pathlib.Path(str(log_path))
        if lp.is_file():
            log_text = lp.read_text(encoding="utf-8", errors="replace")

    records = _build_robotwin_records(
        args, returncode, artifacts, episode_time_sec, result_txt, log_text
    )
    output_dir = pathlib.Path(args.output_dir)
    for record in records:
        append_jsonl(output_dir / "results.jsonl", record)
    write_summary_csv(output_dir / "summary.csv", records)
    write_suite_summary_csv(output_dir / "suite_summary.csv", records)


def run_evaluation(args: argparse.Namespace) -> int:
    """Run run_evaluation."""
    robotwin_path = pathlib.Path(args.robotwin_path)
    eval_script = robotwin_path / "script" / "eval_policy.py"
    if not eval_script.is_file():
        raise FileNotFoundError(f"RoboTwin eval entry does not exist: {eval_script}")
    if not pathlib.Path(args.policy_ckpt_path).exists():
        raise FileNotFoundError(f"Policy checkpoint path does not exist: {args.policy_ckpt_path}")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path: Optional[pathlib.Path] = None
    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    original_step_limit_text: Optional[str] = None
    original_eval_policy_text: Optional[str] = None
    worker_eval_script = _worker_eval_script(args)
    start_time = time.time()
    try:
        if not args.worker_local_files:
            original_step_limit_text = _override_step_limit(robotwin_path, args.task_name, args.step_limit_override)
            original_eval_policy_text = _override_eval_policy_for_smoke(
                robotwin_path,
                args.test_num_override,
                args.disable_expert_check,
                args.start_seed_override,
            )
        if args.keep_config:
            config_path = output_dir / "deploy_policy_vla_eval_robotwin.yml"
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="vla_eval_robotwin_")
            config_path = pathlib.Path(temp_dir.name) / "deploy_policy.yml"
        _write_deploy_policy(args, config_path)

        env = os.environ.copy()
        eval_root = pathlib.Path(getattr(args, "eval_root", "") or pathlib.Path(__file__).resolve().parents[2])
        loongforge_root = pathlib.Path(getattr(args, "loongforge_root", "") or eval_root.parent)
        py_paths = [str(robotwin_path), str(eval_root), str(loongforge_root)]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            py_paths.append(existing_pythonpath)
        env["PYTHONPATH"] = ":".join(py_paths)
        robotwin_bin = str(pathlib.Path(args.robotwin_python).resolve().parent)
        env["PATH"] = f"{robotwin_bin}:{env.get('PATH', '')}"
        if args.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

        command = build_command(args, config_path, worker_eval_script)
        result_dirs_before = _robotwin_result_dirs(args)
        log_path = output_dir / f"{args.task_name}_{args.task_config}_official_eval.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("COMMAND: " + " ".join(command) + "\n")
            log_file.write("PYTHONPATH: " + env["PYTHONPATH"] + "\n")
            log_file.flush()
            try:
                process = subprocess.run(
                    command,
                    cwd=str(robotwin_path),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=args.timeout_sec if args.timeout_sec and args.timeout_sec > 0 else None,
                )
                returncode = int(process.returncode)
            except subprocess.TimeoutExpired:
                log_file.write(f"\nTIMEOUT after {args.timeout_sec} seconds\n")
                log_file.flush()
                returncode = 124
        result_dir = _new_robotwin_result_dir(args, result_dirs_before)
        artifacts = _collect_robotwin_artifacts(args, log_path, config_path, result_dir)
        _write_standard_outputs(args, returncode, artifacts, time.time() - start_time)
        _cleanup_robotwin_work_files(args, log_path, config_path)
        print(
            json.dumps(
                {
                    "returncode": returncode,
                    "artifacts": artifacts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return returncode
    finally:
        if original_eval_policy_text is not None:
            (robotwin_path / "script" / "eval_policy.py").write_text(original_eval_policy_text, encoding="utf-8")
        if original_step_limit_text is not None:
            (robotwin_path / "task_config" / "_eval_step_limit.yml").write_text(
                original_step_limit_text, encoding="utf-8"
            )
        if temp_dir is not None:
            temp_dir.cleanup()


def build_argparser() -> argparse.ArgumentParser:
    """Run build_argparser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--loongforge-root", default=os.environ.get("LOONGFORGE_ROOT", "/workspace/LoongForge-VLA"))
    parser.add_argument(
        "--eval-root",
        default=os.environ.get("EVAL_ROOT", "/workspace/LoongForge-VLA/loongforge/embodied/eval"),
    )
    parser.add_argument("--robotwin-path", default=os.environ.get("ROBOTWIN_PATH", "/workspace/RoboTwin"))
    parser.add_argument("--robotwin-python", default=os.environ.get("ROBOTWIN_PYTHON", sys.executable))
    parser.add_argument("--policy-ckpt-path", default="")
    parser.add_argument("--task-name", default="")
    parser.add_argument("--task-config", choices=["demo_clean", "demo_randomized"], default="demo_clean")
    parser.add_argument("--ckpt-setting", default="loongforge_demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--action-mode", choices=["abs", "delta", "rel"], default="abs")
    # action_bridge / domain_id: YAML only (benchmark.action_bridge, benchmark.domain_id)
    parser.add_argument("--no-action-reorder", action="store_true")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--step-limit-override", type=int, default=None)
    parser.add_argument("--test-num-override", type=int, default=None)
    parser.add_argument("--start-seed-override", type=int, default=None)
    parser.add_argument("--disable-expert-check", action="store_true")
    parser.add_argument("--control-hz", type=int, default=10)
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--disable-action-cache", action="store_true")
    parser.add_argument("--return-action-chunk", action="store_true")
    parser.add_argument("--disable-eval-video-log", action="store_true")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        default="/workspace/LoongForge-VLA/loongforge/embodied/eval/reports/manual/robotwin/official_eval",
    )
    parser.add_argument("--keep-config", action="store_true")
    parser.add_argument("--worker-local-files", action="store_true")
    return parser


def main() -> None:
    """Run main."""
    args = build_argparser().parse_args()
    if args.config:
        args = _apply_config(args, load_config(args.config))
    if not args.policy_ckpt_path:
        raise SystemExit("policy checkpoint must be set by server.ckpt_path in YAML")
    if not args.task_name:
        raise SystemExit("task name must be set by benchmark.task_name in YAML")
    raise SystemExit(run_evaluation(args))


if __name__ == "__main__":
    main()
