# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LIBERO batch evaluator for M1 standalone eval.

Runs multiple LIBERO tasks/episodes against one already-running policy server and
writes per-episode results to results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import signal
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import numpy as np

from loongforge.embodied.eval.adapters.libero import (
    LIBERO_DUMMY_ACTION, LIBERO_ENV_RESOLUTION, SUITE_MAX_STEPS, LiberoAdapter,
)
from loongforge.embodied.eval.metrics import (
    append_jsonl,
    completed_episode_keys,
    load_jsonl,
    write_eval_report,
    write_suite_summary_csv,
    write_summary_csv,
)
from loongforge.embodied.eval.protocol import PROTOCOL_VERSION
from loongforge.embodied.eval.transport import PolicyClient

INFERENCE_CONFIG: dict = {}


class StepTimeoutError(TimeoutError):
    """Provide StepTimeoutError behavior."""

    pass


class EpisodeTimeoutError(TimeoutError):
    """Provide EpisodeTimeoutError behavior."""

    pass


@contextmanager
def _alarm_timeout(seconds: float, error_cls):
    """Run _alarm_timeout."""
    if seconds <= 0:
        yield
        return

    def _handle_timeout(signum, frame):
        """Run _handle_timeout."""
        raise error_cls(f"Timed out after {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _get_libero_env(task, resolution: int, seed: int):
    """Run _get_libero_env."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _extract_primary_frame(obs: Dict[str, Any]) -> np.ndarray:
    """Run _extract_primary_frame."""
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def _save_replay(
    frames: List[np.ndarray], output_dir: str, task_suite: str, task_id: int, episode_idx: int, success: bool
) -> Optional[str]:
    """Run _save_replay."""
    if not frames:
        return None

    import imageio.v2 as imageio

    status = "success" if success else "fail"
    artifact_dir = pathlib.Path(output_dir) / "artifacts" / task_suite / f"task{task_id}" / f"episode{episode_idx}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    replay_path = artifact_dir / f"replay_{status}.gif"
    imageio.mimsave(replay_path, frames, duration=0.1)
    return str(replay_path)


def _save_trace(
    trace: List[Dict[str, Any]], output_dir: str, task_suite: str, task_id: int, episode_idx: int, success: bool
) -> Optional[str]:
    """Run _save_trace."""
    if not trace:
        return None

    status = "success" if success else "fail"
    artifact_dir = pathlib.Path(output_dir) / "artifacts" / task_suite / f"task{task_id}" / f"episode{episode_idx}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / f"trace_{status}.json"
    with trace_path.open("w", encoding="utf-8") as file:
        json.dump(trace, file, ensure_ascii=False, indent=2)
    return str(trace_path)


def _canonical_to_policy_payload(canonical_obs: Dict[str, Any], domain_id: Optional[int] = None) -> Dict[str, Any]:
    """Run _canonical_to_policy_payload."""
    payload = {
        "images": canonical_obs["images"],
        "instruction": canonical_obs["instruction"],
        "episode_id": canonical_obs["meta"]["episode_id"],
        "episode_step": canonical_obs["meta"]["episode_step"],
        "state": canonical_obs.get("model_state"),
    }
    if domain_id is not None:
        payload["domain_id"] = domain_id
    return payload


def _avg(values: List[Optional[float]]) -> Optional[float]:
    """Run _avg."""
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _env_step(env, action, timeout_sec: float):
    """Run _env_step."""
    with _alarm_timeout(timeout_sec, StepTimeoutError):
        return env.step(action)


def _failure_record(
    *,
    args: argparse.Namespace,
    adapter: LiberoAdapter,
    task_suite_name: str,
    task_id: int,
    episode_idx: int,
    task_description: str,
    episode_id: str,
    episode_seed: int,
    episode_start: float,
    reset_time_sec: Optional[float],
    steps: int,
    failure_reason: str,
    error: Exception,
    replay_path: Optional[str] = None,
    trace_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run _failure_record."""
    return {
        "benchmark": "libero",
        "task_suite": task_suite_name,
        "task_id": task_id,
        "task": task_description,
        "seed": episode_seed,
        "episode_idx": episode_idx,
        "episode_id": episode_id,
        "runtime": "sim",
        "success": 0,
        "steps": steps,
        "safety_cost": None,
        "episode_time_sec": time.time() - episode_start,
        "reset_time_sec": reset_time_sec,
        "incidents": [],
        "model_name": args.model_name,
        "checkpoint": args.ckpt_path,
        "robot_setup": "franka",
        "control_hz": adapter.control_hz,
        "avg_inference_latency_ms": None,
        "avg_e2e_latency_ms": None,
        "failure_reason": failure_reason,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "replay_path": replay_path,
        "trace_path": trace_path,
        "server_metadata": None,
        "repro": {
            "git_sha": None,
            "benchmark_commit": None,
            "docker_image": None,
            "python_env_hash": None,
            "global_seed": args.seed,
            "episode_seed": episode_seed,
            "protocol_version": PROTOCOL_VERSION,
            "inference_config": dict(INFERENCE_CONFIG),
        },
    }


def _classify_exception(exc: Exception) -> str:
    """Run _classify_exception."""
    if isinstance(exc, EpisodeTimeoutError):
        return "episode_timeout"
    if isinstance(exc, StepTimeoutError):
        return "env_timeout"
    if isinstance(exc, TimeoutError):
        return "policy_timeout"
    if isinstance(exc, (ConnectionError, OSError)):
        return "server_unreachable"
    return "episode_error"


def run_episode(
    *,
    client: PolicyClient,
    adapter: LiberoAdapter,
    task_suite_name: str,
    task_id: int,
    episode_idx: int,
    task,
    initial_state: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run run_episode."""
    episode_seed = int(args.seed) + task_id * 1000 + episode_idx
    np.random.seed(episode_seed)

    task_description = task.language
    episode_id = f"libero/{task_suite_name}/task={task_id}/episode={episode_idx}"
    reset_time_sec: Optional[float] = None
    steps = 0
    replay_frames: List[np.ndarray] = []
    trace: List[Dict[str, Any]] = []
    inference_latencies: List[Optional[float]] = []
    e2e_latencies: List[Optional[float]] = []
    episode_start = time.time()
    env = None

    try:
        with _alarm_timeout(args.per_episode_timeout_sec, EpisodeTimeoutError):
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, episode_seed)
            reset_start = time.time()
            client.reset(episode_id)
            env.reset()
            obs = env.set_init_state(initial_state)
            reset_time_sec = time.time() - reset_start

            # Explicit benchmark.max_steps takes priority over the suite
            # default so long-horizon policies (e.g. X-VLA, horizon 800) are
            # not clamped by SUITE_MAX_STEPS.
            max_steps = args.max_steps if args.max_steps > 0 else SUITE_MAX_STEPS[task_suite_name]
            done = False
            info: Dict[str, Any] = {}

            for raw_step in range(max_steps + args.num_steps_wait):
                if args.save_replay and raw_step >= args.num_steps_wait:
                    replay_frames.append(_extract_primary_frame(obs))

                if raw_step < args.num_steps_wait:
                    obs, reward, done, info = _env_step(env, LIBERO_DUMMY_ACTION, args.per_step_timeout_sec)
                    continue

                # OSC absolute vs delta: see _resolve_libero_use_delta().
                # Default auto: absolute when action_postprocess is set (xvla),
                # else keep robosuite delta (pi05).
                if raw_step == args.num_steps_wait:
                    use_delta = _resolve_libero_use_delta(args)
                    for robot in env.env.robots:
                        robot.controller.use_delta = use_delta

                episode_step = raw_step - args.num_steps_wait
                controller = env.env.robots[0].controller
                canonical_obs = adapter.obs_to_canonical(
                    obs,
                    {
                        "instruction": task_description,
                        "episode_id": episode_id,
                        "episode_step": episode_step,
                        "ee_pos": np.asarray(controller.ee_pos, dtype=np.float32),
                        "ee_ori_mat": np.asarray(controller.ee_ori_mat, dtype=np.float32),
                    },
                )
                request_start = time.perf_counter()
                with _alarm_timeout(args.policy_call_timeout_ms / 1000.0, TimeoutError):
                    response = client.predict_action(**_canonical_to_policy_payload(
                        canonical_obs, domain_id=getattr(args, "domain_id", None)
                    ))
                e2e_latency_ms = (time.perf_counter() - request_start) * 1000.0
                e2e_latencies.append(e2e_latency_ms)
                if not response.get("ok", False):
                    raise RuntimeError(f"Policy error: {response}")

                data = response["data"]
                inference_latency_ms = data.get("inference_latency_ms")
                inference_latencies.append(inference_latency_ms)
                from loongforge.embodied.eval.servers.predict_action_interface import postprocess_actions
                raw_chunk = np.asarray(data["actions"], dtype=np.float32)
                if raw_chunk.ndim == 1:
                    raw_chunk = raw_chunk.reshape(1, -1)
                postprocess_key = getattr(args, "action_postprocess", "") or ""
                # Log raw grip values before postprocess for debugging
                if postprocess_key and raw_chunk.shape[-1] >= 10:
                    raw_grip = float(raw_chunk[0, 9])
                    logging.info("step %d raw grip logit/sigmoid: %.6f", episode_step, raw_grip)
                processed = postprocess_actions(raw_chunk, postprocess_key)
                # Absolute targets are executed as-is when use_delta=False;
                # delta models (pi05) keep the OSC default use_delta=True.
                flat_action = processed[0]
                env_action = adapter.action_from_canonical({"actions": flat_action})
                obs, reward, done, info = _env_step(env, env_action, args.per_step_timeout_sec)
                steps += 1

                if args.save_trace:
                    trace.append(
                        {
                            "step": episode_step,
                            "request_id": data.get("request_id"),
                            "raw_action": flat_action.tolist(),
                            "env_action": env_action,
                            "inference_latency_ms": inference_latency_ms,
                            "e2e_latency_ms": e2e_latency_ms,
                            "chunk_cache_hit": inference_latency_ms is None,
                            "done": bool(done),
                            "success": bool(info.get("success", False)),
                        }
                    )

                if done or bool(info.get("success", False)):
                    break
    except Exception as exc:
        failure_reason = _classify_exception(exc)
        replay_path = (
            _save_replay(replay_frames, args.output_dir, task_suite_name, task_id, episode_idx, False)
            if args.save_replay
            else None
        )
        trace_path = (
            _save_trace(trace, args.output_dir, task_suite_name, task_id, episode_idx, False)
            if args.save_trace
            else None
        )
        return _failure_record(
            args=args,
            adapter=adapter,
            task_suite_name=task_suite_name,
            task_id=task_id,
            episode_idx=episode_idx,
            task_description=task_description,
            episode_id=episode_id,
            episode_seed=episode_seed,
            episode_start=episode_start,
            reset_time_sec=reset_time_sec,
            steps=steps,
            failure_reason=failure_reason,
            error=exc,
            replay_path=replay_path,
            trace_path=trace_path,
        )
    finally:
        if env is not None:
            env.close()

    success = bool(info.get("success", done))
    failure_reason = None if success else "not_successful_within_max_steps"
    replay_path = (
        _save_replay(replay_frames, args.output_dir, task_suite_name, task_id, episode_idx, success)
        if args.save_replay
        else None
    )
    trace_path = (
        _save_trace(trace, args.output_dir, task_suite_name, task_id, episode_idx, success) if args.save_trace else None
    )

    return {
        "benchmark": "libero",
        "task_suite": task_suite_name,
        "task_id": task_id,
        "task": task_description,
        "seed": episode_seed,
        "episode_idx": episode_idx,
        "episode_id": episode_id,
        "runtime": "sim",
        "success": int(success),
        "steps": steps,
        "safety_cost": None,
        "episode_time_sec": time.time() - episode_start,
        "reset_time_sec": reset_time_sec,
        "incidents": [],
        "model_name": args.model_name,
        "checkpoint": args.ckpt_path,
        "robot_setup": "franka",
        "control_hz": adapter.control_hz,
        "avg_inference_latency_ms": _avg(inference_latencies),
        "avg_e2e_latency_ms": _avg(e2e_latencies),
        "failure_reason": failure_reason,
        "replay_path": replay_path,
        "trace_path": trace_path,
        "server_metadata": client.metadata,
        "repro": {
            "git_sha": None,
            "benchmark_commit": None,
            "docker_image": None,
            "python_env_hash": None,
            "global_seed": args.seed,
            "episode_seed": episode_seed,
            "protocol_version": PROTOCOL_VERSION,
            "inference_config": dict(INFERENCE_CONFIG),
        },
    }


def run_batch(args: argparse.Namespace) -> Dict[str, Any]:
    """Run run_batch."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from libero.libero import benchmark

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.csv"
    suite_summary_path = output_dir / "suite_summary.csv"

    existing_records = load_jsonl(results_path)
    completed = completed_episode_keys(existing_records) if args.resume else set()
    if args.restart and results_path.exists():
        results_path.unlink()
        completed = set()
        existing_records = []

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = task_suite.n_tasks if args.max_tasks <= 0 else min(task_suite.n_tasks, args.max_tasks)
    # Optional benchmark.task_ids: explicit task id list (e.g. rerun only
    # low-success tasks); falls back to the first n_tasks tasks.
    selected_task_ids = [int(t) for t in (getattr(args, "task_ids", None) or [])] or list(range(n_tasks))
    adapter = LiberoAdapter(
        suite_name=args.task_suite_name,
        episodes_per_task=args.episodes_per_task,
        continuous_gripper=getattr(args, "continuous_gripper", False),
        state_format=getattr(args, "state_format", ""),
    )
    server_manager = getattr(args, "server_manager", None)
    if server_manager is not None:
        server_manager.ensure_running()
    client = PolicyClient(host=args.host, port=args.port)
    new_records: List[Dict[str, Any]] = []
    start_time = time.time()

    try:
        for task_id in selected_task_ids:
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            n_episodes = min(args.episodes_per_task, len(initial_states))
            for episode_idx in range(n_episodes):
                key = ("libero", args.task_suite_name, task_id, episode_idx)
                if key in completed:
                    continue
                record = None
                for attempt in range(args.max_retries + 1):
                    if server_manager is not None:
                        server_manager.ensure_running()
                    record = run_episode(
                        client=client,
                        adapter=adapter,
                        task_suite_name=args.task_suite_name,
                        task_id=task_id,
                        episode_idx=episode_idx,
                        task=task,
                        initial_state=initial_states[episode_idx],
                        args=args,
                    )
                    record["attempt"] = attempt
                    if record.get("success") or record.get("failure_reason") not in {
                        "policy_timeout",
                        "env_timeout",
                        "episode_timeout",
                        "server_unreachable",
                        "episode_error",
                    }:
                        break
                    if attempt < args.max_retries:
                        if record.get("failure_reason") == "server_unreachable" and server_manager is not None:
                            client.close()
                            server_manager.ensure_running()
                            client = PolicyClient(host=args.host, port=args.port)
                        time.sleep(min(2**attempt, 8))
                append_jsonl(results_path, record)
                new_records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        client.close()

    all_records = existing_records + new_records
    write_summary_csv(summary_path, all_records)
    write_suite_summary_csv(suite_summary_path, all_records, min_episodes=args.min_episodes_per_task)
    report = None
    if args.generate_report:
        report = write_eval_report(
            results_path,
            output_dir / "report",
            title=f"LIBERO Eval Report: {args.task_suite_name}",
            min_episodes_per_task=args.min_episodes_per_task,
        )
    successes = sum(int(record.get("success", 0)) for record in all_records)
    return {
        "benchmark": "libero",
        "task_suite": args.task_suite_name,
        "n_records": len(all_records),
        "new_records": len(new_records),
        "success_rate": successes / len(all_records) if all_records else 0.0,
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "suite_summary_path": str(suite_summary_path),
        "report": report,
        "elapsed_sec": time.time() - start_time,
    }


def _resolve_libero_use_delta(args: Any) -> bool:
    """Resolve OSC use_delta from benchmark.control_mode (absolute|delta|auto).

    auto (default): absolute when ``action_postprocess`` is set (xvla-style
    absolute EE), otherwise delta (pi05 / robosuite default). Explicit
    absolute/delta overrides the inference for models that need postprocess
    while still using incremental control (or the reverse).
    """
    mode = str(getattr(args, "control_mode", "") or "auto").strip().lower()
    if mode in {"", "auto"}:
        return not bool(getattr(args, "action_postprocess", "") or "")
    if mode in {"absolute", "abs", "absolute_ee", "ee_absolute"}:
        return False
    if mode in {"delta", "relative", "incremental"}:
        return True
    raise ValueError(
        f"Unknown benchmark.control_mode for LIBERO: {mode!r}. "
        "Use auto | absolute | delta."
    )


def build_argparser() -> argparse.ArgumentParser:
    """Run build_argparser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--task-suite-name", default="libero_goal")
    parser.add_argument("--max-tasks", type=int, default=1, help="<=0 means all tasks in suite")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1, help="<=0 means suite default")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unnorm-key", default="franka")
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument("--model-name", default="loongforge")
    parser.add_argument("--save-replay", action="store_true")
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--policy-call-timeout-ms", type=float, default=5000.0)
    parser.add_argument("--per-step-timeout-sec", type=float, default=30.0)
    parser.add_argument("--per-episode-timeout-sec", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--generate-report", action="store_true")
    parser.add_argument("--min-episodes-per-task", type=int, default=1)
    parser.add_argument("--continuous-gripper", action="store_true", default=False,
                        help="Pass raw gripper value directly to LIBERO without binarization")
    parser.add_argument(
        "--control-mode",
        default="auto",
        choices=["auto", "absolute", "delta"],
        help="LIBERO OSC: auto (absolute iff action_postprocess set), absolute, or delta",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/LoongForge-VLA/loongforge/embodied/eval/reports/manual/libero/runner",
    )
    return parser


def main() -> None:
    """Run main."""
    args = build_argparser().parse_args()
    summary = run_batch(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
