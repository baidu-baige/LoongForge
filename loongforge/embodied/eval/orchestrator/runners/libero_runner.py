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
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

from loongforge.embodied.eval.adapters.libero import (
    LIBERO_DUMMY_ACTION, LIBERO_ENV_RESOLUTION, SUITE_MAX_STEPS, LiberoAdapter,
)
from loongforge.embodied.eval.metrics import (
    append_jsonl,
    completed_episode_keys,
    load_jsonl,
    summarize_suites,
    write_eval_report,
    write_suite_summary_csv,
    write_summary_csv,
)
from loongforge.embodied.eval.metrics import repro_metadata
from loongforge.embodied.eval.action_decoders import ActionDecoder
from loongforge.embodied.eval.orchestrator.config import (
    build_rpc_payload,
)
from loongforge.embodied.eval.orchestrator.runners import _common
from loongforge.embodied.eval.orchestrator.runners._common import StepTimeoutError, alarm_timeout, avg as _avg
from loongforge.embodied.eval.orchestrator.runners.libero_gr00t_runner import (
    GROOT_OFFICIAL_PROTOCOL, _run_batch_groot_official,
)
from loongforge.embodied.eval.payload_builders import PayloadBuilder
from loongforge.embodied.eval.protocol import PROTOCOL_VERSION
from loongforge.embodied.eval.transport import PolicyClient

INFERENCE_CONFIG: dict = {}


class EpisodeTimeoutError(TimeoutError):
    """Provide EpisodeTimeoutError behavior."""

    pass


def _render_resolution(args: argparse.Namespace) -> int:
    """Camera render size: optional benchmark.render_resolution, else the shared default.

    Some models' official eval clients render LIBERO natively at a size other
    than ``LIBERO_ENV_RESOLUTION`` (LingBot-VA renders 128), and resizing a
    256-render down is not the same image. Omitting the knob keeps the previous
    behaviour for every existing config.
    """
    value = int(getattr(args, "render_resolution", 0) or 0)
    return value if value > 0 else LIBERO_ENV_RESOLUTION


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


# One env per task, seeded once -- the official eval semantics (`--env-seed-mode per_task`).
# The official harness creates the env outside its 50-trial loop with `cfg.seed` and reuses
# it, so the env's internal RNG advances across trials and each trial's scene is a function
# of (seed, trial order), not of episode_idx alone. LIBERO's own `env.seed()` comment notes
# the seed "seems to affect object positions even when using fixed initial state", so this
# is not cosmetic: with `--env-seed-mode per_episode` (the default, kept for compatibility)
# LoongForge re-creates and re-seeds the env every episode and therefore renders a
# different observation for the same recorded initial state.
_ENV_CACHE: Dict[str, Any] = {"key": None, "env": None}


def _get_cached_env(task, resolution: int, seed: int):
    """Return a per-task cached env, creating it (and seeding it) on first use."""
    key = (str(getattr(task, "language", task)), int(resolution), int(seed))
    if _ENV_CACHE.get("key") != key:
        close_cached_env()
        env, _ = _get_libero_env(task, resolution, seed)
        _ENV_CACHE["key"] = key
        _ENV_CACHE["env"] = env
    return _ENV_CACHE["env"]


def close_cached_env() -> None:
    """Close the cached env if any (called per task switch and at batch exit)."""
    env = _ENV_CACHE.get("env")
    if env is not None:
        try:
            env.close()
        finally:
            _ENV_CACHE["env"] = None
            _ENV_CACHE["key"] = None


def _save_replay(
    frames: List[np.ndarray], output_dir: str, task_suite: str, task_id: int, episode_idx: int, success: bool
) -> Optional[str]:
    """Run _save_replay."""
    if not frames:
        return None
    status = "success" if success else "fail"
    artifact_dir = pathlib.Path(output_dir) / "artifacts" / task_suite / f"task{task_id}" / f"episode{episode_idx}"
    return _common.write_replay_gif(frames, artifact_dir / f"replay_{status}.gif", duration=0.1)


def _save_trace(
    trace: List[Dict[str, Any]], output_dir: str, task_suite: str, task_id: int, episode_idx: int, success: bool
) -> Optional[str]:
    """Run _save_trace."""
    if not trace:
        return None
    status = "success" if success else "fail"
    artifact_dir = pathlib.Path(output_dir) / "artifacts" / task_suite / f"task{task_id}" / f"episode{episode_idx}"
    return _common.write_trace_json(trace, artifact_dir / f"trace_{status}.json")


def _build_rpc_payload(
    payload_builder: PayloadBuilder,
    canonical_obs: Dict[str, Any],
    ctx: Dict[str, Any],
    disable_action_cache: bool = False,
    return_action_chunk: bool = False,
) -> Dict[str, Any]:
    """Thin wrapper preserved for tests that patch this symbol.

    Delegates to the shared ``build_rpc_payload`` in ``orchestrator/config``.
    """
    return build_rpc_payload(
        payload_builder,
        canonical_obs,
        ctx,
        disable_action_cache=disable_action_cache,
        return_action_chunk=return_action_chunk,
    )


def _env_step(env, action, timeout_sec: float):
    """Run _env_step."""
    with alarm_timeout(timeout_sec, StepTimeoutError):
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
            **repro_metadata.repro_fields(("libero",)),
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
    payload_builder: PayloadBuilder,
    action_decoder: ActionDecoder,
    action_decoder_key: str,
    task_suite_name: str,
    task_id: int,
    episode_idx: int,
    task,
    initial_state: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run run_episode."""
    env_seed_mode = str(getattr(args, "env_seed_mode", "per_episode"))
    if env_seed_mode == "per_task":
        # Official semantics: seed once per task (process), env reused across trials.
        episode_seed = int(args.seed)
        if not _ENV_CACHE.get("np_seeded"):
            np.random.seed(episode_seed)
            _ENV_CACHE["np_seeded"] = True
    else:
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
        with alarm_timeout(args.per_episode_timeout_sec, EpisodeTimeoutError):
            if env_seed_mode == "per_task":
                env = _get_cached_env(task, _render_resolution(args), episode_seed)
            else:
                env, task_description = _get_libero_env(task, _render_resolution(args), episode_seed)
            reset_start = time.time()
            client.reset(episode_id)
            payload_builder.reset(episode_id)
            action_decoder.reset()
            env.reset()
            obs = env.set_init_state(initial_state)
            # set_init_state teleports the sim, but the robosuite OSC caches
            # (ee_pos / ee_ori_mat / Jacobians / mass matrix) are only refreshed by
            # `Controller.update()`, which `run_controller()` calls before every sim
            # step and which clears the `new_update` flag. Forcing a refresh here
            # therefore *removes* the refresh the first num_steps_wait dummy step
            # would otherwise do, changing the torques applied during those dummy
            # steps -- i.e. the arm pose and both camera images at the first policy
            # call. The default is therefore ON: that is the legacy LoongForge
            # behaviour every other model's recorded LIBERO numbers were taken with.
            # The official FastWAM eval does NOT force it: measured 2026-09-20 with
            # the same weights/input, the first call's `input_image` was
            # bit-identical to the official harness with the forced update disabled,
            # and differed by max|d|=1.77 (1.3% of pixels) with it enabled -- so the
            # FastWAM eval configs set `benchmark.controller_force_update: false`.
            # LIBERO_CONTROLLER_FORCE_UPDATE=1 forces it on regardless.
            force_controller_update = bool(getattr(args, "controller_force_update", True))
            if os.environ.get("LIBERO_CONTROLLER_FORCE_UPDATE") == "1":
                force_controller_update = True
            if force_controller_update:
                for robot in env.env.robots:
                    robot.controller.update(force=True)
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
                    use_delta = _resolve_libero_use_delta(args, action_decoder_key)
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
                ctx = {
                    "benchmark_name": "libero",
                    "episode_id": episode_id,
                    "episode_step": episode_step,
                    "instruction": task_description,
                    "ee_pos": np.asarray(controller.ee_pos, dtype=np.float32),
                    "ee_ori_mat": np.asarray(controller.ee_ori_mat, dtype=np.float32),
                }
                request_start = time.perf_counter()
                with alarm_timeout(args.policy_call_timeout_ms / 1000.0, TimeoutError):
                    response = client.predict_action(
                        **_build_rpc_payload(
                            payload_builder,
                            canonical_obs,
                            ctx,
                            # Closed-loop-within-chunk models (PayloadBuilder capability)
                            # must be called every env step; others keep the chunk cache.
                            disable_action_cache=bool(
                                getattr(payload_builder, "disable_action_cache", False)
                            ),
                        )
                    )
                e2e_latency_ms = (time.perf_counter() - request_start) * 1000.0
                e2e_latencies.append(e2e_latency_ms)
                if not response.get("ok", False):
                    raise RuntimeError(f"Policy error: {response}")

                data = response["data"]
                payload_builder.update_from_response(data)
                inference_latency_ms = data.get("inference_latency_ms")
                inference_latencies.append(inference_latency_ms)
                raw_chunk = np.asarray(data["actions"], dtype=np.float32)
                if raw_chunk.ndim == 1:
                    raw_chunk = raw_chunk.reshape(1, -1)
                # Log raw grip values before decode for debugging
                if action_decoder_key and raw_chunk.shape[-1] >= 10:
                    raw_grip = float(raw_chunk[0, 9])
                    logging.info("step %d raw grip logit/sigmoid: %.6f", episode_step, raw_grip)
                processed = action_decoder(raw_chunk, {"benchmark_name": "libero"})
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
        # In `per_task` mode the env is cached and reused across trials (official semantics),
        # so it must NOT be closed here -- run_batch closes it on task switch / at exit.
        if env is not None and str(getattr(args, "env_seed_mode", "per_episode")) != "per_task":
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
            **repro_metadata.repro_fields(("libero",)),
            "global_seed": args.seed,
            "episode_seed": episode_seed,
            "protocol_version": PROTOCOL_VERSION,
            "inference_config": dict(INFERENCE_CONFIG),
        },
    }


def _manifest_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    """Return the protocol knobs that affect the measured success rate."""
    model_config = (getattr(args, "raw_config", None) or {}).get("model") or {}
    return {
        "seed": args.seed,
        "episodes_per_task": args.episodes_per_task,
        "num_steps_wait": args.num_steps_wait,
        "max_steps": args.max_steps,
        "env_seed_mode": getattr(args, "env_seed_mode", None),
        "control_mode": getattr(args, "control_mode", None),
        "continuous_gripper": getattr(args, "continuous_gripper", None),
        "prefer_controller_ee": getattr(args, "prefer_controller_ee", None),
        "controller_force_update": getattr(args, "controller_force_update", None),
        "chunk_execute_steps": getattr(args, "chunk_execute_steps", None),
        "noise_seed": model_config.get("noise_seed"),
        "num_inference_steps": model_config.get("num_inference_steps"),
    }


def run_batch(args: argparse.Namespace) -> Dict[str, Any]:
    """Run run_batch."""
    protocol = getattr(args, "libero_protocol", "legacy")
    if protocol == GROOT_OFFICIAL_PROTOCOL:
        return _run_batch_groot_official(args)
    if protocol != "legacy":
        raise ValueError(f"Unknown LIBERO protocol: {protocol!r}")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from libero.libero import benchmark

    output_dir = _common.guard_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.csv"
    suite_summary_path = output_dir / "suite_summary.csv"
    min_episodes = _common.resolve_min_episodes(args)

    existing_records = load_jsonl(results_path)
    if args.restart and results_path.exists():
        results_path.unlink()
        existing_records = []
    completed = completed_episode_keys(existing_records) if args.resume and not args.restart else set()
    expected_counts: Dict[int, int] = {}

    if args.resume and not args.restart and not getattr(args, "_manifest_checked", False):
        repro_metadata.ensure_manifest_matches_checkpoint(output_dir, args.ckpt_path)
    if args.restart or not args.resume or repro_metadata.load_run_manifest(output_dir) is None:
        repro_metadata.write_run_manifest(
            output_dir,
            config=getattr(args, "raw_config", None),
            args=args,
            protocol=_manifest_protocol(args),
            protocol_version=PROTOCOL_VERSION,
            benchmark_modules=("libero",),
        )

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = task_suite.n_tasks if args.max_tasks <= 0 else min(task_suite.n_tasks, args.max_tasks)
    # Optional benchmark.task_ids: explicit task id list (e.g. rerun only
    # low-success tasks); falls back to the first n_tasks tasks.
    selected_task_ids = [int(t) for t in (getattr(args, "task_ids", None) or [])] or list(range(n_tasks))
    adapter = LiberoAdapter(
        suite_name=args.task_suite_name,
        episodes_per_task=args.episodes_per_task,
        continuous_gripper=getattr(args, "continuous_gripper", False),
        resolution=_render_resolution(args),
        prefer_controller_ee=getattr(args, "prefer_controller_ee", True),
    )
    payload_builder, action_decoder_key, action_decoder = _common.build_policy_stack(args, adapter)
    logging.info(
        "libero runner: model_type=%s action_encoding=%s adapter.action_space=%s decoder_key=%s "
        "prefer_controller_ee=%s controller_force_update=%s",
        type(payload_builder).__name__,
        getattr(payload_builder, "action_encoding", ""),
        adapter.action_space,
        action_decoder_key or "<identity>",
        adapter.prefer_controller_ee,
        bool(getattr(args, "controller_force_update", True)),
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
            expected_counts[task_id] = n_episodes
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
                        payload_builder=payload_builder,
                        action_decoder=action_decoder,
                        action_decoder_key=action_decoder_key,
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
        close_cached_env()  # no-op unless --env-seed-mode per_task kept an env alive

    all_records = existing_records + new_records
    write_summary_csv(summary_path, all_records)
    write_suite_summary_csv(
        suite_summary_path,
        all_records,
        min_episodes=min_episodes,
        expected_counts=expected_counts,
    )
    report = None
    if args.generate_report:
        report = write_eval_report(
            results_path,
            output_dir / "report",
            title=f"LIBERO Eval Report: {args.task_suite_name}",
            min_episodes_per_task=min_episodes,
            expected_counts=expected_counts,
        )
    suite_rows = summarize_suites(all_records, min_episodes=min_episodes, expected_counts=expected_counts)
    violations = [row["episode_floor_violations"] for row in suite_rows if not row["episode_floor_met"]]
    successes = sum(int(record.get("success", 0)) for record in all_records)
    infra_failed = sum(int(row.get("infra_failed", 0)) for row in suite_rows)
    evaluated = len(all_records) - infra_failed
    return {
        "benchmark": "libero",
        "task_suite": args.task_suite_name,
        "n_records": len(all_records),
        "new_records": len(new_records),
        "success_rate": successes / evaluated if evaluated else 0.0,
        "infra_failed": infra_failed,
        "ok": not violations,
        "episode_floor_violations": violations,
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "suite_summary_path": str(suite_summary_path),
        "report": report,
        "elapsed_sec": time.time() - start_time,
    }


def _resolve_libero_use_delta(args: Any, action_decoder_key: str = "") -> bool:
    """Resolve OSC use_delta from benchmark.control_mode (absolute|delta|auto).

    auto (default): absolute when an action decoder is composed (xvla-style
    absolute EE via ee6d_to_axis_angle), otherwise delta (pi05 / robosuite
    default). Explicit absolute/delta overrides the inference.
    """
    mode = str(getattr(args, "control_mode", "") or "auto").strip().lower()
    if mode in {"", "auto"}:
        return not bool(action_decoder_key)
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
    parser.add_argument("--libero-protocol", choices=["legacy", "gr00t_official"], default="legacy")
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1, help="<=0 means suite default")
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Global env seed; per-episode seed is seed + task_id*1000 + episode_idx.",
    )
    parser.add_argument(
        "--env-seed-mode",
        default="per_episode",
        choices=["per_episode", "per_task"],
        help=(
            "per_episode (default, legacy): new env + seed per episode "
            "(`seed + task_id*1000 + episode_idx`). "
            "per_task: one env per task seeded with --seed and reused across trials "
            "(official FastWAM eval semantics; see docs §4.8)."
        ),
    )
    parser.add_argument(
        "--prefer-controller-ee",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Source the proprio end-effector position from the OSC controller's cached "
            "ee_pos (default, X-VLA convention) or, with --no-prefer-controller-ee, from the "
            "observation's robot0_eef_pos (official FastWAM convention; the two differ by "
            "sub-millimetre amounts because the controller cache is refreshed inside "
            "run_controller, i.e. before the physics step). Settable from the eval YAML via "
            "benchmark.prefer_controller_ee."
        ),
    )
    parser.add_argument(
        "--controller-force-update",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Force a robosuite OSC cache refresh (ee_pos/ee_ori_mat/Jacobians/mass "
            "matrix) right after set_init_state. Default ON = legacy LoongForge "
            "behaviour; the official FastWAM eval does not force it, so its configs "
            "set benchmark.controller_force_update=false (see the comment in "
            "run_episode: disabling it changes the arm motion during num_steps_wait "
            "and hence the first observation)."
        ),
    )
    parser.add_argument("--unnorm-key", default="franka")
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument("--model-name", default="loongforge")
    parser.add_argument("--save-replay", action="store_true")
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--policy-call-timeout-ms", type=float, default=5000.0)
    parser.add_argument("--per-step-timeout-sec", type=float, default=30.0)
    parser.add_argument("--per-episode-timeout-sec", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--generate-report", action="store_true")
    parser.add_argument(
        "--min-episodes-per-task",
        type=int,
        default=0,
        help="Completeness floor per task; 0 means episodes_per_task.",
    )
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
    if summary.get("ok") is False:
        sys.exit(2)


if __name__ == "__main__":
    main()
