# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CALVIN long-horizon evaluator."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import signal
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from loongforge.embodied.eval.adapters.calvin import CALVIN_MAX_STEPS_PER_SUBTASK, CalvinAdapter
from loongforge.embodied.eval.metrics import append_jsonl, load_jsonl
from loongforge.embodied.eval.protocol import PROTOCOL_VERSION
from loongforge.embodied.eval.transport import PolicyClient

INFERENCE_CONFIG = {
    "do_sample": False,
    "use_ddim": True,
    "num_ddim_steps": 10,
}


class StepTimeoutError(TimeoutError):
    """Provide StepTimeoutError behavior."""

    pass


class SequenceTimeoutError(TimeoutError):
    """Provide SequenceTimeoutError behavior."""

    pass


class _AlarmTimeout:
    """Signal-based timeout context for simulator calls."""

    def __init__(self, seconds: float, error_cls):
        self.seconds = seconds
        self.error_cls = error_cls
        self.previous_handler = None

    def __enter__(self):
        if self.seconds <= 0:
            return self

        def _handle_timeout(signum, frame):
            raise self.error_cls(f"Timed out after {self.seconds} seconds")

        self.previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.previous_handler)
        return False


def _canonical_to_policy_payload(
    canonical_obs: Dict[str, Any],
    unnorm_key: Optional[str],
    domain_id: Optional[int] = None,
    state_override: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run _canonical_to_policy_payload."""
    payload = {
        "images": canonical_obs["images"],
        "instruction": canonical_obs["instruction"],
        "episode_id": canonical_obs["meta"]["episode_id"],
        "episode_step": canonical_obs["meta"]["episode_step"],
        "state": state_override if state_override is not None else canonical_obs.get("model_state"),
        **INFERENCE_CONFIG,
    }
    if unnorm_key is not None:
        payload["unnorm_key"] = unnorm_key
    if domain_id is not None:
        payload["domain_id"] = domain_id
    return payload


def _avg(values: List[Optional[float]]) -> Optional[float]:
    """Run _avg."""
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _make_env(dataset_path: str):
    """Create CALVIN environment from an original-format dataset path."""
    val_folder = pathlib.Path(dataset_path) / "validation"
    config_path = val_folder / ".hydra" / "merged_config.yaml"
    if not config_path.exists():
        from calvin_env.envs.play_table_env import get_env

        return get_env(val_folder, show_gui=False)

    from omegaconf import OmegaConf
    import hydra

    cfg = OmegaConf.load(config_path)
    if hasattr(cfg.env, "cameras") and "tactile" in cfg.env.cameras:
        cfg.env.cameras = OmegaConf.create({key: value for key, value in cfg.env.cameras.items() if key != "tactile"})
    cfg.env.show_gui = False
    cfg.env.use_vr = False
    cfg.env.use_scene_info = True
    if hasattr(cfg.env, "use_egl"):
        cfg.env.use_egl = False
    return hydra.utils.instantiate(cfg.env)


def _load_task_oracle(calvin_config_path: str):
    """Load CALVIN task oracle and validation annotations."""
    from omegaconf import OmegaConf
    import hydra

    conf_dir = pathlib.Path(calvin_config_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    return task_oracle, val_annotations


def _stable_seed(value: str) -> int:
    """Return a deterministic 32-bit seed for CALVIN initial-state shuffling."""
    import hashlib

    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16)


def _initial_condition_to_obs(initial_state: Dict[str, Any]) -> Tuple[Any, Any]:
    """Convert CALVIN eval sequence initial state to robot and scene observations."""
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ],
        dtype=np.float64,
    )
    block_rot_z_range = (np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01], dtype=np.float64)
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01], dtype=np.float64)
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01], dtype=np.float64),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01], dtype=np.float64),
    ]

    rng = np.random.default_rng(_stable_seed(str(initial_state.values())))
    rng.shuffle(block_table)
    scene_obs = np.zeros(24, dtype=np.float64)
    if initial_state["slider"] == "left":
        scene_obs[0] = 0.28
    if initial_state["drawer"] == "open":
        scene_obs[1] = 0.22
    if initial_state["lightbulb"] == 1:
        scene_obs[3] = 0.088
    scene_obs[4] = initial_state["lightbulb"]
    scene_obs[5] = initial_state["led"]

    if initial_state["red_block"] == "slider_right":
        scene_obs[6:9] = block_slider_right
    elif initial_state["red_block"] == "slider_left":
        scene_obs[6:9] = block_slider_left
    else:
        scene_obs[6:9] = block_table[0]
    scene_obs[11] = rng.uniform(*block_rot_z_range)

    if initial_state["blue_block"] == "slider_right":
        scene_obs[12:15] = block_slider_right
    elif initial_state["blue_block"] == "slider_left":
        scene_obs[12:15] = block_slider_left
    elif initial_state["red_block"] == "table":
        scene_obs[12:15] = block_table[1]
    else:
        scene_obs[12:15] = block_table[0]
    scene_obs[17] = rng.uniform(*block_rot_z_range)

    if initial_state["pink_block"] == "slider_right":
        scene_obs[18:21] = block_slider_right
    elif initial_state["pink_block"] == "slider_left":
        scene_obs[18:21] = block_slider_left
    else:
        scene_obs[18:21] = block_table[1]
    scene_obs[23] = rng.uniform(*block_rot_z_range)
    return robot_obs, scene_obs


def _save_replay(frames: List[np.ndarray], output_dir: str, sequence_idx: int, success_count: int) -> Optional[str]:
    """Run _save_replay."""
    if not frames:
        return None
    import imageio.v2 as imageio

    artifact_dir = pathlib.Path(output_dir) / "artifacts" / "calvin" / f"sequence{sequence_idx}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    replay_path = artifact_dir / f"replay_successes{success_count}.gif"
    imageio.mimsave(replay_path, frames, duration=0.1)
    return str(replay_path)


def _save_trace(trace: List[Dict[str, Any]], output_dir: str, sequence_idx: int, success_count: int) -> Optional[str]:
    """Run _save_trace."""
    if not trace:
        return None
    artifact_dir = pathlib.Path(output_dir) / "artifacts" / "calvin" / f"sequence{sequence_idx}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / f"trace_successes{success_count}.json"
    with trace_path.open("w", encoding="utf-8") as file:
        json.dump(trace, file, ensure_ascii=False, indent=2)
    return str(trace_path)


def _classify_exception(exc: Exception) -> str:
    """Run _classify_exception."""
    if isinstance(exc, SequenceTimeoutError):
        return "sequence_timeout"
    if isinstance(exc, StepTimeoutError):
        return "env_timeout"
    if isinstance(exc, TimeoutError):
        return "policy_timeout"
    if isinstance(exc, (ConnectionError, OSError)):
        return "server_unreachable"
    return "sequence_error"


def _failure_record(
    *,
    args: argparse.Namespace,
    sequence_idx: int,
    eval_sequence: List[str],
    episode_id: str,
    sequence_seed: int,
    sequence_start: float,
    reset_time_sec: Optional[float],
    success_count: int,
    steps: int,
    failure_reason: str,
    error: Exception,
    replay_path: Optional[str] = None,
    trace_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run _failure_record."""
    return {
        "benchmark": "calvin",
        "task_suite": args.task_suite_name,
        "sequence_idx": sequence_idx,
        "sequence": eval_sequence,
        "seed": sequence_seed,
        "episode_id": episode_id,
        "runtime": "sim",
        "success": 0,
        "success_count": success_count,
        "steps": steps,
        "episode_time_sec": time.time() - sequence_start,
        "reset_time_sec": reset_time_sec,
        "incidents": [],
        "model_name": args.model_name,
        "checkpoint": args.ckpt_path,
        "robot_setup": "franka",
        "control_hz": args.control_hz,
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
            "episode_seed": sequence_seed,
            "protocol_version": PROTOCOL_VERSION,
            "inference_config": dict(INFERENCE_CONFIG),
        },
    }


def run_sequence(
    *,
    client: PolicyClient,
    adapter: CalvinAdapter,
    env,
    task_oracle,
    val_annotations,
    sequence_idx: int,
    initial_state: Dict[str, Any],
    eval_sequence: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run one CALVIN long-horizon evaluation sequence."""
    sequence_seed = int(args.seed) + sequence_idx
    np.random.seed(sequence_seed)
    episode_id = f"calvin/{args.task_suite_name}/sequence={sequence_idx}"
    reset_time_sec: Optional[float] = None
    steps = 0
    success_count = 0
    replay_frames: List[np.ndarray] = []
    trace: List[Dict[str, Any]] = []
    inference_latencies: List[Optional[float]] = []
    e2e_latencies: List[Optional[float]] = []
    sequence_start = time.time()

    try:
        with _AlarmTimeout(args.per_sequence_timeout_sec, SequenceTimeoutError):
            robot_obs, scene_obs = _initial_condition_to_obs(initial_state)
            reset_start = time.time()
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            client.reset(episode_id)
            reset_time_sec = time.time() - reset_start

            for subtask_idx, subtask in enumerate(eval_sequence):
                obs = env.get_obs()
                lang_annotation = str(val_annotations[subtask][0]).split("\n")[0].replace("\u2019", "'")
                start_info = env.get_info()
                client.reset(f"{episode_id}/subtask={subtask_idx}")
                subtask_success = False
                # Official X-VLA calvin client: proprio starts from the current
                # observation and is then backfilled with the raw predicted
                # action (closed-loop: proprio[:10] = action[:10]).
                proprio = None

                for step in range(args.max_steps_per_subtask):
                    if args.save_replay:
                        replay_frames.append(copy.deepcopy(obs["rgb_obs"]["rgb_static"]))
                    canonical_obs = adapter.obs_to_canonical(
                        obs,
                        {
                            "instruction": lang_annotation,
                            "episode_id": f"{episode_id}/subtask={subtask_idx}",
                            "episode_step": step,
                        },
                    )
                    request_start = time.perf_counter()
                    with _AlarmTimeout(args.policy_call_timeout_ms / 1000.0, TimeoutError):
                        response = client.predict_action(
                            **_canonical_to_policy_payload(
                                canonical_obs,
                                args.unnorm_key,
                                domain_id=getattr(args, "domain_id", None),
                                state_override=proprio,
                            )
                        )
                    e2e_latency_ms = (time.perf_counter() - request_start) * 1000.0
                    e2e_latencies.append(e2e_latency_ms)
                    if not response.get("ok", False):
                        raise RuntimeError(f"Policy error: {response}")

                    data = response["data"]
                    inference_latency_ms = data.get("inference_latency_ms")
                    inference_latencies.append(inference_latency_ms)
                    postprocess_key = getattr(args, "action_postprocess", "") or ""
                    raw_chunk = np.asarray(data["actions"], dtype=np.float32)
                    if raw_chunk.ndim == 1:
                        raw_chunk = raw_chunk.reshape(1, -1)
                    if postprocess_key and raw_chunk.shape[-1] >= 10:
                        if proprio is None:
                            base_state = canonical_obs.get("model_state")
                            proprio = (
                                np.array(base_state, dtype=np.float32)
                                if base_state is not None
                                else np.zeros(20, dtype=np.float32)
                            )
                        proprio[:10] = raw_chunk[0, :10]
                    if postprocess_key:
                        # Absolute-pose models (X-VLA ee6d): decode to
                        # pos(3)+quat(4)+grip(1) and step the env with the
                        # (pos, orn, gripper) tuple accepted by calvin_env
                        # robot.apply_action (len==3 -> absolute mode),
                        # matching the official calvin client.
                        from loongforge.embodied.eval.servers.predict_action_interface import postprocess_actions

                        processed = postprocess_actions(raw_chunk, postprocess_key)
                        flat_action = processed[0]
                        env_action = (
                            np.array(flat_action[:3], dtype=np.float64),
                            np.array(flat_action[3:7], dtype=np.float64),
                            int(flat_action[7]),
                        )
                    else:
                        flat_action = raw_chunk.reshape(-1)[:7]
                        env_action = adapter.action_from_canonical({"actions": flat_action})
                        if not env_action.flags.writeable:
                            env_action = np.array(env_action, copy=True)

                    with _AlarmTimeout(args.per_step_timeout_sec, StepTimeoutError):
                        obs, _, _, current_info = env.step(env_action)
                    steps += 1
                    current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
                    subtask_success = len(current_task_info) > 0

                    if args.save_trace:
                        trace.append(
                            {
                                "subtask_idx": subtask_idx,
                                "subtask": subtask,
                                "instruction": lang_annotation,
                                "step": step,
                                "request_id": data.get("request_id"),
                                "raw_action": flat_action.tolist(),
                                "env_action": [np.asarray(part).tolist() for part in env_action]
                                if isinstance(env_action, tuple)
                                else np.asarray(env_action).tolist(),
                                "inference_latency_ms": inference_latency_ms,
                                "e2e_latency_ms": e2e_latency_ms,
                                "chunk_cache_hit": inference_latency_ms is None,
                                "success": bool(subtask_success),
                            }
                        )
                    if subtask_success:
                        success_count += 1
                        break
                if not subtask_success:
                    break
    except Exception as exc:
        failure_reason = _classify_exception(exc)
        replay_path = (
            _save_replay(replay_frames, args.output_dir, sequence_idx, success_count) if args.save_replay else None
        )
        trace_path = _save_trace(trace, args.output_dir, sequence_idx, success_count) if args.save_trace else None
        return _failure_record(
            args=args,
            sequence_idx=sequence_idx,
            eval_sequence=eval_sequence,
            episode_id=episode_id,
            sequence_seed=sequence_seed,
            sequence_start=sequence_start,
            reset_time_sec=reset_time_sec,
            success_count=success_count,
            steps=steps,
            failure_reason=failure_reason,
            error=exc,
            replay_path=replay_path,
            trace_path=trace_path,
        )

    success = success_count == len(eval_sequence)
    replay_path = (
        _save_replay(replay_frames, args.output_dir, sequence_idx, success_count) if args.save_replay else None
    )
    trace_path = _save_trace(trace, args.output_dir, sequence_idx, success_count) if args.save_trace else None
    return {
        "benchmark": "calvin",
        "task_suite": args.task_suite_name,
        "sequence_idx": sequence_idx,
        "sequence": eval_sequence,
        "seed": sequence_seed,
        "episode_id": episode_id,
        "runtime": "sim",
        "success": int(success),
        "success_count": success_count,
        "steps": steps,
        "episode_time_sec": time.time() - sequence_start,
        "reset_time_sec": reset_time_sec,
        "incidents": [],
        "model_name": args.model_name,
        "checkpoint": args.ckpt_path,
        "robot_setup": "franka",
        "control_hz": adapter.control_hz,
        "avg_inference_latency_ms": _avg(inference_latencies),
        "avg_e2e_latency_ms": _avg(e2e_latencies),
        "failure_reason": None if success else "failed_before_sequence_complete",
        "replay_path": replay_path,
        "trace_path": trace_path,
        "server_metadata": client.metadata,
        "repro": {
            "git_sha": None,
            "benchmark_commit": None,
            "docker_image": None,
            "python_env_hash": None,
            "global_seed": args.seed,
            "episode_seed": sequence_seed,
            "protocol_version": PROTOCOL_VERSION,
            "inference_config": dict(INFERENCE_CONFIG),
        },
    }


def _write_summary_csv(path: pathlib.Path, records: List[Dict[str, Any]]) -> None:
    """Write CALVIN sequence-length success summary."""
    counts = defaultdict(int)
    total = len(records)
    for record in records:
        success_count = int(record.get("success_count", 0))
        for index in range(1, 6):
            if success_count >= index:
                counts[index] += 1
    avg_length = sum(int(record.get("success_count", 0)) for record in records) / total if total else 0.0
    with path.open("w", encoding="utf-8") as file:
        file.write("metric,value\n")
        file.write(f"num_sequences,{total}\n")
        file.write(f"avg_length,{avg_length:.6f}\n")
        for index in range(1, 6):
            rate = counts[index] / total if total else 0.0
            file.write(f"task_{index}_success_rate,{rate:.6f}\n")


def run_batch(args: argparse.Namespace) -> Dict[str, Any]:
    """Run CALVIN evaluation batch."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.csv"

    existing_records = load_jsonl(results_path)
    if args.restart and results_path.exists():
        results_path.unlink()
        existing_records = []
    completed = {int(record["sequence_idx"]) for record in existing_records} if args.resume else set()

    with open(args.eval_sequences_path, "r", encoding="utf-8") as file:
        eval_sequences = json.load(file)
    if args.num_sequences > 0:
        eval_sequences = eval_sequences[: args.num_sequences]

    adapter = CalvinAdapter(
        control_hz=args.control_hz,
        max_steps_per_subtask=args.max_steps_per_subtask,
        continuous_gripper=args.continuous_gripper,
        state_format=getattr(args, "state_format", ""),
    )
    task_oracle, val_annotations = _load_task_oracle(args.calvin_config_path)
    env = _make_env(args.dataset_path)
    server_manager = getattr(args, "server_manager", None)
    if server_manager is not None:
        server_manager.ensure_running()
    client = PolicyClient(host=args.host, port=args.port)
    new_records: List[Dict[str, Any]] = []
    start_time = time.time()

    try:
        for sequence_idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
            if sequence_idx in completed:
                continue
            if server_manager is not None:
                server_manager.ensure_running()
            record = run_sequence(
                client=client,
                adapter=adapter,
                env=env,
                task_oracle=task_oracle,
                val_annotations=val_annotations,
                sequence_idx=sequence_idx,
                initial_state=initial_state,
                eval_sequence=eval_sequence,
                args=args,
            )
            append_jsonl(results_path, record)
            new_records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        client.close()
        env.close()

    all_records = existing_records + new_records
    _write_summary_csv(summary_path, all_records)
    avg_length = (
        sum(int(record.get("success_count", 0)) for record in all_records) / len(all_records) if all_records else 0.0
    )
    return {
        "benchmark": "calvin",
        "task_suite": args.task_suite_name,
        "n_records": len(all_records),
        "new_records": len(new_records),
        "avg_length": avg_length,
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "elapsed_sec": time.time() - start_time,
    }


def build_argparser() -> argparse.ArgumentParser:
    """Run build_argparser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--task-suite-name", default="task_D_D")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--calvin-config-path", default="")
    parser.add_argument("--eval-sequences-path", default="")
    parser.add_argument("--num-sequences", type=int, default=10, help="<=0 means all sequences")
    parser.add_argument("--max-steps-per-subtask", type=int, default=CALVIN_MAX_STEPS_PER_SUBTASK)
    parser.add_argument("--control-hz", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unnorm-key", default="franka")
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument("--model-name", default="loongforge-pi05")
    parser.add_argument("--save-replay", action="store_true")
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--policy-call-timeout-ms", type=float, default=5000.0)
    parser.add_argument("--per-step-timeout-sec", type=float, default=30.0)
    parser.add_argument("--per-sequence-timeout-sec", type=float, default=2400.0)
    parser.add_argument("--continuous-gripper", action="store_true", default=False)
    parser.add_argument(
        "--output-dir",
        default="/workspace/LoongForge-VLA/loongforge/embodied/eval/reports/manual/calvin/runner",
    )
    return parser


def main() -> None:
    """Run main."""
    args = build_argparser().parse_args()
    summary = run_batch(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
