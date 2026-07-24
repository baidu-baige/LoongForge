# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SimplerEnv single-episode runner using the standalone eval module."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import stat
import sys
import time
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np

from loongforge.embodied.eval.adapters.simplerenv import (
    SIMPLERENV_DEFAULT_MAX_STEPS,
    SimplerEnvAdapter,
    build_widowx_initial_model_state,
)
from loongforge.embodied.eval.metrics.results import append_jsonl, write_suite_summary_csv, write_summary_csv
from loongforge.embodied.eval.orchestrator.config import load_config
from loongforge.embodied.eval.transport import PolicyClient


def _canonical_to_legacy_payload(
    canonical_obs: Dict[str, Any],
    args: argparse.Namespace,
    state_override: Any = None,
) -> Dict[str, Any]:
    """Run _canonical_to_legacy_payload."""
    payload = {
        "images": canonical_obs["images"],
        "instruction": canonical_obs["instruction"],
        "episode_id": canonical_obs["meta"]["episode_id"],
        "episode_step": canonical_obs["meta"]["episode_step"],
        "state": state_override if state_override is not None else canonical_obs.get("model_state"),
        "disable_action_cache": args.disable_action_cache,
        "return_action_chunk": args.action_ensemble,
        "cfg_scale": args.cfg_scale,
    }
    domain_id = getattr(args, "domain_id", None)
    if domain_id is not None:
        payload["domain_id"] = domain_id
    return payload


def _vulkan_runtime_env(args: argparse.Namespace) -> Dict[str, str]:
    """Return environment variables needed by SAPIEN's Vulkan renderer."""
    env = os.environ.copy()
    library_paths = env.get("LD_LIBRARY_PATH", "").split(":") if env.get("LD_LIBRARY_PATH") else []
    for path in reversed([args.nvidia_lib_dir or "/ssd1/opt/nvidia_lib", "/usr/lib64"]):
        if path and path not in library_paths:
            library_paths.insert(0, path)
    env["LD_LIBRARY_PATH"] = ":".join(library_paths)
    env["VK_ICD_FILENAMES"] = args.nvidia_icd_json or env.get("VK_ICD_FILENAMES", "/ssd1/opt/nvidia_lib/10_nvidia.json")
    runtime_dir = pathlib.Path(env.get("XDG_RUNTIME_DIR") or f"/tmp/runtime-{os.getuid()}")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.chmod(stat.S_IRWXU)
    env["XDG_RUNTIME_DIR"] = str(runtime_dir)
    return env


def _setup_simplerenv_runtime(args: argparse.Namespace) -> None:
    """Run _setup_simplerenv_runtime."""
    os.environ.update(_vulkan_runtime_env(args))


def _ensure_vulkan_runtime(args: argparse.Namespace) -> None:
    """Re-exec once so LD_LIBRARY_PATH takes effect before importing SAPIEN."""
    marker = "LOONGFORGE_SIMPLERENV_RUNTIME_READY"
    if os.environ.get(marker) == "1":
        return
    env = _vulkan_runtime_env(args)
    env[marker] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def _build_env(args: argparse.Namespace) -> Tuple[Any, str]:
    """Run _build_env."""
    import simpler_env
    from simpler_env.utils.env.env_builder import build_maniskill2_env

    simpler_env_root = pathlib.Path(simpler_env.__file__).resolve().parents[1]
    overlay_path = args.rgb_overlay_path
    if overlay_path is None and args.robot_setup.startswith("widowx"):
        overlay_path = str(simpler_env_root / "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png")

    adapter = SimplerEnvAdapter(
        task_name=args.task_name,
        robot_setup=args.robot_setup,
        control_hz=args.control_freq,
        max_steps=args.max_steps if args.max_steps > 0 else SIMPLERENV_DEFAULT_MAX_STEPS,
        camera_name=args.camera_name,
        action_scale=args.action_scale,
        rotation_mode=args.rotation_mode,
    )
    kwargs: Dict[str, Any] = {
        "obs_mode": "rgbd",
        "robot": args.robot_setup,
        "sim_freq": args.sim_freq,
        "control_mode": args.control_mode,
        "control_freq": args.control_freq,
        "max_episode_steps": args.max_steps if args.max_steps > 0 else SIMPLERENV_DEFAULT_MAX_STEPS,
        "camera_cfgs": {"add_segmentation": True},
    }
    if args.scene_name:
        kwargs["scene_name"] = args.scene_name
    if overlay_path:
        kwargs["rgb_overlay_path"] = overlay_path

    env = build_maniskill2_env(adapter.env_name, **kwargs)
    return env, adapter.env_name


def _reset_env(env: Any, args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run _reset_env."""
    reset_options: Dict[str, Any] = {
        "robot_init_options": {
            "init_xy": np.asarray([args.robot_init_x, args.robot_init_y], dtype=np.float32),
            "init_rot_quat": np.asarray(args.robot_init_quat, dtype=np.float32),
        },
        "obj_init_options": {"episode_id": args.obj_episode_id},
    }
    reset_result = env.reset(seed=args.seed + args.episode_idx, options=reset_options)
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, reset_info = reset_result
    else:
        obs, reset_info = reset_result, {}
    return obs, reset_info


def _get_instruction(env: Any, fallback: str) -> str:
    """Run _get_instruction."""
    if hasattr(env, "get_language_instruction"):
        return str(env.get_language_instruction())
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "get_language_instruction"):
        return str(unwrapped.get_language_instruction())
    return fallback


def _save_replay(frames: List[np.ndarray], output_dir: str, task_name: str, episode_idx: int, success: bool) -> str:
    """Run _save_replay."""
    if not frames:
        raise ValueError("Cannot save replay with no frames")

    import imageio.v2 as imageio

    status = "success" if success else "fail"
    artifact_dir = pathlib.Path(output_dir) / "artifacts" / "simplerenv" / task_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    replay_path = artifact_dir / f"ep{episode_idx}_{status}.gif"
    imageio.mimsave(replay_path, frames, duration=0.2)
    return str(replay_path)


def _save_trace(trace: List[Dict[str, Any]], output_dir: str, task_name: str, episode_idx: int, success: bool) -> str:
    """Run _save_trace."""
    status = "success" if success else "fail"
    artifact_dir = pathlib.Path(output_dir) / "artifacts" / "simplerenv" / task_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / f"ep{episode_idx}_{status}_trace.json"
    with trace_path.open("w", encoding="utf-8") as file:
        json.dump(trace, file, ensure_ascii=False, indent=2)
    return str(trace_path)


def _append_replay_frame(adapter: SimplerEnvAdapter, obs: Dict[str, Any], frames: List[np.ndarray]) -> None:
    """Run _append_replay_frame."""
    frames.append(np.asarray(obs["image"][adapter.camera_name]["rgb"]))


def _ensemble_action(action_history: deque[np.ndarray], current_actions: np.ndarray, alpha: float) -> np.ndarray:
    """Run _ensemble_action."""
    action_history.append(current_actions)
    num_actions = len(action_history)
    current_predictions = np.stack(
        [pred_actions[index] for index, pred_actions in zip(range(num_actions - 1, -1, -1), action_history)]
    )
    reference = current_predictions[num_actions - 1]
    dot_product = np.sum(current_predictions * reference, axis=1)
    norm_previous = np.linalg.norm(current_predictions, axis=1)
    norm_reference = np.linalg.norm(reference)
    cosine_similarity = dot_product / (norm_previous * norm_reference + 1e-7)
    weights = np.exp(alpha * cosine_similarity)
    weights = weights / weights.sum()
    return np.sum(weights[:, None] * current_predictions, axis=0)


def _json_safe(value: Any) -> Any:
    """Run _json_safe."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_standard_outputs(output_dir: str, record: Dict[str, Any]) -> None:
    """Run _write_standard_outputs."""
    out_dir = pathlib.Path(output_dir)
    safe_record = _json_safe(record)
    append_jsonl(out_dir / "results.jsonl", safe_record)
    write_summary_csv(out_dir / "summary.csv", [safe_record])
    write_suite_summary_csv(out_dir / "suite_summary.csv", [safe_record])


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run run_evaluation."""
    _ensure_vulkan_runtime(args)
    np.random.seed(args.seed)

    adapter = SimplerEnvAdapter(
        task_name=args.task_name,
        robot_setup=args.robot_setup,
        control_hz=args.control_freq,
        max_steps=args.max_steps if args.max_steps > 0 else SIMPLERENV_DEFAULT_MAX_STEPS,
        camera_name=args.camera_name,
        action_scale=args.action_scale,
        rotation_mode=args.rotation_mode,
    )
    env, env_name = _build_env(args)
    client = PolicyClient(host=args.host, port=args.port)

    episode_id = f"simplerenv/{args.task_name}/episode={args.episode_idx}"
    client.reset(episode_id)
    obs, reset_info = _reset_env(env, args)
    instruction = _get_instruction(env, args.instruction or args.task_name.replace("_", " "))

    max_steps = args.max_steps if args.max_steps > 0 else SIMPLERENV_DEFAULT_MAX_STEPS
    done = False
    truncated = False
    info: Dict[str, Any] = {}
    steps = 0
    replay_frames: List[np.ndarray] = []
    action_trace: List[Dict[str, Any]] = []
    action_history: deque[np.ndarray] = deque(maxlen=args.action_ensemble_horizon)
    start_time = time.time()

    postprocess_key = getattr(args, "action_postprocess", "") or ""
    # X-VLA protocol: initial proprio from the env, then closed-loop backfill
    # with the raw predicted action (official client: proprio[:10] = action[:10]).
    proprio = build_widowx_initial_model_state(obs) if postprocess_key else None

    try:
        for episode_step in range(max_steps):
            canonical_obs = adapter.obs_to_canonical(
                obs,
                {
                    "instruction": instruction,
                    "episode_id": episode_id,
                    "episode_step": episode_step,
                },
            )
            if args.save_replay:
                _append_replay_frame(adapter, obs, replay_frames)

            response = client.predict_action(
                **_canonical_to_legacy_payload(canonical_obs, args, state_override=proprio)
            )
            if not response.get("ok", False):
                raise RuntimeError(f"Policy error: {response}")

            raw_chunk = np.asarray(response["data"]["actions"], dtype=np.float32)
            if raw_chunk.ndim == 1:
                raw_chunk = raw_chunk.reshape(1, -1)
            if postprocess_key:
                from loongforge.embodied.eval.servers.predict_action_interface import postprocess_actions

                raw_row = raw_chunk[0]
                if proprio is not None and raw_row.size >= 10:
                    proprio[:10] = raw_row[:10]
                env_action = postprocess_actions(raw_chunk[0:1], postprocess_key)[0].astype(np.float32)
                flat_action = raw_row
            else:
                action_chunk = raw_chunk.reshape(-1, 7)
                flat_action = action_chunk[0]
                if args.action_ensemble:
                    flat_action = _ensemble_action(action_history, action_chunk, args.action_ensemble_alpha)
                env_action = adapter.action_from_canonical({"actions": flat_action})
            obs, reward, done, truncated, info = env.step(env_action)
            steps += 1
            if args.save_replay:
                _append_replay_frame(adapter, obs, replay_frames)

            if args.save_trace:
                action_trace.append(
                    {
                        "step": episode_step,
                        "raw_action": flat_action.tolist(),
                        "env_action": np.asarray(env_action).tolist(),
                        "reward": float(reward),
                        "done": bool(done),
                        "truncated": bool(truncated),
                        "success": bool(info.get("success", False)),
                        "inference_latency_ms": response.get("data", {}).get("inference_latency_ms"),
                    }
                )

            if done or truncated or bool(info.get("success", False)):
                if args.save_replay and args.success_settle_steps > 0:
                    settle_action = np.zeros_like(env_action, dtype=np.float32)
                    settle_action[-1] = args.success_settle_gripper
                    for _ in range(args.success_settle_steps):
                        obs, _, _, _, _ = env.step(settle_action)
                        _append_replay_frame(adapter, obs, replay_frames)
                break
    finally:
        client.close()
        env.close()

    success = bool(info.get("success", done))
    replay_path = (
        _save_replay(replay_frames, args.output_dir, args.task_name, args.episode_idx, success)
        if args.save_replay
        else None
    )
    trace_path = (
        _save_trace(action_trace, args.output_dir, args.task_name, args.episode_idx, success)
        if args.save_trace
        else None
    )

    record = {
        "benchmark": "simplerenv",
        "task_suite": args.task_name,
        "task_id": 0,
        "episode_idx": args.episode_idx,
        "episode_id": episode_id,
        "task_description": instruction,
        "env_name": env_name,
        "robot_setup": args.robot_setup,
        "reset_info": reset_info,
        "success": int(success),
        "steps": steps,
        "episode_time_sec": time.time() - start_time,
        "failure_reason": None if success else "not_successful_within_max_steps",
        "replay_path": replay_path,
        "trace_path": trace_path,
    }
    _write_standard_outputs(args.output_dir, record)
    return record


def _apply_config(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    """Run _apply_config."""
    benchmark = config.get("benchmark") or {}
    server = config.get("server") or {}
    run = config.get("run") or {}
    env = config.get("env") or {}

    if not isinstance(benchmark, dict) or not isinstance(server, dict) or not isinstance(run, dict):
        raise ValueError("benchmark, server, and run sections must be mappings")

    mapping = [
        (benchmark, "task_name", "task_name"),
        (benchmark, "robot_setup", "robot_setup"),
        (benchmark, "camera_name", "camera_name"),
        (benchmark, "instruction", "instruction"),
        (benchmark, "scene_name", "scene_name"),
        (benchmark, "rgb_overlay_path", "rgb_overlay_path"),
        (benchmark, "sim_freq", "sim_freq"),
        (benchmark, "control_freq", "control_freq"),
        (benchmark, "control_mode", "control_mode"),
        (benchmark, "action_scale", "action_scale"),
        (benchmark, "rotation_mode", "rotation_mode"),
        (benchmark, "max_steps", "max_steps"),
        (benchmark, "domain_id", "domain_id"),
        (benchmark, "action_postprocess", "action_postprocess"),
        (benchmark, "success_settle_steps", "success_settle_steps"),
        (benchmark, "success_settle_gripper", "success_settle_gripper"),
        (benchmark, "robot_init_x", "robot_init_x"),
        (benchmark, "robot_init_y", "robot_init_y"),
        (benchmark, "robot_init_quat", "robot_init_quat"),
        (server, "host", "host"),
        (server, "port", "port"),
        (run, "seed", "seed"),
        (run, "episode_idx", "episode_idx"),
        (run, "obj_episode_id", "obj_episode_id"),
        (run, "output_dir", "output_dir"),
        (run, "save_replay", "save_replay"),
        (run, "save_trace", "save_trace"),
        (run, "disable_action_cache", "disable_action_cache"),
        (run, "action_ensemble", "action_ensemble"),
        (run, "action_ensemble_horizon", "action_ensemble_horizon"),
        (run, "action_ensemble_alpha", "action_ensemble_alpha"),
        (run, "cfg_scale", "cfg_scale"),
        (env, "simplerenv_root", "simplerenv_root"),
        (env, "nvidia_lib_dir", "nvidia_lib_dir"),
        (env, "nvidia_icd_json", "nvidia_icd_json"),
    ]
    for section, key, attr in mapping:
        value = section.get(key) if isinstance(section, dict) else None
        if value is not None:
            setattr(args, attr, value)
    return args


def build_argparser() -> argparse.ArgumentParser:
    """Run build_argparser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--task-name", default="widowx_put_eggplant_in_basket")
    parser.add_argument("--robot-setup", default="widowx_sink_camera_setup")
    parser.add_argument("--camera-name", default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--scene-name", default="bridge_table_1_v2")
    parser.add_argument("--rgb-overlay-path", default=None)
    parser.add_argument("--sim-freq", type=int, default=500)
    parser.add_argument("--control-freq", type=int, default=5)
    parser.add_argument("--control-mode", default="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--rotation-mode", choices=["euler", "axis_angle"], default="euler")
    parser.add_argument("--domain-id", type=int, default=None)
    parser.add_argument("--action-postprocess", default="")
    parser.add_argument("--disable-action-cache", action="store_true")
    parser.add_argument("--action-ensemble", action="store_true")
    parser.add_argument("--action-ensemble-horizon", type=int, default=7)
    parser.add_argument("--action-ensemble-alpha", type=float, default=0.1)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--robot-init-x", type=float, default=0.127)
    parser.add_argument("--robot-init-y", type=float, default=0.06)
    parser.add_argument("--robot-init-quat", type=float, nargs=4, default=[0.0, 0.0, 0.0, 1.0])
    parser.add_argument("--obj-episode-id", type=int, default=0)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-replay", action="store_true")
    parser.add_argument("--success-settle-steps", type=int, default=12)
    parser.add_argument("--success-settle-gripper", type=float, default=1.0)
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="/workspace/LoongForge-VLA/loongforge/embodied/eval/reports/manual/simplerenv/smoke",
    )
    parser.add_argument("--simplerenv-root", default="")
    parser.add_argument("--nvidia-lib-dir", default=os.environ.get("NVIDIA_LIB_DIR", "/ssd1/opt/nvidia_lib"))
    parser.add_argument(
        "--nvidia-icd-json", default=os.environ.get("NVIDIA_ICD_JSON", "/ssd1/opt/nvidia_lib/10_nvidia.json")
    )
    return parser


def main() -> None:
    """Run main."""
    args = build_argparser().parse_args()
    if args.config:
        args = _apply_config(args, load_config(args.config))
    if args.simplerenv_root:
        root_paths = [args.simplerenv_root, str(pathlib.Path(args.simplerenv_root) / "ManiSkill2_real2sim")]
        for path in reversed(root_paths):
            if path not in sys.path:
                sys.path.insert(0, path)
        os.environ.setdefault("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = ":".join(root_paths + [os.environ["PYTHONPATH"]])
    result = run_evaluation(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
