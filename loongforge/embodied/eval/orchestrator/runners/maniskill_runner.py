# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""ManiSkill single-episode runner using the standalone eval module."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import stat
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from loongforge.embodied.eval.adapters.maniskill import MANISKILL_DEFAULT_MAX_STEPS, ManiSkillAdapter
from loongforge.embodied.eval.metrics.results import append_jsonl, write_suite_summary_csv, write_summary_csv
from loongforge.embodied.eval.orchestrator.config import load_config
from loongforge.embodied.eval.transport import PolicyClient


def _canonical_to_legacy_payload(canonical_obs: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Build the policy RPC payload from a canonical observation."""
    return {
        "images": canonical_obs["images"],
        "instruction": canonical_obs["instruction"],
        "episode_id": canonical_obs["meta"]["episode_id"],
        "episode_step": canonical_obs["meta"]["episode_step"],
        "state": canonical_obs.get("model_state"),
        "disable_action_cache": args.disable_action_cache,
        "return_action_chunk": False,
        "cfg_scale": args.cfg_scale,
    }


def _vulkan_runtime_env(args: argparse.Namespace) -> Dict[str, str]:
    """Build environment variables required before importing SAPIEN."""
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


def _ensure_vulkan_runtime(args: argparse.Namespace) -> None:
    """Restart the process once so Vulkan library paths are loaded early."""
    marker = "LOONGFORGE_MANISKILL_RUNTIME_READY"
    if os.environ.get(marker) == "1":
        return
    env = _vulkan_runtime_env(args)
    env[marker] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def _patch_maniskill_pci_backend_parser(render_backend: str) -> None:
    """Patch ManiSkill's PCI render-backend parser for lavapipe smoke runs."""
    if not render_backend.startswith("pci:"):
        return
    from mani_skill.envs.utils.system import backend as backend_utils

    def parse_backend_device_id(backend: str) -> Tuple[str, None]:
        """Return the full PCI backend string without splitting embedded colons."""
        return backend, None

    backend_utils.parse_backend_device_id = parse_backend_device_id


def _build_env(args: argparse.Namespace) -> Tuple[Any, ManiSkillAdapter]:
    """Create the ManiSkill environment and canonical adapter."""
    _ensure_vulkan_runtime(args)
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    import mani_skill.envs  # noqa: F401

    _patch_maniskill_pci_backend_parser(args.render_backend)
    max_steps = args.max_steps if args.max_steps > 0 else MANISKILL_DEFAULT_MAX_STEPS
    adapter = ManiSkillAdapter(
        task_name=args.task_name,
        robot_uid=args.robot_uid,
        control_hz=args.control_freq,
        max_steps=max_steps,
        camera_name=args.camera_name,
        action_scale=args.action_scale,
        gripper_open_value=args.gripper_open_value,
        gripper_close_value=args.gripper_close_value,
        allow_dummy_image=args.allow_dummy_image,
    )

    env_kwargs: Dict[str, Any] = {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "render_mode": args.render_mode,
        "max_episode_steps": max_steps,
    }
    if args.robot_uid:
        env_kwargs["robot_uids"] = args.robot_uid
    if args.sim_backend:
        env_kwargs["sim_backend"] = args.sim_backend
    if args.render_backend:
        env_kwargs["render_backend"] = args.render_backend
    if args.shader:
        env_kwargs["shader_dir"] = args.shader

    env = gym.make(args.task_name, **env_kwargs)
    return env, adapter


def _reset_env(env: Any, args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Reset a Gym or Gymnasium environment with a deterministic seed."""
    reset_result = env.reset(seed=args.seed + args.episode_idx)
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, reset_info = reset_result
    else:
        obs, reset_info = reset_result, {}
    return obs, reset_info


def _get_instruction(args: argparse.Namespace) -> str:
    """Return the configured instruction or derive one from the task name."""
    if args.instruction:
        return args.instruction
    return args.task_name.replace("-v1", "").replace("-", " ").replace("_", " ")


def _normalize_frame(frame: Any) -> np.ndarray:
    """Normalize renderer output to an HWC uint8 frame."""
    array = _to_numpy(frame)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _render_frame(env: Any, obs: Dict[str, Any], adapter: ManiSkillAdapter) -> np.ndarray:
    """Render a replay frame or fall back to the observation image."""
    if hasattr(env, "render"):
        frame = env.render()
        if frame is not None:
            return _normalize_frame(frame)
    from loongforge.embodied.eval.adapters.maniskill import _extract_image

    return _normalize_frame(_extract_image(obs, adapter.camera_name))


def _save_replay(frames: List[np.ndarray], output_dir: str, task_name: str, episode_idx: int, success: bool) -> str:
    """Write replay frames to a GIF artifact and return its path."""
    if not frames:
        raise ValueError("Cannot save replay with no frames")
    import imageio.v2 as imageio

    status = "success" if success else "fail"
    artifact_dir = pathlib.Path(output_dir) / "artifacts" / "maniskill" / task_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    replay_path = artifact_dir / f"ep{episode_idx}_{status}.gif"
    imageio.mimsave(replay_path, frames, duration=0.2)
    return str(replay_path)


def _save_trace(trace: List[Dict[str, Any]], output_dir: str, task_name: str, episode_idx: int, success: bool) -> str:
    """Write the per-step action trace artifact and return its path."""
    status = "success" if success else "fail"
    artifact_dir = pathlib.Path(output_dir) / "artifacts" / "maniskill" / task_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / f"ep{episode_idx}_{status}_trace.json"
    with trace_path.open("w", encoding="utf-8") as file:
        json.dump(trace, file, ensure_ascii=False, indent=2)
    return str(trace_path)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors or array-like values to NumPy arrays."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_scalar(value: Any) -> Any:
    """Return the first scalar value from tensors or arrays."""
    array = _to_numpy(value)
    if array.size == 0:
        return None
    return array.reshape(-1)[0].item()


def _json_safe(value: Any) -> Any:
    """Convert tensors and arrays in nested values to JSON-safe objects."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_standard_outputs(output_dir: str, record: Dict[str, Any]) -> None:
    """Write standard JSONL and CSV result files for one ManiSkill run."""
    out_dir = pathlib.Path(output_dir)
    safe_record = _json_safe(record)
    append_jsonl(out_dir / "results.jsonl", safe_record)
    write_summary_csv(out_dir / "summary.csv", [safe_record])
    write_suite_summary_csv(out_dir / "suite_summary.csv", [safe_record])


def _is_success(done: Any, reward: Any, info: Dict[str, Any], args: argparse.Namespace) -> bool:
    """Infer episode success from ManiSkill info flags or reward."""
    if "success" in info:
        return bool(_to_scalar(info["success"]))
    if "is_success" in info:
        return bool(_to_scalar(info["is_success"]))
    reward_value = float(_to_scalar(reward) or 0.0)
    if args.success_reward_threshold is not None:
        return reward_value >= float(args.success_reward_threshold)
    return bool(_to_scalar(done)) and reward_value > 0


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run one ManiSkill episode against a policy server."""
    np.random.seed(args.seed)
    env, adapter = _build_env(args)
    client = PolicyClient(host=args.host, port=args.port)

    episode_id = f"maniskill/{args.task_name}/episode={args.episode_idx}"
    client.reset(episode_id)
    obs, reset_info = _reset_env(env, args)
    instruction = _get_instruction(args)

    max_steps = args.max_steps if args.max_steps > 0 else MANISKILL_DEFAULT_MAX_STEPS
    done = False
    truncated = False
    reward = 0.0
    info: Dict[str, Any] = {}
    steps = 0
    replay_frames: List[np.ndarray] = []
    action_trace: List[Dict[str, Any]] = []
    start_time = time.time()

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
                replay_frames.append(_render_frame(env, obs, adapter))

            response = client.predict_action(**_canonical_to_legacy_payload(canonical_obs, args))
            if not response.get("ok", False):
                raise RuntimeError(f"Policy error: {response}")

            # X-VLA ee6d is 20D; pi05 is already 7D. Optional postprocess converts
            # model space → ManiSkill 7D (pos3 + rot_delta3 + grip1) for pd_ee_delta_pose.
            from loongforge.embodied.eval.servers.predict_action_interface import postprocess_actions

            raw_chunk = np.asarray(response["data"]["actions"], dtype=np.float32)
            if raw_chunk.ndim == 1:
                raw_chunk = raw_chunk.reshape(1, -1)
            postprocess_key = getattr(args, "action_postprocess", "") or ""
            processed = postprocess_actions(raw_chunk, postprocess_key)
            if processed.shape[-1] < 7:
                raise ValueError(
                    f"ManiSkill expects >=7D after postprocess, got shape {processed.shape} "
                    f"(action_postprocess={postprocess_key!r})"
                )
            flat_action = processed[0, :7].astype(np.float32)
            env_action = adapter.action_from_canonical({"actions": flat_action})
            obs, reward, done, truncated, info = env.step(env_action)
            done_flag = bool(_to_scalar(done))
            truncated_flag = bool(_to_scalar(truncated))
            reward_value = float(_to_scalar(reward) or 0.0)
            steps += 1

            if args.save_trace:
                action_trace.append(
                    {
                        "step": episode_step,
                        "raw_action": flat_action.tolist(),
                        "env_action": np.asarray(env_action).tolist(),
                        "reward": reward_value,
                        "done": done_flag,
                        "truncated": truncated_flag,
                        "success": _is_success(done, reward, info, args),
                        "inference_latency_ms": response.get("data", {}).get("inference_latency_ms"),
                    }
                )

            if done_flag or truncated_flag or _is_success(done, reward, info, args):
                break
    finally:
        client.close()
        env.close()

    success = _is_success(done, reward, info, args)
    replay_path = None
    if args.save_replay:
        replay_path = _save_replay(replay_frames, args.output_dir, args.task_name, args.episode_idx, success)
    trace_path = None
    if args.save_trace:
        trace_path = _save_trace(action_trace, args.output_dir, args.task_name, args.episode_idx, success)

    record = {
        "benchmark": "maniskill",
        "task_suite": args.task_name,
        "task_id": 0,
        "episode_idx": args.episode_idx,
        "episode_id": episode_id,
        "task_description": instruction,
        "env_name": args.task_name,
        "robot_setup": args.robot_uid,
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
    """Apply YAML config sections to parsed runner arguments."""
    benchmark = config.get("benchmark") or {}
    server = config.get("server") or {}
    run = config.get("run") or {}

    if not isinstance(benchmark, dict) or not isinstance(server, dict) or not isinstance(run, dict):
        raise ValueError("benchmark, server, and run sections must be mappings")

    mapping = [
        (benchmark, "task_name", "task_name"),
        (benchmark, "robot_uid", "robot_uid"),
        (benchmark, "camera_name", "camera_name"),
        (benchmark, "instruction", "instruction"),
        (benchmark, "obs_mode", "obs_mode"),
        (benchmark, "control_mode", "control_mode"),
        (benchmark, "control_freq", "control_freq"),
        (benchmark, "render_mode", "render_mode"),
        (benchmark, "sim_backend", "sim_backend"),
        (benchmark, "render_backend", "render_backend"),
        (benchmark, "shader", "shader"),
        (benchmark, "nvidia_icd_json", "nvidia_icd_json"),
        (benchmark, "nvidia_lib_dir", "nvidia_lib_dir"),
        (benchmark, "action_scale", "action_scale"),
        (benchmark, "gripper_open_value", "gripper_open_value"),
        (benchmark, "gripper_close_value", "gripper_close_value"),
        (benchmark, "allow_dummy_image", "allow_dummy_image"),
        (benchmark, "max_steps", "max_steps"),
        (benchmark, "success_reward_threshold", "success_reward_threshold"),
        (benchmark, "action_postprocess", "action_postprocess"),
        (server, "host", "host"),
        (server, "port", "port"),
        (run, "seed", "seed"),
        (run, "episode_idx", "episode_idx"),
        (run, "output_dir", "output_dir"),
        (run, "save_replay", "save_replay"),
        (run, "save_trace", "save_trace"),
        (run, "disable_action_cache", "disable_action_cache"),
        (run, "cfg_scale", "cfg_scale"),
    ]
    for section, key, attr in mapping:
        value = section.get(key) if isinstance(section, dict) else None
        if value is not None:
            setattr(args, attr, value)
    return args


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI arguments for the ManiSkill runner."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--task-name", default="PickCube-v1")
    parser.add_argument("--robot-uid", default="panda")
    parser.add_argument("--camera-name", default="base_camera")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--obs-mode", default="rgbd")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--control-freq", type=int, default=5)
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--shader", default=None)
    parser.add_argument("--nvidia-icd-json", default=None)
    parser.add_argument("--nvidia-lib-dir", default=None)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--gripper-open-value", type=float, default=-1.0)
    parser.add_argument("--gripper-close-value", type=float, default=1.0)
    parser.add_argument("--success-reward-threshold", type=float, default=None)
    parser.add_argument("--allow-dummy-image", action="store_true")
    parser.add_argument("--disable-action-cache", action="store_true")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument(
        "--action-postprocess",
        default="",
        help="Optional key from ACTION_POSTPROCESS_REGISTRY (e.g. ee6d_to_axis_angle for X-VLA)",
    )
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-replay", action="store_true")
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="/workspace/LoongForge-VLA/loongforge/embodied/eval/reports/manual/maniskill/smoke",
    )
    return parser


def main() -> None:
    """Run ManiSkill evaluation from command-line arguments."""
    args = build_argparser().parse_args()
    if args.config:
        args = _apply_config(args, load_config(args.config))
    result = run_evaluation(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
