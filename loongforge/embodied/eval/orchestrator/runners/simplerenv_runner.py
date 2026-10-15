# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SimplerEnv single-episode runner using the standalone eval module."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import pathlib
import sys
import time
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np

from loongforge.embodied.eval.adapters.simplerenv import (
    SIMPLERENV_DEFAULT_MAX_STEPS,
    SimplerEnvAdapter,
)
from loongforge.embodied.eval.metrics.results import append_jsonl, write_suite_summary_csv, write_summary_csv
from loongforge.embodied.eval.orchestrator.config import (
    build_rpc_payload,
    load_config,
)
from loongforge.embodied.eval.orchestrator.runners import _common
from loongforge.embodied.eval.transport import PolicyClient

logger = logging.getLogger(__name__)


def _ensure_vulkan_runtime(args: argparse.Namespace) -> None:
    """Re-exec once so LD_LIBRARY_PATH takes effect before importing SAPIEN."""
    _common.ensure_vulkan_runtime(args, "LOONGFORGE_SIMPLERENV_RUNTIME_READY")


def _build_env(args: argparse.Namespace, adapter: SimplerEnvAdapter) -> Tuple[Any, str]:
    """Run _build_env."""
    import simpler_env
    from simpler_env.utils.env.env_builder import build_maniskill2_env

    simpler_env_root = pathlib.Path(simpler_env.__file__).resolve().parents[1]

    # Official / strict path: let the env apply its own prepackaged visual-matching
    # config (scene, overlay set, robot, control_mode) and randomize overlay +
    # robot/object init per episode via its episode_rng -- matching the official
    # SimplerEnv eval (`simpler_env.make(task)` sets prepackaged_config=True).
    if getattr(args, "prepackaged_config", False):
        env = build_maniskill2_env(
            adapter.env_name,
            obs_mode="rgbd",
            prepackaged_config=True,
            max_episode_steps=args.max_steps if args.max_steps > 0 else SIMPLERENV_DEFAULT_MAX_STEPS,
        )
        return env, adapter.env_name

    overlay_path = args.rgb_overlay_path
    if overlay_path is None and args.robot_setup.startswith("widowx"):
        overlay_path = str(simpler_env_root / "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png")

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
    # Prepackaged (official) path: reset with only a per-episode seed and let the
    # env randomize overlay + robot/object init itself (do NOT pin robot_init).
    # A None seed matches the official client's default (no --seed): the env's
    # own RNG picks the layout, so runs are non-deterministic like upstream.
    if getattr(args, "prepackaged_config", False):
        reset_result = (
            env.reset()
            if args.seed is None
            else env.reset(seed=args.seed + args.episode_idx)
        )
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            return reset_result
        return reset_result, {}

    reset_options: Dict[str, Any] = {
        "robot_init_options": {
            "init_xy": np.asarray([args.robot_init_x, args.robot_init_y], dtype=np.float32),
            "init_rot_quat": np.asarray(args.robot_init_quat, dtype=np.float32),
        },
        "obj_init_options": {"episode_id": args.obj_episode_id},
    }
    reset_result = (
        env.reset(options=reset_options)
        if args.seed is None
        else env.reset(seed=args.seed + args.episode_idx, options=reset_options)
    )
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
    status = "success" if success else "fail"
    artifact_dir = pathlib.Path(output_dir) / "artifacts" / "simplerenv" / task_name
    return _common.write_replay_gif(frames, artifact_dir / f"ep{episode_idx}_{status}.gif", duration=0.2)


def _save_trace(trace: List[Dict[str, Any]], output_dir: str, task_name: str, episode_idx: int, success: bool) -> str:
    """Run _save_trace."""
    status = "success" if success else "fail"
    artifact_dir = pathlib.Path(output_dir) / "artifacts" / "simplerenv" / task_name
    return _common.write_trace_json(trace, artifact_dir / f"ep{episode_idx}_{status}_trace.json")


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
    return _common.json_safe(value)


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
    if args.seed is not None:
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
    payload_builder, action_decoder_key, action_decoder = _common.build_policy_stack(args, adapter)
    env, env_name = _build_env(args, adapter)
    client = PolicyClient(host=args.host, port=args.port)

    episode_id = f"simplerenv/{args.task_name}/episode={args.episode_idx}"
    client.reset(episode_id)
    payload_builder.reset(episode_id)
    action_decoder.reset()
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

            ctx = {
                "benchmark_name": "simplerenv",
                "episode_id": episode_id,
                "episode_step": episode_step,
                "instruction": instruction,
            }
            extra_control = {
                "cfg_scale": args.cfg_scale,
            }
            response = client.predict_action(
                **build_rpc_payload(
                    payload_builder,
                    canonical_obs,
                    ctx,
                    disable_action_cache=args.disable_action_cache,
                    return_action_chunk=args.action_ensemble,
                    extra_control=extra_control,
                )
            )
            if not response.get("ok", False):
                raise RuntimeError(f"Policy error: {response}")

            data = response["data"]
            raw_chunk = np.asarray(data["actions"], dtype=np.float32)
            if raw_chunk.ndim == 1:
                raw_chunk = raw_chunk.reshape(1, -1)
            # Feed the response back into the PayloadBuilder so stateful
            # encodings (ee6d_widowx) can backfill proprio[:10] = action[:10].
            payload_builder.update_from_response(data)
            if action_decoder_key:
                env_action = action_decoder(raw_chunk[0:1], {"benchmark_name": "simplerenv"})[0].astype(np.float32)
                flat_action = raw_chunk[0]
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


def _make_adapter(args: argparse.Namespace) -> SimplerEnvAdapter:
    """Build a SimplerEnvAdapter from args (fresh = clean sticky-gripper state)."""
    return SimplerEnvAdapter(
        task_name=args.task_name,
        robot_setup=args.robot_setup,
        control_hz=args.control_freq,
        max_steps=args.max_steps if args.max_steps > 0 else SIMPLERENV_DEFAULT_MAX_STEPS,
        camera_name=args.camera_name,
        action_scale=args.action_scale,
        rotation_mode=args.rotation_mode,
    )


def _env_worker(
    conn,
    args: argparse.Namespace,
    seed_base: int | None,
) -> None:
    """One SimplerEnv instance in its own spawned process.

    Owns its env + the per-env episode seed sequence (``seed_base``, then
    ``+= stride`` on each autoreset, matching the ZMQ ``reset_seeds`` stride
    pattern). Sends ``(obs, instruction, completed)`` after each step; the
    parent owns all policy/adapter state and canonicalization.
    """
    env = None
    try:
        adapter = _make_adapter(args)
        env, _ = _build_env(args, adapter)
        max_steps = args.max_steps if args.max_steps > 0 else SIMPLERENV_DEFAULT_MAX_STEPS

        def _do_reset(seed: int | None):
            local = argparse.Namespace(**vars(args))
            local.seed = seed
            local.episode_idx = 0
            obs, _info = _reset_env(env, local)
            instr = _get_instruction(env, args.instruction or args.task_name.replace("_", " "))
            return obs, instr

        seed = seed_base
        steps = 0
        autoreset_pending = False
        obs, instruction = _do_reset(seed)
        episode_seed = getattr(env.unwrapped, "_episode_seed", seed)
        conn.send(("ok", (obs, instruction, None, episode_seed, False)))
        while True:
            msg = conn.recv()
            if msg is None:
                break
            if autoreset_pending:
                steps = 0
                obs, instruction = _do_reset(None)
                episode_seed = getattr(env.unwrapped, "_episode_seed", None)
                autoreset_pending = False
                conn.send(("ok", (obs, instruction, None, episode_seed, True)))
                continue
            action = np.asarray(msg, dtype=np.float32)
            obs, _reward, done, truncated, info = env.step(action)
            steps += 1
            # Some SimplerEnv tasks change their language instruction within an
            # episode (place_in_closed_drawer switches from "open" to "place"
            # after the first subtask). Match GoogleFractalEnv, which reads the
            # instruction into every post-step observation.
            instruction = _get_instruction(
                env, args.instruction or args.task_name.replace("_", " ")
            )
            success = bool(info.get("success", False))
            completed = None
            if done or truncated or success or steps >= max_steps:
                completed = {
                    "success": int(success),
                    "steps": steps,
                    "instruction": instruction,
                    "seed": episode_seed,
                }
                autoreset_pending = True
            conn.send(("ok", (obs, instruction, completed, episode_seed, False)))
    except BaseException as exc:  # noqa: BLE001 - propagate to parent, fail the task
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except (OSError, EOFError):
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                logging.exception("SimplerEnv worker close failed")
        conn.close()


class _Pool:
    """Spawned-process env pool with a barrier step (all send, then all recv)."""

    def __init__(self, args: argparse.Namespace) -> None:
        ctx = multiprocessing.get_context("spawn")
        self.conns: List[Any] = []
        self.procs: List[Any] = []
        base = None if args.seed is None else int(args.seed)
        try:
            for i in range(args.num_envs):
                seed_base = None if base is None else base + i
                parent, child = ctx.Pipe()
                proc = ctx.Process(
                    target=_env_worker,
                    args=(child, args, seed_base),
                    daemon=True,
                )
                self.conns.append(parent)
                proc.start()
                child.close()
                self.procs.append(proc)
        except BaseException:
            self.close()
            raise

    def _recv_all(self):
        out = []
        for conn in self.conns:
            status, value = conn.recv()
            if status != "ok":
                raise RuntimeError(value)
            out.append(value)
        return out

    def initial(self):
        return self._recv_all()

    def step(self, actions):
        for conn, action in zip(self.conns, actions):
            conn.send(action)
        return self._recv_all()

    def close(self):
        for conn in self.conns:
            try:
                conn.send(None)
            except (OSError, EOFError):
                pass
            conn.close()
        for proc in self.procs:
            proc.join(timeout=1)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
                proc.join()


def run_batch(args: argparse.Namespace) -> Dict[str, Any]:
    """Vectorized evaluation of a single task over ``num_envs`` processes.

    The default single-episode path (``run_evaluation``) evaluates ONE episode
    per process. This function adds N_ENVS-style vectorization: one spawned
    process per SimplerEnv instance (isolating SAPIEN + the global NumPy RNG),
    with the parent issuing one batch-of-one RPC per env per step and a barrier
    before stepping all envs. Per-env policy episode state is reset on every
    autoreset: the ``PolicyClient`` episode, the ``PayloadBuilder``, and a fresh
    ``SimplerEnvAdapter`` (so the Google sticky-gripper state starts clean).

    Server inference is NOT batched unless ``batch_inference`` is set (one B=N
    request per vector step); otherwise one B=1 request per env per step, like
    LIBERO. The speedup comes from the N env processes rendering/stepping
    concurrently. Keep ``num_envs`` modest: each env is one SAPIEN Vulkan
    context and too many on one GPU reproduce the ``ErrorDeviceLost`` seen in
    the ZMQ path.

    Per-step behavior (one env action per RPC, first action of the returned
    chunk) is kept byte-identical to ``run_evaluation`` so success rates are
    comparable.
    """
    _common.ensure_vulkan_runtime(args, "LOONGFORGE_SIMPLERENV_RUNTIME_READY")
    if args.simplerenv_root:
        roots = [args.simplerenv_root, str(pathlib.Path(args.simplerenv_root) / "ManiSkill2_real2sim")]
        for path in reversed(roots):
            if path not in sys.path:
                sys.path.insert(0, path)
    if args.num_envs < 1 or args.episodes_per_task < 1:
        raise ValueError("num_envs and episodes_per_task must be positive")

    args.num_envs = min(args.num_envs, args.episodes_per_task)
    target = args.episodes_per_task
    adapters = [_make_adapter(args) for _ in range(args.num_envs)]
    stacks = [_common.build_policy_stack(args, adapters[i]) for i in range(args.num_envs)]
    builders = [s[0] for s in stacks]
    decoder_key = stacks[0][1]
    decoders = [s[2] for s in stacks]

    pool = client = None
    records: List[Dict[str, Any]] = []
    start = time.time()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    results_path.write_text("")  # fresh namespace; append each episode as it completes
    try:
        pool = _Pool(args)
        init = pool.initial()
        obs = [x[0] for x in init]
        instr = [x[1] for x in init]
        current_seeds = [x[3] for x in init]
        env_episode_counts = [0] * args.num_envs
        ids = [
            f"env{i}/seed{current_seeds[i]}"
            if current_seeds[i] is not None
            else f"env{i}/ep0"
            for i in range(args.num_envs)
        ]
        steps = [0] * args.num_envs
        for i in range(args.num_envs):
            client = client or PolicyClient(host=args.host, port=args.port)
            client.reset(ids[i])
            builders[i].reset(ids[i])

        while len(records) < target:
            model_payloads = []
            for i in range(args.num_envs):
                ctx = {"instruction": instr[i], "episode_id": ids[i], "episode_step": steps[i]}
                canonical = adapters[i].obs_to_canonical(obs[i], ctx)
                rpc_ctx = {
                    "benchmark_name": "simplerenv",
                    "episode_id": ids[i],
                    "episode_step": steps[i],
                    "instruction": instr[i],
                }
                model_payloads.append(
                    build_rpc_payload(
                        builders[i],
                        canonical,
                        rpc_ctx,
                        disable_action_cache=args.disable_action_cache,
                        return_action_chunk=False,
                        extra_control={"cfg_scale": args.cfg_scale},
                    )
                )

            if args.batch_inference:
                response = client.predict_action(
                    batch_request=True,
                    disable_action_cache=True,
                    return_action_chunk=False,
                    images=[payload["images"] for payload in model_payloads],
                    instructions=[payload["instructions"][0] for payload in model_payloads],
                    state=np.stack([np.asarray(payload["state"]) for payload in model_payloads]),
                )
                if not response.get("ok", False):
                    raise RuntimeError(f"Policy error: {response}")
                batch_raw = np.asarray(response["data"]["actions"], dtype=np.float32)
                if batch_raw.ndim != 3 or batch_raw.shape[0] != args.num_envs:
                    raise RuntimeError(
                        f"Invalid batched action shape {batch_raw.shape}; expected "
                        f"({args.num_envs}, horizon, action_dim)"
                    )
                response_data = [
                    {**response["data"], "actions": batch_raw[i]}
                    for i in range(args.num_envs)
                ]
            else:
                response_data = []
                for payload in model_payloads:
                    response = client.predict_action(**payload)
                    if not response.get("ok", False):
                        raise RuntimeError(f"Policy error: {response}")
                    response_data.append(response["data"])

            actions = []
            for i, data in enumerate(response_data):
                builders[i].update_from_response(data)
                raw = np.asarray(data["actions"], dtype=np.float32)
                if raw.ndim == 1:
                    raw = raw.reshape(1, -1)
                if decoder_key:
                    env_action = decoders[i](raw[0:1], {"benchmark_name": "simplerenv"})[0].astype(np.float32)
                else:
                    env_action = adapters[i].action_from_canonical({"actions": raw.reshape(-1, 7)[0]})
                actions.append(np.asarray(env_action, dtype=np.float32))

            results = pool.step(actions)
            for i, result in enumerate(results):
                o, ins, completed, current_seed, reset_happened = result
                obs[i] = o
                instr[i] = ins
                current_seeds[i] = current_seed
                if reset_happened:
                    client.reset(ids[i])
                    env_episode_counts[i] += 1
                    ids[i] = (
                        f"env{i}/seed{current_seeds[i]}"
                        if current_seeds[i] is not None
                        else f"env{i}/ep{env_episode_counts[i]}"
                    )
                    adapters[i] = _make_adapter(args)
                    builders[i].reset(ids[i])
                    steps[i] = 0
                    continue
                if completed is not None:
                    if len(records) >= target:
                        continue
                    completed_seed = completed.get("seed")
                    episode_key = completed_seed if completed_seed is not None else len(records)
                    completed_episode_id = ids[i]
                    record = {
                        "benchmark": "simplerenv",
                        "task_suite": args.task_name,
                        "task_id": 0,
                        "episode_idx": int(episode_key),
                        "seed": completed_seed,
                        "env_idx": i,
                        "episode_id": f"simplerenv/{args.task_name}/{completed_episode_id}",
                        "task_description": completed["instruction"],
                        "env_name": adapters[i].env_name,
                        "robot_setup": args.robot_setup,
                        "success": int(completed["success"]),
                        "steps": int(completed["steps"]),
                        "failure_reason": None if completed["success"] else "not_successful_within_max_steps",
                    }
                    records.append(record)
                    append_jsonl(results_path, _common.json_safe(record))
                    succ = sum(r["success"] for r in records)
                    logger.info(
                        "[%s] episode %d/%d done (success=%d/%d, env%d, seed=%s, %d steps)",
                        args.task_name, len(records), target, succ, len(records), i,
                        completed_seed, record["steps"],
                    )
                    print(
                        f"[{args.task_name}] {len(records)}/{target} done  "
                        f"success={succ}/{len(records)}  "
                        f"(env{i}, seed={completed_seed}, {record['steps']} steps)",
                        flush=True,
                    )
                else:
                    steps[i] += 1
    finally:
        if pool is not None:
            pool.close()
        if client is not None:
            try:
                client.close()
            except Exception:
                logging.exception("Policy client close failed")

    if len(records) != target:
        raise RuntimeError(f"Expected {target} episodes, got {len(records)}")
    actual_seeds = [r.get("seed") for r in records]
    if any(seed is None for seed in actual_seeds):
        raise RuntimeError("One or more completed episodes did not expose their actual environment seed")
    if len(set(map(int, actual_seeds))) != target:
        raise RuntimeError("Completed episodes contain duplicate environment seeds")

    out_dir = pathlib.Path(args.output_dir)
    safe = [_common.json_safe(r) for r in records]
    write_summary_csv(out_dir / "summary.csv", safe)
    write_suite_summary_csv(out_dir / "suite_summary.csv", safe)
    n = len(records)
    succ = sum(r["success"] for r in records)
    summary = {
        "benchmark": "simplerenv",
        "task_suite": args.task_name,
        "n_envs": args.num_envs,
        "n_episodes": n,
        "success": succ,
        "success_rate": (succ / n) if n else None,
        "wallclock_sec": time.time() - start,
        "results_path": str(out_dir / "results.jsonl"),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


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
        (benchmark, "prepackaged_config", "prepackaged_config"),
        (benchmark, "sim_freq", "sim_freq"),
        (benchmark, "control_freq", "control_freq"),
        (benchmark, "control_mode", "control_mode"),
        (benchmark, "action_scale", "action_scale"),
        (benchmark, "rotation_mode", "rotation_mode"),
        (benchmark, "max_steps", "max_steps"),
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
    # Preserve the full YAML so build_policy_stack (_common.build_policy_stack)
    # can read model.state_encoding / action_encoding. Without this, raw_config
    # defaults to {} and the GR00T-N1.7 builder falls back to its default
    # state_encoding=libero_ee_axis_angle, sending a LIBERO-format proprio to a
    # SimplerEnv-Bridge model -> every widowx episode fails (0/N).
    args.raw_config = config
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
    parser.add_argument("--prepackaged-config", action="store_true")
    parser.add_argument("--sim-freq", type=int, default=500)
    parser.add_argument("--control-freq", type=int, default=5)
    parser.add_argument("--control-mode", default="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--rotation-mode", choices=["euler", "axis_angle"], default="euler")
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-replay", action="store_true")
    parser.add_argument("--success-settle-steps", type=int, default=12)
    parser.add_argument("--success-settle-gripper", type=float, default=1.0)
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="/workspace/LoongForge-VLA/loongforge/embodied/eval/reports/manual/simplerenv/smoke",
    )
    parser.add_argument("--simplerenv-root", default="")
    parser.add_argument("--nvidia-lib-dir", default=os.environ.get("NVIDIA_LIB_DIR", "/path/to/nvidia_lib"))
    parser.add_argument(
        "--nvidia-icd-json", default=os.environ.get("NVIDIA_ICD_JSON", "/path/to/nvidia_lib/10_nvidia.json")
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Concurrent SimplerEnv env worker processes; 1 = single-episode sequential path.",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=1,
        help="Total episodes for the vectorized path (ignored when num_envs=1).",
    )
    parser.add_argument(
        "--batch-inference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Send one B=num_envs model request per vector step instead of num_envs B=1 requests.",
    )
    return parser


def main() -> None:
    """Run main."""
    args = build_argparser().parse_args()
    if args.config:
        config = load_config(args.config)
        args = _apply_config(args, config)
        run = config.get("run") or {}
        if run.get("num_envs") is not None:
            args.num_envs = int(run["num_envs"])
        if run.get("episodes_per_task") is not None:
            args.episodes_per_task = int(run["episodes_per_task"])
        if run.get("batch_inference") is not None:
            args.batch_inference = bool(run["batch_inference"])
    if args.simplerenv_root:
        root_paths = [args.simplerenv_root, str(pathlib.Path(args.simplerenv_root) / "ManiSkill2_real2sim")]
        for path in reversed(root_paths):
            if path not in sys.path:
                sys.path.insert(0, path)
        os.environ.setdefault("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = ":".join(root_paths + [os.environ["PYTHONPATH"]])
    if int(getattr(args, "num_envs", 1) or 1) > 1:
        result = run_batch(args)
    else:
        result = run_evaluation(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
