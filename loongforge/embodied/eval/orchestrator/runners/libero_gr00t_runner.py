# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Isaac-GR00T under the Apache-2.0 License.

"""Opt-in GR00T official LIBERO protocol (benchmark.libero_protocol: gr00t_official).

Purely additive: reached only through the dispatch at the top of
``libero_runner.run_batch``; the legacy path never calls into this module.

Synchronous macro rounds, single env, serial: one persistent env per task is
seeded once and reused across trials via random resets (no fixed init
states). Each macro round makes one batch-of-one RPC, then advances the env
by up to eight actual steps. Results are committed only after an entire task
succeeds.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import pathlib
import time
import uuid

import numpy as np

from loongforge.embodied.eval.adapters.libero import LiberoAdapter
from loongforge.embodied.eval.metrics import append_jsonl
from loongforge.embodied.eval.orchestrator.config import build_rpc_payload
from loongforge.embodied.eval.orchestrator.runners import _common
from loongforge.embodied.eval.transport import PolicyClient

GROOT_OFFICIAL_PROTOCOL = "gr00t_official"
GROOT_OFFICIAL_MACRO_STEPS = 8


class GrootOfficialAdapter(LiberoAdapter):
    """Keep legacy controller-state preference and gripper threshold untouched."""

    def obs_to_canonical(self, env_obs, context):
        context = {k: v for k, v in context.items() if k not in {"ee_pos", "ee_ori_mat"}}
        return super().obs_to_canonical(env_obs, context)

    def action_from_canonical(self, canonical_action, context=None):
        action = np.asarray(canonical_action["actions"], dtype=np.float32).copy()
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(f"Expected finite GR00T 7D action, got {action.shape}")
        action[-1] = -np.sign(2 * action[-1] - 1)
        return action.tolist()


def _make_groot_official_env(task, resolution):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
        ignore_done=True,
    )


class GrootOfficialEnvSession:
    """One reusable env per task; no per-episode seed or init states."""

    def __init__(self, env, seed, max_steps, save_replay=False, save_trace=False):
        self.env = env
        self.max_steps = max_steps
        self.save_replay = save_replay
        self.save_trace = save_trace
        env.seed(int(seed))
        self.reset()

    def reset(self):
        self.obs = self.env.reset()
        self.reset_success = bool(self.env.check_success())
        # hard_reset can replace the robots/controllers. Never read controller
        # pose as state, nor force-update it: the official OSC updates on step.
        for robot in self.env.env.robots:
            robot.controller.use_delta = True
        self.steps = 0
        self.success = False
        self.frames = []
        self.trace = []

    def step(self, actions):
        if len(actions) != GROOT_OFFICIAL_MACRO_STEPS:
            raise ValueError("Official macro round requires eight actions")
        done = False
        for action in actions:
            if self.save_replay:
                self.frames.append(np.ascontiguousarray(self.obs["agentview_image"][::-1, ::-1]))
            self.obs, _, done, _ = self.env.step(action)
            self.steps += 1
            self.success |= bool(self.env.check_success())
            if self.save_trace:
                self.trace.append({"step": self.steps - 1, "env_action": list(action),
                                   "done": bool(done), "success": self.success})
            # MultiStepWrapper interrupts on inner done / step budget. Its
            # terminate_on_success check is after the macro (LIBERO normally
            # already returns done on success).
            if done or self.steps >= self.max_steps:
                break
        completed = None
        if done or self.success or self.steps >= self.max_steps:
            completed = {"success": int(self.success), "steps": self.steps,
                         "frames": self.frames, "trace": self.trace}
            # Gymnasium 0.29 inline autoreset, including surplus final envs.
            # If this reset raises, the whole task must fail, not emit a rate.
            self.reset()
            # Official vector env returns reset info at the top level and the
            # terminal info in final_info. Its collector ORs BOTH into the
            # completed episode (the initial reset info is otherwise ignored).
            completed["success"] |= int(self.reset_success)
        return self.obs, completed


def _groot_official_rpc(args, fn, *pos, **kwargs):
    with _common.alarm_timeout(args.policy_call_timeout_ms / 1000, TimeoutError):
        response = fn(*pos, **kwargs)
    if not response.get("ok", False):
        raise RuntimeError(f"Policy error: {response}")
    return response["data"]


def collect_groot_official_task(
    args, task, task_id, run_id, env_factory=_make_groot_official_env, client_factory=PolicyClient,
):
    """Return exactly ``episodes_per_task`` completed episodes, or raise."""
    from loongforge.embodied.eval.payload_builders.gr00t_n1_7 import GrootN1d7PayloadBuilder

    target = args.episodes_per_task
    adapter = GrootOfficialAdapter(suite_name=args.task_suite_name)
    builder, key, _ = _common.build_policy_stack(args, adapter)
    if (not isinstance(builder, GrootN1d7PayloadBuilder) or key
            or builder.state_encoding != "libero_ee_axis_angle"):
        raise ValueError("gr00t_official requires GR00T N1.7 LIBERO axis-angle policy")
    seed = int(args.seed)
    episode_timeout = args.per_episode_timeout_sec
    step_timeout = args.per_step_timeout_sec
    macro_timeout = (
        GROOT_OFFICIAL_MACRO_STEPS * step_timeout + episode_timeout
        if step_timeout > 0 and episode_timeout > 0 else episode_timeout
    )
    env = client = None
    records = []
    try:
        with _common.alarm_timeout(episode_timeout, TimeoutError):
            env = env_factory(task, 256)
            session = GrootOfficialEnvSession(env, seed, args.max_steps, args.save_replay, args.save_trace)
        observation = session.obs
        with _common.alarm_timeout(args.policy_call_timeout_ms / 1000, TimeoutError):
            client = client_factory(host=args.host, port=args.port)
        env_id = f"{run_id}/task{task_id}/env0"
        step = 0
        started = time.monotonic()
        _groot_official_rpc(args, client.reset, env_id)
        builder.reset(env_id)
        while len(records) < target:
            if episode_timeout > 0 and time.monotonic() - started > episode_timeout:
                raise TimeoutError("Episode deadline exceeded")
            context = {"instruction": task.language, "episode_id": env_id,
                       "episode_step": step, "benchmark_name": "libero"}
            canonical = adapter.obs_to_canonical(observation, context)
            data = _groot_official_rpc(args, client.predict_action, **build_rpc_payload(
                builder, canonical, context, disable_action_cache=True, return_action_chunk=True,
            ))
            builder.update_from_response(data)
            chunk = np.asarray(data["actions"], dtype=np.float32)
            if chunk.ndim != 2 or chunk.shape[0] < GROOT_OFFICIAL_MACRO_STEPS or chunk.shape[1] != 7:
                raise ValueError(f"Expected >=8 x 7 action chunk, got {chunk.shape}")
            actions = [adapter.action_from_canonical({"actions": action})
                       for action in chunk[:GROOT_OFFICIAL_MACRO_STEPS]]
            with _common.alarm_timeout(macro_timeout, TimeoutError):
                observation, completed = session.step(actions)
            if episode_timeout > 0 and time.monotonic() - started > episode_timeout:
                raise TimeoutError("Episode deadline exceeded during macro round")
            if completed is None:
                step += GROOT_OFFICIAL_MACRO_STEPS
                continue
            records.append({
                **completed, "benchmark": "libero", "protocol": GROOT_OFFICIAL_PROTOCOL,
                "task_suite": args.task_suite_name, "task_id": task_id,
                "task": task.language, "episode_idx": len(records), "env_idx": 0,
                "env_seed": seed, "episode_id": f"{env_id}/completion{len(records)}",
            })
            # The env autoreset inside session.step; clear the server-side
            # episode state so no unused action from the old chunk survives.
            _groot_official_rpc(args, client.reset, env_id)
            builder.reset(env_id)
            step = 0
            started = time.monotonic()
        return records
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                logging.exception("LIBERO env close failed")
        if client is not None:
            try:
                client.close()
            except Exception:
                logging.exception("Policy client close failed")


def summarize_groot_official_tasks(tasks):
    rates = [t["success_rate"] for t in tasks
             if t["status"] == "OK" and isinstance(t.get("success_rate"), (int, float))
             and np.isfinite(t["success_rate"]) and 0 <= t["success_rate"] <= 1]
    return {"success_rate": sum(rates) / len(rates) if rates else None,
            "valid_tasks": len(rates), "total_tasks": len(tasks),
            "failed_tasks": sum(t["status"] == "FAILED" for t in tasks)}


def _write_groot_official_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_batch_groot_official(args):
    if args.episodes_per_task < 1:
        raise ValueError("episodes_per_task must be positive")
    if args.num_steps_wait != 0 or getattr(args, "chunk_execute_steps", 8) != 8:
        raise ValueError("gr00t_official requires num_steps_wait=0 and chunk_execute_steps=8")
    if (getattr(args, "control_mode", "auto") not in {"auto", "delta"}
            or getattr(args, "continuous_gripper", False)
            or getattr(args, "render_resolution", 256) not in {0, 256}):
        raise ValueError("gr00t_official requires delta control, sign gripper and 256px rendering")
    if args.max_steps <= 0:
        args.max_steps = 720
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    count = suite.n_tasks if args.max_tasks <= 0 else min(suite.n_tasks, args.max_tasks)
    task_ids = list(dict.fromkeys(getattr(args, "task_ids", None) or range(count)))
    # Fresh namespace every invocation: never read, append to, delete or resume
    # legacy results. RNG streams cannot be reconstructed from episode records.
    run_id = uuid.uuid4().hex
    output = pathlib.Path(args.output_dir) / GROOT_OFFICIAL_PROTOCOL / run_id
    output.mkdir(parents=True, exist_ok=False)
    results_path = output / "results.jsonl"
    results_path.touch()
    logging.info("GR00T official protocol: single env, serial episodes; "
                 "resume/restart ignored, fresh results: %s", output)
    tasks, records = [], []
    for task_id in task_ids:
        entry = {"task_id": task_id, "task_suite": args.task_suite_name}
        try:
            manager = getattr(args, "server_manager", None)
            if manager is not None:
                manager.ensure_running()
            task_records = collect_groot_official_task(args, suite.get_task(task_id), task_id, run_id)
            rate = sum(r["success"] for r in task_records) / len(task_records)
            entry.update(status="OK", success_rate=rate, n_episodes=len(task_records))
        except Exception as exc:
            logging.exception("Task %s FAILED; excluding all its partial episodes", task_id)
            entry.update(status="FAILED", success_rate=None, n_episodes=0,
                         error_type=type(exc).__name__, error_message=str(exc))
            task_records = []
        for record in task_records:
            frames, trace = record.pop("frames"), record.pop("trace")
            artifact = output / "artifacts" / f"task{task_id}" / f"episode{record['episode_idx']}"
            # Optional artifact IO must not turn a valid policy result into a failure.
            try:
                if frames:
                    artifact.mkdir(parents=True, exist_ok=True)
                    record["replay_path"] = _common.write_replay_gif(frames, artifact / "replay.gif", 0.05)
                if trace:
                    record["trace_path"] = _common.write_trace_json(trace, artifact / "trace.json")
            except Exception as exc:
                record["artifact_error"] = str(exc)
            append_jsonl(results_path, record)
        records.extend(task_records)
        tasks.append(entry)
        append_jsonl(output / "tasks.jsonl", entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    summary = {"benchmark": "libero", "protocol": GROOT_OFFICIAL_PROTOCOL, "task_suite": args.task_suite_name,
               **summarize_groot_official_tasks(tasks), "n_records": len(records), "new_records": len(records),
               "n_envs": 1, "model_batch_size": 1, "macro_steps": GROOT_OFFICIAL_MACRO_STEPS,
               "episodes_per_task": args.episodes_per_task,
               "results_path": str(results_path), "summary_path": str(output / "summary.csv"),
               "suite_summary_path": str(output / "suite_summary.csv"),
               "tasks_path": str(output / "tasks.jsonl"), "report": None}
    _write_groot_official_csv(output / "summary.csv", tasks,
               ["task_suite", "task_id", "status", "n_episodes", "success_rate", "error_message"])
    _write_groot_official_csv(output / "suite_summary.csv", [summary],
               ["task_suite", "success_rate", "valid_tasks", "total_tasks", "failed_tasks", "n_records"])
    return summary
