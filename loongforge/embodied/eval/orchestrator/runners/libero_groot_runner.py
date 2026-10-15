# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in GR00T LIBERO protocol: synchronous macro rounds, NOT model batching.

One persistent spawned process per environment isolates LIBERO's global NumPy
RNG (including reset placement samplers and robot initialization noise). The
parent performs one batch-of-one RPC per env, then advances all envs by up to
eight actual steps. Results are committed only after an entire task succeeds.
"""

from __future__ import annotations

import csv
import json
import logging
import multiprocessing
import os
from pathlib import Path
import time
import uuid

import numpy as np

from loongforge.embodied.eval.adapters.libero import LiberoAdapter
from loongforge.embodied.eval.metrics import append_jsonl
from loongforge.embodied.eval.orchestrator.config import build_rpc_payload
from loongforge.embodied.eval.orchestrator.runners import _common
from loongforge.embodied.eval.payload_builders.groot_n1_7 import GrootN1d7PayloadBuilder
from loongforge.embodied.eval.transport import PolicyClient

PROTOCOL = "groot_official"
MACRO_STEPS = 8


class OfficialAdapter(LiberoAdapter):
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


def _make_env(task, resolution):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
        ignore_done=True,
    )


class EnvSession:
    """Lives only inside its owning process; no per-episode seed or init states."""

    def __init__(self, env, seed, max_steps, save_replay=False, save_trace=False):
        self.env = env
        self.max_steps = max_steps
        self.save_replay = save_replay
        self.save_trace = save_trace
        if seed is not None:
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
        if len(actions) != MACRO_STEPS:
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


def _env_worker(conn, task, seed, max_steps, save_replay, save_trace, factory):
    env = None
    try:
        env = factory(task, 256)
        session = EnvSession(env, seed, max_steps, save_replay, save_trace)
        conn.send(("ok", session.obs))
        while True:
            actions = conn.recv()
            if actions is None:
                break
            conn.send(("ok", session.step(actions)))
    except BaseException as exc:
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except (OSError, EOFError):
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                logging.exception("LIBERO worker close failed")
        conn.close()


class ProcessEnvs:
    """Bounded IPC and unconditional teardown, including partial startup."""

    def __init__(self, args, task, factory=_make_env):
        self.connections = []
        self.processes = []
        # A native reset can swallow Python exceptions; parent-side timeouts
        # and terminate/kill are required rather than only SIGALRM in workers.
        self.timeout = args.per_episode_timeout_sec
        step_timeout = args.per_step_timeout_sec
        self.step_timeout = (
            MACRO_STEPS * step_timeout + self.timeout
            if step_timeout > 0 and self.timeout > 0 else self.timeout
        )
        ctx = multiprocessing.get_context("spawn")
        try:
            for idx in range(args.num_envs):
                parent, child = ctx.Pipe()
                seed = None if args.seed is None else int(args.seed) + idx
                proc = ctx.Process(
                    target=_env_worker,
                    args=(child, task, seed, args.max_steps, args.save_replay, args.save_trace, factory),
                    daemon=True,
                )
                self.connections.append(parent)
                try:
                    proc.start()
                finally:
                    child.close()
                self.processes.append(proc)
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout):
        deadline = time.monotonic() + timeout if timeout > 0 else None
        values = []
        for conn in self.connections:
            if deadline is not None and not conn.poll(max(0, deadline - time.monotonic())):
                raise TimeoutError("LIBERO worker reset/macro round timed out")
            status, value = conn.recv()  # EOF/crashed workers fail the entire task
            if status != "ok":
                raise RuntimeError(value)
            values.append(value)
        return values

    def initial(self):
        return self._receive(self.timeout)

    def step(self, chunks):
        for conn, chunk in zip(self.connections, chunks):
            conn.send(chunk)
        return self._receive(self.step_timeout)

    def close(self):
        for conn in self.connections:
            try:
                conn.send(None)
            except (OSError, EOFError):
                pass
            conn.close()
        for proc in self.processes:
            proc.join(timeout=1)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
                proc.join()


def _rpc(args, fn, *pos, **kwargs):
    with _common.alarm_timeout(args.policy_call_timeout_ms / 1000, TimeoutError):
        response = fn(*pos, **kwargs)
    if not response.get("ok", False):
        raise RuntimeError(f"Policy error: {response}")
    return response["data"]


def collect_task(args, task, task_id, run_id, pool_factory=ProcessEnvs, client_factory=PolicyClient):
    """Return exactly max(requested, n_envs) completed episodes, or raise."""
    target = max(args.episodes_per_task, args.num_envs)
    adapter = OfficialAdapter(suite_name=args.task_suite_name)
    builders = []
    for _ in range(args.num_envs):
        builder, key, _ = _common.build_policy_stack(args, adapter)
        if (not isinstance(builder, GrootN1d7PayloadBuilder) or key
                or builder.state_encoding != "libero_ee_axis_angle"):
            raise ValueError("groot_official requires GR00T N1.7 LIBERO axis-angle policy")
        builders.append(builder)
    pool = client = None
    records = []
    try:
        pool = pool_factory(args, task)
        observations = pool.initial()
        with _common.alarm_timeout(args.policy_call_timeout_ms / 1000, TimeoutError):
            client = client_factory(host=args.host, port=args.port)
        ids = [f"{run_id}/task{task_id}/env{i}" for i in range(args.num_envs)]
        steps = [0] * args.num_envs
        started = [time.monotonic()] * args.num_envs
        for i, builder in enumerate(builders):
            _rpc(args, client.reset, ids[i])
            builder.reset(ids[i])
        while len(records) < target:
            chunks = []
            for i, builder in enumerate(builders):
                if (args.per_episode_timeout_sec > 0
                        and time.monotonic() - started[i] > args.per_episode_timeout_sec):
                    raise TimeoutError(f"Episode deadline exceeded for env {i}")
                context = {"instruction": task.language, "episode_id": ids[i],
                           "episode_step": steps[i], "benchmark_name": "libero"}
                canonical = adapter.obs_to_canonical(observations[i], context)
                data = _rpc(args, client.predict_action, **build_rpc_payload(
                    builder, canonical, context, disable_action_cache=True, return_action_chunk=True,
                ))
                builder.update_from_response(data)
                chunk = np.asarray(data["actions"], dtype=np.float32)
                if chunk.ndim != 2 or chunk.shape[0] < MACRO_STEPS or chunk.shape[1] != 7:
                    raise ValueError(f"Expected >=8 x 7 action chunk, got {chunk.shape}")
                chunks.append([adapter.action_from_canonical({"actions": action})
                               for action in chunk[:MACRO_STEPS]])
            # Barrier: all inference precedes all env stepping, and all env
            # responses (even surplus completions/errors) precede collection.
            results = pool.step(chunks)
            now = time.monotonic()
            if (args.per_episode_timeout_sec > 0
                    and any(now - start > args.per_episode_timeout_sec for start in started)):
                raise TimeoutError("Episode deadline exceeded during macro round")
            for i, (obs, completed) in enumerate(results):
                observations[i] = obs
                if completed is not None:
                    if len(records) < target:
                        records.append({
                            **completed, "benchmark": "libero", "protocol": PROTOCOL,
                            "task_suite": args.task_suite_name, "task_id": task_id,
                            "task": task.language, "episode_idx": len(records), "env_idx": i,
                            "env_seed": None if args.seed is None else int(args.seed) + i,
                            "episode_id": f"{ids[i]}/completion{len(records)}",
                        })
                    # Stable env cache key is cleared on EVERY autoreset; no
                    # unused action from the old chunk survives into next round.
                    _rpc(args, client.reset, ids[i])
                    builders[i].reset(ids[i])
                    steps[i] = 0
                    started[i] = time.monotonic()
                else:
                    steps[i] += MACRO_STEPS
        return records
    finally:
        if pool is not None:
            pool.close()
        if client is not None:
            try:
                client.close()
            except Exception:
                logging.exception("Policy client close failed")


def summarize_tasks(tasks):
    rates = [t["success_rate"] for t in tasks
             if t["status"] == "OK" and isinstance(t.get("success_rate"), (int, float))
             and np.isfinite(t["success_rate"]) and 0 <= t["success_rate"] <= 1]
    return {"success_rate": sum(rates) / len(rates) if rates else None,
            "valid_tasks": len(rates), "total_tasks": len(tasks),
            "failed_tasks": sum(t["status"] == "FAILED" for t in tasks)}


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_batch(args):
    if args.num_envs < 1 or args.episodes_per_task < 1:
        raise ValueError("num_envs and episodes_per_task must be positive")
    if args.num_steps_wait != 0 or getattr(args, "chunk_execute_steps", 8) != 8:
        raise ValueError("groot_official requires num_steps_wait=0 and chunk_execute_steps=8")
    if (getattr(args, "control_mode", "auto") not in {"auto", "delta"}
            or getattr(args, "continuous_gripper", False)
            or getattr(args, "render_resolution", 256) not in {0, 256}):
        raise ValueError("groot_official requires delta control, sign gripper and 256px rendering")
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
    output = Path(args.output_dir) / PROTOCOL / run_id
    output.mkdir(parents=True, exist_ok=False)
    results_path = output / "results.jsonl"
    results_path.touch()
    logging.info("GR00T official protocol: %d isolated envs, per-env RPC (NOT batch inference); "
                 "resume/restart ignored, fresh results: %s", args.num_envs, output)
    tasks, records = [], []
    for task_id in task_ids:
        entry = {"task_id": task_id, "task_suite": args.task_suite_name}
        try:
            manager = getattr(args, "server_manager", None)
            if manager is not None:
                manager.ensure_running()
            task_records = collect_task(args, suite.get_task(task_id), task_id, run_id)
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
    summary = {"benchmark": "libero", "protocol": PROTOCOL, "task_suite": args.task_suite_name,
               **summarize_tasks(tasks), "n_records": len(records), "new_records": len(records),
               "n_envs": args.num_envs, "model_batch_size": 1, "macro_steps": MACRO_STEPS,
               "episodes_per_task": max(args.episodes_per_task, args.num_envs),
               "results_path": str(results_path), "summary_path": str(output / "summary.csv"),
               "suite_summary_path": str(output / "suite_summary.csv"),
               "tasks_path": str(output / "tasks.jsonl"), "report": None}
    _write_csv(output / "summary.csv", tasks,
               ["task_suite", "task_id", "status", "n_episodes", "success_rate", "error_message"])
    _write_csv(output / "suite_summary.csv", [summary],
               ["task_suite", "success_rate", "valid_tasks", "total_tasks", "failed_tasks", "n_records"])
    return summary
