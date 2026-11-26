# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""RoboLab single-episode runner.

RoboLab is an Isaac Sim application: the simulator must own the process, so
this runner launches Isaac Sim in-process (``AppLauncher``) and then drives one
episode against a LoongForge policy server, exactly like the other in-process
benchmark runners (ManiSkill / SimplerEnv).

One invocation plays exactly one episode: the planner turns
``(task, seed)`` into one shard, so an episode is reproducible as
``RoboLab --task <task> --seed <seed> --num-runs 1``.

All Isaac Sim / robolab imports are deferred into the functions below: they are
only valid *after* ``AppLauncher`` has created the app, and the module is
imported by the orchestrator before that.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from loongforge.embodied.eval.adapters.robolab import ROBOLAB_ACTION_DIM, RobolabAdapter
from loongforge.embodied.eval.metrics.results import append_jsonl, write_suite_summary_csv, write_summary_csv
from loongforge.embodied.eval.orchestrator.config import build_rpc_payload, load_config
from loongforge.embodied.eval.orchestrator.runners import _common
from loongforge.embodied.eval.transport import PolicyClient


def _launch_app(args: argparse.Namespace) -> Any:
    """Launch Isaac Sim before any isaaclab/robolab module is imported.

    Teardown is registered with ``atexit`` because ``SimulationApp.close()``
    terminates the process on the spot: closing it inside ``run_evaluation``
    would drop the results file and the JSON summary.
    """
    import atexit

    import cv2  # noqa: F401  RoboLab imports cv2 before isaaclab. Do not remove.

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    app_args = parser.parse_args([])
    app_args.headless = bool(args.headless)
    # RoboLab policies read camera sensors, which need the render pipeline.
    app_args.enable_cameras = True
    app_args.device = str(args.device)
    app = AppLauncher(app_args).app
    atexit.register(app.close)
    return app


def _build_env(args: argparse.Namespace) -> Tuple[Any, Any, str]:
    """Register RoboLab tasks, create the env and return (env, env_cfg, instruction)."""
    import robolab.constants
    from robolab.core.environments.runtime import create_env
    from robolab.core.task.instruction import primary_instruction
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

    # Same camera preset as the official policies/cosmos3/run.py so registered
    # scenes and image keys match the upstream evaluator.
    auto_register_droid_envs(task=args.task_name, cameras=WRIST_LEFT_RIGHT_HEAD)

    scene_dir = pathlib.Path(args.output_dir) / "artifacts" / "robolab" / args.task_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    robolab.constants.set_output_dir(str(scene_dir))

    env, env_cfg = create_env(
        args.task_name,
        device=str(args.device),
        seed=int(args.seed),
        num_envs=max(1, int(args.num_envs)),
        instruction_type=args.instruction_type,
        policy=args.model_name or "loongforge",
        renderer=args.renderer,
        rendering_mode=args.rendering_type,
    )
    _, instruction = primary_instruction(env_cfg.instruction)
    return env, env_cfg, instruction


def _write_standard_outputs(output_dir: str, records: List[Dict[str, Any]]) -> None:
    """Write the standard JSONL and CSV result files for one unit."""
    out_dir = pathlib.Path(output_dir)
    safe_records = [_common.json_safe(record) for record in records]
    for safe_record in safe_records:
        append_jsonl(out_dir / "results.jsonl", safe_record)
    write_summary_csv(out_dir / "summary.csv", safe_records)
    write_suite_summary_csv(out_dir / "suite_summary.csv", safe_records)


def _episode_id(args: argparse.Namespace, run_idx: int, env_id: int) -> str:
    """Stable per-env episode id (globally unique: task x run x env)."""
    return f"robolab/{args.task_name}/run={run_idx}/env={env_id}"


def _task_tags(task_name: str) -> Dict[str, Any]:
    """Difficulty / attribute tags from RoboLab's packaged ``task_metadata.json``.

    Stored on every episode record so result files are self-describing: the report can
    group by difficulty/capability without a RoboLab checkout. Empty when unavailable.
    """
    try:
        import robolab

        path = pathlib.Path(robolab.__file__).parent / "tasks" / "_metadata" / "task_metadata.json"
        for item in json.loads(path.read_text(encoding="utf-8")):
            if item.get("task_name") == task_name:
                return {"difficulty_label": item.get("difficulty_label"), "task_attributes": item.get("attributes")}
    except Exception:  # metadata is optional reporting data, never fail a run over it
        pass
    return {}


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run ``num_runs`` x ``num_envs`` RoboLab episodes against a LoongForge policy server."""
    import torch

    app = _launch_app(args)  # noqa: F841  kept alive for the whole run
    env = None
    client = None
    num_envs = max(1, int(args.num_envs))
    num_runs = max(1, int(args.num_runs))
    records: List[Dict[str, Any]] = []
    action_trace: List[Dict[str, Any]] = []
    action_decoder_key = ""
    server_metadata: Dict[str, Any] = {}
    task_tags = _task_tags(args.task_name)
    start_time = time.time()
    try:
        env, _env_cfg, instruction = _build_env(args)
        action_dim = int(getattr(getattr(env, "action_manager", None), "total_action_dim", ROBOLAB_ACTION_DIM))
        if action_dim != ROBOLAB_ACTION_DIM:
            raise ValueError(
                f"RoboLab env {args.task_name!r} exposes {action_dim}D actions, "
                f"the adapter produces {ROBOLAB_ACTION_DIM}D"
            )

        adapter = RobolabAdapter(
            task_name=args.task_name,
            control_hz=args.control_hz,
            env_id=0,
        )
        payload_builder, action_decoder_key, action_decoder = _common.build_policy_stack(args, adapter)
        client = PolicyClient(host=args.host, port=args.port)
        server_metadata = dict(client.metadata)
        max_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(env.max_episode_length)

        for run_idx in range(num_runs):
            # One RoboLab "run" = one reset of every parallel env. Upstream
            # robolab.eval.episode.run_episode resets twice; keep that so a
            # (task, seed) here lands on the same scene as the official evaluator.
            obs, _ = env.reset()
            obs, _ = env.reset()
            if env.recorder_manager is not None and hasattr(env.recorder_manager, "set_hdf5_file"):
                env.recorder_manager.set_hdf5_file(f"run_{run_idx}.hdf5")
                env.recorder_manager.set_episode_index(run_idx, env_ids=list(range(num_envs)))

            episode_ids = {env_id: _episode_id(args, run_idx, env_id) for env_id in range(num_envs)}
            steppers = {}
            for env_id, episode_id in episode_ids.items():
                client.reset(episode_id)
                payload_builder.reset(episode_id)
                action_decoder.reset()
                steppers[env_id] = _common.EpisodePolicyStepper(
                    chunk_steps=int(client.metadata.get("chunk_execute_steps") or 0),
                    return_action_chunk=bool(getattr(args, "return_action_chunk", False)),
                    disable_action_cache=bool(
                        getattr(args, "disable_action_cache", False)
                        or getattr(payload_builder, "disable_action_cache", False)
                    ),
                    payload_builder=payload_builder,
                )

            def _make_request(env_id: int, episode_step: int):
                """Build the per-env inference request callable the stepper drives."""

                def _request(_request_step: int) -> Tuple[Dict[str, Any], float]:
                    canonical_obs = adapter.obs_to_canonical(
                        obs,
                        {
                            "instruction": instruction,
                            "episode_id": episode_ids[env_id],
                            "episode_step": episode_step,
                            "env_id": env_id,
                        },
                    )
                    ctx = {
                        "benchmark_name": "robolab",
                        "episode_id": episode_ids[env_id],
                        "episode_step": episode_step,
                        "instruction": instruction,
                    }
                    request_start = time.perf_counter()
                    response = client.predict_action(
                        **build_rpc_payload(
                            payload_builder,
                            canonical_obs,
                            ctx,
                            disable_action_cache=steppers[env_id].disable_action_cache,
                            return_action_chunk=steppers[env_id].return_action_chunk,
                        )
                    )
                    latency_ms = (time.perf_counter() - request_start) * 1000.0
                    if not response.get("ok", False):
                        raise RuntimeError(f"Policy error: {response}")
                    return response["data"], latency_ms

                return _request

            for episode_step in range(max_steps):
                active_env_ids = list(env.active_env_ids)
                if not active_env_ids:
                    break
                actions = torch.zeros(num_envs, ROBOLAB_ACTION_DIM, device=env.device)
                for env_id in active_env_ids:
                    decision = steppers[env_id].next_action(_make_request(env_id, episode_step))
                    processed = action_decoder(decision.action, {"benchmark_name": "robolab"})
                    env_action = adapter.action_from_canonical({"actions": processed[0]})
                    actions[env_id] = torch.as_tensor(env_action, device=env.device)
                    if args.save_trace:
                        action_trace.append(
                            {
                                "run": run_idx,
                                "env_id": env_id,
                                "step": episode_step,
                                "env_action": np.asarray(env_action).tolist(),
                                "inference_latency_ms": decision.inference_latency_ms,
                            }
                        )
                obs, _reward, _term, _trunc, _info = env.step(actions)

            for result in env.get_env_results():
                env_id = int(result["env_id"])
                success = bool(result.get("success"))
                records.append(
                    {
                        "benchmark": "robolab",
                        "task_suite": args.task_name,
                        "task_id": 0,
                        "episode_idx": run_idx * num_envs + env_id,
                        "episode_id": episode_ids[env_id],
                        "task_description": instruction,
                        "env_name": args.task_name,
                        "robot_setup": "droid_panda_jointpos",
                        "seed": int(args.seed),
                        "run_idx": run_idx,
                        "env_id": env_id,
                        "success": int(success),
                        "steps": int(result.get("step") or 0),
                        "failure_reason": None if success else "not_successful_within_max_steps",
                        **task_tags,
                    }
                )
            env.reset_eval_state()
    finally:
        if client is not None:
            client.close()
        if env is not None:
            env.close()

    trace_path = None
    if args.save_trace:
        trace_path = _common.write_trace_json(
            action_trace,
            pathlib.Path(args.output_dir)
            / "artifacts"
            / "robolab"
            / args.task_name
            / f"seed_{int(args.seed)}_trace.json",
        )

    elapsed = time.time() - start_time
    for record in records:
        record["episode_time_sec"] = elapsed
        record["action_decoder_key"] = action_decoder_key
        record["trace_path"] = trace_path
        record["server_metadata"] = server_metadata
    if not records:
        raise RuntimeError(f"RoboLab task {args.task_name!r} produced no episode results")
    _write_standard_outputs(args.output_dir, records)
    return {
        "benchmark": "robolab",
        "task_suite": args.task_name,
        "episodes": len(records),
        "successes": sum(int(record["success"]) for record in records),
        "episode_ids": [record["episode_id"] for record in records],
        "episode_time_sec": elapsed,
    }


def _apply_config(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    """Apply YAML config sections to parsed runner arguments."""
    benchmark = config.get("benchmark") or {}
    server = config.get("server") or {}
    run = config.get("run") or {}
    if not isinstance(benchmark, dict) or not isinstance(server, dict) or not isinstance(run, dict):
        raise ValueError("benchmark, server, and run sections must be mappings")

    mapping = [
        (benchmark, "task_name", "task_name"),
        (benchmark, "num_envs", "num_envs"),
        (benchmark, "num_runs", "num_runs"),
        (benchmark, "max_steps", "max_steps"),
        (benchmark, "control_hz", "control_hz"),
        (benchmark, "instruction_type", "instruction_type"),
        (benchmark, "headless", "headless"),
        (benchmark, "device", "device"),
        (benchmark, "renderer", "renderer"),
        (benchmark, "rendering_type", "rendering_type"),
        (benchmark, "return_action_chunk", "return_action_chunk"),
        (server, "host", "host"),
        (server, "port", "port"),
        (server, "random_init", "random_init"),
        (run, "seed", "seed"),
        (run, "output_dir", "output_dir"),
        (run, "save_trace", "save_trace"),
        (run, "disable_action_cache", "disable_action_cache"),
        # ``run:`` wins over ``benchmark:`` when both set it.
        (run, "return_action_chunk", "return_action_chunk"),
    ]
    for section, key, attr in mapping:
        value = section.get(key) if isinstance(section, dict) else None
        if value is not None:
            setattr(args, attr, value)
    # A shard pins its own seed (one episode per seed), so it wins over run.seed.
    if benchmark.get("seed") is not None:
        args.seed = int(benchmark["seed"])
    if run.get("save_replay"):
        print(
            "[robolab] WARNING: run.save_replay is not supported by the LoongForge RoboLab "
            "runner; record videos with the upstream RoboLab evaluator instead.",
            flush=True,
        )
    return args


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI arguments for the RoboLab runner."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--task-name", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel envs (episodes) inside one Isaac Sim process")
    parser.add_argument("--num-runs", type=int, default=1, help="Sequential resets per env; episodes = num_envs * num_runs")
    parser.add_argument("--control-hz", type=int, default=15)
    parser.add_argument("--instruction-type", nargs="*", default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--renderer", default="realtime", choices=["realtime", "pathtracing"])
    parser.add_argument("--rendering-type", default=None, choices=["performance", "balanced", "quality"])
    parser.add_argument("--max-steps", type=int, default=0, help="0 = env.max_episode_length")
    parser.add_argument("--disable-action-cache", action="store_true")
    parser.add_argument("--return-action-chunk", action="store_true")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument("--output-dir", default="/tmp/loongforge_eval/robolab")
    return parser


def main() -> None:
    """Run the RoboLab episode from command-line arguments."""
    args = build_argparser().parse_args()
    if args.config:
        args.raw_config = load_config(args.config)
        args = _apply_config(args, args.raw_config)
    result = run_evaluation(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
