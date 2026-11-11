# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Scene RNG snapshots for LIBERO's ``per_task`` seeding (T-17).

In official ``per_task`` mode the env is seeded once and reused, so the scene an
episode sees depends on the *number of resets already performed in that process*.
That makes episode-level sharding incorrect: every shard would replay the first
scenes. This module records the RNG state just before each reset for a
``(suite, task, seed)`` and lets a worker restore the right state before its own
reset, so any episode can run in any process.

Only ``np.random`` is seeded by LIBERO (``BDDLBaseDomain.seed``); the Python
``random`` state is captured too for safety. Snapshots are tiny (~2.5KB/episode).
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

SNAPSHOT_VERSION = 1


def snapshot_filename(suite: str, task_id: int, seed: int) -> str:
    """Stable file name for one task's snapshot list."""
    return f"{suite}__task{task_id}__seed{seed}.pkl"


def snapshot_path(directory: str | Path, suite: str, task_id: int, seed: int) -> Path:
    """Path of one task's snapshot file."""
    return Path(directory) / snapshot_filename(suite, task_id, seed)


def dump_snapshots(
    env_factory: Callable[[], Any],
    episodes: int,
    path: str | Path,
    reset: Callable[[Any], Any] | None = None,
) -> Dict[str, Any]:
    """Record the RNG state before each of ``episodes`` resets.

    ``env_factory`` must create and seed the env exactly like the runner does
    (create -> seed -> return). ``reset`` defaults to ``env.reset()``; tests can
    inject a cheap consumer instead of a real simulation.
    """
    env = env_factory()
    do_reset = reset or (lambda e: e.reset())
    np_states: List[Any] = []
    py_states: List[Any] = []
    for _ in range(int(episodes)):
        np_states.append(np.random.get_state())
        py_states.append(random.getstate())
        do_reset(env)
    payload = {"version": SNAPSHOT_VERSION, "np": np_states, "py": py_states}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(out)
    return {
        "path": str(out),
        "episodes": int(episodes),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }


def load_snapshots(path: str | Path) -> Tuple[List[Any], List[Any]]:
    """Load ``(np_states, py_states)`` from a snapshot file."""
    payload = pickle.loads(Path(path).read_bytes())
    if payload.get("version") != SNAPSHOT_VERSION:
        raise ValueError(f"unsupported scene snapshot version: {payload.get('version')!r} in {path}")
    return payload["np"], payload["py"]


def apply_snapshot(np_state: Any, py_state: Any) -> None:
    """Restore the RNG states captured before an episode's reset."""
    np.random.set_state(np_state)
    random.setstate(py_state)


def _libero_env_factory(suite_name: str, task_id: int, seed: int, resolution: int) -> Callable[[], Any]:
    """Build the zero-arg env factory the runner would use (imports libero lazily).

    Importing here keeps ``import scene_snapshots`` free of the LIBERO/MuJoCo
    stack, so the planning unit tests can exercise this module without it.
    """
    from libero.libero import benchmark as libero_benchmark

    from loongforge.embodied.eval.orchestrator.runners.libero_runner import _get_libero_env

    task = libero_benchmark.get_benchmark_dict()[suite_name]().get_task(task_id)

    def factory() -> Any:
        env, _ = _get_libero_env(task, resolution, seed)
        return env

    return factory


def _libero_n_tasks(suite_name: str) -> int:
    """Task count of a suite (imports LIBERO lazily)."""
    from libero.libero import benchmark as libero_benchmark

    suites = libero_benchmark.get_benchmark_dict()
    if suite_name not in suites:
        raise SystemExit(f"unknown LIBERO suite {suite_name!r}; available: {sorted(suites)}")
    return int(suites[suite_name]().n_tasks)


def plan_jobs(
    suites: List[str],
    task: int,
    seed: int,
    episodes: int,
    resolution: int,
    out: str,
    n_tasks: Callable[[str], int] | None = None,
) -> List[Tuple[str, int, int, int, int, str]]:
    """Expand ``(--suites, --task)`` into one job per ``(suite, task_id)``.

    Each job writes its own ``{suite}__task{k}__seed{seed}.pkl``, so jobs are
    fully independent and can run in any order / in parallel. ``n_tasks`` is
    injectable for tests; production resolves it from LIBERO.
    """
    if n_tasks is None:
        n_tasks = _libero_n_tasks
    jobs: List[Tuple[str, int, int, int, int, str]] = []
    for suite_name in suites:
        if not suite_name:
            continue
        ids = [int(task)] if task >= 0 else list(range(n_tasks(suite_name)))
        jobs.extend(
            (suite_name, task_id, int(seed), int(episodes), int(resolution), str(out))
            for task_id in ids
        )
    return jobs


def generate_one(
    job: Tuple[str, int, int, int, int, str],
    make_env: Callable[[], Any] | None = None,
) -> Dict[str, Any]:
    """Generate and verify one task's snapshot file.

    ``make_env`` is injectable for tests; production resolves it from LIBERO.
    The file is read back so a truncated write fails here instead of during the
    evaluation run (``has N states but episode_idx=...``).
    """
    suite_name, task_id, seed, episodes, resolution, out = job
    if make_env is None:
        make_env = _libero_env_factory(suite_name, task_id, seed, resolution)
    target = snapshot_path(out, suite_name, task_id, seed)
    info = dump_snapshots(make_env, episodes, target)
    np_states, py_states = load_snapshots(target)
    if len(np_states) != episodes or len(py_states) != episodes:
        raise RuntimeError(
            f"{target}: wrote {len(np_states)} states but {episodes} were requested"
        )
    info.update({"suite": suite_name, "task": task_id, "seed": seed})
    return info


def _run_job(
    job: Tuple[str, int, int, int, int, str],
    make_env: Callable[[], Any] | None = None,
) -> Dict[str, Any]:
    """Pool entry point: never raises, so one bad task does not hide the rest."""
    try:
        return {"ok": True, **generate_one(job, make_env=make_env)}
    except Exception as error:  # noqa: BLE001 - reported in the summary
        suite_name, task_id = job[0], job[1]
        return {"ok": False, "suite": suite_name, "task": task_id, "error": f"{type(error).__name__}: {error}"}


def _report(result: Dict[str, Any], index: int, total: int) -> None:
    """Print one task's progress line (success or failure)."""
    if result["ok"]:
        print(
            f"[snapshots] [{index:3d}/{total}] {result['suite']}/task={result['task']} "
            f"n={result['episodes']} -> {result['path']} ({result['sha256'][:12]})",
            flush=True,
        )
    else:
        print(
            f"[snapshots] [{index:3d}/{total}] FAILED {result['suite']}/task={result['task']}: "
            f"{result['error']}",
            flush=True,
        )


def main() -> None:
    """Generate snapshots for one or more LIBERO suites and tasks."""
    parser = argparse.ArgumentParser(
        description="Record the pre-reset RNG state of every LIBERO episode so a "
        "per_task run can be sharded by episode (see benchmark.scene_snapshot_dir)."
    )
    parser.add_argument("--suite", default="libero_10", help="single suite (kept for compatibility)")
    parser.add_argument("--suites", default="", help="comma-separated suites, e.g. libero_10,libero_object")
    parser.add_argument("--task", type=int, default=-1, help="-1 = every task in the suite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50,
                        help="snapshots per task; must be >= benchmark.episodes_per_task")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--jobs", type=int, default=1, help="parallel task workers (default 1 = serial)")
    parser.add_argument("--out", required=True, help="one directory for every suite (names carry the suite)")
    args = parser.parse_args()

    suites = [name.strip() for name in (args.suites or args.suite).split(",") if name.strip()]
    if not suites:
        parser.error("no suite given")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    jobs = plan_jobs(suites, args.task, args.seed, args.episodes, args.resolution, args.out)
    if not jobs:
        parser.error("no task to snapshot")

    # Suite objects live in the child processes: one task per process keeps the
    # MuJoCo/osmesa state isolated, exactly like running the CLI per task.
    if args.jobs > 1:
        from multiprocessing import Pool

        with Pool(processes=min(args.jobs, len(jobs))) as pool:
            results = []
            for index, result in enumerate(pool.imap_unordered(_run_job, jobs), start=1):
                results.append(result)
                _report(result, index, len(jobs))
    else:
        results = [_run_job(job) for job in jobs]
        for index, result in enumerate(results, start=1):
            _report(result, index, len(jobs))

    failed = [result for result in results if not result["ok"]]
    print(f"[snapshots] {len(results) - len(failed)}/{len(results)} tasks written to {args.out}")
    if failed:
        names = ", ".join(f"{result['suite']}/task={result['task']}" for result in failed)
        print(f"[snapshots] {len(failed)} failed (see the lines above): {names}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
