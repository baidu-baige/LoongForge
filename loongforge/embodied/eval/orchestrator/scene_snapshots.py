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


def main() -> None:
    """Generate snapshots for one or more LIBERO tasks."""
    from libero.libero import benchmark as libero_benchmark

    from loongforge.embodied.eval.orchestrator.runners.libero_runner import _get_libero_env

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task", type=int, default=-1, help="-1 = every task in the suite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    suite = libero_benchmark.get_benchmark_dict()[args.suite]()
    tasks = range(suite.n_tasks) if args.task < 0 else [args.task]
    for task_id in tasks:
        task = suite.get_task(task_id)
        target = snapshot_path(args.out, args.suite, task_id, args.seed)

        def factory(task=task):
            env, _ = _get_libero_env(task, args.resolution, args.seed)
            return env

        info = dump_snapshots(factory, args.episodes, target)
        print(f"[snapshots] {args.suite}/task={task_id} seed={args.seed} -> {info['path']} ({info['sha256'][:12]})")


if __name__ == "__main__":
    main()
