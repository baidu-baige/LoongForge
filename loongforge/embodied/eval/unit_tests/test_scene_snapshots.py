# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per_task RNG snapshots (T-17)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from loongforge.embodied.eval.orchestrator import scene_snapshots


class _FakeEnv:
    """Stands in for a LIBERO env: reset draws the scene from the global RNG."""

    def __init__(self) -> None:
        self.scenes = []

    def reset(self) -> None:
        self.scenes.append(int(np.random.randint(0, 10**6)))


def test_snapshot_roundtrip_reproduces_the_serial_scene_sequence(tmp_path: Path) -> None:
    """Restoring the pre-reset state gives the same scene as a serial run."""
    np.random.seed(42)
    serial = _FakeEnv()
    path = tmp_path / "libero_10__task0__seed42.pkl"
    info = scene_snapshots.dump_snapshots(lambda: serial, 6, path)
    assert info["episodes"] == 6
    assert len(info["sha256"]) == 64

    np_states, py_states = scene_snapshots.load_snapshots(path)
    worker = _FakeEnv()
    for index in range(6):
        scene_snapshots.apply_snapshot(np_states[index], py_states[index])
        worker.reset()

    assert worker.scenes == serial.scenes


def test_snapshot_out_of_range_is_detected(tmp_path: Path) -> None:
    """A shard asking for an episode beyond the snapshot list must fail loudly."""
    np.random.seed(7)
    path = tmp_path / "snap.pkl"
    scene_snapshots.dump_snapshots(lambda: _FakeEnv(), 2, path)
    np_states, _ = scene_snapshots.load_snapshots(path)
    assert len(np_states) == 2


def test_unknown_snapshot_version_is_rejected(tmp_path: Path) -> None:
    """Version drift must not be silently accepted."""
    path = tmp_path / "bad.pkl"
    path.write_bytes(b"\x80\x04}q\x00.")
    with pytest.raises(ValueError, match="unsupported scene snapshot version"):
        scene_snapshots.load_snapshots(path)


def test_plan_jobs_expands_every_suite_and_task() -> None:
    """`--suites a,b` x N tasks becomes one independent job per (suite, task)."""
    sizes = {"libero_10": 10, "libero_object": 10, "libero_spatial": 4}
    jobs = scene_snapshots.plan_jobs(
        ["libero_10", "libero_object"], task=-1, seed=42, episodes=50, resolution=256,
        out="/tmp/snaps", n_tasks=sizes.__getitem__,
    )
    assert len(jobs) == 20
    assert {(suite, task) for suite, task, *_rest in jobs} == {
        (suite, task) for suite in ("libero_10", "libero_object") for task in range(10)
    }
    assert {job[2:] for job in jobs} == {(42, 50, 256, "/tmp/snaps")}

    single = scene_snapshots.plan_jobs(
        ["libero_10", "libero_spatial"], task=3, seed=7, episodes=10, resolution=128,
        out="/tmp/snaps", n_tasks=sizes.__getitem__,
    )
    assert single == [
        ("libero_10", 3, 7, 10, 128, "/tmp/snaps"),
        ("libero_spatial", 3, 7, 10, 128, "/tmp/snaps"),
    ]


def test_generate_one_writes_a_verified_file(tmp_path: Path) -> None:
    """One job writes its own file and is read back before reporting success."""
    np.random.seed(3)
    job = ("libero_10", 3, 42, 6, 256, str(tmp_path))
    info = scene_snapshots.generate_one(job, make_env=lambda: _FakeEnv())

    assert info["path"] == str(tmp_path / "libero_10__task3__seed42.pkl")
    assert (info["suite"], info["task"], info["seed"], info["episodes"]) == ("libero_10", 3, 42, 6)
    np_states, py_states = scene_snapshots.load_snapshots(info["path"])
    assert len(np_states) == len(py_states) == 6


def test_run_job_reports_a_failure_instead_of_raising(tmp_path: Path) -> None:
    """One broken task must not hide the results of the others."""

    def boom():
        raise OSError("osmesa unavailable")

    result = scene_snapshots._run_job(("libero_10", 5, 42, 2, 256, str(tmp_path)), make_env=boom)
    assert result["ok"] is False
    assert result["suite"] == "libero_10" and result["task"] == 5
    assert "osmesa unavailable" in result["error"]

    ok = scene_snapshots._run_job(
        ("libero_object", 1, 42, 2, 256, str(tmp_path)), make_env=lambda: _FakeEnv()
    )
    assert ok["ok"] is True and ok["task"] == 1
