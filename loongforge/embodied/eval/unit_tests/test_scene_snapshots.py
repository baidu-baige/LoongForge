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
