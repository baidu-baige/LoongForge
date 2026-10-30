# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for asynchronous ZeRO optimizer state saving (#308).

Covers the three helpers added to ``checkpoint.py`` — ``_cpu_snapshot``,
``_submit_async_save``, ``_merge_futures`` — and the async branch of
``_save_zero_optimizer_state``, including the snapshot-isolation invariant
(the background ``torch.save`` must not observe post-save mutations) and the
``flush_pending_save`` finalization path. The ZeRO-specific helpers are
monkeypatched so no real ``ZeroRedundancyOptimizer`` / process group is needed.
"""

from concurrent.futures import CancelledError, Future
from types import SimpleNamespace

import pytest
import torch

import loongforge.embodied.distributed.checkpoint as ck
from loongforge.embodied.distributed import DistributedContext


@pytest.fixture(autouse=True)
def _clear_pending():
    ck._pending_async_save = None
    yield
    ck._pending_async_save = None


def _ctx() -> DistributedContext:
    return DistributedContext()


def _patch_zero_helpers(monkeypatch, state):
    """Replace the ZeRO optimizer internals ``_save_zero_optimizer_state`` uses."""
    child = SimpleNamespace(
        param_groups=[{"lr": 0.0}],
        optim=SimpleNamespace(param_groups=[{}]),
    )
    monkeypatch.setattr(
        ck, "_zero_optimizer_children",
        lambda optimizer: [child],
    )
    monkeypatch.setattr(ck, "_copy_param_group_options", lambda source, target: None)
    monkeypatch.setattr(ck, "_local_optimizer_state_dict", lambda optim: state)


def test_cpu_snapshot_preserves_structure_and_isolates():
    live = torch.tensor([1.0, 2.0])
    snap = ck._cpu_snapshot({"a": live, "b": [live, (3, 4)], "c": 5, "d": None})

    assert set(snap) == {"a", "b", "c", "d"}
    assert isinstance(snap["b"], list)
    assert isinstance(snap["b"][1], tuple)
    assert snap["c"] == 5 and snap["d"] is None

    # Tensors are copied, not aliased to the live object.
    assert snap["a"] is not live
    assert snap["b"][0] is not live

    # Mutating the live tensor must not leak into the snapshot.
    live.add_(10.0)
    assert snap["a"].tolist() == [1.0, 2.0]
    assert snap["b"][0].tolist() == [1.0, 2.0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_snapshot_moves_gpu_tensor_to_cpu():
    live = torch.randn(2, device="cuda")
    snap = ck._cpu_snapshot({"t": live})
    assert snap["t"].device.type == "cpu"
    assert snap["t"].tolist() == live.detach().cpu().tolist()


def test_submit_async_save_resolves(tmp_path):
    out = tmp_path / "f.pt"
    fut = ck._submit_async_save(torch.save, {"x": torch.tensor(1.0)}, str(out))
    assert isinstance(fut, Future)
    assert fut.result(timeout=10) is None
    assert out.exists()


def test_merge_futures_all_success():
    f1, f2 = Future(), Future()
    merged = ck._merge_futures([f1, f2])

    f1.set_result(None)
    assert not merged.done()
    f2.set_result(None)
    assert merged.result(timeout=10) is None


def test_merge_futures_waits_for_all_then_propagates_error():
    f1, f2 = Future(), Future()
    merged = ck._merge_futures([f1, f2])

    f1.set_exception(RuntimeError("boom"))
    # Must not resolve until the remaining future also settles.
    assert not merged.done()
    f2.set_result(None)

    with pytest.raises(RuntimeError, match="boom"):
        merged.result(timeout=10)


def test_merge_futures_cancelled_input_resolves():
    f1, f2 = Future(), Future()
    merged = ck._merge_futures([f1, f2])

    f1.cancel()
    f2.set_result(None)

    with pytest.raises(CancelledError):
        merged.result(timeout=10)


def test_save_zero_optimizer_state_async_writes_and_returns_future(tmp_path, monkeypatch):
    ctx = _ctx()
    _patch_zero_helpers(monkeypatch, {"state": {"exp_avg": torch.tensor([1.0, 2.0])}})

    fut = ck._save_zero_optimizer_state(str(tmp_path), object(), ctx, async_save=True)

    assert isinstance(fut, Future)
    assert fut.result(timeout=10) is None

    rank_path = tmp_path / "zero_optimizer" / "rank_0.pt"
    assert rank_path.exists()
    loaded = torch.load(rank_path, map_location="cpu", weights_only=False)
    assert loaded[0]["state"]["exp_avg"].tolist() == [1.0, 2.0]
    assert (tmp_path / "zero_optimizer" / "metadata.json").exists()


def test_save_zero_optimizer_state_snapshot_isolated(tmp_path, monkeypatch):
    """The background write must not observe mutations made after launch."""
    ctx = _ctx()
    live = torch.tensor([1.0, 2.0])
    _patch_zero_helpers(monkeypatch, {"state": {"exp_avg": live}})

    fut = ck._save_zero_optimizer_state(str(tmp_path), object(), ctx, async_save=True)

    # Simulate the next optimizer.step() mutating the live tensor immediately.
    live.add_(100.0)
    fut.result(timeout=10)

    loaded = torch.load(
        tmp_path / "zero_optimizer" / "rank_0.pt", map_location="cpu", weights_only=False
    )
    assert loaded[0]["state"]["exp_avg"].tolist() == [1.0, 2.0]


def test_save_zero_optimizer_state_sync_path(tmp_path, monkeypatch):
    """The synchronous path is unchanged: no future, immediate durable write."""
    ctx = _ctx()
    _patch_zero_helpers(monkeypatch, {"state": {"exp_avg": torch.tensor([3.0, 4.0])}})

    fut = ck._save_zero_optimizer_state(str(tmp_path), object(), ctx, async_save=False)

    assert fut is None
    loaded = torch.load(
        tmp_path / "zero_optimizer" / "rank_0.pt", map_location="cpu", weights_only=False
    )
    assert loaded[0]["state"]["exp_avg"].tolist() == [3.0, 4.0]


def test_async_zero_save_finalizes_via_flush_pending_save(tmp_path, monkeypatch):
    """An async ZeRO save finalizes (writes resume_meta.json) through flush_pending_save."""
    ctx = _ctx()
    _patch_zero_helpers(monkeypatch, {"state": {"exp_avg": torch.tensor([1.0])}})

    fut = ck._save_zero_optimizer_state(str(tmp_path), object(), ctx, async_save=True)
    ck._pending_async_save = {
        "future": fut,
        "path": str(tmp_path),
        "meta": {"completed_steps": 1},
        "is_main": True,
        "rank": 0,
        "save_format": "dcp",
    }

    ck.flush_pending_save(ctx)

    assert (tmp_path / "resume_meta.json").exists()
