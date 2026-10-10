# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for early finalization of async DCP checkpoints.

``maybe_finalize_pending_save`` is the probe the training loop runs every step
so that ``resume_meta.json`` is written as soon as the background write lands,
rather than when the next save starts. These tests drive it single-process
(``is_distributed`` is False, so the cross-rank vote is skipped) and assert the
finalization, the idempotence, and the failure path.
"""

from concurrent.futures import Future

import pytest

import loongforge.engines.torch.checkpointing as ck
from loongforge.engines.torch.distributed import DistributedContext


class _StubAsyncSaveResponse:
    """Stand-in for torch's ``AsyncSaveResponse`` (staging vs upload futures)."""

    def __init__(self, staging, upload):
        self.staging_completion = staging
        self.upload_completion = upload


@pytest.fixture(autouse=True)
def _clear_pending():
    ck._pending_async_save = None
    yield
    ck._pending_async_save = None


def _ctx() -> DistributedContext:
    return DistributedContext()


def _pending(path, handle, is_main=True):
    return {
        "future": handle,
        "path": str(path),
        "meta": {
            "completed_steps": 100,
            "epoch": 0,
            "ckpt_format": "dcp",
            "world_size": 1,
            "use_lora": False,
        },
        "is_main": is_main,
        "rank": 0,
        "save_format": "dcp",
    }


def _done_future():
    future = Future()
    future.set_result(None)
    return future


def test_no_pending_save_is_a_noop(tmp_path):
    assert ck.maybe_finalize_pending_save(_ctx()) is False
    assert not (tmp_path / "resume_meta.json").exists()


def test_finalizes_as_soon_as_the_write_is_done(tmp_path):
    ck._pending_async_save = _pending(tmp_path, _done_future())

    assert ck.maybe_finalize_pending_save(_ctx()) is True
    assert (tmp_path / "resume_meta.json").exists()
    assert ck._pending_async_save is None


def test_finalize_is_idempotent(tmp_path):
    ck._pending_async_save = _pending(tmp_path, _done_future())
    assert ck.maybe_finalize_pending_save(_ctx()) is True

    # Slot cleared -> later steps are free no-ops.
    assert ck.maybe_finalize_pending_save(_ctx()) is False


def test_waits_for_a_slow_write_then_finalizes_the_next_step(tmp_path):
    future = Future()
    ck._pending_async_save = _pending(tmp_path, future)

    # While the write is running the probe leaves the save pending.
    assert ck.maybe_finalize_pending_save(_ctx()) is False
    assert not (tmp_path / "resume_meta.json").exists()

    future.set_result(None)
    assert ck.maybe_finalize_pending_save(_ctx()) is True
    assert (tmp_path / "resume_meta.json").exists()


def test_failed_async_save_aborts_training_when_probed(tmp_path):
    step_dir = tmp_path / "steps_100"
    step_dir.mkdir()
    future = Future()
    future.set_exception(RuntimeError("disk on fire"))
    ck._pending_async_save = _pending(step_dir, future)

    with pytest.raises(RuntimeError, match="async checkpoint commit failed"):
        ck.maybe_finalize_pending_save(_ctx())

    # The incomplete directory is cleaned up and the slot released.
    assert not step_dir.exists()
    assert ck._pending_async_save is None


def test_async_save_response_waits_for_upload_not_staging(tmp_path):
    upload = Future()
    ck._pending_async_save = _pending(
        tmp_path, _StubAsyncSaveResponse(_done_future(), upload)
    )

    # Staging is done but the upload is not -> still pending.
    assert ck.maybe_finalize_pending_save(_ctx()) is False
    assert not (tmp_path / "resume_meta.json").exists()

    upload.set_result(None)
    assert ck.maybe_finalize_pending_save(_ctx()) is True
    assert (tmp_path / "resume_meta.json").exists()
