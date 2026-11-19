# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the opt-in server-side latency recorder."""

from __future__ import annotations

import json
import time

from loongforge.embodied.eval.servers.mock_policy import MockPolicy
from loongforge.embodied.eval.transport.rpc_server import PolicyServer
from loongforge.embodied.eval.transport.timing import LatencyRecorder, percentile, summarize


def test_percentile_interpolates() -> None:
    """Run test_percentile_interpolates."""
    values = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0.0) == 0.0
    assert percentile(values, 1.0) == 40.0
    assert percentile(values, 0.5) == 20.0
    assert percentile(values, 0.25) == 10.0


def test_summarize_known_values() -> None:
    """Run test_summarize_known_values."""
    summary = summarize([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["mean"] == 2.5
    assert summary["p50"] == 2.5
    assert summary["max"] == 4.0


def test_summarize_empty() -> None:
    """Run test_summarize_empty."""
    assert summarize([]) == {"count": 0}


def test_recorder_disabled_is_noop(tmp_path) -> None:
    """Run test_recorder_disabled_is_noop."""
    out = tmp_path / "timing.json"
    recorder = LatencyRecorder(enabled=False, path=str(out))
    recorder.record(12.0)
    recorder.flush()
    assert recorder.summary() == {"count": 0}
    assert not out.exists()


def test_recorder_flush_writes_summary(tmp_path) -> None:
    """Run test_recorder_flush_writes_summary."""
    out = tmp_path / "timing.json"
    recorder = LatencyRecorder(enabled=True, path=str(out))
    for value in (10.0, 20.0, 30.0):
        recorder.record(value)
    recorder.flush()
    payload = json.loads(out.read_text())
    assert payload["count"] == 3
    assert payload["mean"] == 20.0


class _SleepyPolicy(MockPolicy):
    """Mock policy that sleeps a fixed time so service time is measurable."""

    def __init__(self, sleep_sec: float) -> None:
        """Run __init__."""
        super().__init__(action_chunk_size=1)
        self._sleep_sec = sleep_sec

    def predict_action(self, **kwargs):
        """Run predict_action."""
        time.sleep(self._sleep_sec)
        return super().predict_action(**kwargs)


def _infer_message() -> dict:
    return {"type": "infer", "payload": {"episode_id": "e0", "episode_step": 0}}


def test_service_time_is_measured(tmp_path) -> None:
    """Run test_service_time_is_measured."""
    server = PolicyServer(policy=_SleepyPolicy(0.05), host="127.0.0.1", port=0)
    server._latency = LatencyRecorder(enabled=True, path=str(tmp_path / "t.json"))
    response = server._route_message(_infer_message())
    assert response["ok"] is True
    summary = server._latency.summary()
    assert summary["count"] == 1
    assert summary["mean"] >= 45.0  # slept 50 ms
    assert summary["mean"] < 500.0


def test_response_unchanged_when_enabled(tmp_path) -> None:
    """Run test_response_unchanged_when_enabled."""
    off = PolicyServer(policy=MockPolicy(action_chunk_size=1), host="127.0.0.1", port=0)
    off._latency = LatencyRecorder(enabled=False, path=str(tmp_path / "off.json"))
    on = PolicyServer(policy=MockPolicy(action_chunk_size=1), host="127.0.0.1", port=0)
    on._latency = LatencyRecorder(enabled=True, path=str(tmp_path / "on.json"))
    response_off = off._route_message(_infer_message())
    response_on = on._route_message(_infer_message())
    assert response_off["ok"] is True and response_on["ok"] is True
    assert set(response_off["data"]) == set(response_on["data"])
    # Timing must stay out of the wire payload.
    assert "service_time_ms" not in response_on["data"]


def test_error_path_records_and_returns_error(tmp_path) -> None:
    """Run test_error_path_records_and_returns_error."""

    class _Boom(MockPolicy):
        def predict_action(self, **kwargs):
            """Run predict_action."""
            raise RuntimeError("boom")

    server = PolicyServer(policy=_Boom(), host="127.0.0.1", port=0)
    server._latency = LatencyRecorder(enabled=True, path=str(tmp_path / "e.json"))
    response = server._route_message(_infer_message())
    assert response["ok"] is False
    assert server._latency.summary()["count"] == 1


def test_concurrent_flush_is_safe(tmp_path) -> None:
    """Many sims on one server flush concurrently; the temp file must not race."""
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "timing.json"
    recorder = LatencyRecorder(enabled=True, path=str(path), flush_every=1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(recorder.record, range(200)))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["count"] == 200
