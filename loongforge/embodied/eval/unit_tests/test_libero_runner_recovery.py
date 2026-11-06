# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the runner's connection-error classification (server self-healing)."""

from __future__ import annotations

from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from loongforge.embodied.eval.orchestrator.runners.libero_runner import _classify_exception


def test_dropped_websocket_is_a_connection_failure() -> None:
    """A dropped websocket must rebuild the client, not be scored as a model failure."""
    assert _classify_exception(ConnectionClosedError(None, None)) == "server_unreachable"
    assert _classify_exception(ConnectionClosedOK(None, None)) == "server_unreachable"


def test_other_failures_keep_their_classification() -> None:
    """Run test_other_failures_keep_their_classification."""
    assert _classify_exception(ConnectionRefusedError()) == "server_unreachable"
    assert _classify_exception(TimeoutError()) == "policy_timeout"
    assert _classify_exception(ValueError("boom")) == "episode_error"
