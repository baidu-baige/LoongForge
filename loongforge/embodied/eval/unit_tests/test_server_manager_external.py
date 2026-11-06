# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for external-server mode (server.external_url)."""

from __future__ import annotations

import argparse

import pytest

from loongforge.embodied.eval.orchestrator.config import apply_config, parse_external_url
from loongforge.embodied.eval.orchestrator.server_manager import ManagedServer


def _args(external_url: str = "") -> argparse.Namespace:
    return argparse.Namespace(
        host="127.0.0.1", port=1, health_port=2, server_start_timeout_sec=1, external_url=external_url
    )


def test_parse_external_url() -> None:
    """Run test_parse_external_url."""
    assert parse_external_url("ws://127.0.0.1:14560") == ("127.0.0.1", 14560)
    assert parse_external_url("example.com:12345") == ("example.com", 12345)


def test_parse_external_url_rejects_missing_port() -> None:
    """Run test_parse_external_url_rejects_missing_port."""
    with pytest.raises(ValueError):
        parse_external_url("ws://host")


def test_apply_config_derives_host_port() -> None:
    """Run test_apply_config_derives_host_port."""
    out = apply_config(_args(), {"server": {"external_url": "ws://example.com:12345"}})
    assert out.external_url == "ws://example.com:12345"
    assert (out.host, out.port) == ("example.com", 12345)


def test_managed_server_external_is_noop() -> None:
    """Run test_managed_server_external_is_noop."""
    server = ManagedServer(_args("ws://h:1"))
    assert server.external is True
    server.start()  # must not spawn
    assert server.process is None
    server.ensure_running()  # must not restart a shared server
    assert server.process is None
    server.stop()  # must not kill anything
    assert server.process is None


def test_managed_server_internal_does_start(monkeypatch) -> None:
    """Run test_managed_server_internal_does_start."""
    called = {}

    def fake_start_server(args):  # noqa: ANN001
        called["started"] = True
        return object()

    def fake_wait_healthz(url, timeout, process):  # noqa: ANN001
        return None

    monkeypatch.setattr(
        "loongforge.embodied.eval.orchestrator.server_manager.start_server", fake_start_server
    )
    monkeypatch.setattr(
        "loongforge.embodied.eval.orchestrator.server_manager.wait_healthz", fake_wait_healthz
    )
    server = ManagedServer(_args(""))
    assert server.external is False
    server.start()
    assert called.get("started") is True and server.process is not None
