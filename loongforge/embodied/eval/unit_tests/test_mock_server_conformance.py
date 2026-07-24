# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the LoongForge VLA evaluation module."""

from __future__ import annotations

import multiprocessing as mp
import socket
import time

import numpy as np

from loongforge.embodied.eval.protocol import PROTOCOL_VERSION
from loongforge.embodied.eval.servers.mock_policy import MockPolicy
from loongforge.embodied.eval.transport import PolicyClient, PolicyServer


def _free_port() -> int:
    """Run _free_port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _serve(port: int) -> None:
    """Run _serve."""
    policy = MockPolicy(action_chunk_size=4)
    PolicyServer(policy=policy, host="127.0.0.1", port=port, metadata=policy.metadata).serve_forever()


def test_mock_server_conformance() -> None:
    """Run test_mock_server_conformance."""
    port = _free_port()
    proc = mp.Process(target=_serve, args=(port,), daemon=True)
    proc.start()
    try:
        client = PolicyClient(host="127.0.0.1", port=port, timeout=20)
        metadata = client.metadata
        assert metadata["protocol_version"] == PROTOCOL_VERSION
        assert metadata["action_chunk_size"] == 4
        assert metadata["supports_preempt"] is False
        assert "available_unnorm_keys" in metadata
        assert "expected_image_shape" not in metadata

        assert client.ping()["ok"] is True
        assert client.reset("episode-0")["ok"] is True

        image = np.zeros((256, 256, 3), dtype=np.uint8)
        first = client.predict_action(
            images={"primary": image}, instruction="test", episode_id="episode-0", episode_step=0
        )
        assert first["ok"] is True
        assert first["data"]["actions"].shape == (1, 7)
        assert first["data"]["inference_latency_ms"] is not None

        second = client.predict_action(
            images={"primary": image}, instruction="test", episode_id="episode-0", episode_step=1
        )
        assert second["ok"] is True
        assert second["data"]["actions"].shape == (1, 7)
        assert second["data"]["inference_latency_ms"] is None

        third = client.predict_action(
            images={"primary": image}, instruction="test", episode_id="episode-1", episode_step=0
        )
        assert third["ok"] is True
        assert third["data"]["actions"].shape == (1, 7)
        assert third["data"]["inference_latency_ms"] is not None
        client.close()
    finally:
        proc.terminate()
        proc.join(timeout=5)
