# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""PolicyClient request identity and reply validation (issue #343)."""

from __future__ import annotations

import pytest

from loongforge.embodied.eval.transport import msgpack_numpy
from loongforge.embodied.eval.transport.client import PolicyClient


class _FakeWS:
    """Stub websocket: records sent frames and serves pre-packed reply frames."""

    def __init__(self, replies) -> None:
        self.sent = []
        self._replies = list(replies)
        self.closed = False

    def send(self, data) -> None:
        self.sent.append(msgpack_numpy.unpackb(data))

    def recv(self, timeout=None):
        if not self._replies:
            raise TimeoutError("no reply available")
        return self._replies.pop(0)

    def close(self) -> None:
        self.closed = True


def _pack(reply: dict) -> bytes:
    return msgpack_numpy.Packer().pack(reply)


def _make_client(replies):
    """Build a client without a live server, driving a stub connection."""
    client = PolicyClient.__new__(PolicyClient)
    client._packer = msgpack_numpy.Packer()
    client._request_seq = 0
    client._ws = _FakeWS(replies)
    return client, client._ws


def test_request_assigns_monotonic_request_ids() -> None:
    client, ws = _make_client(
        [_pack({"request_id": "1", "ok": True}), _pack({"request_id": "2", "ok": True})]
    )
    client.request("ping")
    client.request("ping")

    assert [frame["request_id"] for frame in ws.sent] == ["1", "2"]


def test_request_raises_on_mismatched_reply() -> None:
    client, _ = _make_client([_pack({"request_id": "not-mine", "ok": True})])

    with pytest.raises(RuntimeError, match="request_id mismatch"):
        client.request("ping")


def test_request_closes_connection_on_timeout() -> None:
    client, ws = _make_client([])  # recv raises TimeoutError

    with pytest.raises(TimeoutError):
        client.request("ping")

    assert ws.closed is True
