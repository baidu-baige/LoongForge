# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Framework-agnostic WebSocket policy client."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import websockets.sync.client

from loongforge.embodied.eval.protocol import PROTOCOL_VERSION
from . import msgpack_numpy


class PolicyClient:
    """Provide PolicyClient behavior."""

    def __init__(self, host: str = "127.0.0.1", port: int = 10093, timeout: float = 300) -> None:
        """Run __init__."""
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server(timeout=timeout)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Run metadata."""
        return self._server_metadata

    def _wait_for_server(self, timeout: float) -> Tuple[websockets.sync.client.ClientConnection, Dict[str, Any]]:
        """Run _wait_for_server."""
        for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(key, None)

        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to server within {timeout} seconds")
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    open_timeout=30,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                if metadata.get("protocol_version") != PROTOCOL_VERSION:
                    raise RuntimeError(
                        f"protocol_version mismatch: {metadata.get('protocol_version')} != {PROTOCOL_VERSION}"
                    )
                return conn, metadata
            except ConnectionRefusedError:
                time.sleep(1)

    def close(self) -> None:
        """Run close."""
        self._ws.close()

    def request(
        self, message_type: str, payload: Optional[Dict[str, Any]] = None, request_id: str = "default"
    ) -> Dict[str, Any]:
        """Run request."""
        message = {
            "type": message_type,
            "request_id": request_id,
            "protocol_version": PROTOCOL_VERSION,
            "payload": payload or {},
        }
        self._ws.send(self._packer.pack(message))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        return msgpack_numpy.unpackb(response)

    def ping(self) -> Dict[str, Any]:
        """Run ping."""
        return self.request("ping")

    def reset(self, episode_id: str) -> Dict[str, Any]:
        """Run reset."""
        return self.request("reset", {"episode_id": episode_id})

    def predict_action(self, **payload: Any) -> Dict[str, Any]:
        """Run predict_action."""
        return self.request("infer", payload)
