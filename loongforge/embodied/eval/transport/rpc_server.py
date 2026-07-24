# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Framework-agnostic WebSocket RPC server."""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any, Dict

import websockets.asyncio.server
import websockets.frames

from loongforge.embodied.eval.protocol import PROTOCOL_VERSION
from . import msgpack_numpy

logger = logging.getLogger(__name__)


class PolicyServer:
    """Provide PolicyServer behavior."""

    def __init__(
        self, policy, host: str = "0.0.0.0", port: int = 10093, metadata: Dict[str, Any] | None = None
    ) -> None:
        """Run __init__."""
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or getattr(policy, "metadata", {})
        self._metadata.setdefault("protocol_version", PROTOCOL_VERSION)

    def serve_forever(self) -> None:
        """Run serve_forever."""
        asyncio.run(self.run())

    async def run(self) -> None:
        """Run run."""
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            logger.info("PolicyServer listening on ws://%s:%s", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        """Run _handler."""
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                response = self._route_message(msg)
                await websocket.send(packer.pack(response))
            except websockets.ConnectionClosed:
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _route_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Run _route_message."""
        request_id = msg.get("request_id", "default")
        message_type = msg.get("type", "infer")
        payload = msg.get("payload", msg)

        if msg.get("protocol_version") not in (None, PROTOCOL_VERSION):
            return {
                "status": "error",
                "ok": False,
                "type": "protocol_error",
                "request_id": request_id,
                "error": {"message": f"protocol_version mismatch: {msg.get('protocol_version')} != {PROTOCOL_VERSION}"},
            }

        if message_type == "ping":
            return {"status": "ok", "ok": True, "type": "pong", "request_id": request_id}

        if message_type == "reset":
            episode_id = payload.get("episode_id")
            if not episode_id:
                return {
                    "status": "error",
                    "ok": False,
                    "type": "reset",
                    "request_id": request_id,
                    "error": {"message": "episode_id required"},
                }
            result = self._policy.reset(episode_id)
            return {"status": "ok", "ok": True, "type": "reset", "request_id": request_id, "data": result}

        if message_type in ("infer", "predict_action"):
            try:
                data = self._policy.predict_action(**payload)
                return {"status": "ok", "ok": True, "type": "inference_result", "request_id": request_id, "data": data}
            except Exception as exc:
                logger.exception("Policy inference error")
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": request_id,
                    "error": {"message": str(exc)},
                }

        return {
            "status": "error",
            "ok": False,
            "type": "unknown",
            "request_id": request_id,
            "error": {"message": f"Unsupported message type {message_type!r}"},
        }
