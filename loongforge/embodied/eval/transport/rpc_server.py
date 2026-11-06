# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Framework-agnostic WebSocket RPC server."""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Any, Dict

import websockets.asyncio.server
import websockets.frames

from loongforge.embodied.eval.protocol import PROTOCOL_VERSION
from . import msgpack_numpy
from .batcher import AsyncBatcher
from .timing import LatencyRecorder

logger = logging.getLogger(__name__)


class PolicyServer:
    """Provide PolicyServer behavior."""

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        metadata: Dict[str, Any] | None = None,
        batching: Dict[str, Any] | None = None,
    ) -> None:
        """Run __init__."""
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or getattr(policy, "metadata", {})
        self._metadata.setdefault("protocol_version", PROTOCOL_VERSION)
        # Opt-in per-request service-time recording (see .timing). A no-op unless
        # LOONGFORGE_EVAL_TIMING is set, so request handling is unchanged by default.
        self._latency = LatencyRecorder.from_env()
        self._batching = dict(batching or {})
        self._batcher: AsyncBatcher | None = None

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
            # Sim clients block for long synchronous stretches (RoboTwin's
            # ~195s check_stable, long renders) and cannot answer the default
            # 20s keepalive ping, so the server would drop the idle connection
            # with 1011. Disable server-side keepalive; liveness is managed by
            # the eval flow + coordinator watchdog, not by websocket pings.
            ping_interval=None,
        ) as server:
            logger.info("PolicyServer listening on ws://%s:%s", self._host, self._port)
            # A non-empty ``batching`` block enables batching unless it says
            # ``enabled: false`` explicitly. Requiring the key silently disabled
            # batching for configs that only set max_batch_size/max_wait_ms.
            batching_on = bool(self._batching) and self._batching.get("enabled", True)
            if batching_on:
                self._batcher = AsyncBatcher(
                    self._policy,
                    max_batch_size=int(self._batching.get("max_batch_size", 8)),
                    max_wait_ms=float(self._batching.get("max_wait_ms", 5.0)),
                    latency_recorder=self._latency,
                    stats_interval_s=float(self._batching.get("stats_interval_s", 30.0)),
                )
                await self._batcher.start()
                logger.info("Server-side batching enabled: %s", self._batching)
            try:
                await server.serve_forever()
            finally:
                if self._batcher is not None:
                    logger.info("Batcher stats: %s", self._batcher.stats)
                    await self._batcher.stop()
                self._latency.flush()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        """Run _handler."""
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                is_infer = msg.get("type", "infer") in ("infer", "predict_action")
                if self._batcher is not None and is_infer:
                    response = await self._route_batched(msg)
                elif is_infer:
                    # Without batching, inference used to run inline on the event
                    # loop: a busy server then starved pings/handshakes and the
                    # watchdog killed it as "hung". The executor keeps the loop free.
                    response = await asyncio.get_running_loop().run_in_executor(
                        None, self._route_message, msg
                    )
                else:
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

    def _predict_action_timed(self, payload: Dict[str, Any]) -> Any:
        """Call ``policy.predict_action`` and record its service time (opt-in)."""
        start = time.perf_counter()
        try:
            return self._policy.predict_action(**payload)
        finally:
            self._latency.record((time.perf_counter() - start) * 1000.0)

    async def _route_batched(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Route an inference message through the batcher."""
        request_id = msg.get("request_id", "default")
        if msg.get("protocol_version") not in (None, PROTOCOL_VERSION):
            return {
                "status": "error",
                "ok": False,
                "type": "protocol_error",
                "request_id": request_id,
                "error": {"message": f"protocol_version mismatch: {msg.get('protocol_version')} != {PROTOCOL_VERSION}"},
            }
        payload = msg.get("payload", msg)
        try:
            data = await self._batcher.submit(payload)
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

        if message_type == "stats":
            return {
                "status": "ok",
                "ok": True,
                "type": "stats",
                "request_id": request_id,
                "data": {"batching": self._batcher.stats if self._batcher is not None else None},
            }

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
                data = self._predict_action_timed(payload)
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
