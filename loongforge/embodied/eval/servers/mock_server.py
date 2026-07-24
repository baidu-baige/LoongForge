# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Run a standalone mock policy server."""

from __future__ import annotations

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from loongforge.embodied.eval.orchestrator.config import load_config
from loongforge.embodied.eval.servers.mock_policy import MockPolicy
from loongforge.embodied.eval.transport.rpc_server import PolicyServer


class ReusableHTTPServer(HTTPServer):
    """HTTPServer variant that can rebind immediately after short smoke runs."""

    allow_reuse_address = True


class HealthHandler(BaseHTTPRequestHandler):
    """Provide HealthHandler behavior."""

    def do_GET(self) -> None:
        """Run do_GET."""
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ready": True, "ckpt_path": "mock://policy"}).encode())
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Run log_message."""
        return


def start_health_server(port: int) -> None:
    """Run start_health_server."""
    server = ReusableHTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


def _apply_config(args: argparse.Namespace) -> argparse.Namespace:
    """Run _apply_config."""
    if not args.config:
        return args
    config = load_config(args.config)
    server = config.get("server") or {}
    model = config.get("model") or {}
    if not isinstance(server, dict) or not isinstance(model, dict):
        raise ValueError("server and model sections must be mappings")
    if server.get("port") is not None:
        args.port = int(server["port"])
    if server.get("health_port") is not None:
        args.health_port = int(server["health_port"])
    if model.get("action_dim") is not None:
        args.action_dim = int(model["action_dim"])
    if model.get("action_chunk_size") is not None:
        args.action_chunk_size = int(model["action_chunk_size"])
    return args


def main() -> None:
    """Run main."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--health-port", type=int, default=10094)
    parser.add_argument("--action-chunk-size", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=7)
    args = _apply_config(parser.parse_args())

    logging.basicConfig(level=logging.INFO, force=True)
    start_health_server(args.health_port)
    policy = MockPolicy(action_chunk_size=args.action_chunk_size, action_dim=args.action_dim)
    PolicyServer(policy=policy, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
