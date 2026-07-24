# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Run a LoongForge-backed policy server for standalone eval."""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from loongforge.embodied.eval.factories.registry import build_model_spec
from loongforge.embodied.eval.servers.eval_server_config import parse_eval_server_config
from loongforge.embodied.eval.servers.loongforge_policy import GenericPredictActionPolicy
from loongforge.embodied.eval.transport.rpc_server import PolicyServer


class ReusableHTTPServer(HTTPServer):
    """HTTPServer variant that can rebind immediately after short smoke runs."""

    allow_reuse_address = True


class HealthHandler(BaseHTTPRequestHandler):
    """Provide HealthHandler behavior."""

    ckpt_path = ""

    def do_GET(self) -> None:
        """Run do_GET."""
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ready": True, "ckpt_path": self.ckpt_path}).encode())
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Run log_message."""
        return


def _warmup_model(model_spec: Any) -> None:
    """Run a single dummy predict_action call to trigger all lazy imports before serving."""
    import numpy as np

    logging.info("Warming up model to resolve lazy imports...")
    try:
        dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
        model_spec.model.predict_action(
            images=[[dummy_image]],
            instructions=["warmup"],
            state=None,
            dataset_stats=None,
        )
        logging.info("Model warmup complete.")
    except Exception as exc:
        logging.warning("Model warmup call raised an exception (ignored): %s", exc)


def start_health_server(port: int, ckpt_path: str) -> None:
    """Run start_health_server."""
    HealthHandler.ckpt_path = ckpt_path
    server = ReusableHTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


def main() -> None:
    """Run main."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to eval YAML config file")
    cli = parser.parse_args()

    server_args, raw_model_dict = parse_eval_server_config(cli.config)

    if server_args.loongforge_root:
        os.chdir(server_args.loongforge_root)
    if not server_args.ckpt_path and not server_args.random_init:
        raise SystemExit("checkpoint must be set by server.ckpt_path in YAML unless server.random_init is true")

    logging.basicConfig(level=logging.INFO, force=True)
    model_spec = build_model_spec(server_args, raw_model_dict)
    _warmup_model(model_spec)
    policy = GenericPredictActionPolicy(
        model=model_spec.model,
        metadata=model_spec.metadata,
        dataset_statistics_path=server_args.dataset_statistics_path,
        action_dim=model_spec.metadata.get("action_dim", 7),
        request_id_prefix=f"loongforge-{server_args.model_type}",
    )
    start_health_server(server_args.health_port, server_args.ckpt_path)
    PolicyServer(policy=policy, port=server_args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
