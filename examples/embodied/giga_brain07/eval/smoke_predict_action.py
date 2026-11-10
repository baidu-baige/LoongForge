#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain-0.7 eval-server smoke client.

Connects to a running LoongForge policy server (started from smoke.yaml),
assembles a dummy observation through the real ``GigaBrain07PayloadBuilder``,
sends one ``predict_action`` RPC, and asserts the returned action chunk has
shape ``[B=1, 50, 32]`` (full chunk via ``return_action_chunk=True``).

Usage (inside the pod, giga PYTHONPATH already exported by the launcher):
    python smoke_predict_action.py --host 127.0.0.1 --port 13391
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=13391)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--action-horizon", type=int, default=50)
    ap.add_argument("--action-dim", type=int, default=32)
    args = ap.parse_args()

    from loongforge.embodied.eval.transport.client import PolicyClient
    from loongforge.embodied.eval.payload_builders.registry import build_payload_builder

    client = PolicyClient(host=args.host, port=args.port, timeout=args.timeout)
    meta = client.metadata  # property, not a method
    print(f"[smoke] connected; server metadata: {meta}")

    # Build the payload through the real GigaBrain07 PayloadBuilder so the eval
    #接入件 (factory + payload builder + action-dim metadata) is exercised end to
    # end, not just the model's predict_action.
    pb = build_payload_builder(
        "giga_brain07",
        yaml_model={"state_encoding": "ee6d", "emb_id": 0},
    )
    canonical = {
        "instruction": "pick up the red block and place it on the plate",
        "images": {
            "primary": np.zeros((224, 224, 3), dtype=np.uint8),
            "left": np.zeros((224, 224, 3), dtype=np.uint8),
            "right": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "state_raw": {"eef_pos": [0.1, 0.2, 0.3], "eef_quat": [0.0, 0.0, 0.0, 1.0]},
        "meta": {"episode_id": "smoke", "episode_step": 0},
    }
    payload = pb.build(canonical, {"benchmark_name": "libero"})
    # Ask for the whole chunk (not just the first action) so we can verify the
    # full [H, D] horizon the model produced.
    payload["return_action_chunk"] = True
    payload["disable_action_cache"] = True
    payload["episode_id"] = "smoke"
    payload["episode_step"] = 0

    print(f"[smoke] payload keys: {sorted(payload)}; "
          f"n_views={len(payload['images'])} emb_id={payload.get('emb_id')}")

    resp = client.predict_action(**payload)
    client.close()

    data = resp.get("data", resp)
    actions = np.asarray(data["actions"], dtype=np.float32)
    latency = data.get("inference_latency_ms")
    print(f"[smoke] raw server action shape: {actions.shape}; latency_ms={latency}")

    # The server normalizes [B,H,D] -> [B*H, D]; with batch=1 that is [H, D].
    if actions.ndim == 2:
        actions = actions[None, ...]  # present as [B=1, H, D]

    B, H, D = actions.shape
    print(f"[smoke] interpreted as [B={B}, H={H}, D={D}]")

    ok = (B == 1 and H == args.action_horizon and D == args.action_dim)
    assert ok, (
        f"expected [B=1, {args.action_horizon}, {args.action_dim}], got {actions.shape}"
    )
    assert np.isfinite(actions).all(), "action chunk contains non-finite values"
    print(f"[smoke] OK — action chunk shape [B, 50, 32] = [{B}, {H}, {D}] verified; "
          f"finite=True, sample action[0,0,:5]={actions[0,0,:5].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
