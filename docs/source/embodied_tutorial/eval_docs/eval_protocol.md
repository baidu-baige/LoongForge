# Evaluation Framework & Protocol

This page describes the LoongForge offline evaluation framework: its architecture, the runtime components, and the wire protocol between the benchmark client and the policy server. It is written for **code contributors** — model integrators and maintainers who touch eval code. End users only need the [user guide](user_guide.md) and the [benchmark pages](benchmarks/index.md).

## Overview

The eval module runs a policy against a simulation benchmark from a single YAML config. It launches the benchmark simulator and an independent model server as separate processes connected by a WebSocket + msgpack-numpy RPC protocol. The model and the benchmark are decoupled: benchmark code never imports model code, and vice versa. (The architecture diagram lives in the [module overview](../eval_overview.md) Architecture section.)

## Components

| Component | Responsibility |
|---|---|
| `adapters/<benchmark>.py` | env observation → canonical dict (`state_raw`); declares matching capabilities (`action_space`, `default_fps`, `cameras`) |
| `payload_builders/<model>.py` | canonical dict → `predict_action(**kwargs)`; declares model capabilities (`state_encoding` / `action_encoding` / `action_dim` / ...) |
| `model.predict_action` (server) | model inference; owns all model-specific normalization/unnormalization |
| `action_decoders/<encoding>.py` | raw model action chunk → benchmark env action space; identity match yields a no-op `IdentityDecoder` |
| `factories/<model>_factory.py` | thin model load/build, registered via `@register_factory("<model_type>")` |
| `transport/` | `PolicyClient` (WebSocket client), msgpack-numpy serialization, RPC server |
| `orchestrator/` | unified YAML entry (`run.py`), server manager, runners, config assembly |

At runtime every benchmark drives the same four-stage chain:

```text
Adapter.obs_to_canonical(env_obs)      # env obs -> canonical dict
  -> PayloadBuilder.build(canonical)   # canonical -> predict_action kwargs
  -> model.predict_action(**kwargs)    # RPC to the policy server
  -> ActionDecoder(action_chunk, ctx)  # raw model chunk -> env action space
```

The ActionDecoder key is auto-composed at startup as `"{model.action_encoding}_to_{adapter.action_space}"`; an identity match yields an empty key and an `IdentityDecoder` no-op.

## Message flow

The client and the server exchange msgpack-numpy-encoded `PolicyRequest` / `PolicyResponse` envelopes (types in `protocol/schema.py`, `PROTOCOL_VERSION = "1.0"`):

```python
# request
{"type": "ping" | "reset" | "infer" | "predict_action",
 "request_id": str, "protocol_version": "1.0", "payload": {...}}
# response
{"status": "ok" | "error", "ok": bool, "type": str,
 "request_id": str, "data": {...}, "error": {...}}
```

1. **Connect** — the client opens a WebSocket to the server; the server first sends its `ServerMetadata` (`protocol_version`, `action_chunk_size`, `available_unnorm_keys`, ...). A `protocol_version` mismatch aborts the connection.
2. **Per step** — the client sends `predict_action` with the PayloadBuilder-produced kwargs as `payload`; the server replies with the raw action chunk (plus model metadata) in `data`.
3. **`ping`** — liveness check; **`reset`** — clears the server's per-episode action chunk cache.
4. **Close** — the client closes the WebSocket; the orchestrator terminates the server process.

The protocol is framework-agnostic: any policy only needs a compatible policy server.

## PolicyClient interface

`transport/client.py` provides `PolicyClient`:

- `ping()` — liveness check
- `request(type, payload, request_id)` — generic request; message types `ping` / `reset` / `infer` / `predict_action`
- `metadata` — server metadata (protocol version, action dim, unnormalization mode, plus factory-declared fields such as action horizon / tokenizer path)

Serialization notes:

- WebSocket payloads do not serialize PIL objects directly; convert images to `np.ndarray` on the client side and restore to PIL inside the framework if the model requires it.
- The `PolicyClient` strips HTTP(S)/ALL proxy environment variables before connecting.

## Running the policy server standalone

The server entry is `loongforge_server.py`, launched by the orchestrator from the YAML config (`--config <yaml>`; the `server:` section holds host/port, `python`, `ckpt_path`, `use_bf16`, ...). For debugging, copy an existing YAML and set `model.backend: mock` to run a pipeline-only check without model weights.
