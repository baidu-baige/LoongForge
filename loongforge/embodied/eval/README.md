# Evaluation Module

> **Note:** This module is still under active development and may see changes or adjustments in the future.

This directory contains the LoongForge-Embodied offline evaluation module. It runs benchmark clients and model policy servers as separate processes connected by a WebSocket/msgpack-numpy RPC protocol.

## Architecture

The benchmark client and the policy server run as separate processes:

```text
Benchmark env (client)                          Policy server
  ├─ Adapter                                      ├─ ModelFactory
  │    env obs → canonical observation            │    loads the model, wraps predict_action
  ├─ PayloadBuilder                               ├─ GenericPredictActionPolicy
  │    canonical → predict_action kwargs          │    RPC / chunk cache / shape check
  └─ ActionDecoder                                └─ model.predict_action
       raw chunk → env action                          model inference + normalization

                ◄──────── WebSocket + msgpack-numpy RPC ────────►
                           PolicyRequest / Response
```

Every benchmark drives the same four-stage chain: `Adapter.obs_to_canonical` → `PayloadBuilder.build` → `model.predict_action` (over RPC) → `ActionDecoder`. Details for code contributors: the [evaluation protocol](docs/eval_protocol.md) and the [model integration guide](docs/model_integration.md).

## Docs

| Doc | Audience | Content |
|---|---|---|
| [User guide](docs/user_guide_en.md) | eval users | Supported models/benchmarks, quick start, config reference, outputs, troubleshooting |
| [Benchmark pages](docs/benchmarks/libero.md) | eval users | Per-benchmark reproduction guides (env setup, run, verification) |
| [Model integration](docs/model_integration.md) | code contributors | New-model integration checklist + `predict_action` contract + development notes |
| [Evaluation protocol](docs/eval_protocol.md) | code contributors | Architecture, components, data protocol, PolicyClient interface |
| [Benchmark environments](docs/benchmark_envs.md) | eval users | Verified per-benchmark env version records (install per the official benchmark homepages) |

## Quick Start

The example below runs LIBERO with pi05:

1. **Set up the LIBERO environment** — please refer to the official [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) for installation; for the eval client deps and common issues, check the [LIBERO guide](docs/benchmarks/libero.md#step-1-environment-setup); the verified environment version lists are in [benchmark_envs.md](docs/benchmark_envs.md).
2. **Get the weights and edit the config** — download [lerobot/pi05_libero_finetuned_v044](https://huggingface.co/lerobot/pi05_libero_finetuned_v044), then fill the `/path/to/...` placeholders in `examples/embodied/pi05/eval/configs/libero/object_smoke.yaml`. A field-by-field example: the [user guide](docs/user_guide_en.md#2-quick-start); the config layout: [user guide §3](docs/user_guide_en.md#3-configuration-reference).

3. **Run** — execute the script inside the **benchmark** environment:

    ```bash
    cd /path/to/LoongForge
    examples/embodied/pi05/eval/run_libero_eval.sh
    ```

The LIBERO simulator runs in the benchmark environment; the policy server is launched from the YAML `server.python` field, pointing at the LoongForge environment. Other models/benchmarks: the [user guide](docs/user_guide_en.md) and the [benchmark pages](docs/benchmarks/libero.md).

## Supported Matrix

| | LIBERO | CALVIN | SimplerEnv (WidowX) | RoboTwin | ManiSkill |
|---|---|---|---|---|---|
| **pi05** | ✅ task success ([lerobot/pi05_libero_finetuned_v044](https://huggingface.co/lerobot/pi05_libero_finetuned_v044)) | connectivity only | connectivity only | ✅ task success ([motus-robotics/pi0.5_robotwin2](https://huggingface.co/motus-robotics/pi0.5_robotwin2)) | connectivity only |
| **xvla** | ✅ task success ([2toINF/X-VLA-LIBERO](https://huggingface.co/2toINF/X-VLA-LIBERO)) | connectivity only | ✅ task success ([2toINF/X-VLA-WidowX](https://huggingface.co/2toINF/X-VLA-WidowX)) | ✅ task success ([2toINF/X-VLA-RoboTwin2](https://huggingface.co/2toINF/X-VLA-RoboTwin2)) | connectivity only |
| **GR00T-N1.6** | — | — | ✅ task success ([nvidia/GR00T-N1.6-bridge](https://huggingface.co/nvidia/GR00T-N1.6-bridge)) | — | — |

- **Task success**: at least one episode passed the benchmark's official success criterion.
- **Weights**: parentheses show the Hugging Face weight (`org/name`) that achieved the run.
- **Connectivity only**: the pipeline runs with `random_init: true` and no score — either no domain weights are released, or the benchmark assets block a full run (e.g. xvla CALVIN: the weights are public, but the official online rollout needs the original-format validation dataset).
- **—**: not supported yet — coming soon.
- **Detailed results**: see the per-benchmark [benchmark pages](docs/benchmarks/libero.md).

## Model Interface

LoongForge model servers share a `predict_action(images, instructions, state=None, dataset_stats=None)` interface instead of a bespoke policy adapter per model. Client-side payload assembly lives in the per-model `PayloadBuilder`; env-side action decoding lives in the `ActionDecoder`; normalization stays inside the model's `predict_action()`. Architecture and protocol: the [evaluation protocol](docs/eval_protocol.md). Step-by-step integration checklist: the [model integration guide](docs/model_integration.md).
