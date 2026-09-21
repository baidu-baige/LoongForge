# LoongForge — Embodied Model Training

This directory contains offline evaluation for embodied models. Training is launched through `loongforge/train.py`, with the Torch runtime in `loongforge/engines/torch/` and model/data implementations in `loongforge/models/embodied/` and `loongforge/data/embodied/`.

- **Benchmark adapters** — LIBERO, CALVIN, SimplerEnv, RoboTwin, and ManiSkill.
- **Model policy servers** — shared `predict_action` RPC protocol with model-specific factories.
- **Action semantics** — payload builders, action decoders, normalization, and metrics stay in the evaluation domain.

---

## Evaluation Scope

The evaluation stack is domain-specific; there is no framework-wide `eval.py` dispatcher.

---

## Quick Start

For the full framework user guide, see [User Manual](../../../docs/source/embodied_tutorial/overview.md). Model-specific quick starts:

- [Pi0.5 (pi05)](../../../docs/source/embodied_tutorial/quick_start_pi05.md)
- [GR00T-N1.6](../../../docs/source/embodied_tutorial/quick_start_groot_n1_6.md)
- [GR00T-N1.7](../../../docs/source/embodied_tutorial/quick_start_groot_n1_7.md)
- [FastWAM](../../../docs/source/embodied_tutorial/quick_start_fastwam.md)
- [DreamZero](../../../docs/source/embodied_tutorial/quick_start_dreamzero.md)
- [Cosmos3](../../../docs/source/embodied_tutorial/quick_start_cosmos3.md)
- [xVLA](../../../docs/source/embodied_tutorial/quick_start_xvla.md)
- [LingBot-VA](../../../docs/source/embodied_tutorial/quick_start_lingbot_va.md)

---

## Performance

Training speedups over mainstream open-source baselines. Performance is still under active optimization, and these numbers will keep improving over time:

| Model | Type | Speedup |
|---|---|---|
| DreamZero (DROID Wan2.2-5B Full) | WAM | **4.38×** |
| Pi0.5 | VLA | **2.80×** |
| GR00T-N1.6 | VLA | **2.31×** |
| FastWAM | WAM | **2.25×** |
| LingBot-VA | WAM | **2.20×** |
| GR00T-N1.7 | VLA | **1.79×** |
| xVLA | VLA | **1.79×** |

Numbers reflect the baseline and LoongForge versions at measurement time and may evolve as implementations change. See the [root README](../../../README.md#-performance) for the full benchmark chart across all model families.

---

## Directory Layout

```
loongforge/
├── train.py
├── models/
│   ├── catalog.py
│   └── embodied/
│       ├── registry.py
│       └── <model>/
├── data/embodied/
│   ├── dataloader.py
│   └── datasets/
├── engines/
│   ├── mcore/
│   └── torch/
│       ├── entrypoint.py
│       ├── parser.py
│       ├── training_args.py
│       ├── trainers/
│       ├── distributed/
│       └── optimizer/
└── evaluation/embodied/
```

---

## Evaluation Architecture

- `orchestrator/` runs a benchmark from YAML and manages the policy server.
- `adapters/` converts benchmark observations and results to the common protocol.
- `payload_builders/` and `action_decoders/` map model actions to benchmark actions.
- `servers/` loads a model factory and exposes `predict_action` over RPC.
- `transport/` owns the WebSocket/msgpack-numpy client and server.
- `metrics/` writes per-run and suite reports.

To add a supported model, implement a factory under `factories/`, add a payload builder or decoder only when its action semantics differ, and add an example YAML under `examples/embodied/*/eval/`.

---

See [Eval User Guide](../../../docs/source/embodied_tutorial/eval_docs/user_guide.md) for benchmark setup and commands.
