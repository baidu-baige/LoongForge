# LoongForge-Embodied — Embodied Model Training Subsystem

`loongforge/embodied/` is a **torch-native training subsystem for embodied models** — Vision-Language-Action (VLA) policies and world-action models (WAM) — combining broad open-source model support with production-grade training performance.

- **Torch-native architecture** — built on vanilla PyTorch around unified `data`, `model`, and `trainer` abstractions that can be flexibly reused or extended.
- **Extensive open-source model support** — Pi0.5, GR00T-N1.6 / N1.7, xVLA, Wall-OSS-0.5, LingBot-VA, FastWAM, DreamZero, Cosmos3, and more — each supports fine-tuning, with accuracy aligned to the official baselines.
- **High training throughput** — up to 2x+ throughput on representative models through `torch.compile`, CUDA Graph, custom kernels, and I/O optimization, plus a full range of distributed strategies: DDP, ZeRO-1, FSDP, and HSDP.

---

## Why a Separate Subsystem?

Embodied models are far smaller than typical LLMs (generally under 10B) — a VLA is essentially a VLM plus an action head — so their bottleneck is not model parameter scale. Megatron's TP/PP/EP model parallelism brings little benefit at this scale.

The subsystem is therefore built on a **torch-native DDP/FSDP** engine with its own configuration, trainer, data, distributed, and evaluation layers. It shares LoongForge's repository, release, and tooling, but stays intentionally decoupled from the Megatron core (no shared args / parser / core) so each stack evolves on its own terms. The core abstractions below all follow from that choice.

---

## Quick Start

For the full framework user guide, see [User Manual](../../docs/source/embodied_tutorial/overview.md). Model-specific quick starts:

- [Pi0.5 (pi05)](../../docs/source/embodied_tutorial/quick_start_pi05.md)
- [GR00T-N1.6](../../docs/source/embodied_tutorial/quick_start_groot_n1_6.md)
- [GR00T-N1.7](../../docs/source/embodied_tutorial/quick_start_groot_n1_7.md)
- [FastWAM](../../docs/source/embodied_tutorial/quick_start_fastwam.md)
- [DreamZero](../../docs/source/embodied_tutorial/quick_start_dreamzero.md)
- [Cosmos3](../../docs/source/embodied_tutorial/quick_start_cosmos3.md)
- [xVLA](../../docs/source/embodied_tutorial/quick_start_xvla.md)
- [LingBot-VA](../../docs/source/embodied_tutorial/quick_start_lingbot_va.md)

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

Numbers reflect the baseline and LoongForge versions at measurement time and may evolve as implementations change. See the [root README](../../README.md#-performance) for the full benchmark chart across all model families.

---

## Directory Layout

```
loongforge/embodied/
├── train.py                                # Entry point: parse args → build trainer → train
├── train/                                  # Config system + trainers
│   ├── parser.py                           # 3-layer config resolution (CLI → YAML → frozen)
│   ├── training_args.py                    # generic training params (single source)
│   ├── config_map.py                       # model-name → (YAML, ModelConfig, DataConfig)
│   ├── global_vars.py                      # frozen global config singletons
│   └── trainers/                           # BaseTrainer (Template Method) + FinetuneTrainer / per-model trainers
├── model/                                  # Model architectures (single model dir expanded)
│   ├── registry.py                         # @register_model + auto module import
│   └── <model>/                            # one dir per model (e.g. pi05); each needs at least:
│       ├── modeling_<model>.py             # model definition (forward / loss)
│       └── model_configuration_<model>.py  # model config dataclass (arch hyperparams)
├── data/                                   # Data pipeline
│   ├── dataloader.py                       # top-level dataloader assembly
│   └── datasets/
│       ├── dataset_builder.py              # dataset construction + registry entry
│       ├── sampler_builder.py              # (stateful) distributed sampler
│       ├── lerobot_dataset.py              # dataset backend (also hdf5 / dummy + video_backends)
│       ├── transforms/                     # shared transform framework: base / pipeline / registry / collator
│       └── <model>/                        # per-model data config + custom data format + processing, e.g. pi05
├── distributed/                            # DDP/FSDP wrap, distributed context, checkpointing
│   ├── context.py                          # DistributedContext
│   ├── parallel.py                         # wrap_model() with DDP / FSDP
│   └── checkpoint.py                       # safetensors / pt / dcp save & load
├── optimizer/                              # AdamW, LR schedulers, grad clipping / NaN cleanup
├── eval/                                   # Offline benchmark eval (see eval/README.md)
└── tools/                                  # helper tools, e.g. dcp_to_safetensors.py
```

---

## Core Abstractions

The entry point `train.py` is self-explanatory (parse configs → build trainer → train), so we skip it. The real core is the four abstractions below.

### 1. Model definition (`model/`)

One directory per model, registered into a single entry via `@register_model`:

- `modeling_<name>.py` — architecture, forward, loss;
- `model_configuration_<name>.py` — model config dataclass (architecture hyperparams);
- exposes a uniform interface upward (trainer / eval), so adding a model requires no change to the training loop.

### 2. Dataset processing (`data/`)

Share the common parts, push per-model differences down:

- **Common parts** — dataset reading backends (`lerobot / hdf5 / dummy`), (stateful) distributed sampling, a composable transform framework;
- **Per-model differences** (`datasets/<name>/`) — how this model's data is read (e.g. `fastwam`'s multi-frame geometry), how actions/images are transformed, how a batch is assembled;
- **Data config** — each model defines a `DataConfig` (e.g. `data_configuration_pi05.py`) listing params like image size, action dim, normalization stats; tweak them via the YAML `data:` section or override with command-line dotlist.

### 3. Training configuration (`train/parser.py`)

Configuration is split into three parts, each producing one object:

- **YAML `model:` section** → `ModelConfig` (defined in `model_configuration_<name>.py`) → parsed into `model_cfg`, holding model architecture params (layers, dims, action head, etc.);
- **YAML `data:` section** → `DataConfig` (defined in `data_configuration_<name>.py`) → parsed into `data_cfg`, holding data params (image size, action dim, normalization stats, etc.);
- **CLI args** → `TrainingArgs` → parsed into `training_args`, holding generic training params (`--train-iters`, `--lr-base`, `--distributed-strategy`, ...).

Resolution flow: `--model-name` routes through `config_map.py` to the model's YAML and its `ModelConfig` / `DataConfig` types; the YAML `model:` / `data:` sections merge into those types, CLI args populate `TrainingArgs`; CLI dotlist can also override YAML fields (e.g. `model.action_horizon=64`). All three are frozen via `to_object()` into immutable objects stored as global singletons.

### 4. Distributed trainer (`train/trainers/`, `distributed/`)

`BaseTrainer` fixes the training lifecycle into a template (`setup → training loop → step → forward/backward → finalize`):

- **Common machinery** — optimizer / LR scheduling, gradient clipping & NaN cleanup, checkpoint save & resume, distributed logging, determinism control;
- **Trainer selection** — standard SFT uses `FinetuneTrainer`; special paradigms (multi-stream, CUDA Graph, ZeRO-1) subclass it and bind `trainer_cls` in `config_map.py`. `--trainer-type` remains the fallback for models that still use the generic trainer;
- **Distribution** — multiple strategies to choose from: `ddp` (data parallel), `ddp` + `--zero-optimizer` (ZeRO Stage-1, sharded optimizer states), `fsdp` (fully sharded), `hsdp` (hybrid sharded, set `--hsdp-shard-size`).

### Adding a new model

1. Add `model/<name>/modeling_<name>.py` + `model_configuration_<name>.py`, register with `@register_model`.
2. Add `data/datasets/<name>/`, including `data_configuration_<name>.py` (DataConfig), transform, and collator.
3. Add a YAML under `configs/models/embodied/` (with `model:` / `data:` sections) and wire it in `config_map.py` (binding YAML + ModelConfig + DataConfig).
4. If the training paradigm differs, subclass `BaseTrainer` and bind it as `trainer_cls` in `config_map.py`; otherwise reuse `FinetuneTrainer`.
5. Add a launch script under `examples/`.

---

## Evaluation

Offline benchmark evaluation (LIBERO / CALVIN / SimplerEnv / RoboTwin / ManiSkill) is a separate module. See [Eval User Guide](../../docs/source/embodied_tutorial/eval_user_guide.md).
