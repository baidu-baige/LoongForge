# LoongForge-Embodied — Embodied Model Training Subsystem

`loongforge/embodied/` is a **self-contained training subsystem for embodied foundation models**: Vision-Language-Action (VLA) policies (e.g. pi0.5) and world-action models (WAM, e.g. FastWAM). It lives inside the LoongForge monorepo alongside the Megatron-based LLM / VLM / Diffusion stacks, but is built on a torch-native DDP/FSDP engine instead.

---

## Why a Separate Subsystem?

Compared with typical large models, embodied models are far smaller (generally under 10B) — a typical VLA is a VLM plus an action head. Their bottleneck is data plumbing and iteration speed, not model parameter scale — so Megatron's TP/PP/EP model parallelism does little here and only adds complexity.

This subsystem is therefore built on **plain PyTorch DDP/FSDP**, with its own configuration, trainer, data, distributed, and evaluation layers. It shares LoongForge's repository, release, and tooling — but not the Megatron core engine. The two stacks are intentionally decoupled (no shared args/parser/core) so each evolves on its own terms:

| Axis | LoongForge core (LLM / VLM / Diffusion) | LoongForge-Embodied |
|------|------------------------------------------|---------------------|
| Compute / distributed | Megatron-LM — TP / PP / EP / CP / FSDP | torch-native DDP / FSDP |
| Workload | large-scale, model-parallel pretrain/SFT | small-to-mid, data-parallel SFT |

The core abstractions below all follow from that choice.

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

### 1. Model networking (`model/`)

One directory per model, registered into a single entry via `@register_model`:

- `modeling_<name>.py` — networking, forward, loss;
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
- **Trainer selection** — standard SFT uses `FinetuneTrainer`; special paradigms (multi-stream, CUDA Graph) subclass it (e.g. `custom/groot_n1_6/`), registered in `trainer_builder.py` and selected with `--trainer-type`;
- **Distribution** — multiple strategies to choose from: `ddp` (data parallel), `ddp` + `--zero-optimizer` (ZeRO Stage-1, sharded optimizer states), `fsdp` (fully sharded), `hsdp` (hybrid sharded, set `--hsdp-shard-size`).

### Adding a new model

1. Add `model/<name>/modeling_<name>.py` + `model_configuration_<name>.py`, register with `@register_model`.
2. Add `data/datasets/<name>/`, including `data_configuration_<name>.py` (DataConfig), transform, and collator.
3. Add a YAML under `configs/models/embodied/` (with `model:` / `data:` sections) and wire it in `config_map.py` (binding YAML + ModelConfig + DataConfig).
4. If the training paradigm differs, subclass `BaseTrainer` and register in `trainer_builder.py`; otherwise reuse `FinetuneTrainer`.
5. Add a launch script under `examples/`.

---

## Quick Start

Training is launched with `torchrun` against `loongforge/embodied/train.py`. The generic shape is:

```bash
export LOONGFORGE_PATH=/workspace/LoongForge

PYTHONPATH=$LOONGFORGE_PATH:$PYTHONPATH \
  torchrun --nproc_per_node 8 --nnodes 1 \
    $LOONGFORGE_PATH/loongforge/embodied/train.py \
    --model-name pi05 \
    --trainer-type FinetuneTrainer \
    --dataset-format lerobot_datasets \
    --distributed-strategy fsdp \
    --train-iters 30000 \
    ...
```

Different models have different data-format support, processing pipelines, and performance-optimization configs, so the exact runnable command differs per model. See the example scripts under `examples/embodied/` and the user guide for details:

```bash
bash examples/embodied/pi05/run_pi05_fsdp_finetune.sh       # also: ddp / ddp_zero1 variants
bash examples/embodied/groot_n1_6/run_groot_n1_6_ddp_finetune.sh
...
```

## Evaluation

Offline benchmark evaluation (LIBERO / CALVIN / SimplerEnv / RoboTwin / ManiSkill) is a separate module. See [`eval/README.md`](eval/README.md).
