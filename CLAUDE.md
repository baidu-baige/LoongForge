# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LoongForge is large-scale transformer training framework built on top of Megatron-LM (as a patched fork: Loong-Megatron) and TransformerEngine. It supports LLMs, VLMs (Vision-Language Models), VLAs (Vision-Language-Action Models), and Diffusion Models across both NVIDIA GPUs and Kunlun XPUs. Training phases supported: pretrain and SFT (supervised fine-tuning).

## Build & Setup

### Quick Start (Docker — recommended)
```bash
git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git
# COMPILE_ENV: ampere | hopper | blackwell
docker build --build-arg COMPILE_ENV=hopper --build-arg ENABLE_LEROBOT=false \
  -t loongforge:latest -f ./LoongForge/docker/Dockerfile .
```

### Source Install
```bash
# 1. Clone with Megatron submodule
git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git
cd LoongForge

# 2. Install LoongForge + dependencies
uv pip install -e ".[gpu]"    # NVIDIA GPU
uv pip install -e ".[xpu]"    # Kunlun XPU

# 3. Setup TransformerEngine (clone, patch, build)
python setup_env.py --te-tag v2.9
```

Note: `setup_env.py` only handles TransformerEngine. Megatron-LM (Loong-Megatron) is a git submodule at `third_party/Loong-Megatron`, initialized via `--recurse-submodules`.

### Build Package
```bash
python -m build --sdist --wheel --outdir dist/
```

## Running Tests

E2E tests use a custom YAML-driven framework (`tests/mcore/main.py`), not pytest.

The entry script does NOT download artifacts. Provision the datasets, HuggingFace base
models, and pre-converted checkpoints referenced by the selected configs first, then run:

```bash
# Run the default CI suite (all models in tests/mcore/configs/)
bash tests/mcore/main_start.sh
```

### Running a Single Model Test

Edit variables in `tests/mcore/main_start.sh`:
```bash
# Run one model from tests/mcore/configs/
model_names="qwen3_14b"

# Run one model from tests/mcore/optional_configs/
model_names="deepseek_v2/deepseek_v2_lite"
include_optional=true

# Run an entire model series from optional_configs/
model_names="NONE"
optional_subdir="internvl2.5"
include_optional=true
```

Test configs: `tests/mcore/configs/` (CI suite) and `tests/mcore/optional_configs/` (regression, organized by model family). Each YAML defines model params and multi-step `scenarios` (checkpoint conversion + training).

### Embodied/VLA Regression Tests

`tests/torch/` is the end-to-end regression suite for training scripts under
`loongforge/engine/torch/` and `examples/{vla,world}/`. Its entry point is
`tests/torch/run.sh`; execution, metric parsing, and baseline comparison are owned by
`tests/torch/cli.py`. Regression targets are registered in
`tests/torch/config/scripts.yaml` and run serially in manifest order.

```bash
# List available torch regression targets
bash tests/torch/run.sh --list_models

# Run the full regression suite on a chip
bash tests/torch/run.sh --chip a

# Run selected targets
bash tests/torch/run.sh --chip a --models fastwam_ddp fastwam_ddp_zero1

# Collect baselines for the current chip
bash tests/torch/run.sh --chip a --auto_collect_baseline

# Artifacts are provisioned by the CI workflow/self-hosted runner before this step.

# Validate commands/configuration without training
bash tests/torch/run.sh --chip a --dry_run
```

Embodied test conventions:

- `tests/torch/config/env.sh` centralizes `TORCH_CI_ROOT`,
  `LOCAL_VLA_ARTIFACTS_ROOT`, log, and baseline paths. Prefer environment
  overrides or this file when moving the suite to another machine.
- Add every new training script to `tests/torch/config/scripts.yaml`; the manifest
  path is relative to `examples/{vla,world}/`. Add a baseline under
  `tests/torch/baseline/<chip>/<name>.json` for each supported chip.
- The executor injects `OUTPUT_DIR`, `TENSORBOARD_DIR`, and model-specific environment
  variables. Training scripts should expose environment overrides for data, checkpoints,
  caches, and output paths instead of relying on the executor to rewrite training args.
- Loss and `grad_norm` are hard-checked by default. Missing baselines, non-zero training
  exits, insufficient metrics, NaN/Inf, or skipped iterations fail the regression.
  Performance regressions warn by default; clear performance improvements may update the
  baseline.
- `--auto_collect_baseline` collects results without comparison. Baselines are chip-specific
  and must not be reused across different hardware without validation.
- For changes to video/action preprocessing, text-embedding caches, or distributed strategy,
  run the affected target's `--dry_run` first and then the real regression when artifacts
  and hardware are available.

## Training Launch Pattern

Training scripts use `torchrun` for distributed execution. The PYTHONPATH must include both Megatron-LM and LoongForge:

```bash
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun --nproc_per_node 8 --nnodes $NNODES ... \
    -m loongforge train --engine mcore \
    --model-name <model-name> \
    --training-phase pretrain|sft \
    ...
```

- **`engine/mcore/entrypoint.py`** / **`engine/torch/entrypoint.py`** — Per-engine `main()` entry points. `engine/dispatch.py` maps a `TrainSpec` to one of them after the CLI resolves the model, recipe, and engine defaults; `engine/mcore/__init__.py` only holds MCore model registration side effects.

Key arguments: `--model-name` (maps to config via `models/catalog.py`) or `--config-file` (direct YAML path), `--training-phase` (pretrain/sft).

## Architecture

### Core Package: `loongforge/`

- **`__main__.py`** — Unified entry point (the only entry file: `python -m loongforge` for torchrun, and the `LoongForge` console script both call its `main()`). Resolves engine defaults, recipe arguments, and CLI overrides through `models/catalog.py`; dispatches a `TrainSpec` to MCore or Torch.
- **`engine/mcore/global_vars.py`** — MCore global state: `get_args()` (Megatron args) plus model/hydra/data config, tokenizer, and chat template singletons. Torch keeps its own typed singletons in `engine/torch/global_vars.py`; `loongforge/utils/` exports no Megatron symbols.
- **`engine/mcore/parser.py`** — MCore argument parsing: merges Megatron CLI args with Hydra YAML configs (OmegaConf). Supports `--model-name` (looked up in `models/catalog.py`) or `--config-file`.
- **`engine/mcore/trainer_builder.py`** — Registry-based trainer dispatch. `register_model_trainer(model_family, training_phase)` decorator registers training functions per model family and phase.
- **`engine/mcore/megatron_trainer.py`** — `MegatronTrainer` wraps model_provider, dataset_provider, and forward_step into Megatron's `pretrain()` loop.
- **`engine/mcore/training_utils.py`** — Extended Megatron pretrain loop (heavily customized).
- **`engine/mcore/arguments.py`** — LoongForge-specific extra CLI arguments added on top of Megatron's.
- **`engine/mcore/validators.py`** — Validation logic for Megatron and LoongForge args.
- **`engine/mcore/pretrain/`** — Pretrain implementations for LLM and VLM.
- **`engine/mcore/sft/`** — SFT implementations for LLM, VLM, InternVL, ERNIE.
- **`engine/mcore/diffusion/`** — Diffusion model trainers (WAN and Qwen-Image).
- **`engine/torch/`** — Torch VLA/WAM training engine, including parser, trainers,
  distributed strategies and optimizers.
- **`datasets/{robotics,world,common}/`** — Torch dataset backends and model-specific
  transforms.
- **`checkpoint/`** — Torch/MCore save and resume, shared metadata, and online HF adapters.
- **`evaluation/`** — Torch model evaluation integrations.

### Model System: `loongforge/models/`

- **`mcore_registry.py`** — MCore model registry. `register_model_config(family, arch)` registers model configs; `register_model_provider(family)` registers model provider functions (accepts a single family string or list of families). Lookups: `get_model_config()`, `get_model_provider()`, `get_model_family()`.
- **`torch_registry.py`** — Torch model registry. `register_model(model_type)` (decorator) fills `MODEL_REGISTRY`; `build_model(model_cfg)` lazily imports only the selected model module and instantiates the registered class. Distinct from `mcore_registry.py`, which serves MCore.
- **`dtype.py`** — `resolve_dtype()`: config dtype string → `torch.dtype`, shared by models and training engines.
- **`dispatch.py`** — Hardware-abstraction layer (`MultiAccModules`). Provides unified access to TransformerEngine or local linear/attention/norm implementations.
- **`llm/`** — LLM backbone implementations: LLaMA, Qwen (all versions through Qwen3-Next), DeepSeek, InternLM, MiniMax, MIMO, GLM. One subdirectory per family, holding that family's config, model, and layer spec.
- **`vision/`** — Vision encoder implementations, one subdirectory per tower: base ViT, Qwen2-VL/3-VL, InternVL, LLaVA-OV, ERNIE-VL.
- **`vlm/`** — Multi-modal model composition: `OmniCombinationModel` assembles encoder + projector + decoder into a unified pipeline, with `model_chunk_schedule_plan.py` for pipeline parallelism scheduling.
- **`common/`** — Shared layers (norms, projectors, PEFT) and MCore model-config helpers (`utils.py`).
- **`diffusion/`** — WAN and Qwen-Image diffusion models.
- **`vla/`** — Torch Pi05, GR00T, X-VLA, and Wall-Oss.
- **`world/`** — Torch DreamZero, FastWAM, Cosmos3, and LingBot-VA.

Module naming: the family name lives in the directory, not the file — `llm/deepseek/config.py`,
`vision/qwen3_vl/vision_model.py`, `world/dreamzero/provider.py`. Files copied from HF upstream
keep their upstream names (`modeling_*.py`, `model_configuration_*.py`, `configuration_*.py`) so
they stay diffable against upstream; those live under `vla/` and `world/`.

### Configuration System: `configs/`

- **`configs/models/<family>/<model>.yaml`** — Hydra/OmegaConf YAML configs defining model architecture params. The `_target_` field maps to a Python config dataclass (e.g., `loongforge.models.llm.LLaMAConfig`).
- **`configs/data/`** — Data configuration templates.
- **`loongforge/models/catalog.py`** — `MCORE_CONFIGS` and `TORCH_CONFIGS` both map `--model-name` strings to `ModelSpec` entries built by `_mcore()` / `_torch()`; the Torch helper also names the model and data config classes.

### Data Pipeline: `loongforge/datasets/`

- SFT datasets with sharegpt/alpaca format support, multimodal data handling, data packing, DP load balancing.
- `multimodal/mm_plugin.py` — Multi-modal data plugin for processing images/video.
- `common/dp_balance/` — Data-parallel load balancing for packed sequences.

### Checkpoint Conversion: `tools/convert_checkpoint/`

Primary entry point: `tools/convert_checkpoint/module_convertor/model.py`.

For LLM models (single step):
```bash
python tools/convert_checkpoint/module_convertor/model.py \
    --load_platform=huggingface --save_platform=mcore \
    --config_file=<yaml> --convert_file=<json> \
    --tensor_model_parallel_size=N --pipeline_model_parallel_size=M \
    --load_ckpt_path=<hf_path> --save_ckpt_path=<mcore_path>
```

For VLM models (multi-step pipeline): convert language model, vision encoder, adapter/projector separately, then merge via `tools/convert_checkpoint/mcore/merge_megatron.py`.

Additional tools: `merge_megatron_expert.py` (MoE expert merging), FP8 conversion support (bf16↔fp8). Example scripts in `examples/<model>/checkpoint_convert/`.

### Custom Ops: `ops/`

Custom CUDA kernels: `sparse_mla_fwd/`, `sparse_mla_bwd/` (sparse MLA attention), `lightning_indexer_bwd/`.

### Examples: `examples/`

Shell scripts for each supported model family with pretrain/SFT/checkpoint-conversion configs. Pattern: `examples/<model>/{pretrain,sft,checkpoint_convert}/`.

### XPU Support: `examples_xpu/`

Kunlun XPU training scripts, mirroring `examples/` structure.

## Key Patterns

### Adding a New Model

1. Create a config dataclass in `loongforge/models/llm/<family>/config.py` (or `models/vision/<family>/` for vision), decorated with `@register_model_config(family, arch)`.
2. Create a model provider function decorated with `@register_model_provider(family)`.
3. Register a trainer function with `@register_model_trainer(family, training_phase)`.
4. Add YAML config under `configs/models/<family>/`.
5. Add entry in `loongforge/models/catalog.py` `MCORE_CONFIGS`.
6. Add example launch scripts under `examples/<model>/`.

### Configuration Flow

CLI args + Hydra YAML config -> `parse_train_args()` -> merged `args` namespace -> `build_model_trainer(args)` dispatches to registered trainer (looks up model_family from Hydra config's `model_type`) -> `MegatronTrainer.train()` runs the Megatron pretrain loop.

### Model Family Constants

Defined in `loongforge/utils/constants.py`. These classes (inheriting `_BaseFamilies`) drive dispatch logic throughout the codebase:
- **`LanguageModelFamilies`**: llama, llama2, llama3, llama3.1, qwen, qwen1.5, qwen2, qwen2.5, qwen3, qwen3_next, deepseek, internlm2.5, minimax, mimo, glm
- **`VisionLanguageModelFamilies`**: qwen2_vl, qwen2_5_vl, qwen3_vl, llava_ov_1_5, vlm, intern_vl, ernie4_5_vl, qwen3_5, kimi_k2_5
- **`CustomModelFamilies`**: wan2_2_i2v
- **`VisionLanguageActionModelFamilies`**: pi05, groot_n1_6

### Dependency Management

Megatron-LM is managed as a git submodule (`third_party/Loong-Megatron` → `baidu-baige/Loong-Megatron`). TransformerEngine is cloned and patched by `setup_env.py`. LoongForge itself is a Python package (`pyproject.toml`, hatchling build backend).

## Patches

`patches/TransformerEngine_v2.9/` contains patch files applied to upstream TransformerEngine during setup. These implement LoongForge-specific optimizations and fixes.

## Code Review

When asked to review a pull request or a diff (e.g. via `@claude review this PR`), follow `skills/code-review/SKILL.md` exactly: its checklist, severity tags (🔴 Critical / 🟠 Major / 🟡 Minor / 🟢 Nit), and output format are the authoritative contract for review output.

**Posting findings — prefer inline comments.** For every finding tied to specific code, call the `mcp__github_inline_comment__create_inline_comment` tool to post it on the exact `file:line` (or `startLine`–`line` range) instead of listing it in the summary. Use the top-level summary comment ONLY for: the overall `Verdict`, the 2–4 sentence `Summary`, the `Tests` line, and the `Checklist` table. Each inline comment body should follow the same severity-tag format as the SKILL spec (`🔴 Critical: ...`, `🟠 Major: ...`, etc.). Set `confirmed: true` only on real review findings — never on probes or self-tests.
