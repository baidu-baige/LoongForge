# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BaigeOmni is Baidu's large-scale transformer training framework built on top of NVIDIA Megatron-LM and TransformerEngine. It supports LLMs, VLMs (Vision-Language Models), VLAs (Vision-Language-Action Models), and Diffusion Models across both NVIDIA GPUs and Baidu Kunlun XPUs. Training phases supported: pretrain and SFT (supervised fine-tuning).

## Build & Setup

### Environment Setup
```bash
# Clone and patch Megatron-LM + TransformerEngine dependencies
python setup_env.py --megatron-tag core_v0.15.0 --te-tag v2.9
```
This clones Megatron-LM and TransformerEngine, checks out specific tags, applies patches from `patches/`, builds TransformerEngine, and installs BaigeOmni dependencies.

### Install Dependencies
```bash
pip install -r requirements.txt       # GPU (NVIDIA)
pip install -r requirements_xpu.txt   # XPU (Kunlun)
```

### Build Package
```bash
sh build.sh    # Creates output tarball
```

## Running Tests

Tests are under `tests/` and use a YAML-driven framework for E2E correctness validation.

```bash
# Download test datasets first
bash tests/download_datasets.sh

# Run default CI test suite (all models in tests/configs/)
bash tests/main_start.sh

# Run optional regression tests
bash tests/main_start.sh --optional
```

Test configuration is in `tests/main_start.sh` via variables: `model_names`, `optional_subdir`, `include_optional`, `tasks`, `training_type`, `chip`.

Unit tests use pytest:
```bash
pytest tests/
```

## Training Launch Pattern

Training scripts use `torchrun` for distributed execution. The PYTHONPATH must include both Megatron-LM and BaigeOmni:

```bash
PYTHONPATH=$MEGATRON_PATH:$BAIGE_OMNI_PATH:$PYTHONPATH \
    torchrun --nproc_per_node 8 --nnodes $NNODES ... \
    $BAIGE_OMNI_PATH/baige_omni/train.py \
    --model-name <model-name> \
    --training-phase pretrain|sft \
    ...
```

Key arguments: `--model-name` (maps to config via `config_map.py`) or `--config-file` (direct YAML path), `--training-phase` (pretrain/sft).

## Architecture

### Core Package: `baige_omni/`

- **`train.py`** — Entry point. Calls `parse_train_args()` then `build_model_trainer(args).train()`.
- **`train/parser.py`** — Argument parsing: merges Megatron CLI args with Hydra YAML configs (OmegaConf). Supports `--model-name` (looked up in `config_map.py`) or `--config-file`.
- **`train/trainer_builder.py`** — Registry-based trainer dispatch. `register_model_trainer(model_family, training_phase)` decorator registers training functions per model family and phase.
- **`train/megatron_trainer.py`** — `MegatronTrainer` wraps model_provider, dataset_provider, and forward_step into Megatron's `pretrain()` loop.
- **`train/training_utils.py`** — Extended Megatron pretrain loop (~87K lines, heavily customized).
- **`train/arguments.py`** — Baige-specific extra CLI arguments added on top of Megatron's.
- **`train/validators.py`** — Validation logic for Megatron and Baige args.
- **`train/pretrain/`** — Pretrain implementations for LLM and VLM.
- **`train/sft/`** — SFT implementations for LLM, VLM, InternVL, ERNIE.
- **`train/custom/`** — Custom model trainers (e.g., WAN diffusion, Pi0.5 VLA).

### Model System: `baige_omni/models/`

- **`factory.py`** — Model registry. `register_model_config(family, arch)` registers model configs; `register_model_provider(family)` registers model provider functions. Lookups: `get_model_config()`, `get_model_provider()`, `get_model_family()`.
- **`dispatch.py`** — Hardware-abstraction layer (`MultiAccModules`). Provides unified access to TransformerEngine or local linear/attention/norm implementations.
- **`foundation/`** — LLM backbone implementations: LLaMA, Qwen2/3, DeepSeek, InternLM, Mixtral, MiniMax, MIMO. Each defines a transformer spec and config dataclass.
- **`encoder/`** — Vision encoder implementations: base ViT, Qwen2-VL/3-VL, InternVL, LLaVA-OV, ERNIE-VL.
- **`omni_models/`** — Multi-modal model composition: `OmniCombinationModel` assembles encoder + projector + decoder into a unified pipeline, with `model_chunk_schedule_plan.py` for pipeline parallelism scheduling.
- **`common/`** — Shared layers (local norms, projectors, etc.).
- **`custom/`** — Non-standard models (WAN diffusion, Pi0.5).
- **`peft/`** — Parameter-efficient fine-tuning (LoRA) support.

### Configuration System: `configs/`

- **`configs/models/<family>/<model>.yaml`** — Hydra/OmegaConf YAML configs defining model architecture params. The `_target_` field maps to a Python config dataclass (e.g., `baige_omni.models.foundation.LLaMAConfig`).
- **`configs/data/`** — Data configuration templates.
- **`baige_omni/utils/config_map.py`** — `MODEL_CONFIG_REGISTRY` maps `--model-name` strings to `(config_path, config_name)` pairs.

### Data Pipeline: `baige_omni/data/`

- SFT datasets with sharegpt/alpaca format support, multimodal data handling, data packing, DP load balancing.
- `mm_plugin.py` — Multi-modal data plugin for processing images/video.
- `dp_balance/` — Data-parallel load balancing for packed sequences.

### Tools: `tools/`

- `convert_checkpoint/` — HuggingFace <-> Megatron checkpoint conversion scripts.
- `data_preprocess/` — Data preprocessing utilities.
- `apply_patches/` — Scripts to apply patches to Megatron-LM and TransformerEngine.
- `fp8_quantization/` — FP8 quantization utilities.

### Custom Ops: `ops/`

Custom CUDA kernels: `sparse_mla_fwd/`, `sparse_mla_bwd/` (sparse MLA attention), `lightning_indexer_bwd/`.

### Examples: `examples/`

Shell scripts for each supported model family with pretrain/SFT/checkpoint-conversion configs. Pattern: `examples/<model>/pretrain/pretrain_<model>.sh`.

### XPU Support: `examples_xpu/`

Baidu Kunlun XPU training scripts, mirroring `examples/` structure.

## Key Patterns

### Adding a New Model

1. Create a config dataclass in `baige_omni/models/foundation/` (or `encoder/` for vision), decorated with `@register_model_config(family, arch)`.
2. Create a model provider function decorated with `@register_model_provider(family)`.
3. Register a trainer function with `@register_model_trainer(family, training_phase)`.
4. Add YAML config under `configs/models/<family>/`.
5. Add entry in `baige_omni/utils/config_map.py` `MODEL_CONFIG_REGISTRY`.
6. Add example launch scripts under `examples/<model>/`.

### Configuration Flow

CLI args + Hydra YAML config -> `parse_train_args()` -> merged `args` namespace -> `build_model_trainer(args)` dispatches to registered trainer -> `MegatronTrainer.train()` runs the Megatron pretrain loop.

### Model Family Constants

Defined in `baige_omni/utils/constants.py`: `LanguageModelFamilies`, `VisionLanguageModelFamilies`, `CustomModelFamilies`, `VisionLanguageActionModelFamilies`. These enums drive dispatch logic throughout the codebase.

## Dependencies

Core external dependencies: Megatron-LM (patched, as `Megatron-LM`), TransformerEngine (patched), PyTorch, Hydra/OmegaConf, HuggingFace Transformers, DeepSpeed, megatron-energon. Python >= 3.10.

## Patches

`patches/Megatron-LM_v0.15.0/` and `patches/TransformerEngine_v2.9/` contain patch files applied to upstream repos during setup. These implement Baige-specific optimizations and fixes.
