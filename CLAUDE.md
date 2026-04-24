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
sh build.sh    # Creates output tarball
```

## Running Tests

E2E tests use a custom YAML-driven framework (`tests/main.py`), not pytest.

```bash
# Download test datasets first
bash tests/download_datasets.sh

# Run default CI test suite (all models in tests/configs/)
bash tests/main_start.sh

# Run optional regression tests (tests/optional_configs/)
bash tests/main_start.sh --optional
```

### Running a Single Model Test

Edit variables in `tests/main_start.sh`:
```bash
# Run one model from tests/configs/
model_names="qwen3_14b"

# Run one model from tests/optional_configs/
model_names="deepseek_v2/deepseek_v2_lite"
include_optional=true

# Run an entire model series from optional_configs/
model_names="NONE"
optional_subdir="internvl2.5"
include_optional=true
```

Test configs: `tests/configs/` (CI suite) and `tests/optional_configs/` (regression, organized by model family). Each YAML defines model params and multi-step `scenarios` (checkpoint conversion + training).

## Training Launch Pattern

Training scripts use `torchrun` for distributed execution. The PYTHONPATH must include both Megatron-LM and LoongForge:

```bash
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun --nproc_per_node 8 --nnodes $NNODES ... \
    $LOONGFORGE_PATH/loongforge/train.py \
    --model-name <model-name> \
    --training-phase pretrain|sft \
    ...
```

Key arguments: `--model-name` (maps to config via `config_map.py`) or `--config-file` (direct YAML path), `--training-phase` (pretrain/sft).

## Architecture

### Core Package: `loongforge/`

- **`train.py`** — Entry point. Calls `parse_train_args()` then `build_model_trainer(args).train()`.
- **`train/parser.py`** — Argument parsing: merges Megatron CLI args with Hydra YAML configs (OmegaConf). Supports `--model-name` (looked up in `config_map.py`) or `--config-file`.
- **`train/trainer_builder.py`** — Registry-based trainer dispatch. `register_model_trainer(model_family, training_phase)` decorator registers training functions per model family and phase.
- **`train/megatron_trainer.py`** — `MegatronTrainer` wraps model_provider, dataset_provider, and forward_step into Megatron's `pretrain()` loop.
- **`train/training_utils.py`** — Extended Megatron pretrain loop (heavily customized).
- **`train/arguments.py`** — LoongForge-specific extra CLI arguments added on top of Megatron's.
- **`train/validators.py`** — Validation logic for Megatron and LoongForge args.
- **`train/pretrain/`** — Pretrain implementations for LLM and VLM.
- **`train/sft/`** — SFT implementations for LLM, VLM, InternVL, ERNIE.
- **`train/custom/`** — Custom model trainers (e.g., WAN diffusion, Pi0.5 VLA).

### Model System: `loongforge/models/`

- **`factory.py`** — Model registry. `register_model_config(family, arch)` registers model configs; `register_model_provider(family)` registers model provider functions (accepts a single family string or list of families). Lookups: `get_model_config()`, `get_model_provider()`, `get_model_family()`.
- **`dispatch.py`** — Hardware-abstraction layer (`MultiAccModules`). Provides unified access to TransformerEngine or local linear/attention/norm implementations.
- **`foundation/`** — LLM backbone implementations: LLaMA, Qwen (all versions through Qwen3-Next), DeepSeek, InternLM, MiniMax, MIMO, GLM. Each defines a transformer spec and config dataclass.
- **`encoder/`** — Vision encoder implementations: base ViT, Qwen2-VL/3-VL, InternVL, LLaVA-OV, ERNIE-VL.
- **`omni_models/`** — Multi-modal model composition: `OmniCombinationModel` assembles encoder + projector + decoder into a unified pipeline, with `model_chunk_schedule_plan.py` for pipeline parallelism scheduling.
- **`common/`** — Shared layers (local norms, projectors, etc.).
- **`custom/`** — Non-standard models (WAN diffusion, Pi0.5).
- **`peft/`** — Parameter-efficient fine-tuning (LoRA) support.

### Configuration System: `configs/`

- **`configs/models/<family>/<model>.yaml`** — Hydra/OmegaConf YAML configs defining model architecture params. The `_target_` field maps to a Python config dataclass (e.g., `loongforge.models.foundation.LLaMAConfig`).
- **`configs/data/`** — Data configuration templates.
- **`loongforge/utils/config_map.py`** — `MODEL_CONFIG_REGISTRY` maps `--model-name` strings to `{"config_path": ..., "config_name": ...}` dicts. Contains 80+ model entries.

### Data Pipeline: `loongforge/data/`

- SFT datasets with sharegpt/alpaca format support, multimodal data handling, data packing, DP load balancing.
- `mm_plugin.py` — Multi-modal data plugin for processing images/video.
- `dp_balance/` — Data-parallel load balancing for packed sequences.

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

1. Create a config dataclass in `loongforge/models/foundation/` (or `encoder/` for vision), decorated with `@register_model_config(family, arch)`.
2. Create a model provider function decorated with `@register_model_provider(family)`.
3. Register a trainer function with `@register_model_trainer(family, training_phase)`.
4. Add YAML config under `configs/models/<family>/`.
5. Add entry in `loongforge/utils/config_map.py` `MODEL_CONFIG_REGISTRY`.
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
