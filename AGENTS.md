# AGENTS.md

LoongForge: large-scale training framework on Loong-Megatron (`third_party/Loong-Megatron`) and TransformerEngine for NVIDIA GPUs and Kunlun XPUs. The Megatron engine serves LLMs, VLMs, and diffusion models; the Torch engine serves embodied (VLA) models. Supported phases: pretrain and SFT.

## Package Management

Dependencies are managed with `uv`:

```bash
uv venv .venv && source .venv/bin/activate
git submodule update --init --recursive
uv pip install -e ".[gpu]"            # ".[xpu]" for Kunlun XPU
python setup_env.py --te-tag v2.9     # GPU only: fetch, patch, and build TransformerEngine
```

## Common Commands

```bash
# Train. Scripts set PYTHONPATH and launch torchrun loongforge/train.py.
bash examples/qwen2.5/pretrain/pretrain_qwen2.5_7b.sh              # LLM/VLM/diffusion (Megatron)
bash examples/pi05/finetuning/sft_pi05_fsdp.sh                # embodied (Torch)

# Test. Details for each suite are in tests/README.md.
python -m pytest tests/test_*.py                                   # CI and release contract tests
python -m pytest tests/llm_vlm/unit                                # Megatron-side unit tests
python -m pytest tests/embodied/unit tests/embodied/evaluation     # Torch-side unit tests and embodied evaluation
bash tests/llm_vlm/main_start.sh                                   # LLM/VLM E2E
bash tests/embodied/run.sh --chip a --list_models                  # list embodied regression targets
bash tests/embodied/run.sh --chip a --models <name> --dry_run      # validate setup, then drop --dry_run
bash tests/mcore_bridge_roundtrip/qwen3/8b_bridge_roundtrip.sh     # HF <-> Mcore checkpoint roundtrip

# Check
ruff check <file>
python3 ci/sensitive_scan.py --paths <path>                        # secrets and internal-info scan
pre-commit run --files <file>                                      # SPDX headers, gitleaks, sensitive scan
```

## Directory Layout

| Path | Purpose |
|---|---|
| `loongforge/train.py` | Training entry point and the tables that map a Megatron family/phase or a Torch `--trainer-type` to a recipe |
| `loongforge/training/` | What one run does: shared args (`megatron_args.py`, `torch_args.py`), per-family recipes (`llm/`, `vlm/`, `diffusion/`, `embodied/`) that combine data, model, and engine, and single-model training flows (`embodied/custom/`) |
| `loongforge/engines/` | How training runs for any model: Megatron and torch wrappers for the parallel runtime, precision, model wrapping, optimizer, step loop, and checkpoint I/O (`megatron/`, `torch/`) |
| `loongforge/models/` | What the network computes (`foundation/`, `encoder/`, `omni_models/`, `diffusion/`, `embodied/`, `common/`), including the model-specific part of engine features such as EP-overlap callables and FSDP/LoRA/FP8 targets; `catalog.py` maps `--model-name` to engine and config |
| `loongforge/data/` | How samples become batches: per-family datasets, transforms, collators, samplers, and chat templates (`chat_templates/`: the registry in `template.py`, multimodal plugins in `plugins/`, Jinja files in `jinja/`); the tokenizer and parallel sizes come in as arguments |
| `loongforge/evaluation/` | Evaluation (`embodied/`) |
| `configs/` | YAML model configs (`configs/models/`) and data configs (`configs/data/`) |
| `examples/` | Per-model training, checkpoint conversion, and evaluation scripts |
| `examples_xpu/` | Kunlun XPU scripts, mirroring `examples/` |
| `tools/` | Checkpoint conversion (`mcore_bridge/`, `mcore_checkpoint_convert/`, `torch_dcp_convert/`), data preprocessing, and benchmarks |
| `ops/` | Custom CUDA and TileLang kernels |
| `tests/` | Unit tests and three E2E suites: `llm_vlm/`, `embodied/`, `mcore_bridge_roundtrip/` |
| `third_party/` | `Loong-Megatron` submodule and TransformerEngine patches |
