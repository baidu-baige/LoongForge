# AGENTS.md

LoongForge: Large-scale training framework built on Loong-Megatron and TransformerEngine for NVIDIA GPUs and Kunlun XPUs.
- **Megatron Engine**: LLMs, VLMs, and diffusion models.
- **Torch Engine**: Embodied AI models.

## Environment & Build

Run commands from the repository root. See `docs/source/get_started/installation.md` for standard setup and `docs/source/kunlun_tutorial/install_p800.md` for Kunlun XPU setup.

```bash
git submodule update --init --recursive
uv pip install -e ".[gpu]"            # Use ".[xpu]" for Kunlun XPU
python setup_env.py --te-tag v2.9
python -m build --sdist --wheel --outdir dist/
```

- `third_party/Loong-Megatron` is a Git submodule.
- `setup_env.py` only fetches, patches, and builds TransformerEngine (patches in `patches/TransformerEngine_v2.9/`).
- **GPU Docker Build** (`COMPILE_ENV`: `ampere`, `hopper`, or `blackwell`):
  ```bash
  docker build --build-arg COMPILE_ENV=hopper -f docker/Dockerfile .
  ```
- **Running Training**: Scripts are in `examples/` (`examples_xpu/` for XPU). Follow the target model's script for dataset, checkpoint, parallelism, and output settings. When launching `loongforge/train.py` with `torchrun`, include both LoongForge and Loong-Megatron in `PYTHONPATH`. `--model-name` selects the engine and config via `loongforge/models/catalog.py`. `--config-file <yaml>` without `--model-name` runs on the Megatron engine; Megatron ignores `--config-file` when `--model-name` is set. On the Torch engine, `--config-file` replaces the catalog YAML and still requires `--model-name` to select the config classes.

## Project Structure

| Path | Responsibility |
|---|---|
| `loongforge/train.py` | Training entrypoint and engine routing |
| `loongforge/training/megatron/`, `loongforge/training/torch/` | Training loop and lifecycle per backend; each owns its runner, recipe/task/data arguments, registry, learning-rate schedules, logging, and profiling |
| `loongforge/training/<backend>/methods/` | Task-specific model/data assembly, forward pass, and loss for that backend |
| `loongforge/engines/` | Megatron / Torch backend arguments, initialization, parallelism, model wrapping, optimizer, step execution, and checkpointing |
| `loongforge/models/` | Model implementations, configs, kernels, and registries (`language`, `vision`, `multimodal`, `diffusion`, `embodied`) |
| `loongforge/accelerations/` | Model × backend adapters for single-model training accelerations; may import `models` and `engines`, never `training` or `data` |
| `loongforge/data/` | Datasets, preprocessing, collators, samplers, and dataloaders |
| `loongforge/chat_templates/` | Chat templates and registry |
| `loongforge/evaluation/` | Inference services and evaluation |
| `configs/` | Model and data configurations |
| `examples/` | Model-specific training, conversion, and data scripts (including single-model/family offline scripts) |
| `tools/` | Shared cross-model offline utilities (weight conversion, data preprocessing) |
| `ops/` | Custom operators |
| `tests/` | Unit tests and training regression suites (see `tests/README.md`) |

MCore weight conversion entrypoint: `tools/mcore_checkpoint_convert/module_convertor/model.py` (see model example scripts for arguments and multi-stage flows).

## Code Organization & Dependencies

- **Scope Isolation**: Keep single-model/subsystem helpers inside their respective modules. Place shared code in well-defined common layers; avoid circular and cross-layer dependencies.
- **`utils/` Convention**: Name all helper directories `utils/`. Keep module-private helpers in the module's `utils/` and cross-module helpers in the shared parent layer's `utils/`, respecting layer dependency boundaries. Name files by purpose (e.g., `io.py`, `state_dict.py`); never dump unrelated helpers into one file.
- **Core vs. Helpers**: Separate model entrypoints and core logic from auxiliary helpers. Base directory depth on cohesion and readability—do not use file count as a hard threshold or flatten helpers just to reduce nesting.
- **Training vs. Engine**: `training/<backend>/` manages loops and lifecycle; `engines/<backend>/` handles backend execution. Each backend subpackage has one method interface and does not import the other backend. `training/<backend>/methods/` assembles models, data, and objectives, and must not manage process groups or checkpoints. Ownership test: would the code still make sense with a different backend? If yes, it belongs to `training/` or `data/` (learning-rate schedules, run length, logging, profiling); if no, it belongs to `engines/<backend>/` (optimizer implementation, parallelism, low precision, step execution, checkpoint format). Define each parameter next to the code that reads it; parameters read only by `data/` are defined in `training/` and passed in explicitly. Upstream Megatron arguments stay upstream.
- **Explicit Configs & Data Decoupling**: Pass runtime info to model and data components via arguments or configs—never read global CLI args directly. Models must not depend on the data layer; assemble embodied inference preprocessing in `evaluation/embodied/factories/` before passing to the model.
- **Acceleration Adapters**: Code that must know both a concrete model and a backend (full-iteration CUDA graph, model-specific FSDP wrapping, custom capturable optimizer) lives in `loongforge/accelerations/<backend>/<model>/`. It implements an engine port (for Torch, `TorchOptimization`) and is selected by `training/<backend>/registry.py`. Models and engines must not import `accelerations`.
- **Stateless Engine Boundaries**: Engines must not depend on private states of datasets or training methods; pass task logic via explicit interfaces or callbacks. Engines import only generic model packages (`models/common`, `models/multimodal`, `models/catalog`), never a model family; model-specific behavior is injected from the model or method side. Reuse family-specific data entrypoints for DataLoaders and samplers, and follow native Megatron interfaces when integrating Megatron data pipelines.
- **Lazy Loading**: Load selected models and backends on demand in configs, registries, and package `__init__.py` files to avoid importing unused weights or optional dependencies. Never auto-register by walking all model packages; raise clear errors naming the missing module when optional dependencies are absent.
- **Upstream Reuse**: Prefer Loong-Megatron and upstream capabilities. Modify upstream behavior via extension hooks first; if maintaining a modified copy, keep original licenses/attribution and document divergences.

## Extending Functionality

Follow existing implementations of similar models or tasks for interfaces and naming.

| Change | Primary Entrypoints |
|---|---|
| New LLM / VLM | Model package, `loongforge/models/catalog.py`, `configs/models/`, `examples/`; VLM data encoders in `loongforge/data/vlm/` (`TASK_ENCODER_REGISTRY`) |
| New Embodied Model | `loongforge/models/embodied/registry.py`, `loongforge/models/catalog.py`, model package; data in `loongforge/data/embodied/registry.py`; optional eval factory |
| New Data Format | Family data module; embodied dataset strategies in `loongforge/data/embodied/registry.py` (`DATASET_STRATEGIES`) |
| New Training Objective | `loongforge/training/<backend>/methods/` and the matching `training/<backend>/registry.py` |
| New CLI Argument | Read by backend code: `loongforge/engines/<backend>/arguments.py`; other recipe, task, or data options: `loongforge/training/megatron/args.py` or `loongforge/training/torch/args.py` |
| New Parallelism / Comm | `distributed/` under the target engine |
| Single-Model Torch Opt | `loongforge/accelerations/torch/<model>/` and `TORCH_TRAINER_TYPES` in `loongforge/training/torch/registry.py`; generic mechanisms go to `engines/torch/`. Model packages keep only structure, kernels, and declarative hooks |
| New Chat Template | `loongforge/chat_templates/jinja/` and `loongforge/chat_templates/registry.py` |

- **File Naming**: Use `modeling_<model>.py`, `configuration_<model>.py`, `processing_<model>.py` for models; `<model>_layer_spec.py` for layer specs; `<phase>_<scope>.py` for training methods. Mirror existing configs, transforms, and collators for new embodied data components.

## Code Style

- Enforced via `pyproject.toml` (Ruff line length: 120). See `.github/workflows/` for CI tool versions and scopes.
- Include required copyright and SPDX license headers in new files. Preserve licenses, attribution, and meaningful comments in upstream-ported code.
- Comment on non-obvious rationale, constraints, and behavior; document purpose, I/O, and side effects in docstrings. No redundant identifier restatements, empty TODOs, or dev logs.

```bash
ruff check path/to/changed_file.py
pre-commit run --files path/to/changed_file.py
```

## Testing & Verification

Match tests to your changes: cover happy paths and critical edge cases, and verify model, data, or distributed changes on target hardware. **Static checks do not replace training/inference verification.** When delivering, report executed commands, results, and unverified areas.

- **Unit Tests**: Run `python -m pytest <path>` (setup in `tests/README.md`).
- **LLM/VLM E2E Tests** (requires pre-staged data, HF models, and converted checkpoints):
  ```bash
  bash tests/llm_vlm/main_start.sh
  ```
  Configs in `tests/llm_vlm/configs/`. Set `model_names` in the script to select models; set `include_optional=true` for `optional_configs/`.
- **Embodied Regression Tests** (registered in `tests/embodied/config/scripts.yaml`, paths relative to `examples/`):
  ```bash
  bash tests/embodied/run.sh --chip a --list_models
  bash tests/embodied/run.sh --chip a --models fastwam_ddp --dry_run
  bash tests/embodied/run.sh --chip a --models fastwam_ddp
  ```
  Select `--chip` by hardware. Register new scripts and add hardware-specific baselines at `tests/embodied/baseline/<chip>/<name>.json` (never reuse baselines across hardware). Set data, checkpoint, cache, and output paths via environment variables. Validates `loss` and `grad_norm`.
- **Docs & CI**: Run `python3 ci/check_doc_links.py` after Markdown edits. Check affected workflows and script tests when modifying CI.

## Security & Collaboration

- **Security**: Never commit secrets, internal hostnames, internal mirrors, or personal absolute paths. Pass data, weights, and artifact paths via configs or environment variables. Run `skills/sensitive-scan/SKILL.md` or `python3 ci/sensitive_scan.py` before committing.
- **Skills**: Maintained in `skills/` (symlinked by both `.claude/skills` and `.agents/skills`).
- **PRs & Reviews**: See `CONTRIBUTING.md` for workflow and PR title rules. Use `skills/submit-pr/SKILL.md` for PR creation and `skills/code-review/SKILL.md` for reviews. Post specific code issues as inline comments; keep the top-level summary to conclusions, verification results, and overall judgment.