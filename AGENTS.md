# AGENTS.md

LoongForge: large-scale training framework on Loong-Megatron and TransformerEngine for NVIDIA GPUs and Kunlun XPUs. The Megatron engine serves LLMs, VLMs, and diffusion models; the Torch engine serves embodied models.

## Environment & Build

Run from the repository root. Installation details are in `docs/source/get_started/installation.md`; Kunlun XPU setup is in `docs/source/kunlun_tutorial/install_p800.md`.

```bash
git submodule update --init --recursive
uv pip install -e ".[gpu]"            # ".[xpu]" for Kunlun XPU
python setup_env.py --te-tag v2.9
python -m build --sdist --wheel --outdir dist/
```

- `third_party/Loong-Megatron` is a Git submodule. `setup_env.py` only fetches, patches, and builds TransformerEngine.
- GPU image: `docker build --build-arg COMPILE_ENV=hopper -f docker/Dockerfile .` (`ampere` / `blackwell` also accepted).
- Launch with `torchrun loongforge/train.py` and both LoongForge and Loong-Megatron on `PYTHONPATH`. `--model-name` selects the engine and config through `loongforge/models/catalog.py`. On Megatron, `--config-file` without `--model-name` runs the Megatron engine; with `--model-name`, Megatron ignores `--config-file`. On Torch, `--config-file` replaces the catalog YAML and still requires `--model-name`.

## Project Structure

| Path | Responsibility |
|---|---|
| `loongforge/train.py` | Training entrypoint and engine routing |
| `loongforge/training/<backend>/` | Loop, lifecycle, registry, non-backend args, schedules, logging, profiling; `methods/` assembles model, data, and objective for a task |
| `loongforge/engines/<backend>/` | Backend args, initialization, parallelism, model wrapping, optimizer, step execution, checkpointing |
| `loongforge/models/` | Model implementations, configs, kernels, and registries (`language`, `vision`, `multimodal`, `diffusion`, `embodied`) |
| `loongforge/extensions/` | Model × backend training extensions; may import `models` and `engines`, never `training` or `data` |
| `loongforge/data/` | Family data packages (`llm/`, `vlm/`, `diffusion/`, `embodied/`): datasets, transforms, collators, samplers, and each family's `dataloader.py`; `chat_templates/` holds the shared chat templates, registry, and multimodal plugins |
| `loongforge/evaluation/` | Inference services and evaluation |
| `configs/` | Model and data configurations |
| `examples/` | One directory per model with the scripts it supports (`pretrain/`, `finetuning/`, `checkpoint_convert/`, `eval/`, `preprocess/`) |
| `tools/` | Self-contained offline tools (weight conversion, data preprocessing, benchmarking) |
| `ops/` | Custom operators |
| `tests/` | Unit tests and training regression suites (see `tests/README.md`) |

MCore weight conversion: `tools/mcore_checkpoint_convert/module_convertor/model.py`.

## Code Organization & Dependencies

- **Four layers**: `data/` owns samples and data formats; `models/` owns network structure; `engines/<backend>/` owns backend execution and performance; `training/<backend>/` owns what this run does.
- **Dependency direction**: `training -> engines -> models`; `data -> models` (declarative modules only); no cycles; backends never import each other. Declarative resources (configs, schemas, registry tables, chat templates) may be read from any layer.
- **Ownership test**: if the code still makes sense with another backend, it belongs to `training/` or `data/`; otherwise to `engines/<backend>/` (parallelism, optimizer, low precision, step execution, checkpoint format).
- **Runtime facts by arguments**: `data/` never reads backend args, global readers, or distributed state; the training method passes them down.
- **Parameter ownership**: backend execution parameters go to `engines/<backend>/arguments.py`; run, task, data, logging, and lifecycle parameters go to `training/<backend>/args.py`. Where a value is read does not decide where it is defined.
- **Non-obvious contracts**: ChunkPipe queue handoff, lazy loading of models and backends, `extensions/<backend>/<model>/` placement, and upstream reuse.

## Extending Functionality

Follow existing models and tasks for interfaces and naming.

| Change | Primary Entrypoints |
|---|---|
| New LLM / VLM / Diffusion | Model package, `models/catalog.py`, `configs/models/`, `examples/`; register with `AutoModel.register(<Config>, <Model>)` in `models/language/__init__.py` or `models/vision/__init__.py`; add the family in `loongforge/constants.py` and, when it has its own method, in `training/megatron/registry.py`; VLM data encoders go in `data/vlm/` (`TASK_ENCODER_REGISTRY`) |
| New Embodied Model | One `MODEL_PACKAGES` entry in `models/embodied/registry.py` plus `models/catalog.py`; a matching `models/embodied/<pkg>/` with `modeling_<pkg>.py` and a `data/embodied/<pkg>/` with `<pkg>_dataset.py`, `data_configuration_<pkg>.py`, `samplers.py`, and `transforms/`; add `DATASET_STRATEGIES` only for a custom strategy; optional eval factory |
| New Data Format | Family data module; embodied formats and strategies in `data/embodied/registry.py` (`DATASET_FORMATS`, `DATASET_STRATEGIES`) |
| New Training Objective | `training/<backend>/methods/` and the matching `training/<backend>/registry.py` |
| New CLI Argument | Backend execution: `engines/<backend>/arguments.py`; recipe, task, data, logging, or lifecycle: `training/<backend>/args.py` |
| New Parallelism / Comm | `distributed/` under the target engine |
| Single-Model Torch Opt | `extensions/torch/<model>/` and `TORCH_TRAINER_TYPES` in `training/torch/registry.py`; generic mechanisms go to `engines/torch/` |
| New Chat Template | `data/chat_templates/jinja/` and `data/chat_templates/registry.py` |

- **File Naming**: Inside `models/`, use `modeling_<model>.py`, `configuration_<model>.py`, and `processing_<model>.py`. Layer specs use `<model>_layer_spec.py`; training methods use `<phase>_<scope>.py`. Embodied data uses `<pkg>_dataset.py`, `data_configuration_<pkg>.py`, `samplers.py`, and under `transforms/`: `<pkg>_transform.py`, `<pkg>_collator.py`, `<pkg>_processor.py`, plus model-specific augmentation helpers. Mirror existing configs, transforms, and collators.

## Code Style

Ruff and pre-commit are configured in `pyproject.toml` (line length 120); CI versions live in `.github/workflows/`.

```bash
ruff check path/to/changed_file.py
pre-commit run --files path/to/changed_file.py
```

- Add copyright and SPDX headers to new files. Preserve upstream licenses, attribution, and meaningful comments.
- Comment non-obvious rationale, constraints, and behavior; document purpose, I/O, and side effects. No redundant identifier restatements, empty TODOs, or development logs.

## Testing & Verification

Match tests to the change, including edge cases. Verify model, data, and distributed changes on target hardware; static checks do not replace training or inference verification. Report commands, results, and unverified areas.

- **Unit tests**: `python -m pytest <path>` (setup in `tests/README.md`).
- **LLM/VLM E2E**: `bash tests/llm_vlm/main_start.sh`; configs in `tests/llm_vlm/configs/`; set `model_names` (and `include_optional=true` for optional configs).
- **Embodied regression**: `bash tests/embodied/run.sh --chip a --list_models`, then `--models <name> --dry_run` and `--models <name>`. Register scripts in `tests/embodied/config/scripts.yaml` and baselines under `tests/embodied/baseline/<chip>/` (never reuse a baseline across hardware). Set data, checkpoint, cache, and output paths via environment variables; validates `loss` and `grad_norm`.
- **Docs & CI**: run `python3 ci/check_doc_links.py` after Markdown edits and check affected workflows.

## Security & Collaboration

- **Security**: Never commit secrets, internal hostnames, mirrors, or personal absolute paths. Pass data, weights, and artifact paths via configs or environment variables. Run `skills/sensitive-scan/SKILL.md` or `python3 ci/sensitive_scan.py` before committing.
- **Skills**: Live in `skills/`, symlinked by `.claude/skills` and `.agents/skills`.
- **PRs & Reviews**: Follow `CONTRIBUTING.md`. Use `skills/submit-pr/SKILL.md` and `skills/code-review/SKILL.md`. Post specific issues as inline comments; keep the top-level summary to conclusions, verification results, and overall judgment.

When a repository’s documentation and its existing code disagree, follow the conventions used in the existing code, and in your final reply point out each inconsistency you found between the documentation and the code.
