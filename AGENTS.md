# AGENTS.md

These instructions apply to the LoongForge repository. Before changing a
covered area, read the relevant `CONTRIBUTING.md`, README, design note, or test
guide. Write code comments in concise English; follow
the language and style of the target document for other documentation.

## Project overview

LoongForge is a large-scale training framework built on Loong-Megatron and
Transformer Engine. The Megatron engine supports language, vision-language,
and diffusion models. The Torch engine supports embodied models such as VLA
and WAM systems. Supported training phases include pretraining and supervised
fine-tuning.

The normal launch path is:

```text
examples/<model>/<phase>/*.sh
    -> loongforge/train.py
    -> engine-specific argument parser
    -> registered recipe or trainer
    -> engine runtime
```

Example scripts provide launch parameters and environment defaults. They do not
own the training loop.

| Area | Main responsibility |
|---|---|
| `loongforge/models/` | Model implementations, providers, configs, and family catalogs |
| `loongforge/data/` | Datasets, preprocessing, collators, samplers, and chat templates |
| `loongforge/training/` | Argument boundaries, validation, recipe registration, and model trainers |
| `loongforge/engines/` | Distributed runtime, precision, optimizers, loops, and checkpoints |
| `loongforge/evaluation/` | Embodied evaluation adapters, servers, and metrics |
| `configs/` | Model and data YAML configuration |
| `examples/` | Training, conversion, and evaluation launch scripts |
| `tools/` | Checkpoint conversion, preprocessing, and benchmark tools |
| `ops/` | CUDA and TileLang kernels |
| `tests/` | Unit, contract, evaluation, and end-to-end tests |
| `third_party/` | Loong-Megatron and external patches or submodules |

## Working rules

- Read the relevant implementation, callers, tests, and configuration before editing.
- Make the smallest change that completely solves the requested behavior.
- Preserve existing interfaces, error behavior, validation, and dependency choices unless the task requires a change.
- Give each concept, parameter, and type one clear owner. Prefer existing factories, registries, and runners over new wrappers.
- Validate unknown input at the boundary that owns it. Pass validated values to internal code and reject unsupported cases clearly.
- Do not add configuration, abstraction, compatibility code, or a helper used once without a concrete need.
- Keep temporary experiments and test-only controls out of production code.
- Do not commit, push, merge, rebase, or publish unless the user explicitly asks.

## Setup and common commands

Use `uv` for Python environments and dependencies. The repository requires
Python `>=3.10`.

```bash
git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git
cd LoongForge
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[gpu]"       # Use ".[xpu]" on Kunlun XPU.
git submodule update --init --recursive
uv run python setup_env.py --te-tag v2.9  # GPU only.
python -m build --sdist --wheel --outdir dist/
```

`setup_env.py` handles Transformer Engine cloning, patching, and installation.
Loong-Megatron is the `third_party/Loong-Megatron` submodule. Docker builds use
`docker/Dockerfile` with `COMPILE_ENV=ampere|hopper|blackwell`; XPU builds use
`docker/Dockerfile.xpu`.

Use existing example scripts as launch references. Megatron examples normally
set both source roots in `PYTHONPATH` and invoke `loongforge/train.py` with
`--model-name` and `--training-phase`. Torch examples use the same entry point
and may add `--trainer-type`.

Run focused checks first:

```bash
uv run ruff check <changed-python-files>
uv run python -m py_compile <changed-python-file.py>
git diff --check
pre-commit run --files <changed-files>
python3 .github/scripts/sensitive_scan.py --paths <changed-files>
```

The Ruff configuration is in `pyproject.toml`: 120-character lines, Python
3.10 target, `third_party` excluded, and the `E4`, `E7`, `E9`, `F`, and `S506`
rule groups enabled. Its documented ignores are intentional; do not add local
formatter or lint settings.

## Python style

- Follow PEP 8 and the surrounding module style. Use four spaces, no tabs, and keep imports grouped as standard library, third-party, and local imports. Avoid wildcard imports except where an existing package export requires them.
- Use `snake_case` for functions, methods, variables, and modules; `PascalCase` for classes; `UPPER_SNAKE_CASE` for constants. Keep model names, model families, model types, and trainer names distinct.
- Add parameter and return annotations to new public functions and to non-trivial internal interfaces. Prefer concrete types and typed dataclasses or schemas at configuration boundaries. Avoid `Any` unless the boundary is genuinely untyped.
- Use built-in generics and union syntax supported by Python 3.10 for new code when consistent with the surrounding file. Do not rewrite existing annotations only for style.
- Keep functions focused and interfaces small. Remove redundant variables, forwarding parameters, wrappers, and calls. Extract a helper when it represents an independent boundary or makes a complex call easier to understand.
- Do not use mutable default arguments. Make ownership of mutable state explicit and avoid hidden module-level state unless the registry or runtime contract requires it.
- Validate and normalize external values once at the owning boundary. Do not repeat checks already guaranteed by a schema, constructor, or type contract.
- Raise a specific exception with useful context for invalid input. Do not silently coerce unsupported values, catch `Exception` without recovery, or discard the original exception context.
- Use `try`/`except` only to recover, add useful context, or translate an exception. Use context managers for files, locks, temporary resources, and other closable objects.
- Use the module logger for library and training diagnostics. Do not use `print` for normal production logging, and never log secrets, complete environments, or sensitive paths.
- Pass subprocess arguments as a list and avoid `shell=True` for user-controlled values. Set timeouts and handle process cleanup where a subprocess can block.
- Keep device, dtype, random-state, and distributed behavior explicit in Torch code. Preserve collective ordering and avoid introducing synchronization or allocation into hot loops without evidence.
- Keep comments and docstrings brief and accurate. Explain a non-obvious reason, constraint, or compatibility requirement; do not restate the code or record editing history. Public APIs should document meaningful arguments, returns, and raised errors.
- Add user-facing configuration only when users must choose between multiple valid behaviors. If one value is always correct, determine it in code.
- Preserve SPDX, copyright, license, and external attribution headers. Record provenance when copying code from another project.

## Architecture and model integration

`loongforge/models/factory.py` registers Megatron model configs and providers.
`loongforge/models/catalog.py` owns model-name routing, including
`MEGATRON_MODEL_CONFIG_REGISTRY`, `EMBODIED_MODEL_NAMES`, and the language,
vision-language, diffusion, and embodied family constants.

The main ownership boundaries are:

- `models/foundation/`, `models/encoder/`, `models/omni_models/`, and `models/diffusion/` contain Megatron model implementations.
- `models/embodied/` contains Torch embodied models and typed configs.
- `data/llm/`, `data/vlm/`, `data/embodied/`, and `data/chat_templates/` own their respective data paths.
- `training/trainer_builder.py` registers and queries both engines. Megatron uses `MEGATRON_TRAINERS_BY_FAMILY`; Torch uses `TORCH_TRAINERS_BY_FAMILY` and the name table `TORCH_TRAINERS_BY_NAME` for `--trainer-type`.
- `engines/megatron/` and `engines/torch/` own runtime execution and checkpoint behavior.

Configuration and dispatch follow this contract:

```text
CLI + YAML
    -> engine parser
    -> merged model/data/training config
    -> build_model_trainer(engine)
    -> registered recipe or trainer
    -> engine runtime
```

Megatron model names resolve through `MEGATRON_MODEL_CONFIG_REGISTRY`. Torch
model names resolve through `MODEL_SCHEMA` in `training/torch_args.py`, which
binds a name to a YAML file and typed model/data config classes. Keep these
registries and family constants consistent when adding a model.

For a new model, first identify the owning model, data, training, and example
modules. Reuse an existing recipe or trainer when its contract fits. Register
only the new config/provider/family or trainer entries required by the call
path, then add the YAML, example, and focused test or regression configuration.
Use `tools/mcore_checkpoint_convert/` for offline HF/Megatron conversion and
`tools/mcore_bridge/` for supported online checkpoint loading or saving.

## Testing and delivery

Choose tests by observable behavior and the smallest useful layer:

- Parser, registry, dispatch, and configuration changes need focused contract tests for default and explicit selection.
- Data or model changes need direct-consumer checks and focused data/model tests.
- Runtime, distributed, GPU, simulator, checkpoint, and numerical changes need corresponding runtime evidence; syntax checks do not prove runtime equivalence.
- Reuse nearby fixtures and helpers. Test behavior through public or real entry points, keep one purpose per test, and avoid tests that only mirror implementation details.
- Put one-off performance experiments in the benchmark area, not the test suite.

The LLM/VLM suite is YAML-driven from `tests/llm_vlm/main_start.sh`; read
`tests/llm_vlm/README.md` for asset preparation and scenario details. The
embodied suite is manifest-driven from `tests/embodied/run.sh`; read
`tests/embodied/README.md` for baselines, dry runs, and hardware requirements.
The Mcore-Bridge roundtrip suite is under `tests/mcore_bridge_roundtrip/`.

Report the commands actually run, failures with their scope, and relevant
untested boundaries. Do not claim GPU, distributed, simulator, checkpoint, or
numerical parity from static checks alone.

## Security and review

- Treat CLI values, YAML values, paths, external model metadata, and downloaded artifacts as untrusted input. Reject unsupported values at the boundary.
- Do not add secrets, credentials, private URLs, private filesystem paths, or sensitive dataset metadata to code, tests, logs, or documentation.
- Do not add unsafe deserialization, arbitrary code execution, or permissive fallback behavior. Keep remote code loading and checkpoint conversion within an explicit project contract.
- For a review, read the full diff and affected files, then inspect direct callers, tests, configs, and public interfaces. Report only changed or directly affected behavior with exact file and line locations, severity, impact, and a concrete fix. Separate findings from probes and self-tests.
- Keep PR and change descriptions concise: state the behavior change, why it is needed, and the checks actually run.
