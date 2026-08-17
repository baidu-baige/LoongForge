# Embodied Regression Test Framework

Manual regression tests for the `loongforge/embodied` subframework: **serially execute the training scripts under
`examples/embodied` directly** -> parse metrics -> compare against the baseline within tolerance -> write `results.json`.

This suite lives under [tests/embodied/](.) and is fully self-contained (own config, executor, and
`baseline/`). The sibling LLM/VLM E2E suite is at [tests/llm_vlm/](../llm_vlm/); see
[tests/README.md](../README.md) for an overview of both.

## Overall Logic

One entry point, one execution layer:

- **Trigger `run.sh`** (inside the container): a thin wrapper that only sources `config/env.sh`
  (so path variables propagate to training subprocesses), then delegates to `cli.py`.
- **Executor `cli.py`**: owns everything else — parameter parsing / defaults / validation,
  optional `--prepare` (bos sync), then serially `bash`-execute the training scripts according to
  the `config/scripts.yaml` manifest -> parse metrics -> compare against the baseline within tolerance -> write `results.json`.

Calling `cli.py` directly is equivalent — it auto-sources `config/env.sh` on first use.

## Workflow

```
tests/embodied/run.sh  (thin wrapper: source config/env.sh, then exec python3 cli.py "$@")
  └─ tests/embodied/cli.py (execution layer)
       1. resolve env-var / CLI-flag defaults (model_names / chip / TIMEOUT / tolerances / ...)
       2. [optional] --prepare / prepare=true → bash config/prepare.sh (bos sync)
       3. read the config/scripts.yaml manifest according to --models (or all)
       4. bash-execute the examples scripts serially one by one:
            inject OUTPUT_DIR / TENSORBOARD_DIR pointing to this run's log directory + model-specific env (EMBODIED_<MODEL>_<VAR>)
       5. metric parsing (metrics.jsonl preferred, fall back to stdout)
            → auto_collect_baseline=true: only collect and write to the baseline root, no comparison
            → otherwise compare against the baseline within tolerance, judge PASS/FAIL
       6. aggregate and write results.json (default under ${EMBODIED_LOG_ROOT}/run_<timestamp>/; override with LOG_DIR)
       7. if --fail_fast and a model fails, break the loop (skip the remaining models)
```

## Path Configuration

All environment-related paths are centralized in `config/env.sh` (sourced automatically whether
you invoke via `run.sh` or `cli.py`). To switch machines, you only need to change
this one file, or temporarily override with environment variables before the command:

- `EMBODIED_CI_ROOT` — default `/ssd2/loongforge_embodied_ci`; unified root directory for out-of-repo mutable state (data, logs, tools). Baselines are **no longer** derived from it by default.
- `LOCAL_VLA_ARTIFACTS_ROOT` — default `${EMBODIED_CI_ROOT}/vla_artifacts`; data/ckpt root, shared by download and training scripts
- `EMBODIED_LOG_ROOT` — default `${EMBODIED_CI_ROOT}/logs`; regression log/result root directory
- `EMBODIED_BASELINE_ROOT` — default **in-repo** at `tests/embodied/baseline` (this suite owns its baselines; the sibling suite owns `tests/llm_vlm/baseline/`). Override to point at a shared out-of-repo collection if multiple checkouts must share baselines.
- `BOS_VLA_ARTIFACTS_ROOT` — BOS source prefix used by `config/prepare.sh`
- `BCECMD_DIR` / `BCECMD` — bcecmd tool path, see file
- `EMBODIED_<MODEL NAME UPPERCASE>_<VAR>` — empty by default; model-specific required paths (e.g. fastwam's TOKENIZER_PATH / DATA_PATH), injected as `<VAR>` before running the corresponding script

`config/prepare.sh` syncs the entire `${LOCAL_VLA_ARTIFACTS_ROOT}` to the local machine in one shot; once done you can regress directly.

> Baseline location changed with the merge into `tests/`: baselines previously collected under
> `${EMBODIED_CI_ROOT}/metrics_baseline/<chip>/` still work if you `export EMBODIED_BASELINE_ROOT=...`
> before running. Otherwise copy them into `tests/embodied/baseline/<chip>/` once, or re-collect on
> the target machine with `--auto_collect_baseline`.

## Prerequisites

The following need to be prepared under `${EMBODIED_CI_ROOT}`:

- **Data and weights** `vla_artifacts/`: organized as `<family>/{models,datasets,tokenizers}`; the training scripts' default paths are derived from
  `LOCAL_VLA_ARTIFACTS_ROOT`. When files are missing or switching machines, use `bash tests/embodied/config/prepare.sh` to sync from BOS.
- **bcecmd + BOS credentials** `tools/bcecmd`: used only when `config/prepare.sh` needs to sync data; you must first
  configure AK/SK with `bcecmd bosconfig`. When data is already local, regression skips prepare by default (needs `--prepare` / `prepare=true` to run).
- **baseline** `tests/embodied/baseline/<chip>/<name>.json` (**critical**): missing baseline means a direct FAIL. Whichever chip you run on,
  first collect all models in the manifest on that chip:
  ```bash
  chip=A800 auto_collect_baseline=true bash tests/embodied/run.sh
  ```
- **Auto-created**: `logs/` is generated automatically at runtime; no manual preparation needed.

> `BOS_VLA_ARTIFACTS_ROOT` in `config/env.sh` is still a `TODO` placeholder; before first onboarding, replace it with the real BOS source prefix.

## Regression Targets

The regression targets are defined by the `config/scripts.yaml` manifest, a YAML mapping of `<name>: <script path relative to examples/embodied>`.
Scripts are **executed verbatim** (`bash <script>`) without injecting training parameters; at runtime, only the environment variables
`OUTPUT_DIR` / `TENSORBOARD_DIR` are pointed to this run's log directory (the examples scripts all support overriding via
`${OUTPUT_DIR:-...}`), so that the `metrics.jsonl` flushed by the trainer can be read;
when missing, it falls back to parsing the stdout training log.

To add a regression model: add an executable training script under `examples/embodied/`, then add a line in
`config/scripts.yaml`; on the data side, you only need to upload it to the
corresponding family directory under `${BOS_VLA_ARTIFACTS_ROOT}`, and `config/prepare.sh` will sync it as a whole.
Also collect the baseline for the new name once with `auto_collect_baseline=true`.

## Directory Structure

```
tests/embodied/
├── run.sh                      # thin wrapper: source config/env.sh, then exec cli.py
├── cli.py                      # regression executor: parameter parsing + serial execution per manifest
├── config/                     # centralized configuration (the only directory that needs per-environment changes)
│   ├── env.sh                  #   centralized path configuration (data/logs/baseline/tools)
│   ├── scripts.yaml            #   regression script manifest: name → script under examples/embodied
│   └── prepare.sh              #   regression environment preparation: bcecmd bos sync vla_artifacts
├── execution/                  # execution layer: load manifest → execute scripts → parse metrics
│   ├── manifest.py             #   config/scripts.yaml manifest loading
│   ├── executor.py             #   single-script execution (timeout control, model-specific env injection)
│   └── metrics.py              #   metrics.jsonl / stdout log metric parsing
└── reporting/                  # result layer: baseline comparison + colored logging
    └── baseline.py             #   baseline read/write, tolerance comparison, write back baseline on performance improvement

# In-repo baseline collection (per-chip):
tests/embodied/baseline/<chip>/<name>.json

# Mutable per-run state (default outside the repo under ${EMBODIED_CI_ROOT}, see config/env.sh):
${EMBODIED_CI_ROOT}/logs/run_<timestamp>/    # per-run logs + results.json (EMBODIED_LOG_ROOT)
```

## Usage

Two entry points divide the work: `config/prepare.sh` prepares the environment (one-off/idempotent), and `run.sh`
runs the regression inside the container.

```bash
# Full regression (inside the container)
bash tests/embodied/run.sh --chip A800

# Run only one script (name from config/scripts.yaml) — choose either environment variable or command-line flag
chip=A800 model_names="pi05_ddp" bash tests/embodied/run.sh
bash tests/embodied/run.sh --chip A800 --models pi05_ddp

# Run multiple scripts
chip=A800 model_names="pi05_ddp groot_n1_6_ddp" bash tests/embodied/run.sh
bash tests/embodied/run.sh --chip A800 --models pi05_ddp groot_n1_6_ddp

# First-time baseline collection (no comparison, write the current result into tests/embodied/baseline/<chip>/)
chip=A800 auto_collect_baseline=true bash tests/embodied/run.sh
bash tests/embodied/run.sh --chip A800 --auto_collect_baseline

# Prepare the environment on-site (bos sync data), skipped by default, needs explicit enabling
bash tests/embodied/run.sh --chip A800 --prepare

# Skip the grad_norm hard check and only enforce loss (both are checked by default)
bash tests/embodied/run.sh --chip A800 --check_loss_only

# Only print commands without training, to verify the pipeline
chip=A800 dry_run=true bash tests/embodied/run.sh
bash tests/embodied/run.sh --chip A800 --dry_run

# Stop at the first FAIL (skip the remaining models, useful for debugging)
bash tests/embodied/run.sh --chip A800 --fail_fast

# List available names
bash tests/embodied/run.sh --chip A800 --list_models
```

## Parameter Interface

`run.sh` is a thin wrapper that only sources `config/env.sh` (to propagate path variables
to the training subprocesses) and delegates all parameter parsing, defaults, validation, and
optional `--prepare` execution to `cli.py`. Calling `cli.py` directly is equivalent —
it auto-sources `env.sh` on first use.

Every option supports two forms, with **CLI flags taking precedence over environment variables**:

| Environment variable | CLI flag | Default | Description |
|---|---|---|---|
| `model_names` | `--models MODEL ...` | empty (=all) | Names to regress (must be in `config/scripts.yaml`); env-var form is a space-separated string, e.g. `model_names="pi05_ddp groot_n1_6_ddp"` |
| `chip` | `--chip` | **required** | Chip model, `A800` / `P6K`; determines the baseline subdirectory `tests/embodied/baseline/<chip>/` |
| `TIMEOUT` | `--timeout` | `3600` | Per-script timeout (seconds); on timeout, SIGTERM→SIGKILL the process group |
| `accuracy_relative_tolerance` | `--accuracy_relative_tolerance` | `0.02` | Per-iteration relative-error threshold for loss-type metrics (hard check) |
| `performance_relative_tolerance` | `--performance_relative_tolerance` | `0.05` | Iteration time/throughput degradation threshold (soft check / used to write back the baseline) |
| `check_loss_only` | `--check_loss_only` | `false` | By default hard-check both loss and grad_norm; when set, grad_norm is skipped and only loss is hard-checked |
| `auto_collect_baseline` | `--auto_collect_baseline` | `false` | Collect the current result as the baseline, no comparison |
| `dry_run` | `--dry_run` | `false` | Only print commands, no training |
| `fail_fast` | `--fail_fast` | `false` | Abort the run on the first FAIL; the remaining models are skipped |
| `prepare` | `--prepare` | `false` | When set, run `config/prepare.sh` (bos sync) before regressing; skipped by default |
| `LOG_DIR` | `--log_dir` | `${EMBODIED_LOG_ROOT}/run_<ts>` | Log/result output directory |
| — | `--results_file` | `<log_dir>/results.json` | Result JSON output path (CLI only) |
| — | `--list_models` | off | List available names and exit; does not require `--chip` |

Boolean env vars accept `true` / `1` / `yes` / `on` (case-insensitive) as truthy; anything else is falsy.

Exit codes: `0` all passed, `1` run completed with failures, `2` the executor itself crashed
(or `--prepare` failed, or a preflight baseline check refused the run).

> Note: by default, both loss and grad_norm are hard-checked with the same tolerance. Pass `--check_loss_only`
> (or `check_loss_only=true`) to skip the grad_norm hard check and only enforce loss.

## Result Determination

- Non-zero script exit code -> FAIL
- No iteration metrics parsed, or fewer iterations than the baseline -> FAIL
- Loss-type metric per-iteration relative error exceeds `accuracy_relative_tolerance` (default 0.02) -> FAIL
  (grad_norm is hard-checked as well when `check_loss_only=false`)
- Numerical anomalies -> **FAIL** (hard check, independent of tolerance):
  - any hard-checked metric (loss / grad_norm) with a non-finite value (NaN / Inf) in either the actual run or the baseline
  - trainer-reported `nan_iterations > 0` or `skipped_iterations > 0` on any iteration (read from `metrics.jsonl`)
  - such runs will also never trigger the automatic performance-baseline write-back
- Iteration time/throughput degradation exceeds `performance_relative_tolerance` (default 0.05) -> warning only
- Regression passes and iteration time/throughput **improves** beyond that tolerance (with no performance metric degradation) -> automatically write back
  the performance metrics in the baseline to this run's better values (accuracy metrics are never auto-updated)
- Missing `tests/embodied/baseline/<chip>/<name>.json` -> **preflight refusal**: `cli.py` checks all selected
  models' baselines before launching any training subprocess. If any is missing, it lists the missing paths
  and exits with code `2` without running. Collect first with `--auto_collect_baseline`, or pass `--dry_run`
  to only verify the pipeline. When `--auto_collect_baseline` is set and a baseline already exists, a warning
  is emitted (the baseline will be overwritten).

Log directory layout: `<log_dir>/<name>/train.log` (stdout), `<log_dir>/<name>/output/`
(the script's OUTPUT_DIR, containing metrics.jsonl), `<log_dir>/results.json`,
`<log_dir>/regression.log` (executor-level log).
