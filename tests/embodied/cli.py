#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Embodied regression executor (runs inside the container, manual trigger entry point).

The regression targets are defined by the config/scripts.yaml manifest (name -> script relative path); scripts are executed verbatim.

Usage examples:
    python3 tests/embodied/cli.py --models pi05 groot_n1_6
    python3 tests/embodied/cli.py --auto_collect_baseline   # collect baseline on first run
    python3 tests/embodied/cli.py --dry_run                 # only print commands, do not train
"""

import argparse
import json
import os
import subprocess
import sys
import time

# This file lives at tests/embodied/cli.py; EMBODIED_ROOT is this suite's self-contained
# root (config/ + execution/ + reporting/). The embodied suite has no dependency on the
# sibling tests/llm_vlm/ suite.
EMBODIED_ROOT = os.path.dirname(os.path.abspath(__file__))
if EMBODIED_ROOT not in sys.path:
    sys.path.insert(0, EMBODIED_ROOT)

from reporting.log import create_logger  # noqa: E402
from reporting.baseline import baseline_path, load_baseline  # noqa: E402
from execution.manifest import examples_dir, load_manifest  # noqa: E402
from execution.executor import BASELINE_KEY, run_script  # noqa: E402

# Assigned in main() once the run directory is known, so the logger's file handler
# can point at <log_dir>/regression.log.
logger = None


def _ensure_env_loaded():
    """Idempotently source config/env.sh into os.environ.

    run.sh sources env.sh before exec-ing this file; when this file is called
    directly (python3 tests/embodied/cli.py ...), env.sh has not been sourced
    yet, so training subprocesses would miss ${LOCAL_VLA_ARTIFACTS_ROOT} and
    friends. This function fills that gap. No-op when env.sh has already been
    sourced (detected via the EMBODIED_CI_ROOT marker).
    """
    if os.environ.get("EMBODIED_CI_ROOT"):
        return
    env_sh = os.path.join(EMBODIED_ROOT, "config", "env.sh")
    if not os.path.exists(env_sh):
        return
    try:
        out = subprocess.check_output(
            ["bash", "-c", f'set -a; source "{env_sh}"; env -0'])
    except (subprocess.CalledProcessError, OSError):
        return
    for chunk in out.split(b"\0"):
        if not chunk:
            continue
        try:
            key, val = chunk.decode("utf-8", errors="replace").split("=", 1)
        except ValueError:
            continue
        # Do not overwrite variables the user has already set explicitly.
        os.environ.setdefault(key, val)


def _env_flag(name, default=False):
    """Read a boolean-like environment variable ('true'/'1'/'yes'/'on', case-insensitive)."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "on")


# Load env.sh at import time so DEFAULT_LOG_ROOT and other env-var-derived
# defaults below reflect the same paths that run.sh would have sourced.
_ensure_env_loaded()

# Root directory for regression logs (override per run with --log_dir)
DEFAULT_LOG_ROOT = os.environ.get(
    "EMBODIED_LOG_ROOT", "/ssd2/loongforge_embodied_ci/logs")


def parse_args(available_models):
    parser = argparse.ArgumentParser(
        description="Embodied regression runner. All flags below also accept an environment-variable form "
                    "(shown in each help string); CLI flags take precedence over env vars.")

    # --- Env-var defaults (mirror the historical shell-entry conventions) ---
    env_models = os.environ.get("model_names", "").split() or None
    env_chip = os.environ.get("chip") or None
    env_timeout = int(os.environ.get("TIMEOUT") or "3600")
    env_acc_tol = float(os.environ.get("accuracy_relative_tolerance") or "0.02")
    env_perf_tol = float(os.environ.get("performance_relative_tolerance") or "0.05")
    env_check_loss = _env_flag("check_loss_only", default=False)
    env_auto_collect = _env_flag("auto_collect_baseline")
    env_dry_run = _env_flag("dry_run")
    env_fail_fast = _env_flag("fail_fast")
    env_prepare = _env_flag("prepare")
    env_log_dir = os.environ.get("LOG_DIR") or None

    parser.add_argument("--models", nargs="*", default=env_models,
                        choices=available_models, metavar="MODEL",
                        help="Names to regress (defined in config/scripts.yaml); "
                             f"options: {', '.join(available_models)}; default is all. "
                             "Env var: model_names (space-separated string).")
    parser.add_argument("--chip", default=env_chip,
                        choices=["A800", "P6K"],
                        help="Chip model; determines the baseline subdirectory "
                             "metrics_baseline/<chip>/. Env var: chip. Required.")
    parser.add_argument("--timeout", type=int, default=env_timeout,
                        help="Per-script timeout (seconds). Env var: TIMEOUT.")
    parser.add_argument("--accuracy_relative_tolerance", type=float, default=env_acc_tol,
                        help="Env var: accuracy_relative_tolerance.")
    parser.add_argument("--performance_relative_tolerance", type=float, default=env_perf_tol,
                        help="Env var: performance_relative_tolerance.")
    parser.add_argument("--check_loss_only", action="store_true", default=env_check_loss,
                        help="Hard-check loss only for accuracy (grad_norm skipped). "
                             "Default: off (both loss and grad_norm are hard-checked). "
                             "Env var: check_loss_only=true.")
    parser.add_argument("--auto_collect_baseline", action="store_true", default=env_auto_collect,
                        help="Collect the current result as the baseline; do not compare. "
                             "Env var: auto_collect_baseline.")
    parser.add_argument("--dry_run", action="store_true", default=env_dry_run,
                        help="Only print commands, do not run training. Env var: dry_run.")
    parser.add_argument("--fail_fast", action="store_true", default=env_fail_fast,
                        help="Abort the run as soon as any model fails (skip the remaining models). "
                             "Env var: fail_fast.")
    parser.add_argument("--prepare", action="store_true", default=env_prepare,
                        help="Run config/prepare.sh (bcecmd bos sync vla_artifacts) before regressing. "
                             "Env var: prepare.")
    parser.add_argument("--log_dir", default=env_log_dir,
                        help=f"Log/result output directory, default {DEFAULT_LOG_ROOT}/run_<ts>. "
                             "Env var: LOG_DIR.")
    parser.add_argument("--results_file", default=None,
                        help="results.json output path, default <log_dir>/results.json")
    parser.add_argument("--list_models", action="store_true",
                        help="List available names and exit (does not require --chip)")
    args = parser.parse_args()

    # --list_models is a pure query; skip the rest of the validation.
    if args.list_models:
        return args

    # --models given but without any name -> raise an explicit error to avoid silently running everything.
    if args.models is not None and len(args.models) == 0:
        parser.error("--models requires at least one name (or omit it to regress all)")

    # argparse's choices only validates values coming from the command line; env-derived
    # defaults skip that check, so re-verify them here.
    if args.models:
        invalid = [m for m in args.models if m not in available_models]
        if invalid:
            parser.error(
                f"unknown model(s) from model_names env var: {invalid}; "
                f"available: {', '.join(available_models)}")

    if args.chip is None:
        parser.error("--chip is required (or set the 'chip' environment variable to A800/P6K)")
    if args.chip not in ("A800", "P6K"):
        parser.error(
            f"invalid chip {args.chip!r} from env var; allowed: A800, P6K")

    return args


def main():
    manifest = load_manifest(EMBODIED_ROOT)
    available = list(manifest)
    args = parse_args(available)

    if args.list_models:
        print("\n".join(available))
        return 0

    # Resolve the run directory before anything logs, so create_logger can
    # persist the whole run (preflight included) into <log_dir>/regression.log
    # alongside results.json instead of only writing to stdout.
    if not args.log_dir:
        args.log_dir = os.path.join(
            DEFAULT_LOG_ROOT, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(args.log_dir, exist_ok=True)

    global logger
    logger = create_logger(log_dir=args.log_dir,
                           name="embodied_regression",
                           logfile_name="regression.log")

    if args.prepare:
        prepare_sh = os.path.join(EMBODIED_ROOT, "config", "prepare.sh")
        if os.path.exists(prepare_sh):
            logger.info(f"Running preparation script: {prepare_sh}")
            rc = subprocess.run(["bash", prepare_sh]).returncode
            if rc != 0:
                logger.error(f"Preparation script failed (exit={rc}); aborting")
                return 2
        else:
            logger.warning(
                f"--prepare requested but {prepare_sh} not found; skipping")

    models = args.models if args.models else available

    # Preflight baseline check: fail fast before spawning any training subprocess.
    # - normal regression run: every selected model must have a baseline; missing -> exit 2
    # - --auto_collect_baseline: warn on overwriting any existing baseline (never silent clobber)
    # - --dry_run: skipped (pipeline verification should not require baselines)
    if args.auto_collect_baseline:
        for m in models:
            if load_baseline(args.chip, m, BASELINE_KEY) is not None:
                logger.warning(
                    f"[{m}] baseline already exists at {baseline_path(args.chip, m)}; "
                    "--auto_collect_baseline will overwrite it")
    elif not args.dry_run:
        missing = [(m, baseline_path(args.chip, m)) for m in models
                   if load_baseline(args.chip, m, BASELINE_KEY) is None]
        if missing:
            logger.error(f"Missing baseline for chip={args.chip}:")
            for m, p in missing:
                logger.error(f"  - {m} -> {p}")
            logger.error(
                "Refusing to run. Either collect baselines first with "
                "--auto_collect_baseline, or use --dry_run to only verify the pipeline.")
            return 2

    scripts_root = examples_dir(EMBODIED_ROOT)
    logger.info(f"Regression script directory: {scripts_root}")
    logger.info(f"Regression targets: {models}")
    logger.info(f"Log directory: {args.log_dir}  chip={args.chip}  "
                f"acc_tol={args.accuracy_relative_tolerance}  "
                f"perf_tol={args.performance_relative_tolerance}  "
                f"dry_run={args.dry_run}")

    results = []
    for index, model_name in enumerate(models, 1):
        logger.info(f"[{index}/{len(models)}] Starting regression for {model_name}")
        script_path = os.path.join(scripts_root, manifest[model_name])
        result = run_script(model_name, script_path, args, args.log_dir, logger)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        logger.info(f"[{index}/{len(models)}] {model_name} finished: {status} "
                    f"({result['duration_sec']}s)")
        if not result["passed"] and args.fail_fast:
            remaining = len(models) - index
            if remaining > 0:
                logger.warning(
                    f"--fail_fast: aborting run after {model_name} FAIL "
                    f"({remaining} model(s) skipped)")
            break

    results_file = args.results_file or os.path.join(args.log_dir, "results.json")
    summary = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chip": args.chip,
        "log_dir": args.log_dir,
        "auto_collect_baseline": args.auto_collect_baseline,
        "results": results,
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Results written to {results_file}")

    failed = [r["model_name"] for r in results if not r["passed"]]
    total_run = len(results)
    logger.info(f"Regression complete: ran {total_run}, failed {len(failed)}"
                + (f" ({failed})" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        # On the executor's own exception, return exit code 2 to distinguish it from
        # "run completed but with failures" (0/1).
        if logger is not None:
            logger.exception("Executor exited abnormally")
        else:
            import traceback
            traceback.print_exc()
        sys.exit(2)
