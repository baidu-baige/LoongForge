# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Single-script regression execution: directly run a script under examples/embodied -> timeout control -> metric parsing and baseline comparison.

Scripts are executed verbatim (bash <script>) without injecting training parameters; only the
OUTPUT_DIR / TENSORBOARD_DIR environment variables are pointed to this run's log directory (the examples
scripts all support overriding via ``${OUTPUT_DIR:-...}``), so that the metrics.jsonl flushed by the trainer can be read.
"""

import os
import re
import signal
import subprocess
import time

from execution import metrics as metrics_mod
from reporting import baseline as baseline_mod

# The top-level key of the baseline JSON (was training_type in the original framework, now unified to train)
BASELINE_KEY = "train"

# Valid shell variable name: starts with a letter/underscore, followed by letters/digits/underscores.
# When a model name starts with a digit (e.g. "5b..."), the derived variable name starts with a digit and bash cannot reference it,
# so it is skipped here with a warning, to avoid silently injecting a variable that the script cannot use.
_VALID_BASH_VAR = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _inject_model_env(env, model_name, logger):
    """Inject model-specific paths: EMBODIED_<MODEL NAME UPPERCASE>_<VAR> -> <VAR> (see config/env.sh)."""
    prefix = "EMBODIED_" + re.sub(r"[^0-9A-Za-z]", "_", model_name).upper() + "_"
    for key, value in os.environ.items():
        if key.startswith(prefix) and value:
            var = key[len(prefix):]
            if not _VALID_BASH_VAR.match(var):
                logger.warning(
                    f"[{model_name}] Skipping invalid variable name {var} (from {key}): "
                    "a model name starting with a digit makes bash unable to reference it; ignored")
                continue
            env[var] = value
            logger.info(f"[{model_name}] Injected model-specific environment variable {var}={value}")


def _run_with_timeout(cmd, env, cwd, log_file, timeout, logger):
    """Run the command, flushing stdout/stderr to log_file; on timeout, SIGTERM->SIGKILL the whole process group."""
    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd, env=env, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout ({timeout}s); terminating process group pgid={proc.pid}")
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=60)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return -signal.SIGTERM


def run_script(model_name, script_path, args, log_dir, logger):
    """Execute one regression script and return a result dict.

    The baseline path is determined by reporting.baseline (EMBODIED_BASELINE_ROOT env var
    when set, otherwise the in-repo tests/baseline/embodied default), so no repo-root
    argument is needed here.
    """
    out_dir = os.path.join(log_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)
    output_dir = os.path.join(out_dir, "output")

    result = {
        "model_name": model_name,
        "script": script_path,
        "passed": True,
        "failed_metrics": [],
        "warnings": [],
        "error": "",
        "log_dir": out_dir,
        "duration_sec": 0,
        "metrics": [],
    }
    start = time.time()

    if not os.path.exists(script_path):
        result["passed"] = False
        result["error"] = f"Script does not exist: {script_path}"
        logger.error(result["error"])
        return result

    env = os.environ.copy()
    env["OUTPUT_DIR"] = output_dir
    env["TENSORBOARD_DIR"] = os.path.join(out_dir, "tensorboard")
    _inject_model_env(env, model_name, logger)

    cmd = ["bash", script_path]
    log_file = os.path.join(out_dir, "train.log")
    logger.info(f"[{model_name}] Executing script {script_path}, log: {log_file}")

    if args.dry_run:
        logger.info(f"[{model_name}] dry_run: OUTPUT_DIR={output_dir} bash {script_path}")
        result["duration_sec"] = round(time.time() - start, 1)
        return result

    rc = _run_with_timeout(cmd, env, os.path.dirname(script_path),
                           log_file, args.timeout, logger)
    if rc != 0:
        result["passed"] = False
        result["error"] = f"Script exit code {rc}, see {log_file}"
        logger.error(result["error"])
        result["duration_sec"] = round(time.time() - start, 1)
        return result

    # Metric parsing: prefer the metrics.jsonl flushed by the trainer, fall back to the stdout log when missing
    records = []
    jsonl_file = os.path.join(output_dir, "metrics.jsonl")
    if os.path.exists(jsonl_file):
        records = metrics_mod.parse_jsonl(jsonl_file)
        logger.info(f"[{model_name}] Parsed {len(records)} metrics from {jsonl_file}")
    if not records:
        records = metrics_mod.parse_log(log_file)
    result["metrics"] = records
    if not records:
        result["passed"] = False
        result["error"] = "No iteration metrics were parsed (both jsonl and log are empty)"
        logger.error(f"[{model_name}] {result['error']}")
        result["duration_sec"] = round(time.time() - start, 1)
        return result

    if args.auto_collect_baseline:
        path = baseline_mod.save_baseline(
            args.chip, model_name, BASELINE_KEY, records)
        logger.info(f"[{model_name}] baseline collected: {path}")
        result["duration_sec"] = round(time.time() - start, 1)
        return result

    expected = baseline_mod.load_baseline(
        args.chip, model_name, BASELINE_KEY)
    if expected is None:
        result["passed"] = False
        result["error"] = (
            f"Missing baseline: {baseline_mod.baseline_path(args.chip, model_name)} "
            f"(collect it with --auto_collect_baseline)"
        )
        logger.error(f"[{model_name}] {result['error']}")
        result["duration_sec"] = round(time.time() - start, 1)
        return result

    failed, warnings = baseline_mod.compare(
        records, expected,
        accuracy_tol=args.accuracy_relative_tolerance,
        performance_tol=args.performance_relative_tolerance,
        check_loss_only=args.check_loss_only,
    )
    result["failed_metrics"] = failed
    result["warnings"] = warnings
    for w in warnings:
        logger.warning(f"[{model_name}] {w}")
    if failed:
        result["passed"] = False
        for item in failed:
            logger.error(f"[{model_name}] {item}")
    else:
        logger.info(f"[{model_name}] Metric validation passed ({len(records)} iterations)")
        # When performance improved (beyond tolerance and with no degradation), update the baseline's performance metrics to this run's values
        updated = baseline_mod.update_perf_baseline(
            args.chip, model_name, BASELINE_KEY, records, expected,
            performance_tol=args.performance_relative_tolerance)
        if updated:
            logger.info(f"[{model_name}] Performance better than baseline; performance metrics written back: {updated}")

    result["duration_sec"] = round(time.time() - start, 1)
    return result
