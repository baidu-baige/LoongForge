# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training output -> structured metrics.

Two sources (the execution layer prefers JSONL, falling back to the stdout log when missing):

1. <OUTPUT_DIR>/metrics.jsonl (flushed by the embodied trainer, the most reliable):
   {"step_time": 15.098, "step": 1, "action_loss": 0.4337, "lr": 5e-08,
    "samples_per_sec": 0.53, "grad_norm": 4.1888, ...}

2. stdout training log (aligned with the Megatron format, loongforge/embodied/train/utils/logging.py):
   iteration        1/      20 | consumed samples: 8 | elapsed time per iteration (ms): 15098.4 |
   throughput (samples/sec/per_device): 0.530 | learning rate: 5.000000E-08 |
   global batch size: 64 | action loss: 4.337645E-01 | loss scale: 1.0 | grad norm: 4.188855 | ...

   Note: loss keys are printed with spaces in the log (key.replace("_", " ")), and are restored to the underscore form after parsing.

Unified output record format (aligned with the baseline JSON):
   {"iteration": 1, "elapsed_time_ms": 15098.4, "throughput": 0.53,
    "learning_rate": 5e-08, "action_loss": 0.4337, "grad_norm": 4.1888}
"""

import json
import re

_NUM = r"([-+]?[\d.]+(?:[eE][-+]?\d+)?)"

ITER_RE = re.compile(r"iteration\s+(\d+)\s*/\s*\d+\s*\|")
FIELD_RES = {
    "elapsed_time_ms": re.compile(r"elapsed time per iteration \(ms\):\s*" + _NUM),
    "throughput": re.compile(r"throughput \(samples/sec/per_device\):\s*" + _NUM),
    "grad_norm": re.compile(r"grad norm:\s*" + _NUM),
    "learning_rate": re.compile(r"learning rate:\s*" + _NUM),
}
# Match "| xxx loss: 1.2E+00"; the lookbehind keeps the separator for matching adjacent loss fields;
# "loss scale" is not matched because there is no colon after loss
LOSS_RE = re.compile(r"(?<=\|)\s*((?:[a-z][a-z ]*?)?loss):\s*" + _NUM)


def parse_line(line):
    m = ITER_RE.search(line)
    if not m:
        return None
    record = {"iteration": int(m.group(1))}
    for key, pattern in FIELD_RES.items():
        fm = pattern.search(line)
        if fm:
            record[key] = float(fm.group(1))
    for lm in LOSS_RE.finditer(line):
        key = lm.group(1).strip().replace(" ", "_")
        record[key] = float(lm.group(2))
    return record


def parse_log(log_file):
    """Parse the stdout training log and return a list of metric records ordered by iteration.

    When the same iteration appears multiple times, the last one is kept (consistent with parse_jsonl behavior;
    the more complete record appended by a resume rerun overrides an earlier partial write).
    """
    records = {}
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            record = parse_line(line)
            if record and len(record) > 1:
                records[record["iteration"]] = record
    return [records[it] for it in sorted(records)]


# metrics.jsonl field -> unified record field
_JSONL_KEY_MAP = {
    "lr": "learning_rate",
    "samples_per_sec": "throughput",
}
# Fields we deliberately drop from records: step is already the iteration key, step_time is
# renamed to elapsed_time_ms, consumed_samples is bookkeeping. nan_iterations /
# skipped_iterations are trainer-reported instability signals — kept so compare() can hard-fail
# when either is >0.
_JSONL_SKIP = {"step", "step_time", "consumed_samples"}


def parse_jsonl(jsonl_file):
    """Parse the metrics.jsonl flushed by the trainer.

    Duplicate steps (appended by a resume rerun) keep the last one.
    """
    records = {}
    with open(jsonl_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" not in raw:
                continue
            record = {"iteration": int(raw["step"])}
            if isinstance(raw.get("step_time"), (int, float)):
                record["elapsed_time_ms"] = raw["step_time"] * 1000.0
            for key, value in raw.items():
                # Exclude bool (in Python bool is a subclass of int, and float(True)=1.0 would pollute the metrics)
                if key in _JSONL_SKIP or not isinstance(value, (int, float)) \
                        or isinstance(value, bool):
                    continue
                record[_JSONL_KEY_MAP.get(key, key)] = float(value)
            records[record["iteration"]] = record
    return [records[it] for it in sorted(records)]
