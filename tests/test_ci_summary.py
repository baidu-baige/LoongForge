# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/summarize_suite_results.py"
_SPEC = importlib.util.spec_from_file_location("summarize_suite_results", _SCRIPT)
summarize = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summarize)


def _result(**overrides):
    base = {
        "model_name": "pi05_ddp",
        "script": "/repo/examples/embodied/pi05/run.sh",
        "passed": True,
        "failed_metrics": [],
        "warnings": [],
        "error": "",
        "log_dir": "/out/run_1/pi05_ddp",
        "duration_sec": 716.2,
        "metrics": [
            {"iteration": 1, "action_loss": 0.4305, "grad_norm": 1.6127},
            {"iteration": 2, "action_loss": 0.2690, "grad_norm": 1.7000},
        ],
    }
    base.update(overrides)
    return base


def test_condense_warning_matches_metric_pattern():
    note = (
        "throughput: actual_mean=4.40 baseline_mean=14.47 degraded 69.6% > 5% "
        "(soft check, warning only)"
    )
    assert summarize._condense_warning(note) == "throughput degraded 69.6%"


def test_condense_warning_keeps_direction_for_time_metrics():
    # baseline.py reports a direction-normalised degradation magnitude, so a
    # leading "-" would render this 248% slowdown as an improvement.
    note = (
        "elapsed_time_ms: actual_mean=10.00 baseline_mean=2.87 degraded 248.4% > 5% "
        "(soft check, warning only)"
    )
    assert summarize._condense_warning(note) == "elapsed_time_ms degraded 248.4%"


def test_condense_warning_falls_back_to_excerpt_for_unknown_text():
    note = "something unexpected happened in the framework just now"
    condensed = summarize._condense_warning(note)
    assert condensed == note  # short enough to pass through unchanged
    long_note = "x" * (summarize.NOTE_EXCERPT_CHARS + 10)
    assert summarize._condense_warning(long_note).endswith("...")


def test_condense_warning_ignores_empty_entries():
    assert summarize._condense(["", "  "]) == ""


def test_render_omits_chip_and_keeps_loss_series():
    summary = {
        "finished_at": "2026-08-28 12:12:39",
        "chip": "p",
        "log_dir": "/out/run_1",
        "auto_collect_baseline": False,
        "results": [_result()],
    }
    text = summarize.render(summary)
    assert "Regression results:" in text
    assert "chip" not in text
    assert "action_loss: 0.4305 → 0.269" in text
    assert "PASS" in text
    assert "1/1 models passed" in text


def test_render_marks_failures_with_notes():
    summary = {
        "chip": "p",
        "results": [
            _result(
                passed=False,
                failed_metrics=["action_loss@iter2: rel diff 0.21 exceeds tolerance"],
                warnings=[
                    "throughput: actual_mean=4.40 baseline_mean=14.47 degraded 69.6% > 5% (soft check, warning only)"
                ],
                error="train exit 1",
            )
        ],
    }
    text = summarize.render(summary)
    assert "FAIL" in text
    assert "failed: action_loss@iter2" in text
    assert "warn: throughput degraded 69.6%" in text
    assert "error: train exit 1" in text


def test_main_prints_fallback_without_failing(tmp_path, capsys, monkeypatch):
    broken = tmp_path / "results.json"
    broken.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["summarize", str(broken)])
    assert summarize.main() == 0
    assert "Suite detail unavailable" in capsys.readouterr().out
