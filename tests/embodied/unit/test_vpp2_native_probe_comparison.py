# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic comparator contract tests; never treated as training evidence."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "compare_native_probe", Path(__file__).resolve().parents[3] / "examples/embodied/vpp2/compare_native_probe.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture
def runs(tmp_path):
    roots = [tmp_path / "base", tmp_path / "trans"]
    for root in roots:
        root.mkdir()
        (root / "scope.json").write_text(json.dumps({"world": 2}))
        (root / "initial_trainable.json").write_text(json.dumps({"sha256": "fixture"}))
        torch.save({"weight": torch.ones(2)}, root / "final.pt")
        for rank in range(2):
            (root / f"inputs_rank{rank}.jsonl").write_text(
                "\n".join(json.dumps({"x": {"sha256": str(i)}}) for i in range(3))
            )
    base_rows = [{"train": {"iteration": i, "action_loss": i * 0.1, "video_loss": 0,
                              "learning_rate": 1e-4, "grad_norm": 2}} for i in range(1, 4)]
    trans_rows = [dict(step=i, action_loss=i * 0.1, video_loss=0, lr=1e-4, grad_norm=2,
                       nan_iterations=0, skipped_iterations=0) for i in range(1, 4)]
    for root, rows in zip(roots, (base_rows, trans_rows)):
        (root / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
    return roots


def test_normalizes_native_logger_fields_without_rounding(runs):
    report = probe.compare(*runs, steps=3)
    assert report["exact_native_tiny_pass"]
    assert report["action_loss_mae"] == 0
    assert report["canonical_zero2_pass"] is False


@pytest.mark.parametrize("problem", ["duplicate", "missing", "nan", "skipped"])
def test_rejects_invalid_training_updates(runs, problem):
    path = runs[1] / "metrics.jsonl"
    rows = probe.lines(path)
    if problem == "duplicate":
        rows.append(rows[0])
    elif problem == "missing":
        rows.pop()
    elif problem == "nan":
        rows[0]["action_loss"] = float("nan")
    else:
        rows[0]["skipped_iterations"] = 1
    path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError):
        probe.compare(*runs, steps=3)


def test_resume_checks_inputs_not_just_metrics_and_final_weights(runs):
    trans = runs[1]
    for name in ("metrics.jsonl", "inputs_rank0.jsonl", "inputs_rank1.jsonl"):
        path = trans / name
        path.write_text("\n".join(path.read_text().splitlines()[1:]))
    assert probe.compare(*runs, steps=3, start=2)["exact_native_tiny_pass"]
    (trans / "inputs_rank1.jsonl").write_text('{}\n{}\n')
    report = probe.compare(*runs, steps=3, start=2)
    assert not report["exact_native_tiny_pass"]
    assert not report["input_exact_by_rank"]["1"]


def test_detects_final_weight_difference(runs):
    torch.save({"weight": torch.ones(2) + 0.1}, runs[1] / "final.pt")
    report = probe.compare(*runs, steps=3)
    assert not report["exact_native_tiny_pass"]
    assert report["final_parameters_max_abs"] > 0
