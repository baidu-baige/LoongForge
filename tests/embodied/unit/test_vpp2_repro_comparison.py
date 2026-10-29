# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic fixtures test gate validation only; they are NOT baselines."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "compare_repro", Path(__file__).resolve().parents[3] / "examples/embodied/vpp2/compare_repro.py"
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


@pytest.fixture
def runs(tmp_path):
    paths = [tmp_path / "base", tmp_path / "trans"]
    for side, path in enumerate(paths):
        path.mkdir()
        manifest = {
            key: 1
            for key in [
                "steps",
                "schedule_steps",
                "world_size",
                "per_device_batch",
                "ga",
                "global_batch",
                "seed",
                "num_workers",
                "internal_chunk",
                "dtype",
                "autocast",
                "engine",
                "dataset_roots",
                "stats_sha256",
            ]
        }
        manifest["side"] = "base" if side == 0 else "loongforge"
        (path / "run_manifest.json").write_text(json.dumps(manifest))
        for name in [
            "asset_sha256.json",
            "dataset_meta_sha256.json",
            "initial_trainable.json",
            "effective_deepspeed.json",
        ]:
            (path / name).write_text("{}")
        (path / "pip_freeze.txt").write_text(
            "\n".join(
                n + "==1"
                for n in ["torch", "numpy", "deepspeed", "accelerate", "diffusers", "transformers", "torchvision", "av"]
            )
        )
        (path / "dependency_versions.json").write_text(
            json.dumps(
                {
                    name: "1"
                    for name in [
                        "torch",
                        "numpy",
                        "deepspeed",
                        "accelerate",
                        "diffusers",
                        "transformers",
                        "torchvision",
                        "av",
                    ]
                }
            )
        )
        rows = [
            dict(iteration=i, action_loss=1.0 + side * 1e-4, video_loss=0, learning_rate=1e-6, grad_norm=2)
            for i in range(1, 101)
        ]
        (path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return paths


def test_gate_accepts_complete_pair_and_rejects_partial_or_nan(runs):
    assert comparison.compare(*runs, 100, 0.001)["pass_100_step_gate"]
    path = runs[1] / "metrics.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:99]))
    with pytest.raises(ValueError, match="missing optimizer"):
        comparison.compare(*runs, 100, 0.001)
    path.write_text("\n".join(lines).replace("1.0001", "NaN"))
    with pytest.raises(ValueError, match="nonfinite"):
        comparison.compare(*runs, 100, 0.001)


def test_gate_rejects_configuration_or_weight_mismatch(runs):
    (runs[1] / "initial_trainable.json").write_text('{"sha256":"different"}')
    with pytest.raises(ValueError, match="inputs/weights"):
        comparison.compare(*runs, 100, 0.001)


def test_gate_loss_above_threshold_is_not_a_pass(runs):
    path = runs[1] / "metrics.jsonl"
    path.write_text(path.read_text().replace("1.0001", "1.1"))
    assert not comparison.compare(*runs, 100, 0.001)["pass_100_step_gate"]


def test_gate_cannot_compare_a_run_to_itself(runs):
    with pytest.raises(ValueError, match="independent run"):
        comparison.compare(runs[0], runs[0], 100, 0.001)


def test_gate_rejects_different_deterministic_mode(runs):
    path = runs[1] / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["deterministic"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="deterministic"):
        comparison.compare(*runs, 100, 0.001)
