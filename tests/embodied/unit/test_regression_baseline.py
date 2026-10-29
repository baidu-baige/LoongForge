# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the embodied metric-comparison gates."""

from tests.embodied.reporting.baseline import compare


def _records(values):
    return [
        {"iteration": index, "action_loss": value}
        for index, value in enumerate(values, start=1)
    ]


def test_loss_mae_gate_passes_at_threshold():
    failed, _ = compare(
        _records([1.001, 1.001]),
        _records([1.0, 1.0]),
        accuracy_tol=0.02,
        performance_tol=0.05,
        loss_mae_tolerance=0.001,
    )
    assert failed == []


def test_loss_mae_gate_fails_above_threshold():
    failed, _ = compare(
        _records([1.0021, 1.0021]),
        _records([1.0, 1.0]),
        accuracy_tol=0.02,
        performance_tol=0.05,
        loss_mae_tolerance=0.001,
    )
    assert failed == [
        "action_loss: MAE=0.0021 > 0.001 over 2 aligned iterations"
    ]


def test_loss_mae_gate_is_reported_with_pointwise_failure():
    failed, _ = compare(
        _records([1.1, 1.1]),
        _records([1.0, 1.0]),
        accuracy_tol=0.02,
        performance_tol=0.05,
        loss_mae_tolerance=0.001,
    )
    assert failed[-1] == "action_loss: MAE=0.1 > 0.001 over 2 aligned iterations"
