# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the custom cosine scheduler's lr_decay_start_iters parameter.

The default lr_decay_start_iters=0 must reproduce the pre-change schedule
(cosine decay beginning immediately after warmup); a value above the warmup
length adds a constant-LR hold before the decay.
"""

import math
from types import SimpleNamespace

import torch

from loongforge.embodied.optimizer.lr_scheduler import build_scheduler

PEAK_LR = 5e-5
MIN_LR = 1e-5
WARMUP_ITERS = 10000
TRAIN_ITERS = 30000


def _args(lr_decay_start_iters):
    return SimpleNamespace(
        lr_decay_style="cosine_with_min_lr",
        custom_lr_lambda=True,
        lr_warmup_iters=WARMUP_ITERS,
        min_lr=MIN_LR,
        lr_decay_iters=None,
        train_iters=TRAIN_ITERS,
        lr_decay_start_iters=lr_decay_start_iters,
    )


def _build_lambda(lr_decay_start_iters):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=PEAK_LR)
    scheduler = build_scheduler(optimizer, _args(lr_decay_start_iters))
    assert len(scheduler.lr_lambdas) == 1
    return scheduler.lr_lambdas[0]


def _reference_lambda(step):
    """Pre-change schedule: warmup, then immediate cosine decay to end_lr."""
    if step < WARMUP_ITERS:
        init_lr = PEAK_LR / (WARMUP_ITERS + 1)
        current_lr = init_lr + (PEAK_LR - init_lr) * step / WARMUP_ITERS
        return current_lr / PEAK_LR
    decay_steps = TRAIN_ITERS - WARMUP_ITERS
    progress = min(1.0, (step - WARMUP_ITERS) / max(1, decay_steps))
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    current_lr = MIN_LR + (PEAK_LR - MIN_LR) * cos
    return current_lr / PEAK_LR


def test_default_decay_start_matches_pre_change_curve():
    """lr_decay_start_iters=0 reproduces the original curve at every step."""
    current = _build_lambda(0)
    for step in range(TRAIN_ITERS + 1):
        assert math.isclose(
            current(step), _reference_lambda(step), rel_tol=0.0, abs_tol=1e-12
        ), f"LR multiplier diverged at step {step}"


def test_nonzero_decay_start_holds_peak_then_differs():
    """A decay start above warmup holds the peak LR and departs from the original."""
    held = _build_lambda(20000)
    # Warmup end through the decay start stays exactly at the peak LR.
    assert held(WARMUP_ITERS) == 1.0
    assert held(15000) == 1.0
    assert held(19999) == 1.0
    # The hold makes it strictly higher than the original, which is already decaying.
    assert _reference_lambda(15000) < 1.0
    assert held(15000) > _reference_lambda(15000)
    # Decay begins at the configured step and reaches min_lr at the end.
    assert held(20000) == 1.0
    assert held(20001) < 1.0
    assert math.isclose(held(TRAIN_ITERS), MIN_LR / PEAK_LR, rel_tol=1e-9)
