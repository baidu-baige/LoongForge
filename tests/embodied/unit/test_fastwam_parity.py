# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FastWAM parity with the official implementation.

Covers the FastWAM-style action/proprio normalization (``data.norm_stats_path``),
the FastWAM action scheduler default and the ``linear_warmup_cosine_annealing``
LR schedule.
"""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from loongforge.embodied.data.datasets.fastwam.transforms.fastwam_transform import (
    FastWAMLinearNormalizeTransform,
    _fastwam_linear_params,
)
from loongforge.embodied.model.fastwam.modeling_configuration_fastwam import FastWAMModelConfig


def _stats(lo, hi, extra=None):
    out = {"global_min": lo, "global_max": hi, "global_mean": [0.0] * len(lo), "global_std": [1.0] * len(lo)}
    out.update(extra or {})
    return out


def test_minmax_params_map_range_to_unit_interval_and_skip_constant_dims():
    scale, offset = _fastwam_linear_params(_stats([0.0, -2.0, 3.0], [4.0, 2.0, 3.0]), "min/max")
    x = torch.tensor([[0.0, -2.0, 3.0], [4.0, 2.0, 3.0]])
    y = x * scale + offset
    assert torch.allclose(y[:, :2], torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))
    # Constant dim (range < 1e-4): scale 1, offset -min -> value shifted to 0.
    assert scale[2] == 1.0 and y[0, 2] == 0.0


def test_unknown_norm_mode_raises():
    with pytest.raises(ValueError):
        _fastwam_linear_params(_stats([0.0], [1.0]), "q99")


@pytest.fixture
def stats_file(tmp_path):
    stats = {
        "action": {"default": _stats([0.0, 0.0], [2.0, 2.0])},
        "state": {"default": _stats([-1.0], [1.0])},
    }
    path = tmp_path / "dataset_stats.json"
    path.write_text(json.dumps(stats))
    return str(path)


def test_normalize_zeroes_padded_delta_dims_and_clamps(stats_file):
    transform = FastWAMLinearNormalizeTransform(stats_file, delta_action_dim_mask=[True, False])
    data = {
        "action": torch.tensor([[2.0, 2.0], [2.0, 2.0], [100.0, 0.0]]),
        "action_is_pad": torch.tensor([False, True, False]),
        "observation.state": torch.tensor([[0.5]]),
    }
    out = transform.apply(data)
    # Padded step: delta dim 0 is zeroed before normalization (0 -> -1); dim 1 is kept.
    expected = torch.tensor([[1.0, 1.0], [-1.0, 1.0], [5.0, -1.0]])
    assert torch.allclose(out["action"], expected)
    assert torch.allclose(out["observation.state"], torch.tensor([[0.5]]))


def test_normalize_unapply_roundtrip(stats_file):
    transform = FastWAMLinearNormalizeTransform(stats_file)
    action = torch.tensor([[0.5, 1.5]])
    out = transform.unapply(transform.apply({"action": action.clone()}))
    assert torch.allclose(out["action"], action)


def test_action_scheduler_defaults_match_fastwam():
    cfg = FastWAMModelConfig.__dataclass_fields__
    action = cfg["action_scheduler"].default_factory()
    video = cfg["video_scheduler"].default_factory()
    assert action["train_shift"] == 1.0 and action["infer_shift"] == 1.0
    assert video["train_shift"] == 5.0 and video["infer_shift"] == 5.0


def _lr_trace(warmup, total, lr=1e-4, min_lr=1e-6):
    from loongforge.embodied.optimizer.lr_scheduler import build_scheduler

    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    args = SimpleNamespace(lr_decay_style="linear_warmup_cosine_annealing", lr_decay_iters=None,
                           train_iters=total, lr_warmup_iters=warmup, min_lr=min_lr)
    sched = build_scheduler(opt, args)
    trace = []
    for _ in range(total):
        trace.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return trace


def test_linear_warmup_cosine_annealing_matches_fastwam_schedule():
    # FastWAM: warmup = int(0.05 * total), LinearLR(1/W -> 1) then CosineAnnealingLR(T=total-W) to min_lr.
    lr, min_lr, warmup, total = 1e-4, 1e-6, 20, 400
    trace = _lr_trace(warmup, total, lr, min_lr)
    assert trace[0] == pytest.approx(lr / warmup)
    k = warmup // 2
    assert trace[k] == pytest.approx(lr * (1 / warmup + (1 - 1 / warmup) * k / warmup))
    assert trace[warmup] == pytest.approx(lr)
    mid = warmup + (total - warmup) // 2
    assert trace[mid] == pytest.approx(min_lr + (lr - min_lr) / 2, rel=1e-6)
    last = math.cos(math.pi * (total - 1 - warmup) / (total - warmup))
    assert trace[-1] == pytest.approx(min_lr + (lr - min_lr) * (1 + last) / 2, rel=1e-6)


def test_linear_warmup_cosine_annealing_without_warmup():
    trace = _lr_trace(0, 10)
    assert trace[0] == pytest.approx(1e-4)
    assert all(a >= b for a, b in zip(trace, trace[1:]))
