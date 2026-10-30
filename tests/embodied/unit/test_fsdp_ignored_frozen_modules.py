# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FSDP2 frozen-module parameter replication."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from loongforge.embodied.distributed.fsdp_utils.builders import build_ignored_params
from loongforge.embodied.distributed.fsdp_utils.units import resolve_wrap_runs
from loongforge.embodied.train.validators import _validate_fsdp_ignored_frozen_args


class FrozenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, inputs):
        return self.proj(inputs)


class TrainableBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, inputs):
        return self.proj(inputs)


class EmptyBlock(nn.Module):
    def forward(self, inputs):
        return inputs


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = FrozenBlock().requires_grad_(False)
        self.trainable = TrainableBlock()
        self.empty = EmptyBlock()

    def forward(self, inputs):
        return self.trainable(self.frozen(inputs))


def _ignored_args(*classes, dtype=None):
    return SimpleNamespace(
        fsdp_ignored_param_names=[],
        fsdp_ignore_frozen_module_classes=list(classes),
        fsdp_ignored_frozen_param_dtype=dtype,
    )


def _wrap_args(*classes):
    return SimpleNamespace(
        fsdp_wrap_modules=list(classes),
        fsdp_no_wrap_modules=None,
        fsdp_min_param_num=1,
    )


def _validation_args(
    *, classes=None, dtype=None, compute_dtype="bfloat16", strategy="fsdp", init_on_meta=False
):
    return SimpleNamespace(
        fsdp_ignore_frozen_module_classes=classes,
        fsdp_ignored_frozen_param_dtype=dtype,
        dtype=compute_dtype,
        distributed_strategy=strategy,
        init_on_meta=init_on_meta,
    )


def test_build_ignored_params_matches_frozen_class_and_casts_dtype():
    model = ToyModel()
    ignored = build_ignored_params(
        _ignored_args("FrozenBlock", dtype="bf16"),
        model,
        SimpleNamespace(device=torch.device("cpu")),
    )

    assert ignored == set(model.frozen.parameters())
    assert {param.dtype for param in ignored} == {torch.bfloat16}
    assert {param.dtype for param in model.trainable.parameters()} == {torch.float32}


def test_build_ignored_params_rejects_trainable_match():
    model = ToyModel()

    with pytest.raises(ValueError, match="cannot ignore trainable parameters"):
        build_ignored_params(
            _ignored_args("TrainableBlock"),
            model,
            SimpleNamespace(device=torch.device("cpu")),
        )


def test_build_ignored_params_rejects_missing_or_empty_class():
    model = ToyModel()
    ctx = SimpleNamespace(device=torch.device("cpu"))

    with pytest.raises(ValueError, match="matched no modules: MissingBlock"):
        build_ignored_params(_ignored_args("MissingBlock"), model, ctx)
    with pytest.raises(ValueError, match="matched no parameters: EmptyBlock"):
        build_ignored_params(_ignored_args("FrozenBlock", "EmptyBlock"), model, ctx)


def test_resolve_wrap_runs_skips_unit_with_only_ignored_parameters():
    model = ToyModel()
    ignored = set(model.frozen.parameters())

    runs = resolve_wrap_runs(model, _wrap_args("FrozenBlock", "TrainableBlock"), ignored)
    units = [module for run in runs for unit in run.units for module in unit]

    assert model.frozen not in units
    assert model.trainable in units


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (_validation_args(dtype="bf16"), "--fsdp-ignored-frozen-param-dtype requires"),
        (_validation_args(classes=["FrozenBlock"], strategy="ddp"), "requires --distributed-strategy fsdp"),
        (_validation_args(classes=["FrozenBlock"], init_on_meta=True), "incompatible with --init-on-meta"),
        (_validation_args(classes=["FrozenBlock"], dtype="fp32"), "must match the training compute dtype"),
    ],
)
def test_validate_frozen_ignore_rejects_unsupported_combinations(args, message):
    with pytest.raises(ValueError, match=message):
        _validate_fsdp_ignored_frozen_args(args)


def test_validate_frozen_ignore_accepts_supported_or_disabled_config():
    _validate_fsdp_ignored_frozen_args(_validation_args())
    _validate_fsdp_ignored_frozen_args(_validation_args(classes=["FrozenBlock"], dtype="bf16"))


def test_ignored_frozen_dtype_matches_compute_dtype_without_autocast():
    model = ToyModel()
    build_ignored_params(
        _ignored_args("FrozenBlock", dtype="bf16"),
        model,
        SimpleNamespace(device=torch.device("cpu")),
    )
    inputs = torch.randn(2, 4, dtype=torch.bfloat16)
    with torch.no_grad():
        output = model.frozen(inputs)
    assert output.dtype == torch.bfloat16
