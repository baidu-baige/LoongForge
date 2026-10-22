# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Torch trainer registry.

Statically declared: ``--trainer-type`` selects a method module plus an optional
single-model optimization factory. All heavy imports (torch, method modules,
optimization modules) are deferred to call time via importlib.import_module.

Usage:
    method, build_optimization = resolve_torch_trainer("GrootN1d6Trainer")
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple


TORCH_TRAINER_TYPES: Dict[str, Tuple[str, str | None]] = {
    "FinetuneTrainer": ("sft_embodied", None),
    "VLDAFinetuneTrainer": ("sft_vlda", None),
    "GrootN1d6Trainer": (
        "sft_embodied", "loongforge.extensions.torch.groot_n1_6.factory"),
    "GrootN1d7Trainer": (
        "sft_embodied", "loongforge.extensions.torch.groot_n1_7.factory"),
    "LingBotFinetuneTrainer": (
        "sft_lingbot_va", "loongforge.extensions.torch.lingbot_va.factory"),
    "WallOss05Trainer": (
        "sft_embodied", "loongforge.extensions.torch.wall_oss_0_5.factory"),
}


def resolve_torch_trainer(trainer_type: str):
    """Lazily import and return the method module + optimization factory.

    Returns ``(method_module, build_optimization_callable)``.
    Raises ValueError for unknown --trainer-type values.
    """
    if not trainer_type:
        raise ValueError(
            "--trainer-type is required. "
            f"Available: {sorted(TORCH_TRAINER_TYPES.keys())}"
        )

    entry = TORCH_TRAINER_TYPES.get(trainer_type)
    if entry is None:
        raise ValueError(
            f"Unknown --trainer-type '{trainer_type}'. "
            f"Available: {sorted(TORCH_TRAINER_TYPES.keys())}"
        )

    method_name, factory_path = entry
    method = importlib.import_module(f"loongforge.training.torch.methods.{method_name}")
    if factory_path is None:
        from loongforge.engines.torch.train_step import TorchOptimization
        return method, lambda _args: TorchOptimization()
    return method, importlib.import_module(factory_path).build_optimization


def resolve_torch_runner(trainer_type: str):
    """Select the lifecycle for the recipe's checkpoint contract."""
    if trainer_type == "VLDAFinetuneTrainer":
        from loongforge.training.torch.vlda_runner import VLDARunner
        return VLDARunner
    from loongforge.training.torch.runner import TorchRunner
    return TorchRunner
