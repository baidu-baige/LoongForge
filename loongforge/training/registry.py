# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training dispatch: family/phase and trainer type to executable recipes.

The tables are declarative. Heavy imports (megatron, torch, method modules,
optimization modules) are deferred to call time via ``importlib.import_module``
so importing this module stays cheap.

Usage:
    run = resolve("megatron", args)
    run()
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict, Tuple

from loongforge.constants import DiffusionModelFamilies, LanguageModelFamilies, VisionLanguageModelFamilies

# ---------------------------------------------------------------------------
# Megatron: family -> group
# ---------------------------------------------------------------------------

FAMILY_GROUP: Dict[str, str] = {f: "llm" for f in LanguageModelFamilies.names()}
FAMILY_GROUP.update({f: "vlm" for f in VisionLanguageModelFamilies.names()})
FAMILY_GROUP[VisionLanguageModelFamilies.INTERN_VL] = "intern_vl"
FAMILY_GROUP[VisionLanguageModelFamilies.ERNIE4_5_VL] = "ernie4_5_vl"
FAMILY_GROUP[DiffusionModelFamilies.WAN2_1_I2V] = "wan"
FAMILY_GROUP[DiffusionModelFamilies.WAN2_2_I2V] = "wan"
FAMILY_GROUP[DiffusionModelFamilies.QWEN_IMAGE] = "qwen_image"

# (group, phase) -> method module path
METHOD_REGISTRY: Dict[Tuple[str, str], str] = {
    ("llm", "pretrain"): "loongforge.training.llm.pretrain",
    ("llm", "sft"): "loongforge.training.llm.sft",
    ("vlm", "pretrain"): "loongforge.training.vlm.pretrain_vlm",
    ("vlm", "sft"): "loongforge.training.vlm.sft_vlm",
    ("intern_vl", "sft"): "loongforge.training.vlm.sft_internvl",
    ("ernie4_5_vl", "sft"): "loongforge.training.vlm.sft_ernie",
    ("wan", "pretrain"): "loongforge.training.diffusion.pretrain_wan",
    ("qwen_image", "pretrain"): "loongforge.training.diffusion.pretrain_qwen_image",
}

# ---------------------------------------------------------------------------
# Torch: trainer type -> (method, optional optimization factory)
# ---------------------------------------------------------------------------

TORCH_TRAINER_TYPES: Dict[str, Tuple[str, str | None]] = {
    "FinetuneTrainer": ("sft", None),
    "VLDAFinetuneTrainer": ("sft_vlda", None),
    "Gr00tN1d6Trainer": ("sft", "loongforge.training.embodied.optimizations.gr00t_n1_6.factory"),
    "Gr00tN1d7Trainer": ("sft", "loongforge.training.embodied.optimizations.gr00t_n1_7.factory"),
    "LingBotFinetuneTrainer": ("sft_lingbot_va", "loongforge.training.embodied.optimizations.lingbot_va.factory"),
    "WallOss05Trainer": ("sft", "loongforge.training.embodied.optimizations.wall_oss_0_5.factory"),
}


def resolve_method(model_family: str, training_phase: str):
    """Lazily import and return the Megatron method module for *model_family* + *phase*."""
    group = FAMILY_GROUP.get(model_family, model_family)
    key = (group, training_phase)
    module_path = METHOD_REGISTRY.get(key)
    if module_path is None:
        raise ValueError(
            f"No method registered for ({model_family!r}, {training_phase!r}) "
            f"[resolved group={group!r}]. "
            f"Available: {sorted(METHOD_REGISTRY.keys())}"
        )
    return importlib.import_module(module_path)


def resolve_torch_trainer(trainer_type: str):
    """Lazily import and return the Torch method module and optimization factory."""
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
    method = importlib.import_module(f"loongforge.training.embodied.{method_name}")
    if factory_path is None:
        from loongforge.engines.torch.train_step import TorchOptimization
        return method, lambda _args: TorchOptimization()
    return method, importlib.import_module(factory_path).build_optimization


def resolve_torch_runner(trainer_type: str):
    """Select the lifecycle for the recipe's checkpoint contract."""
    if trainer_type == "VLDAFinetuneTrainer":
        from loongforge.engines.torch.vlda_runner import VLDARunner
        return VLDARunner
    from loongforge.engines.torch.runner import TorchRunner
    return TorchRunner


def resolve(engine: str, parsed) -> Callable[[], object]:
    """Return a zero-argument callable that runs the selected recipe.

    ``parsed`` is the Megatron namespace or the Torch
    ``(training_args, model_cfg, data_cfg)`` tuple.
    """
    if engine == "megatron":
        method = resolve_method(parsed.model_family, parsed.training_phase)
        return lambda: method.train(parsed)

    if engine == "torch":
        training_args, model_cfg, data_cfg = parsed
        method, build_optimization = resolve_torch_trainer(training_args.trainer_type)
        runner_type = resolve_torch_runner(training_args.trainer_type)
        return lambda: runner_type(
            training_args, model_cfg, data_cfg, method, build_optimization(training_args)
        ).train()

    raise ValueError(f"Unknown engine: {engine!r}")


__all__ = [
    "FAMILY_GROUP",
    "METHOD_REGISTRY",
    "TORCH_TRAINER_TYPES",
    "resolve",
    "resolve_method",
    "resolve_torch_runner",
    "resolve_torch_trainer",
]
