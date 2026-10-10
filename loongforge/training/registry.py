# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unified training registry.

Replaces the per-engine trainer registries (decorator-based registration and
class lookup) with a single, statically-declared registry.  All heavy imports (torch, megatron, method
modules) are deferred to call time via importlib.import_module.

Usage (MCore):
    method = resolve_method("qwen2_vl", "pretrain")
    # method is the lazily-imported module (e.g. pretrain_vlm)

Usage (Torch):
    trainer_cls = resolve_torch_trainer("GrootN1d6Trainer")
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

from loongforge.constants import LanguageModelFamilies, VisionLanguageModelFamilies

# ---------------------------------------------------------------------------
# Family -> group mapping.
# Built from loongforge/constants.py families.  Each concrete model family
# maps to a "group" that selects the method module.  Families that need their
# own method (intern_vl, ernie4_5_vl, wan, qwen_image) map to themselves.
# ---------------------------------------------------------------------------

FAMILY_GROUP: Dict[str, str] = {f: "llm" for f in LanguageModelFamilies.names()}
FAMILY_GROUP.update({f: "vlm" for f in VisionLanguageModelFamilies.names()})
# Families with dedicated methods keep their own name as group key.
FAMILY_GROUP["intern_vl"] = "intern_vl"
FAMILY_GROUP["ernie4_5_vl"] = "ernie4_5_vl"
FAMILY_GROUP["wan2_1_i2v"] = "wan"
FAMILY_GROUP["wan2_2_i2v"] = "wan"
FAMILY_GROUP["qwen_image"] = "qwen_image"

# ---------------------------------------------------------------------------
# (group, phase) -> method module path   (MCore engine)
# ---------------------------------------------------------------------------

METHOD_REGISTRY: Dict[Tuple[str, str], str] = {
    # LLM
    ("llm", "pretrain"): "loongforge.training.methods.pretrain_llm",
    ("llm", "sft"):      "loongforge.training.methods.sft_llm",
    # VLM  (sft_vlm delegates to pretrain_vlm callbacks)
    ("vlm", "pretrain"): "loongforge.training.methods.pretrain_vlm",
    ("vlm", "sft"):      "loongforge.training.methods.sft_vlm",
    # Dedicated overrides
    ("intern_vl", "sft"):     "loongforge.training.methods.sft_internvl",
    ("ernie4_5_vl", "sft"):   "loongforge.training.methods.sft_ernie",
    ("wan", "pretrain"):      "loongforge.training.methods.pretrain_wan",
    ("qwen_image", "pretrain"): "loongforge.training.methods.pretrain_qwen_image",
}


def resolve_method(model_family: str, training_phase: str):
    """Lazily import and return the method module for *model_family* + *phase*.

    Raises ValueError for unsupported combinations.
    """
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


# ---------------------------------------------------------------------------
# Torch engine: --trainer-type -> (method_module, optimization_package)
# All old values stay valid. Lazy imports; no torch at module level.
# ---------------------------------------------------------------------------

TORCH_TRAINER_TYPES: Dict[str, Tuple[str, str | None]] = {
    "FinetuneTrainer": ("sft_embodied", None),
    "GrootN1d6Trainer": ("sft_embodied", "groot_n1_6"),
    "GrootN1d7Trainer": ("sft_embodied", "groot_n1_7"),
    "LingBotFinetuneTrainer": ("sft_lingbot_va", "lingbot_va"),
}

# ponytail: legacy TORCH_TRAINER_REGISTRY removed — trainers/ directory deleted


def resolve_torch_trainer(trainer_type: str):
    """Lazily import and return the method module + optimization factory.

    Returns ``(method_module, build_optimization_callable)``.
    Raises ValueError for unknown --trainer-type values, preserving the
    original error messages from trainer_builder.py.
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

    method_name, opt_name = entry
    method = importlib.import_module(f"loongforge.training.methods.{method_name}")
    if opt_name is None:
        from loongforge.engines.torch.train_step import TorchOptimization
        return method, lambda _args: TorchOptimization()
    opt_module = importlib.import_module(
        f"loongforge.engines.torch.optimizations.{opt_name}.optimization"
    )
    return method, opt_module.build_optimization
