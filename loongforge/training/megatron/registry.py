# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Megatron method registry.

Statically declared: (model family, phase) selects a method module. All heavy
imports (megatron, method modules) are deferred to call time via
importlib.import_module.

Usage:
    method = resolve_method(args.model_family, "pretrain")
    method.train(args)
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
# (group, phase) -> method module path   (Megatron engine)
# ---------------------------------------------------------------------------

METHOD_REGISTRY: Dict[Tuple[str, str], str] = {
    # LLM
    ("llm", "pretrain"): "loongforge.training.megatron.methods.pretrain_llm",
    ("llm", "sft"):      "loongforge.training.megatron.methods.sft_llm",
    # VLM  (sft_vlm delegates to pretrain_vlm callbacks)
    ("vlm", "pretrain"): "loongforge.training.megatron.methods.pretrain_vlm",
    ("vlm", "sft"):      "loongforge.training.megatron.methods.sft_vlm",
    # Dedicated overrides
    ("intern_vl", "sft"):     "loongforge.training.megatron.methods.sft_internvl",
    ("ernie4_5_vl", "sft"):   "loongforge.training.megatron.methods.sft_ernie",
    ("wan", "pretrain"):      "loongforge.training.megatron.methods.pretrain_wan",
    ("qwen_image", "pretrain"): "loongforge.training.megatron.methods.pretrain_qwen_image",
}



def resolve_method(model_family: str, training_phase: str):
    """Lazily import and return the method module for *model_family* + *phase*.

    The module's ``train(args)`` runs the training. Raises ValueError for
    unsupported combinations.
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
