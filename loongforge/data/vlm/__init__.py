# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal data utilities and task encoder registry."""

import importlib

# Sample types live in ``flavors``; importing them here would pull in
# megatron.energon for every import of this package.

# Registry for multimodal task encoders so configs can swap them without
# touching individual encoder modules.
TASK_ENCODER_REGISTRY = {
    "vlmtaskencoder": "loongforge.data.vlm.vlm_task_encoder.VLMTaskEncoder",
    "internvltaskencoder": "loongforge.data.vlm.internvl_task_encoder.InternVLTaskEncoder",
    "llavaov15taskencoder": "loongforge.data.vlm.llava_ov_task_encoder.LLavaOv15TaskEncoder",
    "ernietaskencoder": "loongforge.data.vlm.ernie.ernie_task_encoder.ErnieTaskEncoder",
    "kimitaskencoder": "loongforge.data.vlm.kimi_task_encoder.KimiTaskEncoder",
    "minicpmv46taskencoder": (
        "loongforge.data.vlm.minicpm_v_4_6_task_encoder.MiniCPMV46TaskEncoder"
    ),
}


def resolve_task_encoder(name: str):
    """Resolve and import a task encoder class by registry name."""
    normalized = name.lower()
    if normalized not in TASK_ENCODER_REGISTRY:
        available = sorted(TASK_ENCODER_REGISTRY)
        raise ValueError(f"Unknown task encoder '{name}'. Available: {available}")
    module_path, cls_name = TASK_ENCODER_REGISTRY[normalized].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def build_task_encoder(args, tokenizer, chat_template=None):
    """
    Factory that builds a task encoder instance based on args.task_encoder.

    Defaults to VLMTaskEncoder when unspecified, while keeping registry-based
    extensibility for other encoder classes.
    """
    encoder_name = getattr(args, "task_encoder", None) or "VLMTaskEncoder"
    encoder_cls = resolve_task_encoder(encoder_name)
    return encoder_cls(args, tokenizer, chat_template)


__all__ = [
    "TASK_ENCODER_REGISTRY",
    "resolve_task_encoder",
    "build_task_encoder",
]
