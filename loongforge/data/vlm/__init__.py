# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal (Energon) data pipeline and task encoder registry.

Pipeline: ``samples.py`` (sample types) -> ``cookers.py`` (raw WebDataset to
samples) -> ``encoders/`` (samples to model inputs) -> ``dataloader.py``
(collation and Energon iterators). ``packer.py`` and
``length_sort_dataset.py`` control packing and batch order.
``task_encoder_base.py`` holds the base sample types and ``BaseTaskEncoder``.
"""

import importlib

# Sample types live in ``samples``; importing them here would pull in
# megatron.energon for every import of this package.

# Registry for multimodal task encoders so configs can swap them without
# touching individual encoder modules.
TASK_ENCODER_REGISTRY = {
    "vlmtaskencoder": "loongforge.data.vlm.encoders.vlm.VLMTaskEncoder",
    "internvltaskencoder": "loongforge.data.vlm.encoders.internvl.task_encoder.InternVLTaskEncoder",
    "llavaov15taskencoder": "loongforge.data.vlm.encoders.llava_ov.LLaVAOv15TaskEncoder",
    "ernietaskencoder": "loongforge.data.vlm.encoders.ernie.task_encoder.ErnieTaskEncoder",
    "kimitaskencoder": "loongforge.data.vlm.encoders.kimi.KimiTaskEncoder",
    "minicpmv46taskencoder": (
        "loongforge.data.vlm.encoders.minicpm_v_4_6.MiniCPMV46TaskEncoder"
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


def build_task_encoder(config, tokenizer, chat_template=None):
    """
    Factory that builds a task encoder instance based on config.task_encoder.

    Defaults to VLMTaskEncoder when unspecified, while keeping registry-based
    extensibility for other encoder classes.
    """
    encoder_name = getattr(config, "task_encoder", None) or "VLMTaskEncoder"
    encoder_cls = resolve_task_encoder(encoder_name)
    return encoder_cls(config, tokenizer, chat_template)


__all__ = [
    "TASK_ENCODER_REGISTRY",
    "resolve_task_encoder",
    "build_task_encoder",
]
