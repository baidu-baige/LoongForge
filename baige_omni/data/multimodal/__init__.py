# Copyright 2026 The BaigeOmni Authors.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal data utilities and task encoder registry."""

import importlib

from baige_omni.data.multimodal.flavors import (
    PackedCaptioningSample,
    PackedVQASample,
    PackedMultiMixQASample,
    MultiVidQASample,
    MultiMixQASample,
)

# Registry for multimodal task encoders so configs can swap them without
# touching individual encoder modules.
TASK_ENCODER_REGISTRY = {
    "vlmtaskencoder": "baige_omni.data.multimodal.vlm_task_encoder.VLMTaskEncoder",
    "internvltaskencoder": "baige_omni.data.multimodal.internvl.internvl_task_encoder.InternVLTaskEncoder",
    "llavaov15taskencoder": "baige_omni.data.multimodal.llava_ov_task_encoder.LLavaOv15TaskEncoder",
    "ernietaskencoder": "baige_omni.data.multimodal.ernie_task_encoder.ErnieTaskEncoder",
}


def resolve_task_encoder(name: str):
    """Resolve and import a task encoder class by registry key or class name."""
    normalized = name.lower()
    if normalized not in TASK_ENCODER_REGISTRY:
        available = [
            path.rsplit(".", 1)[-1] for path in TASK_ENCODER_REGISTRY.values()
        ]
        raise ValueError(f"Unknown task encoder '{name}'. Available: {available}")
    module_path, cls_name = TASK_ENCODER_REGISTRY[normalized].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def build_task_encoder(args, *encoder_args, **encoder_kwargs):
    """
    Factory that builds a task encoder instance based on args.task_encoder.

    Defaults to VLMTaskEncoder when unspecified, while keeping registry-based
    extensibility for other encoder classes.
    """
    encoder_name = getattr(args, "task_encoder", None) or "VLMTaskEncoder"
    try:
        encoder_cls = resolve_task_encoder(encoder_name)
    except ValueError:
        # Fallback: keep training running even if config typo slips in.
        from baige_omni.data.multimodal.vlm_task_encoder import VLMTaskEncoder
        encoder_cls = VLMTaskEncoder
    return encoder_cls(args, *encoder_args, **encoder_kwargs)


__all__ = [
    "PackedCaptioningSample",
    "PackedVQASample",
    "PackedMultiMixQASample",
    "MultiVidQASample",
    "MultiMixQASample",
    "TASK_ENCODER_REGISTRY",
    "resolve_task_encoder",
    "build_task_encoder",
]
