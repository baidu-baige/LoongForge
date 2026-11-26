# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MCore module specifications for the DeepSeek-V4.1 decoder."""

from __future__ import annotations

from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from loongforge.models.common.clamped_swiglu import ClampedSwiGLUSharedExpertMLP

from .deepseek_v41_layers import DeepseekV41MoELayer, DeepseekV41SequentialMLP


def get_deepseek_v41_moe_spec(config: TransformerConfig, backend) -> ModuleSpec:
    """Reuse MCore experts and dispatch with modality-aware V4.1 routing."""
    spec = get_moe_module_spec_for_backend(
        backend=backend,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        use_te_activation_func=config.use_te_activation_func,
    )
    spec.module = DeepseekV41MoELayer
    spec.submodules.shared_experts.module = ClampedSwiGLUSharedExpertMLP
    if spec.submodules.experts.module is SequentialMLP:
        spec.submodules.experts.module = DeepseekV41SequentialMLP
    return spec
