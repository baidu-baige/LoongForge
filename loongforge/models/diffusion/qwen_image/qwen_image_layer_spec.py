# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image layer spec."""

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELinear,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec

from .qwen_image_layer import QwenImageLayer, QwenImageLayerSubmodules


def get_qwen_image_layer_with_te_spec() -> ModuleSpec:
    """Layer spec that hands the block TE Linear/DotProductAttention.

    Building the layer through a ModuleSpec is required for
    ``TransformerBlock`` and for ``recompute_granularity=full`` to work.
    """
    return ModuleSpec(
        module=QwenImageLayer,
        params={"attn_mask_type": AttnMaskType.no_mask},
        submodules=QwenImageLayerSubmodules(
            linear=TELinear,
            column_linear=TEColumnParallelLinear,
            row_linear=TERowParallelLinear,
            core_attention=TEDotProductAttention,
        ),
    )
