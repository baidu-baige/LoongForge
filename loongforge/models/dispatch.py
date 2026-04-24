# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch Module"""

from typing import Any
from dataclasses import dataclass
from functools import cached_property

from loongforge.utils import get_args


@dataclass
class MultiAccModules:
    """MultiAccModules"""

    # dense linear impl
    TELayerNormColumnParallelLinear: Any = None
    TEColumnParallelLinear: Any = None
    TERowParallelLinear: Any = None
    # group-gemm linear impl
    TEColumnParallelGroupedLinear: Any = None
    TERowParallelGroupedLinear: Any = None
    # local linear
    ColumnParallelLinear: Any = None
    RowParallelLinear: Any = None
    # attention impl
    DotProductAttention: Any = None
    # norm impl
    TENorm: Any = None
    LocalNorm: Any = None
    # other ops impl
    get_bias_dropout_add: Any = None
    apply_rotary_pos_emb: Any = None
    # some flags
    TELinear: Any = None


def _gpu_backend_transformer_layer_modules() -> MultiAccModules:
    """define gpu transformer layer modules"""
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TEColumnParallelLinear,
        TERowParallelLinear,
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear,
        TENorm,
        TELinear,
    )

    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.models.common.embeddings.rotary_pos_embedding import (
        apply_rotary_pos_emb,
    )
    from loongforge.models.common.local_layers.local_norm import LocalNorm

    args = get_args()

    return MultiAccModules(
        # dense linear
        TELayerNormColumnParallelLinear=TELayerNormColumnParallelLinear,
        TEColumnParallelLinear=TEColumnParallelLinear,
        TERowParallelLinear=TERowParallelLinear,
        # group-gemm linear
        TEColumnParallelGroupedLinear=TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear=TERowParallelGroupedLinear,
        # local linear
        ColumnParallelLinear=ColumnParallelLinear,
        RowParallelLinear=RowParallelLinear,
        # attn
        DotProductAttention=TEDotProductAttention,
        # norm
        TENorm=TENorm,
        LocalNorm=LocalNorm,
        # ops
        get_bias_dropout_add=get_bias_dropout_add,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
        TELinear=TELinear,
    )


class Dispatch:
    """dispatch transformer layer module"""

    @cached_property
    def settings(self):
        """stting attr"""
        return _gpu_backend_transformer_layer_modules()

    def __getattr__(self, name):
        return getattr(self.settings, name)


multiacc_modules: MultiAccModules = Dispatch()
