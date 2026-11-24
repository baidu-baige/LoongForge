# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Linear layers used by the native DeepSeek-V4.1 modules."""

from __future__ import annotations

import copy

from torch import nn


class _LocalLinear(nn.Linear):
    """TP=1 fallback with the same two-value return contract as TE linear."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config=None,
        init_method=None,
        bias: bool = False,
        **kwargs,
    ):
        del kwargs
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            device=getattr(config, "device", None),
            dtype=getattr(config, "params_dtype", None),
        )
        if init_method is not None:
            init_method(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, hidden_states):
        return super().forward(hidden_states), None


def build_deepseek_v41_linear(
    config,
    input_size: int,
    output_size: int,
    *,
    mode: str,
    init_method,
    bias: bool = False,
    skip_bias_add: bool = False,
    tp_group=None,
    **kwargs,
):
    """Build a duplicated, column-parallel, or row-parallel V4.1 projection."""
    del kwargs

    if mode not in ("duplicated", "column", "row"):
        raise ValueError(f"Unsupported DeepSeek-V4.1 linear mode: {mode}")

    transformer_impl = getattr(config, "transformer_impl", "transformer_engine")
    if transformer_impl != "transformer_engine":
        if getattr(config, "tensor_model_parallel_size", 1) != 1:
            raise ValueError("DeepSeek-V4.1 local linear fallback only supports tensor-model-parallel-size=1")
        return _LocalLinear(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            bias=bias,
        )

    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELinear,
        TERowParallelLinear,
    )

    # V4.1 owns a batch-first decoder path.  TE's sequence-parallel
    # communication assumes [sequence, batch, hidden] inputs, which would
    # gather the batch dimension when these projections are called here.
    te_config = copy.copy(config)
    te_config.sequence_parallel = False

    if mode == "duplicated":
        return TELinear(
            input_size,
            output_size,
            parallel_mode="duplicated",
            config=te_config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=False,
        )
    if mode == "column":
        return TEColumnParallelLinear(
            input_size,
            output_size,
            config=te_config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=False,
            gather_output=False,
            tp_group=tp_group,
        )
    return TERowParallelLinear(
        input_size,
        output_size,
        config=te_config,
        init_method=init_method,
        bias=bias,
        input_is_parallel=True,
        skip_bias_add=skip_bias_add,
        is_expert=False,
        tp_group=tp_group,
    )


__all__ = ["build_deepseek_v41_linear"]
