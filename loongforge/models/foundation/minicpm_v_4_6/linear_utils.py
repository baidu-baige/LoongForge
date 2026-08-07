# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM linear fallbacks that preserve attached LoRA adapters."""

from typing import Optional, Tuple

import torch
from torch import nn

from loongforge.models.common.peft.adapter_wrapper import AdapterWrapper

from .peft import lora_linear_forward


def torch_linear_forward(
    module: nn.Module,
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run a torch linear projection and its optional adapter branch."""
    wrapper = module if isinstance(module, AdapterWrapper) else None
    base = wrapper.to_wrap if wrapper is not None else module
    bias = getattr(base, "bias", None)
    if bias is not None and bias.numel() == 0:
        bias = None
    skip_bias_add = getattr(base, "skip_bias_add", False)
    projection_input = inputs.to(dtype=base.weight.dtype)
    output = torch.nn.functional.linear(
        projection_input,
        base.weight,
        None if skip_bias_add else bias,
    )

    if wrapper is not None and wrapper._adapter_enabled:
        adapter_output = wrapper.adapter(projection_input.contiguous())
        output = output + adapter_output.reshape(output.shape).to(dtype=output.dtype)
    elif (
        wrapper is None
        and getattr(module, "_adapter_enabled", False)
        and hasattr(module, "linear_in")
        and hasattr(module, "linear_out")
    ):
        adapter_input = projection_input.to(dtype=module.linear_in.weight.dtype)
        if module.dropout_position == "pre":
            adapter_input = module.dropout(adapter_input)
        adapter_output = lora_linear_forward(
            module.linear_in,
            module.linear_out,
            adapter_input,
            module.scale,
            output.dtype,
        )
        if module.dropout_position == "post":
            adapter_output = module.dropout(adapter_output)
        output = output + adapter_output.to(dtype=output.dtype)

    return output, bias if skip_bias_add else None
