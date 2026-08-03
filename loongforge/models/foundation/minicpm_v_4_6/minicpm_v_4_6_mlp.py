# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-V-4.6 dense MLP."""

import torch
import torch.nn.functional as F

from megatron.core.transformer.mlp import MLP
from megatron.core.utils import nvtx_range_pop, nvtx_range_push
from loongforge.models.common.peft.lora_layers import torch_linear_forward


class MiniCPMV46DenseMLP(MLP):
    """Run gate/up projections separately to match the reference BF16 order."""

    def forward(self, hidden_states, per_token_scale=None):
        if per_token_scale is not None:
            raise NotImplementedError("MiniCPMV46DenseMLP does not support per_token_scale.")
        if self.config.add_bias_linear:
            raise NotImplementedError("MiniCPMV46DenseMLP expects bias-free projections.")
        if self.config.activation_func_clamp_value is not None:
            raise NotImplementedError("MiniCPMV46DenseMLP does not support activation clamping.")
        if not self.config.gated_linear_unit:
            return super().forward(hidden_states, per_token_scale=per_token_scale)

        nvtx_range_push(suffix="linear_fc1_split")
        if hasattr(self.linear_fc1, "forward_split"):
            gate, up = self.linear_fc1.forward_split(hidden_states)
        else:
            linear_fc1 = getattr(self.linear_fc1, "to_wrap", self.linear_fc1)
            gate_weight, up_weight = torch.chunk(linear_fc1.weight, 2, dim=0)
            gate = F.linear(hidden_states, gate_weight)
            up = F.linear(hidden_states, up_weight)
        intermediate_parallel = self.config.activation_func(gate) * (
            up + self.config.glu_linear_offset
        )
        nvtx_range_pop(suffix="linear_fc1_split")

        nvtx_range_push(suffix="linear_fc2")
        if self.config.mlp_linear_backend == "torch":
            if self.config.tensor_model_parallel_size != 1:
                raise ValueError("The torch MLP backend requires tensor parallel size 1")
            output, output_bias = torch_linear_forward(
                self.linear_fc2, intermediate_parallel
            )
        else:
            output, output_bias = self.linear_fc2(intermediate_parallel)
        nvtx_range_pop(suffix="linear_fc2")
        return output, output_bias
