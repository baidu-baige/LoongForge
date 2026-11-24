# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from copy import copy
from dataclasses import replace

import torch
import torch.nn.functional as F
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn


class ClampedSwiGLUSharedExpertMLP(SharedExpertMLP):
    """Compute clamped SwiGLU in FP32 with or without fusion."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        gate: bool,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        if (
            not config.bias_activation_fusion
            and config.activation_func_clamp_value is not None
            and config.activation_func is F.silu
            and config.gated_linear_unit
        ):
            config = copy(config)
            config.use_te_activation_func = True
            submodules = replace(submodules, activation_func=_ClampedSwiGLU)
        super().__init__(config=config, submodules=submodules, gate=gate, pg_collection=pg_collection)


class _ClampedSwiGLU(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.limit = config.activation_func_clamp_value
        self.offset = config.glu_linear_offset

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = hidden_states.float().chunk(2, dim=-1)
        up = up.clamp(-self.limit, self.limit)
        if self.offset:
            up = up + self.offset
        return (F.silu(gate.clamp(max=self.limit)) * up).to(hidden_states.dtype)
