# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Parameter semantics for LingBot VLA v2 distributed strategies."""

from __future__ import annotations

import torch
from torch import nn


class LingbotVlaV2ParameterPolicy:
    """Preserve the validated LingBot ZeRO-1 ownership and precision contract."""

    _ACTION_FP32_MARKERS = (
        "qwenvl_with_expert.qwen_expert.model.layers.",
        "qwenvl_with_expert.qwen_expert.model.norm.",
    )
    _DEFAULT_ADAMW_MARKERS = (
        "embed_tokens",
        "embedding",
        "lm_head",
        "output_layer",
    )
    # Expert selection is the one place where a rounding error changes the
    # computation instead of perturbing it: the MoE gate already runs its matmul
    # in true fp32 (qwen2_action_expert.py) because bf16 logits flip top-k. A
    # router weight that arrives bf16- or fp8-rounded reintroduces exactly that
    # failure, so router/gate weights keep full-precision collectives even when
    # everything else is downcast. 1-D tensors (norms, biases) are lumped in:
    # they are precision sensitive and too small for the bytes to matter.
    _COMM_FP32_MARKERS = (
        ".gate.weight",
        "shared_expert_gate",
    )

    def __init__(self, model_cfg=None):
        extra_markers = (
            () if model_cfg is None else tuple(model_cfg.muon_exclude_name_patterns or ())
        )
        self._adamw_markers = self._DEFAULT_ADAMW_MARKERS + extra_markers

    def is_expert_shard(self, name: str, parameter: nn.Parameter) -> bool:
        """Keep expert tensors owned whole instead of sharded along dim 0."""
        # Expert sharding is off: 3-D expert tensors are owned whole like every
        # other parameter. Measured on 8 GPUs at GBS80 it was worth 6.2ms
        # (0.4%) with the collectives overlapped and -29ms without them, so the
        # extra dim-0 ownership path does not pay for itself here.
        return False

    def compute_dtype(self, name: str, parameter: nn.Parameter) -> torch.dtype:
        """Return fp32 for action-expert parameters and bf16 for the rest."""
        if any(marker in name for marker in self._ACTION_FP32_MARKERS):
            return torch.float32
        return torch.bfloat16

    def is_comm_precision_critical(self, name: str, parameter: nn.Parameter) -> bool:
        """Return whether the parameter must keep full-precision collectives."""
        if parameter.ndim < 2:
            return True
        return any(marker in name for marker in self._COMM_FP32_MARKERS)

    def optimizer_kind(self, name: str, parameter: nn.Parameter) -> str:
        """Route 2-D and 3-D tensors to muon unless an adamw marker matches."""
        lowered = name.lower()
        if parameter.ndim not in (2, 3):
            return "adamw"
        if any(marker.lower() in lowered for marker in self._adamw_markers):
            return "adamw"
        return "muon"


__all__ = ["LingbotVlaV2ParameterPolicy"]
