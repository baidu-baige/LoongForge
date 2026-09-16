# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU-owned MoE group metadata helpers."""

import torch


def compute_moe_group_counts(token_types: torch.Tensor) -> tuple[int, ...]:
    """Count observed expert token types before batch device transfer."""
    if token_types.device.type != "cpu":
        raise ValueError("MoE group counts must be computed before batch device transfer")

    flat_token_types = token_types.reshape(-1).to(dtype=torch.long)
    if flat_token_types.numel() > 0:
        min_type = int(flat_token_types.min())
        if min_type < 0:
            raise ValueError(f"MoE token types must be non-negative, got {min_type}")

    return tuple(torch.bincount(flat_token_types).tolist())


def build_moe_group_indices(
    group_counts: tuple[int, ...], num_experts: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Expand CPU counts to boundaries using the model-owned expert count."""
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if len(group_counts) > num_experts:
        raise ValueError(
            f"Observed {len(group_counts)} MoE token types for {num_experts} experts"
        )
    if any(count < 0 for count in group_counts):
        raise ValueError(f"MoE group counts must be non-negative, got {group_counts}")

    padded_counts = list(group_counts) + [0] * (num_experts - len(group_counts))
    start_indices = []
    end_indices = []
    running_total = 0
    for count in padded_counts:
        start_indices.append(running_total)
        running_total += count
        end_indices.append(running_total)

    return tuple(start_indices), tuple(end_indices)
