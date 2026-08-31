# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Python interface for the GR00T-N1.7 precision-compatible AdamW kernel."""

import torch

from . import _groot_n1_7_fused_adamw


def _extension():
    return _groot_n1_7_fused_adamw


def eager_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    exp_avgs: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    *,
    decay_factor: float,
    beta2: float,
    first_moment_weight: float,
    second_moment_weight: float,
    eps: float,
    bias_correction1: float,
    bias_correction2_sqrt: float,
    lr: float,
) -> None:
    """Apply the eager update using host-computed AdamW scalars."""
    _extension().groot_n1_7_fused_adamw_eager_step(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        decay_factor,
        beta2,
        first_moment_weight,
        second_moment_weight,
        eps,
        bias_correction1,
        bias_correction2_sqrt,
        lr,
    )


def capturable_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    exp_avgs: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    *,
    lr: torch.Tensor,
    step: torch.Tensor,
    bias_correction1: torch.Tensor,
    bias_correction2_sqrt: torch.Tensor,
    beta2: float,
    first_moment_weight: float,
    second_moment_weight: float,
    eps: float,
    weight_decay: float,
) -> None:
    """Apply the graph-stable update using CUDA scalar inputs."""
    _extension().groot_n1_7_fused_adamw_capturable_step(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        lr,
        step,
        bias_correction1,
        bias_correction2_sqrt,
        beta2,
        first_moment_weight,
        second_moment_weight,
        eps,
        weight_decay,
    )


def capturable_grad_scaled_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    exp_avgs: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    *,
    lr: torch.Tensor,
    step: torch.Tensor,
    bias_correction1: torch.Tensor,
    bias_correction2_sqrt: torch.Tensor,
    grad_scale: torch.Tensor,
    beta2: float,
    first_moment_weight: float,
    second_moment_weight: float,
    eps: float,
    weight_decay: float,
) -> None:
    """Apply the graph-stable update while scaling each loaded gradient once."""
    _extension().groot_n1_7_fused_adamw_capturable_grad_scaled_step(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        lr,
        step,
        bias_correction1,
        bias_correction2_sqrt,
        grad_scale,
        beta2,
        first_moment_weight,
        second_moment_weight,
        eps,
        weight_decay,
    )
