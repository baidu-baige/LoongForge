# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SFT method for DreamZero: the common embodied SFT plus two training-start policies.

To match upstream DreamZero numerics, the run rounds the trainable parameters once
after the optimizer exists, and resets the training RNG once after the first batch.
``DreamZeroConfig`` holds the switches for both.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from loongforge.training.embodied.sft import (  # noqa: F401
    build_dataloaders,
    build_model,
    forward,
    on_train_begin,
)

logger = logging.getLogger(__name__)

# One training run per process, so each policy runs at most once.
_initial_param_rounding_done = False
_train_rng_reset_after_first_batch_done = False


def on_after_data_iterators_initialized(model, *, training_args, completed_steps, optimizer, ctx) -> None:
    """Apply one-time DreamZero FSDP initialization compatibility rounding."""
    global _initial_param_rounding_done
    from loongforge.engines.torch.distributed.utils import unwrap_model

    raw = unwrap_model(model)
    if optimizer is None or _initial_param_rounding_done:
        return
    if completed_steps != 0:
        _initial_param_rounding_done = True
        return

    dtype_name = raw.config.fsdp_initial_param_rounding_dtype
    target_dtype = _resolve_initial_param_rounding_dtype(dtype_name)
    if target_dtype is None:
        _initial_param_rounding_done = True
        return

    rounded_params = 0
    rounded_numel = 0
    fallback_params = 0
    fallback_numel = 0
    skipped_params = 0

    with torch.no_grad():
        for group in optimizer.param_groups:
            for param in group.get("params", []):
                if param is None or not param.requires_grad:
                    continue
                if not torch.is_floating_point(param):
                    skipped_params += 1
                    continue

                local = _mutable_local_tensor(param)
                if local is not None:
                    if not torch.is_floating_point(local):
                        skipped_params += 1
                        continue
                    local.copy_(local.to(dtype=target_dtype).to(dtype=local.dtype))
                    rounded_params += 1
                    rounded_numel += int(local.numel())
                    continue

                param.data.copy_(
                    param.data.to(dtype=target_dtype).to(dtype=param.dtype)
                )
                fallback_params += 1
                fallback_numel += int(param.numel())

    _initial_param_rounding_done = True

    rank_zero = not torch.distributed.is_available() or not torch.distributed.is_initialized()
    if rank_zero or torch.distributed.get_rank() == 0:
        logger.info(
            "DreamZero initial FSDP param rounding: dtype=%s "
            "local_params=%d local_numel=%d fallback_params=%d "
            "fallback_numel=%d skipped=%d",
            str(target_dtype).replace("torch.", ""),
            rounded_params,
            rounded_numel,
            fallback_params,
            fallback_numel,
            skipped_params,
        )


def _resolve_initial_param_rounding_dtype(dtype_name: Any) -> torch.dtype | None:
    """Resolve the configured initial parameter-rounding dtype."""
    if dtype_name is None:
        return None
    mode = str(dtype_name).strip().lower()
    if mode in {"", "0", "false", "no", "off", "none", "null"}:
        return None
    if mode in {"1", "true", "yes", "on", "bf16", "bfloat16"}:
        return torch.bfloat16
    if mode in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(
        "DreamZeroConfig.fsdp_initial_param_rounding_dtype must be one of "
        "null/bf16/bfloat16/fp16/float16"
    )


def _mutable_local_tensor(param: torch.Tensor) -> torch.Tensor | None:
    """Return the mutable local tensor backing a regular or DTensor parameter."""
    data = param.data
    local = getattr(data, "_local_tensor", None)
    if torch.is_tensor(local):
        return local
    detached = param.detach()
    local = getattr(detached, "_local_tensor", None)
    if torch.is_tensor(local):
        return local
    if torch.is_tensor(data) and not hasattr(data, "to_local"):
        return data
    return None


def on_after_train_batch_fetch(model, batch, *, training_args, completed_steps, micro_step) -> None:
    """Reset the training RNG once after the first batch fetch."""
    global _train_rng_reset_after_first_batch_done
    from loongforge.engines.torch.distributed.utils import unwrap_model
    from loongforge.engines.torch.utils.utils import set_seed

    raw = unwrap_model(model)
    if _train_rng_reset_after_first_batch_done:
        return
    if not raw.config.train_rng_reset_after_first_batch:
        _train_rng_reset_after_first_batch_done = True
        return
    if training_args.resume or completed_steps != 0:
        _train_rng_reset_after_first_batch_done = True
        return

    seed = raw.config.train_rng_seed
    if seed is None:
        seed = int(training_args.seed or 0)
    seed = int(seed)

    cpu_burn = int(raw.config.train_rng_cpu_burn)
    if cpu_burn < 0:
        raise ValueError("DreamZeroConfig.train_rng_cpu_burn must be >= 0")

    _train_rng_reset_after_first_batch_done = True
    set_seed(seed)
    if cpu_burn:
        torch.rand(cpu_burn)
    logger.info(
        "%s reset training RNG after first batch fetch: seed=%d cpu_burn=%d",
        raw.__class__.__name__,
        seed,
        cpu_burn,
    )
