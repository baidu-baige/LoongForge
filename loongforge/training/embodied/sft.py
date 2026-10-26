# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SFT method for embodied models (Pi05 / GR00T / etc).

Module-level hooks that TorchRunner calls to build the model and dataloaders
and run the forward. The method module does NOT do autocast, backward, or optimizer steps.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def build_model(model_cfg, training_args) -> "nn.Module":
    """Build the embodied model from YAML model_cfg and this run's training args."""
    from loongforge.models.embodied.registry import build_model as _build
    return _build(model_cfg, training_args)


def build_dataloaders(model_cfg, data_cfg, training_args, ctx) -> Dict[str, "DataLoader"]:
    """Build dataloaders. Returns {"vla": dl}."""
    from loongforge.data.embodied.registry import DataRuntime
    from loongforge.data.embodied.dataloader import build_dataloader

    runtime = DataRuntime(
        rank=ctx.rank,
        world_size=ctx.world_size,
        is_distributed=ctx.is_distributed,
        grad_accum_steps=training_args.gradient_accumulation_steps,
        deterministic=training_args.deterministic_mode,
        graph_enabled=training_args.cuda_graph_impl == "local",
        graph_pad_length=training_args.cuda_graph_pad_length,
    )
    dl = build_dataloader(model_cfg, data_cfg, training_args, runtime)
    return {"vla": dl}


@functools.lru_cache(maxsize=16)
def _select_forward_kwargs_info(model_type: type) -> Tuple[frozenset, bool]:
    """Inspect forward signature and return (param_names, accepts_var_kw).

    Cached by model type to avoid repeated introspection.
    """
    params = list(inspect.signature(model_type.forward).parameters.values())
    names = frozenset(
        param.name
        for param in params
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          inspect.Parameter.KEYWORD_ONLY)
    )
    accepts_var_kw = any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in params
    )
    return names, accepts_var_kw


def _select_forward_kwargs(model, **candidates) -> dict:
    """Keep the candidate kwargs the model forward can actually accept."""
    from loongforge.engines.torch.distributed.utils import unwrap_model
    core = unwrap_model(model)
    for attr in ("_orig_mod",):
        core = getattr(core, attr, core)
    names, accepts_var_kw = _select_forward_kwargs_info(type(core))
    if accepts_var_kw:
        return dict(candidates)
    return {
        name: value
        for name, value in candidates.items()
        if name in names
    }


def forward(model, batch, *, iteration: int, training_args) -> Tuple["torch.Tensor", dict]:
    """Single forward: call model(batch) with filtered kwargs.

    Returns (loss, log_loss_dict). No autocast, no backward.
    """
    fwd_kwargs = _select_forward_kwargs(model, iteration=iteration)
    loss, log_loss_dict = model(batch, **fwd_kwargs)
    return loss, log_loss_dict


def on_train_begin(model, ctx) -> None:
    """Hook before training loop starts."""
    from loongforge.engines.torch.distributed.utils import unwrap_model
    raw = unwrap_model(model)
    hook = getattr(raw, "on_train_begin", None)
    if callable(hook):
        hook(ctx=ctx)
    if ctx.is_main:
        logger.info(f"Model: {raw.__class__.__name__}")


def on_after_data_iterators_initialized(model, *, training_args, completed_steps, optimizer, ctx) -> None:
    """Hook after dataloader iterators are initialized."""
    from loongforge.engines.torch.distributed.utils import unwrap_model
    raw = unwrap_model(model)
    hook = getattr(raw, "on_after_data_iterators_initialized", None)
    if callable(hook):
        hook(
            args=training_args,
            completed_steps=completed_steps,
            optimizer=optimizer,
        )


def on_after_train_batch_fetch(model, batch, *, training_args, completed_steps, micro_step) -> None:
    """Hook after a training batch is fetched.

    A model hook may return ``(seed, cpu_burn)`` to request an RNG reset. The
    reset is applied here so models stay free of global RNG side effects.
    """
    from loongforge.engines.torch.distributed.utils import unwrap_model
    raw = unwrap_model(model)
    hook = getattr(raw, "on_after_train_batch_fetch", None)
    if not callable(hook):
        return
    rng_reset = hook(
        args=training_args,
        completed_steps=completed_steps,
        micro_step=micro_step,
        batch=batch,
    )
    if rng_reset is None:
        return
    import torch

    from loongforge.engines.torch.initialize import set_seed

    seed, cpu_burn = rng_reset
    seed, cpu_burn = int(seed), int(cpu_burn)
    if cpu_burn < 0:
        raise ValueError("RNG reset cpu_burn must be >= 0")
    set_seed(seed)
    if cpu_burn:
        torch.rand(cpu_burn)
    logger.info(
        "%s reset training RNG after first batch fetch: seed=%d cpu_burn=%d",
        raw.__class__.__name__,
        seed,
        cpu_burn,
    )
