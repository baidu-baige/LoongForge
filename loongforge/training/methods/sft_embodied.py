# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SFT method for embodied models (Pi05 / GR00T / etc).

Module-level functions replacing FinetuneTrainer's model/data/forward hooks.
The method module does NOT do autocast, backward, or optimizer steps.
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


def build_model(model_cfg) -> "nn.Module":
    """Build the embodied model from YAML model_cfg."""
    from loongforge.models.embodied.registry import build_model as _build
    return _build(model_cfg)


def build_dataloaders(model_cfg, data_cfg, training_args, ctx) -> Dict[str, "DataLoader"]:
    """Build dataloaders. Returns {"vla": dl}."""
    from loongforge.data.embodied_dataloader import build_dataloader
    dl = build_dataloader(model_cfg, data_cfg, training_args, ctx)
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
            ctx=ctx,
        )


def on_after_train_batch_fetch(model, batch, *, training_args, completed_steps, micro_step) -> None:
    """Hook after a training batch is fetched."""
    from loongforge.engines.torch.distributed.utils import unwrap_model
    raw = unwrap_model(model)
    hook = getattr(raw, "on_after_train_batch_fetch", None)
    if callable(hook):
        hook(
            args=training_args,
            completed_steps=completed_steps,
            micro_step=micro_step,
            batch=batch,
        )
