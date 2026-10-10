# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Single optimizer step: dataclasses, forward/backward, grad clip, update.

All functions operate on an explicit ``StepContext`` rather than ``self.*``.
``TorchOptimization`` (frozen dataclass, all-None = vanilla FinetuneTrainer)
lets per-model optimizations override any slice without subclassing.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════


@dataclass
class LossHealth:
    nan: bool = False
    spiked: bool = False


@dataclass
class StepResult:
    log_dict: Dict[str, Any]
    grad_norm: float
    updated: bool
    nan: bool
    skipped: bool


@dataclass
class StepContext:
    """Explicit runtime state for one training run.

    Built once by the runner; ``iteration`` is updated every step.
    """
    model: Any  # nn.Module
    optimizer: Any  # torch.optim.Optimizer
    lr_scheduler: Any
    training_args: Any
    model_cfg: Any
    ctx: Any  # DistributedContext
    timers: Any  # StageTimers
    method: ModuleType
    optimization: "TorchOptimization"
    fetch_cpu_batch: Callable[[], Any]
    log_loss_spike: Callable[[int, float], None]
    iteration: int = 0
    static_graph_bootstrapped: bool = False


@dataclass(frozen=True)
class TorchOptimization:
    # consulted by torch_runner
    prepare_args: Optional[Callable] = None
    configure_gc: Optional[Callable] = None
    wrap_model: Optional[Callable] = None
    build_optimizer: Optional[Callable] = None
    after_setup: Optional[Callable] = None
    on_step_end: Optional[Callable] = None
    close: Optional[Callable] = None
    # consulted by train_step
    move_batch_to_device: Optional[Callable] = None
    forward_backward: Optional[Callable] = None
    clean_nan_gradients: Optional[Callable] = None
    clip_gradients: Optional[Callable] = None
    run_step: Optional[Callable] = None


# ═══════════════════════════════════════════════════════════════════
# Module-level functions
# ═══════════════════════════════════════════════════════════════════


def _cfg_bool(cfg, key: str, default: bool = False) -> bool:
    value = getattr(cfg, key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_dtype(dtype_str: str):
    import torch as _torch
    mapping = {
        "bfloat16": _torch.bfloat16,
        "float16": _torch.float16,
        "float32": _torch.float32,
    }
    return mapping[dtype_str]


def train_autocast(step_ctx: StepContext):
    """Return the autocast context for training forward passes."""
    import torch as _torch
    if _cfg_bool(step_ctx.model_cfg, "disable_train_autocast", False):
        return nullcontext()
    dtype = _resolve_dtype(step_ctx.training_args.dtype)
    return _torch.autocast("cuda", dtype=dtype)


def train_fp8_ctx(step_ctx: StepContext):
    """Return the FP8 forward context, or a no-op when ``--fp8`` is off.

    ``te.fp8_autocast`` has to enclose the whole model call, so both the
    standard forward and the CUDA-graph capture go through this.
    """
    from loongforge.engines.torch.distributed.fp8_utils import resolve_fp8_forward_ctx

    return resolve_fp8_forward_ctx(step_ctx.training_args)


def forward(step_ctx: StepContext, batch):
    """Forward pass with autocast: delegates to method.forward."""
    with train_fp8_ctx(step_ctx), train_autocast(step_ctx):
        loss, log_loss_dict = step_ctx.method.forward(
            step_ctx.model, batch,
            iteration=step_ctx.iteration,
            training_args=step_ctx.training_args,
        )
    return loss, log_loss_dict


def move_batch(step_ctx: StepContext, batch):
    """Move a fetched batch to the model device."""
    opt = step_ctx.optimization
    if opt.move_batch_to_device is not None:
        return opt.move_batch_to_device(batch, step_ctx.model)
    return _move_batch_to_device(batch, step_ctx.model)


def _move_batch_to_device(batch, model):
    """Default batch-to-device transfer."""
    device = next(model.parameters()).device
    if hasattr(batch, "to"):
        batch = batch.to(device)
    return batch


def prepare_model_for_train_step(model) -> None:
    """Put model in train mode and let policies re-freeze eval-only modules."""
    from loongforge.engines.torch.distributed.utils import unwrap_model
    model.train()
    raw = unwrap_model(model)
    if hasattr(raw, "set_frozen_modules_to_eval_mode"):
        raw.set_frozen_modules_to_eval_mode()


def before_microbatch(step_ctx: StepContext, batch, micro_step: int) -> None:
    """Per-microbatch hook: model callbacks then train-mode setup."""
    step_ctx.method.on_after_train_batch_fetch(
        step_ctx.model, batch,
        training_args=step_ctx.training_args,
        completed_steps=step_ctx.iteration,
        micro_step=micro_step,
    )
    prepare_model_for_train_step(step_ctx.model)


@contextmanager
def grad_sync_ctx(step_ctx: StepContext, sync_grads: bool):
    """Gate cross-rank gradient sync for one micro-step.

    Must wrap the forward as well as the backward.
    """
    if sync_grads or not step_ctx.ctx.is_distributed:
        yield
    elif hasattr(step_ctx.model, "no_sync"):
        if getattr(step_ctx.model, "static_graph", False) and not step_ctx.static_graph_bootstrapped:
            yield
            step_ctx.static_graph_bootstrapped = True
        else:
            with step_ctx.model.no_sync():
                yield
    else:
        # FSDP2 (fully_shard)
        step_ctx.model.set_requires_gradient_sync(False)
        yield
        step_ctx.model.set_requires_gradient_sync(True)


def backward_loss(
    step_ctx: StepContext,
    loss,
    log_loss_dict: dict,
    log_dict: dict,
    health: LossHealth,
) -> None:
    """Scale + spike-guard + backward, accumulating losses into log_dict."""
    import torch as _torch
    grad_accum = step_ctx.training_args.gradient_accumulation_steps
    threshold = step_ctx.training_args.loss_spike_threshold
    with step_ctx.timers("backward-compute"):
        raw_loss = loss
        loss = raw_loss / grad_accum
        loss_val = loss.detach().item()
        is_nan = bool(_torch.isnan(loss) or _torch.isinf(loss))
        if is_nan or loss_val > threshold:
            step_ctx.log_loss_spike(step_ctx.iteration, loss_val)
            if is_nan:
                health.nan = True
            health.spiked = True
            loss = loss * 0.0

        loss.backward()

    for key, value in log_loss_dict.items():
        import torch as _torch2
        v = value.detach().item() if isinstance(value, _torch2.Tensor) else float(value)
        log_dict[key] = log_dict.get(key, 0.0) + v / grad_accum


def forward_backward(step_ctx: StepContext, backward=None) -> Tuple[dict, LossHealth]:
    """Zero-grad + gradient-accumulation loop.

    Returns (log_dict, LossHealth).
    """
    if backward is None:
        backward = backward_loss
    st = step_ctx.timers
    args = step_ctx.training_args
    grad_accum = args.gradient_accumulation_steps
    log_dict: Dict[str, float] = {}
    health = LossHealth()
    with st("forward-backward"):
        for micro in range(grad_accum):
            with st("batch-generator"):
                cpu_batch = step_ctx.fetch_cpu_batch()
                batch = move_batch(step_ctx, cpu_batch)
            before_microbatch(step_ctx, batch, micro)
            sync_grads = micro == grad_accum - 1
            with grad_sync_ctx(step_ctx, sync_grads):
                with st("forward-compute"):
                    loss, log_loss_dict = forward(step_ctx, batch)
                backward(step_ctx, loss, log_loss_dict, log_dict, health)
    return log_dict, health


def _clip_gradients(model, optimizer, max_norm: float) -> float:
    """Gradient clipping. Returns the pre-clip global gradient norm."""
    if hasattr(optimizer, "clip_grad_norm"):
        return optimizer.clip_grad_norm(max_norm)
    from loongforge.engines.torch.optimizer import clip_gradients
    return clip_gradients(model, max_norm)


def _clean_nan_gradients(model, optimizer) -> None:
    """Replace NaN/Inf gradients with 0."""
    from loongforge.engines.torch.optimizer import clean_nan_gradients
    clean_nan_gradients(model)


def train_step(step_ctx: StepContext) -> StepResult:
    """One optimizer step. Returns StepResult."""
    opt = step_ctx.optimization
    if opt.run_step is not None:
        return opt.run_step(step_ctx)
    args, st = step_ctx.training_args, step_ctx.timers

    with st("optimizer-zero-grad"):
        step_ctx.optimizer.zero_grad()

    log_dict, health = (opt.forward_backward or forward_backward)(step_ctx)

    if args.check_for_nan_in_loss_and_grad:
        with st("nan-grad-cleanup"):
            (opt.clean_nan_gradients or _clean_nan_gradients)(step_ctx.model, step_ctx.optimizer)

    with st("grad-clip"):
        if args.clip_grad > 0:
            grad_norm = (opt.clip_gradients or _clip_gradients)(step_ctx.model, step_ctx.optimizer, args.clip_grad)
        else:
            from loongforge.engines.torch.optimizer import get_grad_norm
            grad_norm = get_grad_norm(step_ctx.model)

    with st("optimizer"):
        with st("optimizer-inner-step"):
            step_ctx.optimizer.step()
        with st("optimizer-scheduler-step"):
            step_ctx.lr_scheduler.step()

    return StepResult(log_dict, grad_norm, True, health.nan, health.spiked)
