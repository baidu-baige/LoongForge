# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VA optimization factory.

Device-side loss guard, GC suppression, nested FSDP2 wrap, DTensor-aware
clip/clean. Returns a TorchOptimization that overrides the vanilla trainer
behavior for LingBot-specific needs.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

logger = logging.getLogger(__name__)


def build_optimization(training_args):
    """Return a TorchOptimization for LingBot-VA."""
    from loongforge.engines.torch.train_step import TorchOptimization
    import loongforge.engines.torch.train_step as ts

    from loongforge.models.embodied.lingbot_va.features import (
        GC_GENERATION2_THRESHOLD,
        feature_enabled,
    )

    state = SimpleNamespace(gc_thresholds=None)

    def prepare_args(args):
        return replace(args, batch_drop_last=False)

    def wrap_model(model, args, ctx):
        if args.distributed_strategy != "fsdp":
            raise RuntimeError(
                "LingBot native nested FSDP2 requires embodied FSDP strategy"
            )
        from loongforge.engines.torch.optimizations.lingbot_va.fsdp2_adapter import (
            wrap_lingbot_torch_nested_fsdp2,
        )
        return wrap_lingbot_torch_nested_fsdp2(model, args, ctx)

    def build_optimizer_fn(model, args, ctx):
        if args.distributed_strategy != "fsdp":
            raise RuntimeError(
                "LingBot native nested FSDP2 requires embodied FSDP strategy"
            )
        from loongforge.engines.torch.optimizations.lingbot_va.fsdp2_adapter import (
            apply_lingbot_fsdp2_tuning,
            register_lingbot_post_step_reshard,
        )
        from loongforge.engines.torch.optimizer import build_optimizer

        apply_lingbot_fsdp2_tuning(model)
        optimizer = build_optimizer(model, args)
        if feature_enabled("LINGBOT_FSDP_RESHARD"):
            reshard_module_count = sum(
                1
                for module in model.modules()
                if hasattr(module, "set_reshard_after_backward")
            )
            reshard_mode = "framework-default"
        else:
            _, reshard_module_count = register_lingbot_post_step_reshard(
                model, optimizer,
            )
            reshard_mode = "post-step"
        if ctx is not None and ctx.is_main:
            logger.info("Using common TorchFusedAdamW.")
            logger.info(
                "LingBot FSDP reshard mode=%s modules=%d.",
                reshard_mode,
                reshard_module_count,
            )
        return optimizer

    def clip_gradients(model, optimizer, max_norm):
        from loongforge.engines.torch.optimizations.lingbot_va.fsdp2_adapter import (
            clip_lingbot_optimizer_gradients,
        )
        return clip_lingbot_optimizer_gradients(optimizer, max_norm)

    def clean_nan_gradients(model, optimizer):
        from loongforge.engines.torch.optimizations.lingbot_va.fsdp2_adapter import (
            clean_lingbot_optimizer_gradients,
        )
        clean_lingbot_optimizer_gradients(optimizer)

    # Device-side loss guard functions
    def _device_loss_guard(loss, grad_accum, threshold):
        import torch
        scaled = loss / grad_accum
        detached = scaled.detach()
        invalid = ~torch.isfinite(detached)
        spiked = invalid | (detached > threshold)
        guarded = torch.where(spiked, scaled * 0.0, scaled)
        return guarded, detached, invalid, spiked

    import torch as _torch
    _compiled_guard = (
        _torch.compile(_device_loss_guard, dynamic=False, fullgraph=True)
        if getattr(_torch, "compile", None) is not None
        else _device_loss_guard
    )

    def _device_guard_backward(records, step_ctx, loss, log_loss_dict, log_dict, health):
        import torch
        grad_accum = step_ctx.training_args.gradient_accumulation_steps
        threshold = step_ctx.training_args.loss_spike_threshold
        with step_ctx.timers("backward-compute"):
            loss, raw_loss, invalid, spiked = _compiled_guard(
                loss, grad_accum, threshold
            )
            records.append(
                (
                    raw_loss,
                    invalid,
                    spiked,
                    tuple(
                        (
                            key,
                            value.detach()
                            if isinstance(value, torch.Tensor)
                            else torch.as_tensor(
                                value, device=loss.device, dtype=torch.float32
                            ),
                        )
                        for key, value in log_loss_dict.items()
                    ),
                )
            )
            loss.backward()

    def _drain_loss_guard_records(records, step_ctx, log_dict, health):
        import torch
        if not records:
            return
        grad_accum = step_ctx.training_args.gradient_accumulation_steps
        log_keys = tuple(key for key, _ in records[0][3])
        for _, _, _, log_items in records:
            if tuple(key for key, _ in log_items) != log_keys:
                raise RuntimeError("LingBot loss log keys changed within an optimizer step")
        packed = torch.stack(
            [
                torch.stack(
                    (
                        loss_val.float(),
                        inv.float(),
                        sp.float(),
                        *(value.float() for _, value in log_items),
                    )
                )
                for loss_val, inv, sp, log_items in records
            ]
        )
        rows = packed.detach().cpu().tolist()
        records.clear()
        for row in rows:
            loss_value, invalid_value, spiked_value, *log_values = row
            for key, value in zip(log_keys, log_values):
                log_dict[key] = log_dict.get(key, 0.0) + value / grad_accum
            if not bool(spiked_value):
                continue
            step_ctx.log_loss_spike(step_ctx.iteration, loss_value)
            if bool(invalid_value):
                health.nan = True
            health.spiked = True

    def forward_backward(step_ctx):
        records = []
        log_dict, health = ts.forward_backward(
            step_ctx, backward=partial(_device_guard_backward, records)
        )
        _drain_loss_guard_records(records, step_ctx, log_dict, health)
        return log_dict, health

    def configure_gc(args, ctx):
        if not args.manual_gc:
            # Fall back to base manual GC
            if not args.manual_gc:
                return
            interval = int(args.manual_gc_interval)
            if interval < 0:
                raise ValueError("--manual-gc-interval must be >= 0")
            gc.disable()
            gc.collect()
            if ctx.is_main:
                cadence = "startup only" if interval == 0 else f"every {interval} steps"
                logger.info("Manual Python GC enabled (%s)", cadence)
            return
        interval = int(args.manual_gc_interval)
        if interval < 0:
            raise ValueError("--manual-gc-interval must be >= 0")
        state.gc_thresholds = gc.get_threshold()
        gc.collect()
        gc.enable()
        generation0, generation1, generation2 = state.gc_thresholds
        suppressed_generation2 = max(GC_GENERATION2_THRESHOLD, generation2 + 1)
        gc.set_threshold(generation0, generation1, suppressed_generation2)
        if ctx.is_main:
            logger.info(
                "LingBot generation-2 automatic GC suppressed (threshold=%d, manual interval=%d)",
                suppressed_generation2,
                interval,
            )

    def close():
        if state.gc_thresholds is not None:
            gc.set_threshold(*state.gc_thresholds)
            state.gc_thresholds = None

    return TorchOptimization(
        prepare_args=prepare_args,
        configure_gc=configure_gc,
        wrap_model=wrap_model,
        build_optimizer=build_optimizer_fn,
        forward_backward=forward_backward,
        clip_gradients=clip_gradients,
        clean_nan_gradients=clean_nan_gradients,
        close=close,
    )
