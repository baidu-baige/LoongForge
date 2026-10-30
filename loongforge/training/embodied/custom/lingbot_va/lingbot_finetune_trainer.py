# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VA adapter around the common FinetuneTrainer lifecycle."""

from __future__ import annotations

import gc
import logging
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import torch

from loongforge.models.embodied.lingbot_va.features import (
    GC_GENERATION2_THRESHOLD,
    feature_enabled,
)

from loongforge.training.embodied.sft import (  # noqa: F401
    build_model,
    build_dataloaders,
    on_train_begin,
    on_after_data_iterators_initialized,
    on_after_train_batch_fetch,
)
from loongforge.training.embodied import sft


def _device_loss_guard(loss, grad_accum, threshold):
    scaled = loss / grad_accum
    detached = scaled.detach()
    invalid = ~torch.isfinite(detached)
    spiked = invalid | (detached > threshold)
    guarded = torch.where(spiked, scaled * 0.0, scaled)
    return guarded, detached, invalid, spiked


_COMPILED_DEVICE_LOSS_GUARD = (
    torch.compile(_device_loss_guard, dynamic=False, fullgraph=True)
    if getattr(torch, "compile", None) is not None
    else _device_loss_guard
)

logger = logging.getLogger(__name__)


def _map_loss_log_dict(log_loss_dict, *, backward_loss, gradient_accumulation_steps: int):
    """Map LingBot loss names to the keys the common trainer logs."""
    metric_key_map = {
        "total loss": "lingbot_total_loss",
        "video loss": "lingbot_video_loss",
        "action loss": "lingbot_diffusion_action_loss",
    }
    mapped_log_dict = {}
    for key, value in log_loss_dict.items():
        metric_key = metric_key_map.get(key, f"lingbot_{key.replace(' ', '_')}")
        if metric_key == "action_loss":
            metric_key = "lingbot_logged_action_loss"
        mapped_log_dict[metric_key] = value
    mapped_log_dict["action_loss"] = backward_loss.detach() * float(
        max(1, int(gradient_accumulation_steps))
    )
    return mapped_log_dict


def forward(model, batch, *, iteration: int, training_args) -> tuple:
    """LingBot forward: delegates to the common embodied SFT, then maps loss names."""
    loss, logs = sft.forward(model, batch, iteration=iteration, training_args=training_args)
    return loss, _map_loss_log_dict(
        logs,
        backward_loss=loss,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    )


def build_hooks(training_args):
    """Return the TrainerHooks for LingBot-VA."""
    from loongforge.engines.torch.runner import TrainerHooks
    import loongforge.engines.torch.runner as ts

    state = SimpleNamespace(gc_thresholds=None)

    def prepare_args(args):
        # Match the community baseline's DistributedSampler/DataLoader epoch
        # boundary; LingBot must retain the final padded distributed sample.
        return replace(args, batch_drop_last=False)

    def wrap_model(model, args, ctx):
        if args.distributed_strategy != "fsdp":
            raise RuntimeError(
                "LingBot native nested FSDP2 requires embodied FSDP strategy"
            )
        from loongforge.training.embodied.custom.lingbot_va.fsdp2_adapter import (
            wrap_lingbot_torch_nested_fsdp2,
        )
        return wrap_lingbot_torch_nested_fsdp2(model, args, ctx)

    def build_optimizer_fn(model, args, ctx):
        if args.distributed_strategy != "fsdp":
            raise RuntimeError(
                "LingBot native nested FSDP2 requires embodied FSDP strategy"
            )
        from loongforge.engines.torch.distributed.fsdp_utils.optimizer_grads import (
            register_post_step_reshard,
        )
        from loongforge.training.embodied.custom.lingbot_va.fsdp2_adapter import (
            apply_lingbot_fsdp2_tuning,
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
            _, reshard_module_count = register_post_step_reshard(
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
        """Clip RAB=false gradients through the DTensor helper."""
        from loongforge.engines.torch.distributed.fsdp_utils.optimizer_grads import (
            clip_optimizer_gradients,
        )
        return clip_optimizer_gradients(optimizer, max_norm)

    def clean_nan_gradients(model, optimizer):
        """Clean the optimizer-owned DTensor gradients used by LingBot."""
        from loongforge.engines.torch.distributed.fsdp_utils.optimizer_grads import (
            clean_optimizer_gradients,
        )
        clean_optimizer_gradients(optimizer)

    def _device_guard_backward(records, step_ctx, loss, log_loss_dict, log_dict, health):
        """Run the LingBot batched device-side loss guard.

        Same contract as the common trainer, except the spike/NaN check stays on
        the device: the per-microbatch records are drained once per optimizer
        step in ``_finish_device_loss_guard``, which replaces the base
        implementation's per-microbatch ``.item()`` synchronization. Gradient
        sync gating is the caller's ``_grad_sync_ctx``, as upstream.
        """
        grad_accum = step_ctx.training_args.gradient_accumulation_steps
        threshold = step_ctx.training_args.loss_spike_threshold
        with step_ctx.timers("backward-compute"):
            loss, raw_loss, invalid, spiked = _COMPILED_DEVICE_LOSS_GUARD(
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
        """Finish LingBot-only loss diagnostics after final backward."""
        records = []
        log_dict, health = ts.forward_backward(
            step_ctx, backward=partial(_device_guard_backward, records)
        )
        _drain_loss_guard_records(records, step_ctx, log_dict, health)
        return log_dict, health

    def configure_gc(args, ctx):
        """Suppress expensive full GC while preserving young-generation cleanup."""
        if not args.manual_gc:
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
        """Restore the process-wide GC policy before the common teardown."""
        if state.gc_thresholds is not None:
            gc.set_threshold(*state.gc_thresholds)
            state.gc_thresholds = None

    return TrainerHooks(
        prepare_args=prepare_args,
        configure_gc=configure_gc,
        wrap_model=wrap_model,
        build_optimizer=build_optimizer_fn,
        forward_backward=forward_backward,
        clip_gradients=clip_gradients,
        clean_nan_gradients=clean_nan_gradients,
        close=close,
    )
