# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VA adapter around the common FinetuneTrainer lifecycle."""

from __future__ import annotations

import gc
import logging
from dataclasses import replace
from typing import Tuple

import torch

from loongforge.embodied.model.lingbot_va.features import (
    GC_GENERATION2_THRESHOLD,
    feature_enabled,
)

from loongforge.embodied.train.trainers.supervised.finetune_trainer import (
    FinetuneTrainer,
)


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


class LingBotFinetuneTrainer(FinetuneTrainer):
    """LingBot-specific construction and diagnostics over the common trainer.

    The optimizer-step orchestration, gradient accumulation, loss-health guard,
    logging, checkpoint cadence, and scheduler lifecycle are inherited from
    ``FinetuneTrainer``/``BaseTrainer``. Only LingBot's nested FSDP2, DTensor
    gradient helper wiring, precision diagnostics, and final-checkpoint policy
    remain here.
    """

    def __init__(self, training_args, model_cfg, data_cfg):
        # Match the community baseline's DistributedSampler/DataLoader epoch
        # boundary; LingBot must retain the final padded distributed sample.
        training_args = replace(training_args, batch_drop_last=False)
        self._lingbot_post_step_reshard_hook = None
        self._lingbot_gc_thresholds = None
        self._lingbot_loss_guard_records = []
        super().__init__(training_args, model_cfg, data_cfg)

    def _wrap_model_for_training(self):
        if self.training_args.distributed_strategy != "fsdp":
            raise RuntimeError(
                "LingBot native nested FSDP2 requires embodied FSDP strategy"
            )
        from loongforge.embodied.model.lingbot_va.lingbot_fsdp2_adapter import (
            wrap_lingbot_torch_nested_fsdp2,
        )

        self.model = wrap_lingbot_torch_nested_fsdp2(
            self.model, self.training_args, self.ctx
        )

    def _build_optimizer(self):
        if self.training_args.distributed_strategy != "fsdp":
            raise RuntimeError(
                "LingBot native nested FSDP2 requires embodied FSDP strategy"
            )

        from loongforge.embodied.model.lingbot_va.lingbot_fsdp2_adapter import (
            apply_lingbot_fsdp2_tuning,
            register_lingbot_post_step_reshard,
        )

        apply_lingbot_fsdp2_tuning(self.model)
        optimizer = super()._build_optimizer()
        if feature_enabled("LINGBOT_FSDP_RESHARD"):
            reshard_module_count = sum(
                1
                for module in self.model.modules()
                if hasattr(module, "set_reshard_after_backward")
            )
            reshard_mode = "framework-default"
        else:
            self._lingbot_post_step_reshard_hook, reshard_module_count = (
                register_lingbot_post_step_reshard(
                    self.model,
                    optimizer,
                )
            )
            reshard_mode = "post-step"
        if self.ctx is not None and self.ctx.is_main:
            logger.info("Using common TorchFusedAdamW through FinetuneTrainer.")
            logger.info(
                "LingBot FSDP reshard mode=%s modules=%d.",
                reshard_mode,
                reshard_module_count,
            )
        return optimizer

    def _backward_loss(
        self,
        loss: torch.Tensor,
        log_loss_dict: dict,
        log_dict: dict,
        grad_accum: int,
    ) -> None:
        """Run the LingBot batched device-side loss guard.

        Same contract as the common trainer, except the spike/NaN check stays on
        the device: the per-microbatch records are drained once per optimizer
        step in ``_finish_device_loss_guard``, which replaces the base
        implementation's per-microbatch ``.item()`` synchronization. Gradient
        sync gating is the caller's ``_grad_sync_ctx``, as upstream.
        """
        threshold = self.training_args.loss_spike_threshold
        with self._stage_timers("backward-compute"):
            loss, raw_loss, invalid, spiked = _COMPILED_DEVICE_LOSS_GUARD(
                loss, grad_accum, threshold
            )
            self._lingbot_loss_guard_records.append(
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

    def _finish_device_loss_guard(self, log_dict: dict, grad_accum: int) -> None:
        records = self._lingbot_loss_guard_records
        if not records:
            return
        log_keys = tuple(key for key, _ in records[0][3])
        for _, _, _, log_items in records:
            if tuple(key for key, _ in log_items) != log_keys:
                raise RuntimeError("LingBot loss log keys changed within an optimizer step")
        packed = torch.stack(
            [
                torch.stack(
                    (
                        loss.float(),
                        invalid.float(),
                        spiked.float(),
                        *(value.float() for _, value in log_items),
                    )
                )
                for loss, invalid, spiked, log_items in records
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
            self.logger.log_loss_spike(self.completed_steps, loss_value)
            if bool(invalid_value):
                self._step_loss_is_nan = True
            self._step_loss_spiked = True

    def _clip_gradients(self, max_norm: float) -> float:
        """Clip RAB=false gradients through the LingBot DTensor helper."""
        from loongforge.embodied.model.lingbot_va.lingbot_fsdp2_adapter import (
            clip_lingbot_optimizer_gradients,
        )

        return clip_lingbot_optimizer_gradients(self.optimizer, max_norm)

    def _clean_nan_gradients(self) -> None:
        """Clean the optimizer-owned DTensor gradients used by LingBot."""
        from loongforge.embodied.model.lingbot_va.lingbot_fsdp2_adapter import (
            clean_lingbot_optimizer_gradients,
        )

        clean_lingbot_optimizer_gradients(self.optimizer)

    def _forward_backward(self) -> dict:
        """Finish LingBot-only loss diagnostics after final backward."""
        # Drop anything a previous, aborted step left behind: those records hold
        # device tensors, and mixing them into this step's drain would corrupt
        # the averaged loss logs.
        self._lingbot_loss_guard_records.clear()
        log_dict = super()._forward_backward()
        self._finish_device_loss_guard(
            log_dict, self.training_args.gradient_accumulation_steps
        )
        return log_dict

    def _train_forward(self, batch) -> Tuple[torch.Tensor, dict]:
        """Map LingBot loss names onto the trainer's logging keys."""
        loss, log_loss_dict = super()._train_forward(batch)
        return loss, _map_loss_log_dict(
            log_loss_dict,
            backward_loss=loss,
            gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
        )


    def _configure_manual_gc(self) -> None:
        """Suppress expensive full GC while preserving young-generation cleanup."""
        if not self.training_args.manual_gc:
            return super()._configure_manual_gc()
        interval = int(self.training_args.manual_gc_interval)
        if interval < 0:
            raise ValueError("--manual-gc-interval must be >= 0")
        self._lingbot_gc_thresholds = gc.get_threshold()
        gc.collect()
        gc.enable()
        generation0, generation1, generation2 = self._lingbot_gc_thresholds
        suppressed_generation2 = max(GC_GENERATION2_THRESHOLD, generation2 + 1)
        gc.set_threshold(generation0, generation1, suppressed_generation2)
        if self.ctx.is_main:
            logger.info(
                "LingBot generation-2 automatic GC suppressed (threshold=%d, manual interval=%d)",
                suppressed_generation2,
                interval,
            )

    def _restore_lingbot_gc_thresholds(self) -> None:
        if self._lingbot_gc_thresholds is None:
            return
        gc.set_threshold(*self._lingbot_gc_thresholds)
        self._lingbot_gc_thresholds = None

    def _finalize(self):
        """Restore the process-wide GC policy before the common teardown."""
        self._restore_lingbot_gc_thresholds()
        return super()._finalize()
