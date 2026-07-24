# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VA adapter around the common FinetuneTrainer lifecycle."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Tuple

import torch

from loongforge.embodied.model.lingbot_va.features import feature_enabled

from loongforge.embodied.model.lingbot_va.training_diagnostics import (
    LingBotTrainingDiagnostics,
)
from loongforge.embodied.train.trainers.supervised.finetune_trainer import (
    FinetuneTrainer,
)

logger = logging.getLogger(__name__)


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
        self._lingbot_training_diagnostics = LingBotTrainingDiagnostics()
        self._lingbot_post_step_reshard_hook = None
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
        self._lingbot_post_step_reshard_hook, reshard_module_count = (
            register_lingbot_post_step_reshard(self.model, optimizer)
        )
        if self.ctx is not None and self.ctx.is_main:
            logger.info("Using common TorchFusedAdamW through FinetuneTrainer.")
            logger.info(
                "Registered LingBot post-step reshard hook for %d FSDP modules.",
                reshard_module_count,
            )
        return optimizer

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

    def _train_forward(self, batch) -> Tuple[torch.Tensor, dict]:
        diagnostics = self._lingbot_training_diagnostics
        diagnostics.before_forward(
            batch,
            ctx=self.ctx,
            completed_steps=self.completed_steps,
            gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
        )
        loss, log_loss_dict = super()._train_forward(batch)
        diagnostics.after_forward(
            log_loss_dict,
            ctx=self.ctx,
            gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
        )
        if diagnostics.use_baseline_style_loss_log():
            return loss, {}
        return loss, diagnostics.map_loss_log_dict(
            log_loss_dict,
            backward_loss=loss,
            gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
        )

    def _finalize(self):
        if not feature_enabled("LINGBOT_SKIP_FINAL_CHECKPOINT"):
            return super()._finalize()

        from loongforge.embodied.distributed.checkpoint import flush_pending_save

        if self.ctx is not None and self.ctx.is_main:
            logger.info(
                "LingBot final checkpoint disabled by LINGBOT_SKIP_FINAL_CHECKPOINT."
            )
        flush_pending_save(self.ctx)
        self.logger.finish()
        self.ctx.barrier()
        self.ctx.destroy()
