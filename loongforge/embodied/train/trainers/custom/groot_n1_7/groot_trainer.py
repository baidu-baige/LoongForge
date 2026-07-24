# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-N1.7 finetune trainer with per-microbatch CUDA graph support."""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from loongforge.embodied.distributed.parallel import wrap_model
from loongforge.embodied.distributed.utils import unwrap_model
from loongforge.embodied.train.trainers.custom.groot_n1_7.per_microbatch_cuda_graph import (
    GrootN1d7PerMicrobatchCudaGraphRunner,
)
from loongforge.embodied.train.trainers.supervised.finetune_trainer import FinetuneTrainer
from loongforge.embodied.train.utils.utils import resolve_dtype

logger = logging.getLogger(__name__)


class GrootN1d7Trainer(FinetuneTrainer):
    """GR00T-N1.7 finetune trainer with optional per-microbatch CUDA graph."""

    def __init__(self, training_args, model_cfg, data_cfg):
        super().__init__(training_args, model_cfg, data_cfg)
        self._train_step_runner: GrootN1d7PerMicrobatchCudaGraphRunner | None = None

    def _wrap_model_for_training(self) -> None:
        """Wrap the model, installing the CUDA graph runner when enabled."""
        if not GrootN1d7PerMicrobatchCudaGraphRunner.is_enabled(self):
            super()._wrap_model_for_training()
            return

        training_args = self.training_args
        ctx = self.ctx
        if (
            ctx.is_distributed
            and ctx.world_size > 1
            and training_args.cuda_graph_ddp_sync_in_graph
        ):
            self.model = wrap_model(self.model, training_args, ctx)
        else:
            self.model = self.model.to(dtype=resolve_dtype(training_args.dtype), device=ctx.device)

        if ctx.is_distributed and ctx.world_size > 1 and not hasattr(self.model, "module"):
            with torch.no_grad():
                for tensor in list(self.model.parameters()) + list(self.model.buffers()):
                    if tensor is not None and tensor.device.type == "cuda":
                        dist.broadcast(tensor, src=0)

        self._train_step_runner = GrootN1d7PerMicrobatchCudaGraphRunner(self)
        logger.info(
            "Using model-managed train step runner: %s",
            self._train_step_runner.__class__.__name__,
        )

    def _run_forward_backward_block(self) -> dict:
        """Route forward/backward through the CUDA graph runner when active."""
        if self._train_step_runner is None:
            return super()._run_forward_backward_block()

        with self._stage_timers("forward-backward"):
            self._train_step_runner.zero_grad()
            output, _loss_val = self._train_step_runner.step()
        return {
            key: (value.detach().item() if isinstance(value, torch.Tensor) else float(value))
            for key, value in output.items()
        }

    def _move_batch_to_device(self, batch):
        """Move batch to device and precompute Qwen metadata before graph capture."""
        batch = super()._move_batch_to_device(batch)
        if self._train_step_runner is None:
            return batch

        raw_model = unwrap_model(self.model)
        backbone = getattr(getattr(raw_model, "model", None), "backbone", None)
        prepare = getattr(backbone, "prepare_cuda_graph_batch", None)
        if callable(prepare):
            prepare(batch)
        return batch

    def _clean_nan_gradients(self) -> None:
        """Skip host-side gradient cleanup while graph-owned grad buffers are active."""
        if self._train_step_runner is not None:
            return
        super()._clean_nan_gradients()
