# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-N1.6 finetune trainer.

Extends :class:`FinetuneTrainer` with the GR00T-specific per-microbatch CUDA
graph runner. The runner itself lives next to this file
(``per_microbatch_cuda_graph.py``) because it depends on GR00T-only types
(``GrootN1d6PreparedBatch``) and probes for the GR00T action head via
``action_encoder``/``sample_time``.

Only the wrapping step (``_wrap_model_for_training``) and the inner
forward/backward block (``_run_forward_backward_block``) differ from the
generic FinetuneTrainer; all data / optimizer / checkpoint paths are inherited
unchanged. The base trainer remains unaware of any CUDA-graph or runner
concepts.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from loongforge.embodied.distributed.parallel import wrap_model
from loongforge.embodied.train.utils.utils import resolve_dtype
from loongforge.embodied.train.trainers.supervised.finetune_trainer import FinetuneTrainer
from loongforge.embodied.train.trainers.custom.groot_n1_6.per_microbatch_cuda_graph import (
    GrootN1d6PerMicrobatchCudaGraphRunner,
)

logger = logging.getLogger(__name__)


class GrootN1d6Trainer(FinetuneTrainer):
    """GR00T-N1.6 finetune trainer with per-microbatch CUDA graph support.

    Behavior is identical to :class:`FinetuneTrainer` except that when CUDA
    graph is enabled (``--cuda-graph-impl local --cuda-graph-scope
    per_microbatch``), a :class:`GrootN1d6PerMicrobatchCudaGraphRunner` takes
    over zero_grad + forward + backward + gradient sync for each optimizer
    step. The rest of the training loop (clipping, optimizer, scheduler,
    logging, checkpointing) is inherited as-is from the base loop.
    """

    def __init__(self, training_args, model_cfg, data_cfg):
        super().__init__(training_args, model_cfg, data_cfg)
        self._train_step_runner: GrootN1d6PerMicrobatchCudaGraphRunner | None = None

    def _wrap_model_for_training(self) -> None:
        """Wrap the model for training, installing a CUDA graph runner when enabled.

        When CUDA graph is disabled this delegates to the base implementation
        (standard DDP/FSDP wrapping). When CUDA graph is enabled, this method
        takes over wrapping: it either calls :func:`wrap_model` (DDP sync inside
        the graph) or simply moves the model to the training device + dtype and
        broadcasts parameters from rank 0 (when DDP sync is handled outside the
        graph by the runner's manual all-reduce path). Finally it instantiates
        the runner.
        """
        if not GrootN1d6PerMicrobatchCudaGraphRunner.is_enabled(self):
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

        # When the model is not wrapped by DDP, parameters were not broadcast
        # from rank 0 by the framework — do it manually so every rank starts
        # CUDA graph capture from identical weights.
        if ctx.is_distributed and ctx.world_size > 1 and not hasattr(self.model, "module"):
            with torch.no_grad():
                for tensor in list(self.model.parameters()) + list(self.model.buffers()):
                    if tensor is not None and tensor.device.type == "cuda":
                        dist.broadcast(tensor, src=0)

        self._train_step_runner = GrootN1d6PerMicrobatchCudaGraphRunner(self)
        logger.info(
            "Using model-managed train step runner: %s",
            self._train_step_runner.__class__.__name__,
        )

    def _run_forward_backward_block(self) -> dict:
        """Route the inner forward/backward block through the CUDA graph runner.

        When the runner is active, it manages zero_grad + forward + backward +
        gradient sync internally and returns an ``output`` dict whose tensor
        values are converted into the same ``log_dict`` shape that
        :meth:`FinetuneTrainer._forward_backward` produces. Otherwise the base
        implementation runs unchanged.
        """
        if self._train_step_runner is None:
            return super()._run_forward_backward_block()

        with self._stage_timers("forward-backward"):
            self._train_step_runner.zero_grad()
            output, _loss_val = self._train_step_runner.step()
        return {
            key: (value.detach().item() if isinstance(value, torch.Tensor) else float(value))
            for key, value in output.items()
        }

    def _clean_nan_gradients(self) -> None:
        """Skip host-side NaN/Inf gradient cleanup when the CUDA graph runner is active.

        Under per-microbatch CUDA graph capture/replay the parameter ``.grad``
        tensors are static buffers tied to the captured graphs. Mutating them
        outside the graph (e.g. via :func:`torch.nan_to_num`) would violate the
        static-buffer invariant. :mod:`validators` warns when CUDA graph mode is
        enabled, so this override matches that contract by no-oping while the
        runner is installed and otherwise delegating to the base implementation.
        """
        if self._train_step_runner is not None:
            return
        super()._clean_nan_gradients()
