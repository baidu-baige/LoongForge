# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Reusable trainer integration for the optimizer-state-shard manager.

A model adopts the manager by subclassing ``OptimizerStateShardTrainerMixin`` and
implementing two hooks:

    _build_parameter_policy()   the model's precision/ownership policy
    _prepare_shard_module()     the nn.Module whose parameters are managed

The generic manager lifecycle -- gradient-reduction arming, parameter
publication, gradient clipping, nan scrubbing and the dcp-only training-state
guard -- lives here so a new model reuses the wiring instead of copying it. The
remaining hooks (``_after_shard_manager_built``, ``_create_inner_optimizer``,
``_after_optimizer_built``, ``_on_gradients_clipped``, ``_run_train_step``)
default to no-ops; a model overrides only what it needs.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

from loongforge.embodied.optimizer import build_optimizer
from loongforge.embodied.train.trainers.supervised.finetune_trainer import (
    FinetuneTrainer,
)

from .checkpoint_io import OptimizerStateShardCheckpointIO
from .parameter_manager import OptimizerStateShardManager

logger = logging.getLogger(__name__)

class OptimizerStateShardTrainerMixin(FinetuneTrainer):
    """Own the optimizer-state-shard manager lifecycle inside the finetune loop."""

    # -- model-supplied hooks -------------------------------------------------

    def _build_parameter_policy(self):
        """Return the model's parameter policy (precision + ownership)."""
        raise NotImplementedError

    def _prepare_shard_module(self):
        """Return the nn.Module whose parameters the manager should own."""
        raise NotImplementedError

    def _after_shard_manager_built(self) -> None:
        """Run model-specific setup once the manager exists (default: none)."""

    def _create_inner_optimizer(self):
        """Build the optimizer over the fp32 masters (default: standard build)."""
        return build_optimizer(self.model, self.training_args)

    def _after_optimizer_built(self) -> None:
        """Wire model-specific optimizer callbacks (default: none)."""

    def _on_gradients_clipped(self) -> None:
        """Hook fired right after clipping, before the norm is returned."""

    def _run_train_step(self):
        """Run the underlying training step (override to wrap it)."""
        return super()._train_step()

    # -- manager construction -------------------------------------------------

    def _build_state_shard_manager(self, module) -> OptimizerStateShardManager:
        """Build the manager from the model config's collective-precision knobs."""
        cfg = self.model_cfg
        return OptimizerStateShardManager(
            module,
            group=None,
            rank=self.ctx.rank,
            world_size=self.ctx.world_size,
            parameter_policy=self._parameter_policy,
            grad_reduce_dtype=cfg.grad_reduce_dtype,
            param_sync_precision=cfg.param_sync_precision,
            param_sync_fp8_block=cfg.param_sync_fp8_block,
            param_sync_fp8_reprime_interval=cfg.param_sync_fp8_reprime_interval,
            param_sync_fp8_include=cfg.param_sync_fp8_include,
            param_sync_bf16_with_fp8=cfg.param_sync_bf16_with_fp8,
            grad_overlap=cfg.grad_overlap,
            param_overlap=cfg.param_overlap,
            bucket_mb=cfg.comm_bucket_mb,
            grad_inflight_mb=cfg.grad_inflight_mb,
        )

    def _log_precision_summary(self) -> None:
        """Log the resolved per-class compute/collective precision on the main rank."""
        logger.info(
            "optimizer-state-shard collective precision: grad_reduce=%s "
            "parameter_sync=%s (router/gate and 1-D tensors stay fp32)",
            self._dist_optimizer.grad_reduce_mode,
            self._dist_optimizer.param_sync_precision,
        )
        for row in self._dist_optimizer.precision_summary():
            logger.info(
                "  %-16s compute=%-4s grad_reduce=%-14s param_sync=%-14s "
                "params=%d numel=%d (%.1f%%)",
                row["parameter_class"],
                row["compute"],
                row["grad_reduce"],
                row["param_sync"],
                row["params"],
                row["numel"],
                100.0 * row["numel_ratio"],
            )
        logger.info(
            "optimizer-state-shard overlap: grad=%s param=%s bucket=%dMiB",
            self.model_cfg.grad_overlap,
            self.model_cfg.param_overlap,
            self._dist_optimizer.bucket_mb,
        )

    # -- training-loop integration --------------------------------------------

    def _wrap_model_for_training(self) -> None:
        if getattr(self, "_parameter_policy", None) is None:
            self._parameter_policy = self._build_parameter_policy()
        module = self._prepare_shard_module()
        self._dist_optimizer = self._build_state_shard_manager(module)
        self._optimizer_parameter_model = self._dist_optimizer.optimizer_view()
        if self.ctx.is_main:
            self._log_precision_summary()
        self._after_shard_manager_built()

    def _should_sync_grads(self, micro: int, grad_accum: int) -> bool:
        sync = super()._should_sync_grads(micro, grad_accum)
        if micro == grad_accum - 1:
            self._dist_optimizer.begin_gradient_sync()
        return sync

    @contextmanager
    def _grad_sync_ctx(self, sync_grads: bool):
        """Let accumulation micro-steps run without any wrapper-level gating.

        The base implementation gates through the model wrapper: ``no_sync()`` for
        DDP, otherwise ``set_requires_gradient_sync`` for FSDP2. The
        replicated-compute model is neither, so it has no such method. No gating is
        needed here: backward only accumulates into ``.grad`` and the reduction is
        armed exactly once per step by ``_should_sync_grads`` on the last
        micro-step.
        """
        del sync_grads
        yield

    def _run_forward_backward_block(self):
        result = super()._run_forward_backward_block()
        self._dist_optimizer.finish_gradient_sync()
        return result

    def _train_step(self):
        self._dist_optimizer.begin_parameter_sync()
        result = self._run_train_step()
        # Not in a finally: a failed step leaves the publish plan for
        # ``clear_pending_work`` to unwind rather than draining half-issued work.
        self._dist_optimizer.finish_parameter_sync()
        return result

    def _build_optimizer(self):
        optimizer = self._create_inner_optimizer()
        self.optimizer = optimizer
        # The shared checkpoint backend owns the file layout and the aux files;
        # this object owns the rank-local FP32 masters and Muon/AdamW state.
        self.optimizer.zero_checkpoint_io = OptimizerStateShardCheckpointIO(
            self._dist_optimizer, optimizer
        )
        self._after_optimizer_built()
        return optimizer

    def _clip_gradients(self, max_norm: float) -> float:
        norm = self._dist_optimizer.clip_grad_norm_(max_norm)
        self._on_gradients_clipped()
        return norm

    def _clean_nan_gradients(self):
        for parameter in self._dist_optimizer.master.values():
            if parameter.grad is not None:
                parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    def _save_checkpoint(self):
        # The fp32 masters and their optimizer state are rank-local, and only the
        # dcp path keeps per-rank files. A legacy-format training-state save would
        # map them through the model's parameter FQNs and silently write rank0's
        # shard alone, so refuse it rather than emit a checkpoint that only looks
        # resumable.
        if self.training_args.save_training_state and (
            self.training_args.use_lora or self.training_args.save_format != "dcp"
        ):
            raise ValueError(
                "optimizer-state-shard training-state checkpoints require "
                "--save-format=dcp; pass --no-save-training-state to export "
                "weights only."
            )
        super()._save_checkpoint()


__all__ = ["OptimizerStateShardTrainerMixin"]

