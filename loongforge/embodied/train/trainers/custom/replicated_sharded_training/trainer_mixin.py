# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Reusable trainer integration for the replicated-sharded manager.

A model adopts the manager by subclassing ``ReplicatedShardedTrainer`` and
implementing two hooks:

    _build_parameter_policy()   the model's precision/ownership policy
    _prepare_shard_module()     the nn.Module whose parameters are managed

The generic manager lifecycle -- gradient-reduction arming, parameter
publication, gradient clipping, nan scrubbing and the dcp-only training-state
guard -- lives here so a new model reuses the wiring instead of copying it. The
remaining hooks (``_after_shard_manager_built``, ``_create_inner_optimizer``,
``_after_optimizer_built``, ``_on_gradients_clipped``, ``_run_train_step``)
default to no-ops; a model overrides only what it needs.

This class wires the distributed optimizer runtime (and the optional look-ahead
pipeline) directly and drives them from the finetune loop's hook points. The
runtime (``self._optimizer_runtime``) is the single seam: model code reaches the
resolved policy, the compute replicas and the partial-publish hooks through it,
never through the manager it wraps.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

from loongforge.embodied.optimizer import build_optimizer
from loongforge.embodied.train.trainers.supervised.finetune_trainer import (
    FinetuneTrainer,
)

from .parameter_manager import CollectiveConfig
from .optimizer_runtime import (
    DistributedContext,
    build_optimizer_runtime,
)

logger = logging.getLogger(__name__)


class ReplicatedShardedTrainer(FinetuneTrainer):
    """Own the replicated-sharded manager lifecycle inside the finetune loop."""

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

    # -- runtime construction -------------------------------------------------

    def _build_async_provider(self):
        """Return an ``AsyncTargetProvider``, or ``None`` for no look-ahead."""
        return None

    def _build_optimizer_runtime(self, module):
        """Build the distributed optimizer runtime over ``module``."""
        return build_optimizer_runtime(
            module,
            self._parameter_policy,
            DistributedContext(rank=self.ctx.rank, world_size=self.ctx.world_size),
            CollectiveConfig.from_model_config(self.model_cfg),
        )

    def _build_async_pipeline(self):
        """Build the bounded look-ahead pipeline, or ``None`` without a provider.

        A model without look-ahead targets should not pay for a worker thread, so
        the pipeline is only built when ``_build_async_provider`` returns one.
        """
        provider = self._build_async_provider()
        if provider is None:
            return None
        from .async_runtime import BoundedAsyncPipeline

        return BoundedAsyncPipeline(
            provider,
            max_depth=int(self.training_args.gradient_accumulation_steps),
        )

    def _log_precision_summary(self) -> None:
        """Log the resolved per-class compute/collective precision on the main rank."""
        logger.info(
            "replicated-sharded collective precision: grad_reduce=%s "
            "parameter_sync=%s (router/gate and 1-D tensors stay fp32)",
            self._optimizer_runtime.grad_reduce_mode,
            self._optimizer_runtime.param_sync_precision,
        )
        for row in self._optimizer_runtime.precision_summary():
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
        plan = self._optimizer_runtime.summary()
        logger.info(
            "replicated-sharded overlap: grad=%s param=%s bucket=%dMiB",
            plan["grad_overlap"],
            plan["param_overlap"],
            plan["bucket_mb"],
        )

    # -- training-loop integration --------------------------------------------

    def _wrap_model_for_training(self) -> None:
        if getattr(self, "_parameter_policy", None) is None:
            self._parameter_policy = self._build_parameter_policy()
        module = self._prepare_shard_module()
        self._optimizer_runtime = self._build_optimizer_runtime(module)
        self._async_pipeline = self._build_async_pipeline()
        self._optimizer_parameter_model = self._optimizer_runtime.optimizer_view()
        if self.ctx.is_main:
            self._log_precision_summary()
        self._after_shard_manager_built()

    def _should_sync_grads(self, micro: int, grad_accum: int) -> bool:
        sync = super()._should_sync_grads(micro, grad_accum)
        if micro == grad_accum - 1:
            self._optimizer_runtime.begin_gradient_sync()
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
        self._optimizer_runtime.finish_gradient_sync()
        return result

    def _train_step(self):
        self._optimizer_runtime.begin_parameter_sync()
        result = self._run_train_step()
        # Not in a finally: a failed step leaves the publish plan for
        # ``_close_runtime`` to unwind rather than draining half-issued work.
        self._optimizer_runtime.finish_parameter_sync()
        return result

    def _build_optimizer(self):
        optimizer = self._create_inner_optimizer()
        self.optimizer = optimizer
        # The shared checkpoint backend owns the file layout and the aux files; the
        # IO the runtime installs owns the rank-local FP32 masters and the
        # Muon/AdamW state that follows them.
        self._optimizer_runtime.attach_optimizer(optimizer)
        self._after_optimizer_built()
        return optimizer

    def _clip_gradients(self, max_norm: float) -> float:
        norm = self._optimizer_runtime.clip_grad_norm_(max_norm)
        self._on_gradients_clipped()
        return norm

    def _clean_nan_gradients(self):
        self._optimizer_runtime.scrub_nan_gradients()

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
                "replicated-sharded training-state checkpoints require "
                "--save-format=dcp; pass --no-save-training-state to export "
                "weights only."
            )
        super()._save_checkpoint()

    def _finalize(self):
        # After the base finalize, not before: the final checkpoint is written in
        # there and closing the runtime abandons any half-issued collective.
        try:
            return super()._finalize()
        finally:
            self._close_runtime()

    def _close_runtime(self) -> None:
        """Shut the pipeline and optimizer runtime down, failures and all.

        A partially built or already failed trainer still has to release its
        worker thread and abandon half-issued collectives, so no single failure
        may skip the rest.
        """
        errors = []
        for shutdown in (
            lambda: getattr(self, "_async_pipeline", None)
            and self._async_pipeline.close(),
            lambda: getattr(self, "_optimizer_runtime", None)
            and self._optimizer_runtime.close(),
        ):
            try:
                shutdown()
            except Exception as exc:  # keep closing the rest
                logger.warning(
                    "replicated-sharded shutdown step failed", exc_info=True
                )
                errors.append(exc)
        if errors:
            raise errors[0]


__all__ = ["ReplicatedShardedTrainer"]

