# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Reusable trainer integration for ``CustomDpOptimizer``.

A model can run on custom_dp with no code at all (``--trainer-type CustomDpTrainer``): the
two hooks below have neutral defaults, and the collective knobs come from the
``--custom-dp-*`` arguments. A model that wants a precision policy of its own
subclasses ``CustomDpTrainer`` and overrides:

    _build_parameter_policy()   the model's precision/ownership policy
    _prepare_module()       the nn.Module whose parameters are managed

The strategy itself -- gradient reduction, parameter publication, clipping,
nan scrubbing and the Muon publish wiring -- lives in ``CustomDpOptimizer``.
This class builds it and routes the finetune loop's hook points to it, plus the
dcp-only training-state guard. The remaining hooks (``_after_manager_built``,
``_muon_lr_scale_for``, ``_register_optimizer_hooks``, ``_after_grad_norm``) default to no-ops; a model
overrides only what it needs.

The optimizer is built in ``_wrap_model_for_training``, not ``_build_optimizer``:
its parameter plan has to capture the module tree before a model hook (e.g.
regional compile) touches it.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

from loongforge.embodied.distributed.custom_dp_utils.precision_policy import (
    DefaultParameterPolicy,
)
from loongforge.embodied.optimizer import build_optimizer
from loongforge.embodied.train.trainers.supervised.finetune_trainer import (
    FinetuneTrainer,
)

from .custom_dp_optimizer import CollectiveConfig, CustomDpOptimizer

logger = logging.getLogger(__name__)


class CustomDpTrainer(FinetuneTrainer):
    """Drive ``CustomDpOptimizer`` from the finetune loop."""

    # True only while ``_train_step`` runs. Evaluation reads the same loader names
    # as training, so a subclass that serves look-ahead batches from
    # ``_fetch_batch`` uses this to keep them out of evaluation.
    _in_train_step = False

    # -- model-supplied hooks (all optional) ----------------------------------

    def _build_parameter_policy(self):
        """Return the model's parameter policy (precision + ownership).

        The default keeps every parameter's dtype, holds 1-D tensors at fp32
        collectives and routes everything to the registry optimizer. A model that
        needs fp32 compute, fp32-critical collectives or Muon overrides this and
        returns a ``ParameterClassPolicy``.
        """
        return DefaultParameterPolicy()

    def _prepare_module(self):
        """Return the module custom_dp manages, already on the training device.

        custom_dp replaces ``wrap_model``, so this default repeats the steps of it that
        are not DDP/FSDP specific: activation checkpointing, then the cast to
        ``--dtype`` (left alone for mixed-dtype models, as ``wrap_model`` does).
        FP8 conversion is not repeated; the validator rejects ``--fp8`` under custom_dp.
        A model managing a sub-module, or checkpointing on its own, overrides this.
        """
        from loongforge.embodied.distributed.activation_checkpointing import (
            apply_activation_checkpointing,
        )
        from loongforge.embodied.distributed.utils import is_mixed_param_dtype
        from loongforge.embodied.train.training_args import parse_dtype_from_str

        args = self.training_args
        apply_activation_checkpointing(
            self.model,
            args.activation_checkpoint_module_patterns,
            args.activation_checkpoint_skip_modules,
            fp8_backend=None,
        )
        if is_mixed_param_dtype(self.model, trainable_only=False):
            return self.model.to(device=self.ctx.device)
        return self.model.to(
            dtype=parse_dtype_from_str(args.dtype), device=self.ctx.device
        )

    def _after_manager_built(self) -> None:
        """Run model-specific setup once the optimizer exists (default: none)."""

    def _create_inner_optimizer(self, dp_optimizer):
        """Build the optimizer over the fp32 masters (default: standard build).

        The masters are the optimizer's parameters, not the compute replicas:
        the replicas' ``.grad`` is consumed and cleared by the reduce-to-owner
        step, so an optimizer built on ``self.model`` would see no gradients and
        silently leave the run untrained.
        """
        if self.training_args.optimizer.lower() == "muon":
            return self._build_muon_optimizer(dp_optimizer)
        return build_optimizer(dp_optimizer.optimizer_view(), self.training_args)

    def _build_muon_optimizer(self, dp_optimizer):
        """Build Muon over the masters plus its AdamW fallback groups.

        Muon belongs to this strategy rather than to the optimizer registry: it
        optimizes the fp32 masters, which only exist here. The Muon/AdamW split
        comes from the plan, so the builder is handed two ready groups.
        """
        from .optimizer_builder import build_muon_optimizer

        args = self.training_args
        muon_parameters, adamw_parameters = (
            dp_optimizer.parameter_plan.split_masters_by_optimizer()
        )
        optimizer = build_muon_optimizer(
            muon_parameters,
            lr=args.lr_base,
            adamw_parameters=adamw_parameters,
            weight_decay=args.weight_decay,
            momentum=args.muon_momentum,
            nesterov=args.muon_nesterov,
            ns_steps=args.muon_ns_steps,
            adjust_lr_fn=args.muon_adjust_lr_fn,
            lr_scale_for=self._muon_lr_scale_for(),
        )
        self._register_optimizer_hooks(optimizer)
        return optimizer

    def _muon_lr_scale_for(self):
        """Return ``lr_scale_for(name)``, or ``None`` for a single LR.

        A model whose parameters need per-parameter LR scaling (e.g. MoE experts)
        overrides this; the default applies the base LR to everything.
        """
        return None

    def _register_optimizer_hooks(self, optimizer) -> None:
        """Register model-specific optimizer hooks (default: none)."""

    def _after_grad_norm(self) -> None:
        """Hook fired right after the pre-clip global norm was computed."""

    # -- optimizer construction -----------------------------------------------

    def _shut_down_lookahead(self) -> None:
        """Hook: release whatever a subclass started ahead of the step."""

    def _build_custom_dp_optimizer(self, module):
        """Build ``CustomDpOptimizer`` over ``module``; the inner one comes from the hooks."""
        return CustomDpOptimizer(
            module,
            self._parameter_policy,
            self._create_inner_optimizer,
            rank=self.ctx.rank,
            world_size=self.ctx.world_size,
            collective_config=CollectiveConfig.from_sources(
                self.model_cfg, self.training_args
            ),
        )

    def _log_precision_summary(self) -> None:
        """Log the resolved per-class compute/collective precision on the main rank."""
        logger.info(
            "custom_dp collective precision: grad_reduce=%s "
            "parameter_sync=%s (router/gate and 1-D tensors stay fp32)",
            self._custom_dp.grad_reduce_mode,
            self._custom_dp.param_sync_precision,
        )
        for row in self._custom_dp.precision_summary():
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
        plan = self._custom_dp.collective_summary()
        logger.info(
            "custom_dp overlap: grad=%s param=%s bucket=%dMiB",
            plan["grad_overlap"],
            plan["param_overlap"],
            plan["bucket_mb"],
        )

    # -- training-loop integration --------------------------------------------

    def _wrap_model_for_training(self) -> None:
        if getattr(self, "_parameter_policy", None) is None:
            self._parameter_policy = self._build_parameter_policy()
        module = self._prepare_module()
        self._custom_dp = self._build_custom_dp_optimizer(module)
        if self.ctx.is_main:
            self._log_precision_summary()
        self._after_manager_built()

    @contextmanager
    def _grad_sync_ctx(self, sync_grads: bool):
        """Arm reduce-to-owner on the syncing micro-step; let the others accumulate.

        The base implementation gates through the model wrapper (``no_sync()`` for
        DDP, ``set_requires_gradient_sync`` for FSDP2); the replicated-compute
        model has neither. Accumulation micro-steps need no gating at all, since
        the reduce hooks only launch while armed.
        """
        if sync_grads:
            self.optimizer.prepare_for_backward()
        yield

    def _train_step(self):
        self._in_train_step = True
        try:
            return super()._train_step()
        finally:
            self._in_train_step = False

    def _build_optimizer(self):
        return self._custom_dp

    def _clip_gradients(self, max_norm: float) -> float:
        norm = self.optimizer.clip_grad_norm_(max_norm)
        self._after_grad_norm()
        return norm

    def _compute_grad_norm(self) -> float:
        """Norm over the fp32 master gradients, without clipping.

        ``self.model`` holds the compute replicas, whose gradients the
        reduce-to-owner step already consumed, so the base implementation would
        report a norm of zero whenever ``--clip-grad`` is off.
        """
        norm = self.optimizer.clip_grad_norm_(0.0)
        self._after_grad_norm()
        return norm

    def _clean_nan_gradients(self):
        self.optimizer.scrub_nan_gradients()

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
                "custom_dp training-state checkpoints require "
                "--save-format=dcp; pass --no-save-training-state to export "
                "weights only."
            )
        super()._save_checkpoint()

    def _finalize(self):
        # After the base finalize, not before: the final checkpoint is written in
        # there and abandoning the collectives must not run before it.
        try:
            return super()._finalize()
        finally:
            self._shut_down()

    def _shut_down(self) -> None:
        """Shut the look-ahead and the optimizer's collectives down, failures and all.

        A partially built or already failed trainer still has to release its
        worker thread and abandon half-issued collectives, so no single failure
        may skip the rest.
        """
        errors = []
        for shutdown in (
            self._shut_down_lookahead,
            lambda: getattr(self, "_custom_dp", None) and self._custom_dp.close(),
        ):
            try:
                shutdown()
            except Exception as exc:  # keep closing the rest
                logger.warning(
                    "custom_dp shutdown step failed", exc_info=True
                )
                errors.append(exc)
        if errors:
            raise errors[0]


__all__ = ["CustomDpTrainer"]

