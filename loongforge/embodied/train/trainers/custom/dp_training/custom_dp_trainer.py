# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Reusable trainer integration for the custom_dp manager.

A model can run on custom_dp with no code at all (``--trainer-type CustomDpTrainer``): the
two hooks below have neutral defaults, and the collective knobs come from the
``--custom-dp-*`` arguments. A model that wants a precision policy of its own
subclasses ``CustomDpTrainer`` and overrides:

    _build_parameter_policy()   the model's precision/ownership policy
    _prepare_module()       the nn.Module whose parameters are managed

The generic manager lifecycle -- gradient-reduction arming, parameter
publication, gradient clipping, nan scrubbing, Muon wiring and the dcp-only
training-state guard -- lives here so a new model reuses the wiring instead of
copying it. The remaining hooks (``_after_manager_built``,
``_muon_lr_scale_for``, ``_register_optimizer_hooks``, ``_after_grad_norm``) default to no-ops; a model
overrides only what it needs.

This class builds the manager (and the optional look-ahead pipeline) and drives
them from the finetune loop's hook points. ``self._parameter_manager`` is the single
seam: model code reaches the resolved plan, the compute replicas and the
partial-publish hooks through it and nothing else.
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

from .parameter_manager import CollectiveConfig, build_parameter_manager

logger = logging.getLogger(__name__)


class CustomDpTrainer(FinetuneTrainer):
    """Own the custom_dp manager lifecycle inside the finetune loop."""

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
        """Return the module the custom_dp manager manages, already on the training device.

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
        """Run model-specific setup once the manager exists (default: none)."""

    def _create_inner_optimizer(self):
        """Build the optimizer over the fp32 masters (default: standard build).

        The masters are the optimizer's parameters, not the compute replicas:
        the replicas' ``.grad`` is consumed and cleared by the reduce-to-owner
        step, so an optimizer built on ``self.model`` would see no gradients and
        silently leave the run untrained.
        """
        if self.training_args.optimizer.lower() == "muon":
            return self._build_muon_optimizer()
        return build_optimizer(self._optimizer_parameter_model, self.training_args)

    def _build_muon_optimizer(self):
        """Build Muon over the masters plus its AdamW fallback groups.

        Muon belongs to this strategy rather than to the optimizer registry: it
        optimizes the fp32 masters, which only exist here. The Muon/AdamW split
        comes from the plan, so the builder is handed two ready groups.
        """
        from .optimizer_builder import build_muon_optimizer

        args = self.training_args
        muon_parameters, adamw_parameters = (
            self._parameter_manager.parameter_plan.split_masters_by_optimizer()
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

    def _after_optimizer_built(self) -> None:
        """Publish the Muon-managed masters through the manager's callback.

        The manager needs the set of parameters Muon updates *inside* the step so
        it publishes exactly those, and Muon reports each master update through
        ``param_update_callback``. The name set is rank-independent on purpose:
        the publish plan is compared across ranks.
        """
        from loongforge.embodied.optimizer.muon import DistributedMuon

        inner = getattr(self.optimizer, "optimizers", None)
        if not inner:
            return
        muon = next((opt for opt in inner if isinstance(opt, DistributedMuon)), None)
        if muon is None:
            return
        self._parameter_manager.set_in_step_updated_names(
            self._parameter_manager.parameter_plan.muon_parameter_names()
        )
        muon.param_update_callback = self._parameter_manager.on_master_updated

    def _after_grad_norm(self) -> None:
        """Hook fired right after the pre-clip global norm was computed."""

    # -- manager construction -------------------------------------------------

    def _shut_down_lookahead(self) -> None:
        """Hook: release whatever a subclass started ahead of the step."""

    def _build_parameter_manager(self, module):
        """Build the custom_dp manager over ``module``."""
        return build_parameter_manager(
            module,
            self._parameter_policy,
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
            self._parameter_manager.grad_reduce_mode,
            self._parameter_manager.param_sync_precision,
        )
        for row in self._parameter_manager.precision_summary():
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
        plan = self._parameter_manager.collective_summary()
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
        self._parameter_manager = self._build_parameter_manager(module)
        self._optimizer_parameter_model = self._parameter_manager.optimizer_view()
        if self.ctx.is_main:
            self._log_precision_summary()
        self._after_manager_built()

    def _should_sync_grads(self, micro: int, grad_accum: int) -> bool:
        sync = super()._should_sync_grads(micro, grad_accum)
        if micro == grad_accum - 1:
            self._parameter_manager.begin_gradient_sync()
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
        self._parameter_manager.finish_gradient_sync()
        return result

    def _train_step(self):
        self._parameter_manager.begin_parameter_sync()
        self._in_train_step = True
        try:
            result = super()._train_step()
        finally:
            self._in_train_step = False
        # Not in a finally: a failed step leaves the publish plan for
        # ``_shut_down`` to unwind rather than draining half-issued work.
        self._parameter_manager.finish_parameter_sync()
        return result

    def _build_optimizer(self):
        optimizer = self._create_inner_optimizer()
        self.optimizer = optimizer
        # The shared checkpoint backend owns the file layout and the aux files; the
        # IO the manager installs owns the rank-local FP32 masters and the
        # Muon/AdamW state that follows them.
        self._parameter_manager.attach_optimizer(optimizer)
        self._after_optimizer_built()
        return optimizer

    def _clip_gradients(self, max_norm: float) -> float:
        norm = self._parameter_manager.clip_grad_norm_(max_norm)
        self._after_grad_norm()
        return norm

    def _compute_grad_norm(self) -> float:
        """Norm over the fp32 master gradients, without clipping.

        ``self.model`` holds the compute replicas, whose gradients the
        reduce-to-owner step already consumed, so the base implementation would
        report a norm of zero whenever ``--clip-grad`` is off.
        """
        norm = self._parameter_manager.clip_grad_norm_(0.0)
        self._after_grad_norm()
        return norm

    def _clean_nan_gradients(self):
        self._parameter_manager.scrub_nan_gradients()

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
        """Shut the look-ahead and manager down, failures and all.

        A partially built or already failed trainer still has to release its
        worker thread and abandon half-issued collectives, so no single failure
        may skip the rest.
        """
        errors = []
        for shutdown in (
            self._shut_down_lookahead,
            lambda: getattr(self, "_parameter_manager", None)
            and self._parameter_manager.clear_pending_work(),
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

