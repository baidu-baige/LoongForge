# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Finetune trainer on the tensor_wise distributed optimizer.

Selecting a tensor_wise trainer is what enables tensor_wise: it always builds
``TensorWiseOptimizer`` under ``--distributed-strategy ddp``.
``--trainer-type TensorWiseTrainer`` runs any model that ``FinetuneTrainer``
runs, with the parameters in ``--dtype`` and the ``--optimizer`` optimizer over
the fp32 masters, including ``--optimizer muon``; a model that needs more (a
precision policy, per-parameter LRs, compile) subclasses it, as
``LingbotVlaV2Trainer`` does.

The optimizer is built in ``_wrap_model_for_training``, not ``_build_optimizer``:
its parameter plan has to capture the module tree before anything (e.g. regional
compile) changes it.
"""

import logging
from contextlib import contextmanager

from loongforge.embodied.distributed.activation_checkpointing import (
    apply_activation_checkpointing,
)
from loongforge.embodied.distributed.utils import is_mixed_param_dtype
from loongforge.embodied.optimizer import build_optimizer
from loongforge.embodied.optimizer.muon_builder import (
    DEFAULT_MUON_NDIMS,
    build_muon_optimizer_from_args,
)
from loongforge.embodied.optimizer.tensor_wise_optimizer.precision_policy import (
    DefaultParameterPolicy,
)
from loongforge.embodied.optimizer.tensor_wise_optimizer.tensor_wise_optimizer import (
    CollectiveConfig,
    TensorWiseOptimizer,
)
from loongforge.embodied.train.trainers.supervised.finetune_trainer import (
    FinetuneTrainer,
)
from loongforge.embodied.train.utils.utils import resolve_dtype

logger = logging.getLogger(__name__)


class TensorWiseTrainer(FinetuneTrainer):
    """Finetune loop on ``TensorWiseOptimizer``; subclass it for a new model.

    Extension points (the defaults train every parameter in ``--dtype`` with
    ``--optimizer``):

        _build_parameter_policy()     per-parameter compute dtype, wire precision
                                      and optimizer route
        _prepare_module()             the module handed to the optimizer
        _create_inner_optimizer(tw)   the optimizer over the fp32 masters
        _muon_lr_scale_for()          per-parameter LR multipliers for Muon
        _after_optimizer_built()      work that must follow the parameter plan,
                                      e.g. regional compile

    The rest is the loop contract of ``TensorWiseOptimizer``. A subclass is
    registered in ``trainer_builder.py`` and in ``TENSOR_WISE_TRAINERS``
    (``tensor_wise_optimizer/validation.py``).
    """

    # -- extension points -------------------------------------------------------

    def _build_parameter_policy(self):
        """Keep every parameter's dtype; under ``--optimizer muon`` route 2-D weights to Muon."""
        if self.training_args.optimizer.lower() != "muon":
            return DefaultParameterPolicy()
        return DefaultParameterPolicy(muon_ndims=DEFAULT_MUON_NDIMS)

    def _prepare_module(self):
        """Prepare the model as ``wrap_model`` does for DDP, minus the wrapper:
        activation checkpointing, then ``--dtype`` unless the model mixes dtypes.

        ``--fp8`` is rejected at startup, so the FP8 conversion step is absent.
        """
        args = self.training_args
        apply_activation_checkpointing(
            self.model,
            args.activation_checkpoint_module_patterns,
            args.activation_checkpoint_skip_modules,
        )
        if is_mixed_param_dtype(self.model, trainable_only=False):
            return self.model.to(device=self.ctx.device)
        return self.model.to(dtype=resolve_dtype(args.dtype), device=self.ctx.device)

    def _create_inner_optimizer(self, tensor_wise):
        """Build the optimizer over the fp32 masters, not the compute replicas.

        The replicas' ``.grad`` is consumed and cleared by the reduce-to-owner
        step, so an optimizer built on ``self.model`` would see no gradients and
        silently leave the run untrained. Muon is built here, not by the
        optimizer registry, because it needs the masters, already split into the
        Muon and AdamW routes by the parameter policy.
        """
        args = self.training_args
        if args.optimizer.lower() != "muon":
            return build_optimizer(tensor_wise.optimizer_view(), args)
        muon_parameters, adamw_parameters = (
            tensor_wise.parameter_plan.split_masters_by_optimizer()
        )
        return build_muon_optimizer_from_args(
            muon_parameters, adamw_parameters, args,
            lr_scale_for=self._muon_lr_scale_for(),
        )

    def _muon_lr_scale_for(self):
        """``name -> LR multiplier`` for Muon, or ``None`` for a uniform LR."""
        return None

    def _after_optimizer_built(self) -> None:
        """Hook for work that has to run after the parameter plan is captured."""

    # -- model and optimizer setup ----------------------------------------------

    def _wrap_model_for_training(self) -> None:
        self._parameter_policy = self._build_parameter_policy()
        module = self._prepare_module()
        self._tensor_wise = TensorWiseOptimizer(
            module,
            self._parameter_policy,
            self._create_inner_optimizer,
            rank=self.ctx.rank,
            world_size=self.ctx.world_size,
            collective_config=CollectiveConfig.from_model_config(self.model_cfg),
        )
        if self.ctx.is_main:
            self._log_precision_summary()
        self._after_optimizer_built()

    def _log_precision_summary(self) -> None:
        """Log the resolved per-class compute/collective precision."""
        logger.info(
            "tensor_wise collective precision: grad_reduce=%s "
            "parameter_sync=%s (precision-critical tensors stay fp32)",
            self._tensor_wise.grad_reduce_mode,
            self._tensor_wise.param_sync_precision,
        )
        for row in self._tensor_wise.precision_summary():
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
        plan = self._tensor_wise.collective_summary()
        logger.info(
            "tensor_wise overlap: grad=%s param=%s bucket=%dMiB",
            plan["grad_overlap"],
            plan["param_overlap"],
            plan["bucket_mb"],
        )

    def _build_optimizer(self):
        return self._tensor_wise

    # -- tensor_wise loop contract ------------------------------------------------

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

    def _clip_gradients(self, max_norm: float) -> float:
        return self.optimizer.clip_grad_norm_(max_norm)

    def _compute_grad_norm(self) -> float:
        """Norm over the fp32 master gradients, without clipping.

        ``self.model`` holds the compute replicas, whose gradients the
        reduce-to-owner step already consumed, so the base implementation would
        report a norm of zero whenever ``--clip-grad`` is off.
        """
        return self.optimizer.clip_grad_norm_(0.0)

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
                "tensor_wise training-state checkpoints require "
                "--save-format=dcp; pass --no-save-training-state to export "
                "weights only."
            )
        super()._save_checkpoint()

    def _finalize(self):
        # The final checkpoint is written inside ``super()._finalize()``; the
        # half-issued collectives are abandoned only after it, and also when it
        # fails.
        try:
            return super()._finalize()
        finally:
            tensor_wise = getattr(self, "_tensor_wise", None)
            if tensor_wise is not None:
                tensor_wise.close()


__all__ = ["TensorWiseTrainer"]
