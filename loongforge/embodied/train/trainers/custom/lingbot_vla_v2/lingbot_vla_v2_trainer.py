# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot VLA v2 trainer, on the tensor_wise distributed optimizer.

Selecting a tensor_wise trainer is what enables tensor_wise: it always builds
``TensorWiseOptimizer`` under ``--distributed-strategy ddp``.

``TensorWiseTrainer`` drives ``TensorWiseOptimizer`` from the finetune loop (the
contract is listed in that class's docstring); a new model subclasses it and
overrides the extension points. ``LingbotVlaV2Trainer`` adds what is specific to
LingBot: the parameter policy, Muon, the model-side batch and compile helpers,
and the teacher look-ahead (the teachers run a step early, and the queue of
started batches and the dataloader position it hides live here).

The optimizer is built in ``_wrap_model_for_training``, not ``_build_optimizer``:
its parameter plan has to capture the module tree before anything (e.g. regional
compile) changes it.
"""

import logging
from collections import deque
from contextlib import contextmanager

import torch

from loongforge.embodied.train.trainers.custom.lingbot_vla_v2.batch_enricher import (
    LingbotVlaV2BatchEnricher,
)
from loongforge.embodied.model.lingbot_vla_v2.fm_seed import seed_fm_step
from loongforge.embodied.model.lingbot_vla_v2.modeling_lingbot_vla_v2 import (
    apply_regional_compile,
)
from loongforge.embodied.model.lingbot_vla_v2.moe_load_balance import (
    moe_expert_lr_scale_for,
    register_moe_load_balance_hook,
)
from loongforge.embodied.model.lingbot_vla_v2.parameter_policy import (
    LingbotVlaV2ParameterPolicy,
)
from loongforge.embodied.optimizer.tensor_wise_optimizer.precision_policy import (
    DefaultParameterPolicy,
)
from loongforge.embodied.optimizer.tensor_wise_optimizer.tensor_wise_optimizer import (
    CollectiveConfig,
    TensorWiseOptimizer,
)
from loongforge.embodied.optimizer import build_optimizer
from loongforge.embodied.optimizer.tensor_wise_optimizer.optimizer_builder import (
    build_muon_optimizer,
)
from loongforge.embodied.train.trainers.supervised.finetune_trainer import (
    FinetuneTrainer,
)

logger = logging.getLogger(__name__)


class TensorWiseTrainer(FinetuneTrainer):
    """Finetune loop on ``TensorWiseOptimizer``; subclass it for a new model.

    Extension points (the defaults train every parameter in its own dtype with
    ``--optimizer``):

        _build_parameter_policy()     per-parameter compute dtype, wire precision
                                      and optimizer route
        _prepare_module()             the module handed to the optimizer
        _create_inner_optimizer(tw)   the optimizer over the fp32 masters
        _after_optimizer_built()      work that must follow the parameter plan,
                                      e.g. regional compile

    The rest is the loop contract of ``TensorWiseOptimizer``. A subclass is
    registered in ``trainer_builder.py`` and in ``TENSOR_WISE_TRAINERS``
    (``tensor_wise_optimizer/validation.py``).
    """

    # -- extension points -------------------------------------------------------

    def _build_parameter_policy(self):
        return DefaultParameterPolicy()

    def _prepare_module(self):
        return self.model.to(self.ctx.device)

    def _create_inner_optimizer(self, tensor_wise):
        """Build the optimizer over the fp32 masters, not the compute replicas.

        The replicas' ``.grad`` is consumed and cleared by the reduce-to-owner
        step, so an optimizer built on ``self.model`` would see no gradients and
        silently leave the run untrained.
        """
        return build_optimizer(tensor_wise.optimizer_view(), self.training_args)

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


class LingbotVlaV2Trainer(TensorWiseTrainer):
    """LingBot batch/teacher pipeline on the tensor_wise loop."""

    # True only while ``_train_step`` runs. Evaluation reads the same loader name
    # as training, and must not be served look-ahead batches.
    _in_train_step = False

    def __init__(self, training_args, model_cfg, data_cfg):
        super().__init__(training_args, model_cfg, data_cfg)
        self._batch_enricher = LingbotVlaV2BatchEnricher(model_cfg)
        self._teacher_for_batch = None
        self._pipeline_teacher = bool(getattr(model_cfg, "pipeline_teacher", True))
        # ``(batch, teacher handle)`` pairs whose teachers were started a step
        # early, in fetch order, and the dataloader position from before they were
        # read (``None`` while the queue is empty).
        self._lookahead = deque()
        self._lookahead_position = None

    # -- TensorWiseTrainer extension points --------------------------------------

    def _build_parameter_policy(self):
        return LingbotVlaV2ParameterPolicy(self.model_cfg)

    def _prepare_module(self):
        module = self.model.policy.to(self.ctx.device)
        if self.model_cfg.gradient_checkpointing and hasattr(
            module, "gradient_checkpointing_enable"
        ):
            module.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        return module

    def _create_inner_optimizer(self, tensor_wise):
        """Add Muon over the masters the policy routes to it.

        Muon is built here rather than by the optimizer registry because it
        needs the masters, which only exist inside ``TensorWiseOptimizer``.
        """
        args = self.training_args
        if args.optimizer.lower() != "muon":
            return super()._create_inner_optimizer(tensor_wise)
        muon_parameters, adamw_parameters = (
            tensor_wise.parameter_plan.split_masters_by_optimizer()
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
            lr_scale_for=moe_expert_lr_scale_for(self.model_cfg),
        )
        register_moe_load_balance_hook(optimizer, self.model, self.model_cfg)
        return optimizer

    def _after_optimizer_built(self) -> None:
        # The executor is built before the compile so the order matches the
        # previous recipe wiring; regional compile has to run after the optimizer
        # captured the module tree (see ``apply_regional_compile``).
        teacher_executor = self._build_teacher_executor()
        apply_regional_compile(self.model.policy, self.model_cfg.regional_compile)
        self._batch_enricher.setup(self.ctx, self.model_cfg, teacher_executor)
        if self.ctx.is_main:
            logger.info(
                "Teacher targets start %s",
                "one step early, in the optimizer window"
                if self._pipeline_teacher
                else "at the head of their own step",
            )

    # -- LingBot additions to the loop ------------------------------------------

    def _train_step(self):
        self._in_train_step = True
        try:
            return super()._train_step()
        finally:
            self._in_train_step = False

    def _clip_gradients(self, max_norm: float) -> float:
        norm = super()._clip_gradients(max_norm)
        self._after_grad_norm()
        return norm

    def _compute_grad_norm(self) -> float:
        norm = super()._compute_grad_norm()
        self._after_grad_norm()
        return norm

    def _after_grad_norm(self):
        # The muon + adamw + parameter_sync region is mostly NCCL, so its tensor
        # cores sit idle. Fill it with the next step's teachers.
        self._start_next_step_teachers()

    def _shut_down_lookahead(self) -> None:
        """Drop the queued batches and stop the teachers, also on a failed run."""
        self._lookahead.clear()
        self._lookahead_position = None
        self._batch_enricher.close()

    # -- teacher pipeline -----------------------------------------------------

    def _build_teacher_executor(self):
        """Side-stream executor for the frozen teachers, or ``None`` to run them
        synchronously in the training thread."""
        if not (self.model_cfg.async_teacher and torch.cuda.is_available()):
            return None
        from loongforge.embodied.distributed.cuda_stream_executor import (
            CudaStreamExecutor,
        )

        # ``torch.fx`` patches ``nn.Module.__call__`` process-wide while the main
        # thread traces, so it can catch the teacher mid-forward. The teacher is
        # frozen and side-effect-free, so recomputing it on the training thread is
        # safe -- hence ``fallback_to_sync``.
        return CudaStreamExecutor(
            error_policy="fallback_to_sync", name="lingbot-teacher"
        )

    def _train_forward(self, batch):
        self._batch_enricher.prepare_batch(
            self.model.policy, batch, self._teacher_for_batch
        )
        seed_fm_step(
            self.training_args.seed,
            self.completed_steps,
            getattr(self.ctx, "local_rank", 0),
        )
        return self.model(batch)

    def _finalize(self):
        self._teacher_for_batch = None
        self._batch_enricher.close()
        # The final checkpoint is written inside ``super()._finalize()``, so the
        # look-ahead bookkeeping has to stay intact until it has been saved.
        try:
            return super()._finalize()
        finally:
            self._shut_down_lookahead()

    def _fetch_batch(self, dl_name: str):
        """Serve a batch whose teachers were already started, when one is queued.

        Only during a training step: evaluation reads the same loader name, and
        handing it a queued training batch would silently train and evaluate on
        different data.
        """
        if self._in_train_step and dl_name == "vla" and self._lookahead:
            batch, self._teacher_for_batch = self._lookahead.popleft()
            return batch
        self._teacher_for_batch = None
        return super()._fetch_batch(dl_name)

    def _start_next_step_teachers(self) -> None:
        """Fetch the next step's batches and start their teachers, here in the
        optimizer window.

        The window is dominated by NCCL, so it has tensor-core and latency slack
        the forward does not, and the teacher hides behind the step's collectives
        rather than competing with them.

        Requires the async executor: without it ``submit`` blocks, which would put
        the whole teacher inside the window instead of alongside it.
        """
        if not self._pipeline_teacher or self._lookahead:
            return
        if not self._batch_enricher.has_executor:
            return
        # The look-ahead read pulls the next step's batches off the loader, so from
        # here on the loader sits ahead of the training position. A checkpoint
        # written at the end of this step must resume from *this* position, not the
        # advanced one, or the resumed run skips the prefetched batches.
        self._lookahead_position = super()._get_dataloader_state()
        for _ in range(self.training_args.gradient_accumulation_steps):
            batch = super()._fetch_batch("vla")
            self._lookahead.append((batch, self._batch_enricher.submit(batch)))

    def _get_dataloader_state(self):
        """Report the training position, not the look-ahead read position."""
        if self._lookahead:
            return self._lookahead_position
        return super()._get_dataloader_state()


__all__ = ["LingbotVlaV2Trainer", "TensorWiseTrainer"]
