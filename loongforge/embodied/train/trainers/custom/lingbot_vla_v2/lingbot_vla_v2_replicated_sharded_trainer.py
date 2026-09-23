# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Replicated-sharded trainer for LingBot VLA v2.

Everything generic lives one level up: the manager lifecycle in
``ReplicatedShardedTrainer``, the look-ahead bound and its position
bookkeeping in ``BoundedAsyncPipeline``. This subclass adds only what is specific
to LingBot -- the recipe, and the decision to run the teachers a step early --
through the mixin's hooks.
"""

import logging

from loongforge.embodied.model.lingbot_vla_v2.parameter_policy import (
    LingbotVlaV2ParameterPolicy,
)
from loongforge.embodied.model.lingbot_vla_v2.recipe import LingbotVlaV2Recipe
from loongforge.embodied.train.trainers.custom.replicated_sharded_training.trainer_mixin import (
    ReplicatedShardedTrainer,
)

logger = logging.getLogger(__name__)

class LingbotVlaV2ReplicatedShardedTrainer(ReplicatedShardedTrainer):
    """LingBot recipe and teacher pipeline on the replicated-sharded loop."""

    def __init__(self, training_args, model_cfg, data_cfg):
        super().__init__(training_args, model_cfg, data_cfg)
        self._recipe = LingbotVlaV2Recipe(model_cfg)
        self._teacher_for_batch = None
        self._in_train_step = False
        self._pipeline_teacher = bool(getattr(model_cfg, "pipeline_teacher", True))

    # -- replicated-sharded hooks ------------------------------------------

    def _build_parameter_policy(self):
        return LingbotVlaV2ParameterPolicy(self.model_cfg)

    def _build_async_provider(self):
        """The batch enricher is the LingBot ``AsyncTargetProvider``."""
        return self._recipe.batch_enricher

    def _prepare_shard_module(self):
        self._recipe.setup_parallel_state(self)
        policy = self.model.policy.to(self.ctx.device)
        if self.model_cfg.gradient_checkpointing and hasattr(
            policy, "gradient_checkpointing_enable"
        ):
            policy.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.model.policy = policy
        return policy

    def _after_shard_manager_built(self):
        self._recipe.setup(self)
        if self.ctx.is_main:
            logger.info(
                "Teacher targets start %s",
                "one step early, in the optimizer window"
                if self._pipeline_teacher
                else "at the head of their own step",
            )

    def _create_inner_optimizer(self):
        optimizer = self._recipe.build_optimizer(self)
        if optimizer is None:
            optimizer = super()._create_inner_optimizer()
        return optimizer

    def _after_optimizer_built(self):
        self._recipe.wire_parameter_sync(self)

    def _run_train_step(self):
        self._in_train_step = True
        try:
            return super()._run_train_step()
        finally:
            self._in_train_step = False

    def _on_gradients_clipped(self):
        # The muon + adamw + parameter_sync region is mostly NCCL, so its tensor
        # cores sit idle. Fill it with the next step's teachers.
        self._start_next_step_teachers()

    # -- teacher pipeline -----------------------------------------------------

    @property
    def _pipeline(self):
        """The look-ahead pipeline, or ``None`` before the runtime is built."""
        return getattr(self, "_async_pipeline", None)

    def _train_forward(self, batch):
        return self._recipe.forward(self, batch, self._teacher_for_batch)

    def _finalize(self):
        self._teacher_for_batch = None
        self._recipe.close()
        # The final checkpoint is written inside ``super()._finalize()``, so the
        # look-ahead bookkeeping has to stay intact until it has been saved; the
        # mixin closes the runtime (and with it the pipeline) after that.
        return super()._finalize()

    def _fetch_batch(self, dl_name: str):
        """Serve a batch whose teachers were already started, when one is queued.

        Only during a training step: evaluation reads the same loader name, and
        handing it a queued training batch would silently train and evaluate on
        different data.
        """
        pipeline = self._pipeline
        if self._in_train_step and dl_name == "vla" and pipeline and pipeline.pending():
            batch, self._teacher_for_batch = pipeline.consume()
            return batch
        self._teacher_for_batch = None
        return super()._fetch_batch(dl_name)

    def _start_next_step_teachers(self) -> None:
        """Fetch the next step's batches and start their teachers, here in the
        optimizer window.

        Measured on 8 GPUs: this window absorbs 79% of the teacher's 145.7 ms
        against 34% when the teacher runs into the forward, because 375.9 of its
        389.4 ms is NCCL and it therefore has slack the forward does not.

        Requires the async runner: without it ``submit`` blocks, which would put
        the whole teacher inside the window instead of alongside it.
        """
        pipeline = self._pipeline
        if not self._pipeline_teacher or pipeline is None or pipeline.pending():
            return
        if not self._recipe.batch_enricher.has_runner:
            return
        # The look-ahead read pulls the next step's batches off the loader, so from
        # here on the loader sits ahead of the training position. A checkpoint
        # written at the end of this step must resume from *this* position, not the
        # advanced one, or the resumed run skips the prefetched batches.
        pipeline.mark_position(super()._get_dataloader_state())
        for _ in range(self.training_args.gradient_accumulation_steps):
            pipeline.submit(super()._fetch_batch("vla"))

    def _get_dataloader_state(self):
        """Report the training position, not the look-ahead read position."""
        pipeline = self._pipeline
        if pipeline is not None and pipeline.position is not None:
            return pipeline.position
        return super()._get_dataloader_state()


__all__ = ["LingbotVlaV2ReplicatedShardedTrainer"]
