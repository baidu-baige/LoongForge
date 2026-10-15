# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""custom_dp trainer for LingBot VLA v2.

Everything generic lives one level up: the optimizer wiring in
``CustomDpTrainer``. This subclass adds only what is specific to
LingBot -- the model-side batch, optimizer and compile helpers, and the teacher
look-ahead (the teachers run a step early, and the queue of started batches and
the dataloader position it hides live here) -- through the mixin's hooks.
"""

import logging
from collections import deque

import torch

from loongforge.embodied.model.lingbot_vla_v2.batch_enricher import (
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
from loongforge.embodied.train.trainers.custom.dp_training.custom_dp_trainer import (
    CustomDpTrainer,
)

logger = logging.getLogger(__name__)

class LingbotVlaV2CustomDpTrainer(CustomDpTrainer):
    """LingBot batch/teacher pipeline on the custom_dp loop."""

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

    # -- custom_dp hooks ------------------------------------------

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

    def _after_manager_built(self):
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

    def _build_teacher_executor(self):
        """Side-stream executor for the frozen teachers, or ``None`` to run them
        synchronously in the training thread."""
        if not (self.model_cfg.async_teacher and torch.cuda.is_available()):
            return None
        from loongforge.embodied.train.trainers.custom.async_runtime import (
            CudaStreamExecutor,
        )

        # ``torch.fx`` patches ``nn.Module.__call__`` process-wide while the main
        # thread traces, so it can catch the teacher mid-forward. The teacher is
        # frozen and side-effect-free, so recomputing it on the training thread is
        # safe -- hence ``fallback_to_sync``.
        return CudaStreamExecutor(
            error_policy="fallback_to_sync", name="lingbot-teacher"
        )

    def _muon_lr_scale_for(self):
        return moe_expert_lr_scale_for(self.model_cfg)

    def _register_optimizer_hooks(self, optimizer):
        register_moe_load_balance_hook(optimizer, self.model, self.model_cfg)

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
        # look-ahead bookkeeping has to stay intact until it has been saved; the
        # mixin shuts the look-ahead and the optimizer down after that.
        return super()._finalize()

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


__all__ = ["LingbotVlaV2CustomDpTrainer"]
