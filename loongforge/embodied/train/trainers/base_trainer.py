# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""BaseTrainer — pure native PyTorch distributed training skeleton."""

import gc
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loongforge.embodied.distributed import DistributedContext
from loongforge.embodied.distributed.checkpoint import (
    flush_pending_save,
    get_latest_checkpoint,
    restore_rank_rng_state,
    resume_training_state,
)
from loongforge.embodied.distributed.parallel import wrap_model
from loongforge.embodied.train.lora import (
    apply_lora,
    is_lora_enabled,
    load_adapter_into_model,
)
from loongforge.embodied.train.utils.logging import TrainingLogger, StageTimers, log_effective_config
from loongforge.embodied.train.utils.utils import (
    log_stage,
    set_deterministic,
    set_precision,
    set_seed,
    setup_logging,
    Profiler,
)
from loongforge.embodied.optimizer import get_grad_norm

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Training skeleton — Template Method pattern.

    Lifecycle: __init__(training_args) → train() → [_setup → _training_loop → _finalize]

    The base class holds ONLY:
      - abstract methods: model/data/paradigm-dependent compute and training
        infrastructure (subclasses must implement) — see the "Abstract methods"
        section below.
      - shared functions: lifecycle orchestration, timing instrumentation,
        loss/backward routing, one-shot RNG restore, dataloader state save/load
        — all model-independent and reused by every subclass.

    The training loop is split into three layers (Template Method):
      - Layer 1 `_training_loop`: fixed orchestration (shared).
      - Layer 2 `_train_step`: one optimizer-step skeleton (shared, rarely
        overridden).
      - Layer 3 `_forward_backward`: gradient-accumulation + fetch + forward +
        backward + loss combination (ABSTRACT — subclass implements).
    """

    def __init__(self, training_args, model_cfg, data_cfg):
        self.training_args = training_args
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg

        # Initialized in _setup()
        self.ctx: Optional[DistributedContext] = None
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler = None
        self.dataloaders: Dict[str, DataLoader] = {}
        self.logger: Optional[TrainingLogger] = None

        # Training state
        self.completed_steps: int = 0
        self.current_epoch: int = 0
        self.train_iters: int = training_args.train_iters

        # Cumulative iteration-health counters (reported in the training log).
        # nan: steps whose loss was NaN/Inf. skipped: steps whose loss-spike
        # guard fired (NaN/Inf loss or loss above --loss-spike-threshold), so
        # the loss was zeroed and the update contributed no real gradient.
        # Not persisted across checkpoint resume (reset to 0 on restart).
        self.nan_iterations: int = 0
        self.skipped_iterations: int = 0

        # Data iterators (managed by _fetch_batch for epoch cycling)
        self._data_iters: Dict[str, Any] = {}
        self._resume_dataloader_state: Dict[str, Dict[str, Any]] = {}
        self._resume_rng_per_rank = None
        self._lora_resume_adapter_path: Optional[str] = None

        # The primary loader "vla" mirrors self.current_epoch for checkpoint/log display.
        self._epochs: Dict[str, int] = {}

        # Per-stage timing — created in _training_loop, shared via this attr so
        # the timed helper functions and all layers can read it without passing
        # it through every signature.
        self._stage_timers: Optional[StageTimers] = None

    # ═══════════════════════════════════════════════
    # Public interface
    # ═══════════════════════════════════════════════

    def train(self):
        """Main entry point."""
        self._setup()
        self._training_loop()
        self._finalize()

    # ═══════════════════════════════════════════════
    # Setup
    # ═══════════════════════════════════════════════

    def _setup(self):
        """One-shot initialization of all training resources."""
        training_args = self.training_args

        # 1. Distributed context
        self.ctx = DistributedContext()
        self.ctx.init()

        # 2. Seed — use the same seed on all ranks (align with lerobot/accelerate baseline).
        # DistributedSampler handles per-rank data partitioning internally via its own seed+rank offset.
        set_seed(training_args.seed)
        if training_args.deterministic_mode:
            set_deterministic()
        if training_args.disable_tf32:
            set_precision(allow_tf32=False)

        # 3. Output directories + logging
        self.output_dir = training_args.output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        if self.ctx.is_main:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.ctx.barrier()
        setup_logging(self.output_dir, self.ctx.rank)
        self._configure_backend_precision()

        # Dump fully-resolved CLI training_args + model config now that the file
        # handler is attached, so the effective config also lands in the log.
        log_effective_config(training_args, self.model_cfg, self.data_cfg)
        self._configure_manual_gc()

        # 4. TrainingLogger (initialize early for logging during setup)
        self.logger = TrainingLogger(
            output_dir=self.output_dir,
            wandb_project=training_args.wandb_project,
            wandb_mode=training_args.wandb_mode,
            is_main=self.ctx.is_main,
            model_cfg=self.model_cfg,
            tensorboard_dir=training_args.tensorboard_dir,
            tensorboard_queue_size=training_args.tensorboard_queue_size,
            run_name=os.path.basename(self.output_dir),
        )

        # 5. Build model (from YAML model_cfg)
        with log_stage(
            "model",
            start_msg=(
                f"start building model: model_type={self.model_cfg.model_type}, "
                f"class={self.model_cfg.__class__.__name__}"
            ),
            end_msg="model built in {elapsed}",
        ):
            self.model = self._build_model()

        # 5. Pretrained weights / Resume (before wrapping)
        latest_path = None
        if training_args.resume:
            latest_path, latest_step, latest_epoch = get_latest_checkpoint(self.checkpoint_dir)
            assert latest_path, (
                f"--resume set but no checkpoint was found in {self.checkpoint_dir}. "
                f"Point --output-dir to a run whose checkpoints/ directory contains "
                f"a checkpoint, or drop --resume to start from scratch."
            )
            with log_stage(
                "ckpt",
                start_msg=f"resume requested: dir=={latest_path}",
                end_msg="resume done in {elapsed}",
            ):
                self._handle_resume(latest_path, latest_step, latest_epoch)
        elif training_args.pretrained_checkpoint:
            if not training_args.init_on_meta:
                with log_stage(
                    "ckpt",
                    start_msg=f"loading pretrained: {training_args.pretrained_checkpoint}",
                    end_msg="pretrained loaded in {elapsed}",
                ):
                    self._load_pretrained(training_args.pretrained_checkpoint)
        else:
            logger.info("No pretrained weights or resume checkpoint found. Using random initialization.")

        self.model = self._apply_lora_before_wrap(self.model)

        # 6. Freeze modules
        self._freeze_modules(training_args.freeze_modules)

        with log_stage(
            "wrap_model",
            start_msg=f"wrap_model: strategy={training_args.distributed_strategy}, dtype={training_args.dtype}",
            end_msg="done in {elapsed}",
        ):
            # 7. Parallel wrapping (DDP/FSDP + mixed precision via policy).
            # Subclasses may override _wrap_model_for_training to take over the
            # wrapping step (e.g. to install a CUDA graph runner instead).
            self._wrap_model_for_training()

        # 7.5 Deferred materialize + load_pretrained
        if training_args.init_on_meta and not training_args.resume:
            with log_stage(
                "materialize",
                start_msg=f"materializing meta tensors on {self.ctx.device}",
                end_msg="materialized in {elapsed}",
            ):
                self.model.materialize(self.ctx.device)
            if training_args.pretrained_checkpoint:
                with log_stage(
                    "ckpt",
                    start_msg=f"loading pretrained (sharded): {training_args.pretrained_checkpoint}",
                    end_msg="pretrained loaded in {elapsed}",
                ):
                    self.model.load_pretrained(training_args.pretrained_checkpoint, device=self.ctx.device)

        with log_stage(
            "optimizer",
            start_msg="building optimizer", end_msg="optimizer built in {elapsed}"
        ):
            # 8. Optimizer + Scheduler (after wrapping; FSDP use_orig_params=True)
            self.optimizer = self._build_optimizer()
            self.lr_scheduler = self._build_scheduler()

        # 9. Resume optimizer/scheduler/RNG state (after wrapping + optimizer creation).
        # Gate on `training_args.resume` only — step==0 is a valid resume point and must
        # still restore optimizer/scheduler/RNG/dataloader, otherwise we'd run
        # with resumed weights but a fresh optimizer (half-resume).
        if training_args.resume and latest_path:
            with log_stage(
                "ckpt",
                start_msg=f"restoring optimizer/scheduler/RNG state from {latest_path}",
                end_msg="optimizer/scheduler/RNG state restored in {elapsed}",
            ):
                saved_epoch, dataloader_state, rng_per_rank = resume_training_state(
                    self.model, self.optimizer, self.lr_scheduler, latest_path, self.ctx,
                    restore_rng=False,
                )
                # Trust the epoch from training_state.pt over resume_meta.json
                # when present (training_state is the freshest source). Use
                # `is not None` so a legitimate saved_epoch=0 still overrides.
                if saved_epoch is not None:
                    self.current_epoch = saved_epoch
                self._resume_dataloader_state = dataloader_state or {}
                self._resume_rng_per_rank = rng_per_rank

        # 10. Data
        with log_stage("data", start_msg="building dataloaders"):
            self.dataloaders = self._build_dataloaders()
            self._restore_dataloader_states()

        # 11. Print stats
        self.logger.log_param_stats(self.model)

        # Hook
        self._on_train_begin()

    # ═══════════════════════════════════════════════
    # Training loop — Layer 1 (fixed orchestration, shared)
    # ═══════════════════════════════════════════════

    def _training_loop(self):
        training_args = self.training_args
        log_interval = training_args.log_interval
        detail_log_interval = training_args.detail_log_interval
        save_interval = training_args.save_interval

        # ── Profiler setup ──
        prof = Profiler(training_args, self.ctx, self.output_dir)
        prof.start()

        # ── Per-stage timing (shared via instance attr) ──
        self._stage_timers = StageTimers()

        # Initialize iterators for all loaders ("vla" first to keep the primary
        # epoch semantics consistent — see _init_data_iterator).
        for name in self.dataloaders:
            self._init_data_iterator(name)
        self._on_after_data_iterators_initialized()

        while self.completed_steps < self.train_iters:

            prof.step(self.completed_steps)
            # Detailed per-stage timing is enabled only on the step that will be
            # logged, so the cuda.synchronize() inside the timers does not slow
            # down steady-state training.
            enable_detail = (
                detail_log_interval > 0
                and (self.completed_steps + 1) % detail_log_interval == 0
            )
            self._stage_timers.set_enabled(enable_detail)

            t0 = time.perf_counter()

            # ── One optimizer step (Layer 2) ──
            log_dict, grad_norm = self._train_step()

            self.completed_steps += 1

            # ── Cross-rank loss aggregation ──
            # log_dict losses are per-rank local values, controlled by
            # --loss-log-rank (a list of ints):
            #   contains -1 (default [-1]) -> all-reduce mean: the reported loss
            #       reflects the global batch (logged on rank 0). This is a
            #       collective op — EVERY rank must call it (it runs before the
            #       rank-0-only logging below), otherwise NCCL hangs.
            #   one or more non-negative ranks -> NO communication: each listed
            #       rank prints its own local loss (tagged with its rank); the
            #       rank-0 backends (W&B/TB/JSONL) below are unaffected.
            # grad_norm is already global (clip_gradients / get_grad_norm all-reduce
            # internally), so it is NOT touched here.
            loss_log_ranks = training_args.loss_log_rank
            if any(r < 0 for r in loss_log_ranks):
                for key in list(log_dict.keys()):
                    if "loss" in key:
                        log_dict[key] = self.ctx.all_reduce_mean(log_dict[key])
            elif (self.ctx.rank in loss_log_ranks
                  and self.completed_steps % log_interval == 0):
                self._log_local_loss(log_dict)

            # ── Metrics ──
            step_time = time.perf_counter() - t0
            local_batch_size = training_args.gradient_accumulation_steps * training_args.per_device_batch_size
            global_batch_size = local_batch_size * self.ctx.world_size
            consumed_samples = self.completed_steps * global_batch_size
            metrics = self.logger.collect_metrics(
                log_dict, step_time,
                self.completed_steps, self.lr_scheduler,
                consumed_samples,
                self.model, local_batch_size, grad_norm,
            )
            metrics["nan_iterations"] = self.nan_iterations
            metrics["skipped_iterations"] = self.skipped_iterations

            # ── Step-end hook (optional; default pass) ──
            self._on_step_end(metrics)
            self._maybe_collect_manual_gc()

            # ── Profiler stop ──
            if prof.should_stop(self.completed_steps):
                prof.stop()

            # ── Logging ──
            if self.completed_steps % log_interval == 0:
                self.logger.log_metrics(
                    metrics, self.completed_steps, self.train_iters,
                    training_args.per_device_batch_size, self.ctx.world_size, self.ctx.is_distributed,
                    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                )

            # ── Per-stage timing log (all ranks call; rank 0 emits) ──
            if enable_detail:
                self.logger.log_stage_times(
                    self._stage_timers, self.ctx, log_level=training_args.timing_log_level
                )
                self._stage_timers.reset()

            # ── Checkpoint ──
            if save_interval and self.completed_steps % save_interval == 0:
                self._save_checkpoint()

        # Final cleanup if loop exited before profile_step_end was reached.
        prof.stop()

    def _log_local_loss(self, log_dict: dict):
        """Print this rank's own local loss (no cross-rank communication).

        Used when ``--loss-log-rank`` lists one or more non-negative ranks: each
        listed rank calls this and prints its own loss tagged with its rank,
        instead of routing through the rank-0-only backends (W&B/TB/JSONL). Uses
        ``logger.warning`` so a non-rank-0 target still emits — ``setup_logging``
        sets non-rank-0 loggers to WARNING level, which would filter
        ``logger.info``.
        """
        loss_str = " ".join(
            f"{k}={v:.6f}" for k, v in log_dict.items()
            if "loss" in k and isinstance(v, (int, float))
        )
        logger.warning(
            "[rank %d][step %d] %s", self.ctx.rank, self.completed_steps, loss_str
        )

    # ═══════════════════════════════════════════════
    # Training loop — Layer 2 (one optimizer step, shared)
    # ═══════════════════════════════════════════════

    def _train_step(self) -> Tuple[dict, float]:
        """One optimizer step. Returns (log_dict, grad_norm).

        Fixed skeleton: zero_grad → _forward_backward (subclass) → nan cleanup
        → grad clip → optimizer/scheduler step. All instrumentation lives here
        so subclasses only implement _forward_backward.

        _forward_backward returns log_dict containing the loss and any other
        scalar metrics to log.
        """
        st = self._stage_timers
        grad_clip = self.training_args.clip_grad

        # Per-step health flags; subclass _backward_loss sets these when the
        # loss spike/NaN guard fires during gradient accumulation. Reset each
        # step so they reflect only the current iteration.
        self._step_loss_is_nan = False
        self._step_loss_spiked = False

        # ── Gradient accumulation + forward + backward (Layer 3, abstract) ──
        # Subclasses may override _run_forward_backward_block to take over the
        # zero_grad + forward + backward + grad sync block (e.g. for CUDA graph
        # runners that manage these steps internally).
        log_dict = self._run_forward_backward_block()

        if self._step_loss_is_nan:
            self.nan_iterations += 1
        if self._step_loss_spiked:
            self.skipped_iterations += 1

        # ── NaN gradient cleanup ──
        if self.training_args.check_for_nan_in_loss_and_grad:
            with st("nan-grad-cleanup"):
                self._clean_nan_gradients()

        # ── Gradient clipping (returns pre-clip global grad norm) ──
        with st("grad-clip"):
            if grad_clip > 0:
                grad_norm = self._clip_gradients(grad_clip)
            else:
                grad_norm = get_grad_norm(self.model)

        # ── Optimizer step ──
        with st("optimizer"):
            with st("optimizer-inner-step"):
                self.optimizer.step()
            with st("optimizer-scheduler-step"):
                self.lr_scheduler.step()

        return log_dict, grad_norm

    # ═══════════════════════════════════════════════
    # Timed helper functions (shared) — logging routed here once
    # ═══════════════════════════════════════════════

    def _should_sync_grads(self, micro: int, grad_accum: int) -> bool:
        """Whether the gradient sync (all-reduce) should happen on this micro step.

        Only the last accumulation step syncs. Kept separate so that a
        micro-step containing multiple backward passes can sync exactly once (on
        the last loss of the last accum step).
        """
        return micro == grad_accum - 1

    # ═══════════════════════════════════════════════
    # Abstract methods — subclass must implement
    # ═══════════════════════════════════════════════
    
    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Build model from self.model_cfg. Return unwrapped model."""
        ...

    @abstractmethod
    def _build_dataloaders(self) -> Dict[str, DataLoader]:
        """Build dataloaders from self.training_args. Must include 'vla' key."""
        ...

    @abstractmethod
    def _train_forward(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Single forward pass. Returns (loss, log_loss_dict).

        loss is the single scalar tensor that needs backward (raw, un-scaled).
        log_loss_dict carries model-specific component metrics for reporting.
        """
        ...

    @abstractmethod
    def _forward_backward(self) -> dict:
        """Run one optimizer step's gradient-accumulation loop.

        Returns log_dict — a per-step accumulator of model-provided reporting
        losses. May call
        shared helpers: _fetch_batch_timed / _should_sync_grads / _backward_loss /
        self._stage_timers, to reuse unified timing stages.
        """
        ...

    @abstractmethod
    def _backward_loss(self, loss: torch.Tensor,
                       log_loss_dict: Dict[str, torch.Tensor],
                       log_dict: Dict[str, float],
                       grad_accum: int, sync: bool) -> None:
        """Scale + spike-guard + backward routing, accumulating losses into log_dict.

        ``loss`` is the single scalar to backpropagate. ``log_loss_dict`` holds
        losses used ONLY for printing/reporting (not backward). ``log_dict`` is the
        running per-step accumulator that all scalars are summed into (in-place).

        Contract (subclass implementation must honor it):
          - Scale loss by 1/grad_accum before backward.
          - Apply NaN/Inf/threshold spike protection (log + zero-out).
          - Run backward under the "backward-compute" timing stage
            (self._stage_timers).
          - All-reduce EXACTLY ONCE per optimizer step: only the last accumulation
            step (sync is True) may sync gradients; otherwise skip sync via
            no_sync (DDP) / set_requires_gradient_sync(False) (FSDP2).
          - Accumulate log_loss_dict scalars under their own keys, summed across
            micro-steps.
        """
        ...

    # ── Infrastructure (model-independent plumbing, implemented per subclass) ──

    @abstractmethod
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer (e.g. AdamW with per-module LR groups)."""
        ...

    @abstractmethod
    def _build_scheduler(self):
        """Build the LR scheduler."""
        ...

    @abstractmethod
    def _clip_gradients(self, max_norm: float) -> float:
        """Gradient clipping. Returns the pre-clip global gradient norm."""
        ...

    @abstractmethod
    def _clean_nan_gradients(self):
        """Replace NaN/Inf gradients with 0."""
        ...

    @abstractmethod
    def _load_pretrained(self, path: str):
        """Load pretrained weights."""
        ...

    @abstractmethod
    def _handle_resume(self, path: str, step: int, epoch: int):
        """Resume model weights from a discovered checkpoint, set step/epoch."""
        ...

    @abstractmethod
    def _freeze_modules(self, freeze_str: str):
        """Freeze specified modules."""
        ...

    @abstractmethod
    def _save_checkpoint(self):
        """Persist model/optimizer/scheduler/dataloader state to a checkpoint."""
        ...

    # ── Data / state (iterator lifecycle, implemented per subclass) ──

    @abstractmethod
    def _init_data_iterator(self, name: str):
        """Initialize the iterator for the named dataloader.

        Implementations should honor per-loader epoch semantics and call
        self._maybe_restore_rng_once() once (see multi-dataloader design).
        """
        ...

    @abstractmethod
    def _advance_epoch(self, name: str):
        """Advance the named dataloader to its next epoch and reset its iterator."""
        ...

    @abstractmethod
    def _fetch_batch(self, dl_name: str):
        """Fetch the next batch, cycling the iterator at epoch boundaries."""
        ...

    # ═══════════════════════════════════════════════
    # Optional hooks
    # ═══════════════════════════════════════════════

    def _wrap_model_for_training(self):
        """Wrap ``self.model`` for distributed training.

        Default: invoke :func:`wrap_model` so the model is DDP/FSDP-wrapped with
        mixed precision applied. Subclasses may override to take over wrapping
        entirely (e.g. install a custom train-step runner that manages its own
        wrapping policy).
        """
        self.model = wrap_model(self.model, self.training_args, self.ctx)

    def _run_forward_backward_block(self) -> dict:
        """Run zero_grad + one optimizer step's forward/backward block.

        Default: ``optimizer.zero_grad`` followed by the abstract
        :meth:`_forward_backward` gradient-accumulation loop. Returns
        ``log_dict``. Subclasses may override to take over this block (e.g. a
        CUDA graph runner that fuses zero_grad + forward + backward + grad sync
        and returns the same ``log_dict`` shape).
        """
        with self._stage_timers("optimizer-zero-grad"):
            self.optimizer.zero_grad()
        return self._forward_backward()

    def _configure_backend_precision(self):
        """Hook for model-specific backend precision controls."""
        pass

    def _apply_lora_before_wrap(self, model: nn.Module) -> nn.Module:
        """Apply generic LoRA injection before distributed wrapping."""
        if not is_lora_enabled(self.training_args):
            return model

        resuming_adapter = self._lora_resume_adapter_path is not None
        with log_stage(
            "lora",
            start_msg="applying LoRA" + (" (resume)" if resuming_adapter else ""),
            end_msg="LoRA applied in {elapsed}",
        ):
            model = apply_lora(
                model,
                self.training_args,
                require_base=not resuming_adapter,
                adapter_path=self._lora_resume_adapter_path,
            )
            if resuming_adapter:
                load_adapter_into_model(model, self._lora_resume_adapter_path)
        return model

    def _on_after_data_iterators_initialized(self):
        """Hook after dataloader iterators are initialized."""
        pass

    def _on_train_begin(self):
        """Hook before training loop starts."""
        pass

    def _on_step_end(self, metrics: Dict[str, float]):
        """Hook after each training step. Default: no-op."""
        pass

    # ═══════════════════════════════════════════════
    # Shared functions (model-independent, reused by subclasses)
    # ═══════════════════════════════════════════════

    def _maybe_restore_rng_once(self):
        """Restore per-rank RNG state exactly once, guarding against multiple
        loaders each triggering a restore on their first iterator init."""
        if self._resume_rng_per_rank is None:
            return
        restore_rank_rng_state(self._resume_rng_per_rank, self.ctx)
        if self.ctx.is_main:
            logger.info("RNG state resumed successfully after dataloader iterator init")
        self._resume_rng_per_rank = None

    def _restore_dataloader_states(self):
        """Restore full dataloader states when checkpoints provide them."""
        if not self._resume_dataloader_state:
            return
        for name, state in self._resume_dataloader_state.items():
            dl = self.dataloaders.get(name)
            if dl is None:
                if self.ctx.is_main:
                    logger.warning(f"Checkpoint has dataloader state for unknown loader: {name}")
                continue
            if hasattr(dl, "load_state_dict"):
                dl.load_state_dict(state)
                if self.ctx.is_main:
                    logger.info(f"Restored dataloader state: {name}")
            elif self.ctx.is_main:
                logger.warning(
                    f"Dataloader '{name}' does not support load_state_dict(); "
                    "dataloader state in checkpoint will be ignored."
                )

    def _get_dataloader_state(self) -> Dict[str, Dict[str, Any]]:
        """Return full dataloader states for exact checkpoint resume when supported."""
        states = {}
        for name, dl in self.dataloaders.items():
            if name in self._data_iters and hasattr(dl, "state_dict"):
                states[name] = dl.state_dict()
            elif self.ctx.is_main:
                logger.warning(
                    f"Dataloader '{name}' has not been iterated or does not support state_dict(); "
                    "dataloader state will not be saved in this checkpoint."
                )
        return states

    def _configure_manual_gc(self) -> None:
        """Configure optional explicit Python GC cadence."""
        if not self.training_args.manual_gc:
            return
        interval = int(self.training_args.manual_gc_interval)
        if interval < 0:
            raise ValueError("--manual-gc-interval must be >= 0")
        gc.disable()
        gc.collect()
        if self.ctx.is_main:
            cadence = "startup only" if interval == 0 else f"every {interval} steps"
            logger.info("Manual Python GC enabled (%s)", cadence)

    def _maybe_collect_manual_gc(self) -> None:
        if not self.training_args.manual_gc:
            return
        interval = int(self.training_args.manual_gc_interval)
        if interval > 0 and self.completed_steps % interval == 0:
            with self._stage_timers("manual-gc"):
                gc.collect()

    def _shutdown_dataloaders(self) -> None:
        """Explicitly stop persistent DataLoader workers before process teardown."""
        seen: set[int] = set()

        def shutdown_iterator(name: str, iterator: Any) -> None:
            if iterator is None:
                return
            iterator_id = id(iterator)
            if iterator_id in seen:
                return
            seen.add(iterator_id)
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:  # pragma: no cover - best-effort cleanup
                    logger.warning("DataLoader iterator shutdown failed for %s: %s", name, exc)

        for name, iterator in list(self._data_iters.items()):
            shutdown_iterator(name, iterator)
        self._data_iters.clear()

        for name, dataloader in list(self.dataloaders.items()):
            iterator = getattr(dataloader, "_iterator", None)
            shutdown_iterator(name, iterator)
            if hasattr(dataloader, "_iterator"):
                dataloader._iterator = None

    # ═══════════════════════════════════════════════
    # Finalize
    # ═══════════════════════════════════════════════
    def _finalize(self):
        """End of training: save final model, close W&B."""
        # Final checkpoint — skip when checkpoint saves are disabled or when
        # the last loop iteration already saved at this exact step.
        save_interval = self.training_args.save_interval
        if save_interval and self.completed_steps % save_interval != 0:
            self._save_checkpoint()

        # Wait for any in-flight async DCP save before tearing down the
        # process group / NCCL — otherwise the background writer may race
        # with destroy() and leave an unfinalized checkpoint.
        flush_pending_save(self.ctx)

        # Close W&B / TensorBoard
        self.logger.finish()

        self._shutdown_dataloaders()

        if self.training_args.manual_gc:
            gc.enable()

        self.ctx.barrier()
        self.ctx.destroy()
