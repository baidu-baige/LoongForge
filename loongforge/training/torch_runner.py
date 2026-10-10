# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""TorchRunner — lifecycle orchestration for Torch-native training.

Ordinary class (no base class). Owns setup sequence, training loop, epoch/
iterator management, logging, save/restore timing, GC, and finalize.
All forward/backward/optimizer logic lives in ``train_step``.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from functools import partial
from types import ModuleType
from typing import Any, Dict, Optional

from loongforge.engines.torch.train_step import (
    StepContext,
    TorchOptimization,
    train_step,
)

logger = logging.getLogger(__name__)


class TorchRunner:
    """Pure-Python training runner for the Torch engine."""

    def __init__(
        self,
        training_args,
        model_cfg,
        data_cfg,
        method: ModuleType,
        optimization: TorchOptimization,
    ):
        self.training_args = training_args
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.method = method
        self.optimization = optimization

        self.ctx = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.dataloaders: Dict[str, Any] = {}
        self.logger = None

        # Training state
        self.completed_steps: int = 0
        self.current_epoch: int = 0
        self.train_iters: int = training_args.train_iters
        self.nan_iterations: int = 0
        self.skipped_iterations: int = 0

        # Data iterators
        self._data_iters: Dict[str, Any] = {}
        self._resume_dataloader_state: Dict[str, Dict[str, Any]] = {}
        self._resume_rng_per_rank = None
        self._lora_resume_adapter_path: Optional[str] = None
        self._epochs: Dict[str, int] = {}
        self._stage_timers = None

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
        from loongforge.engines.torch.distributed import DistributedContext
        from loongforge.engines.torch.initialize import (
            set_seed, set_deterministic, set_precision, set_backend_precision,
        )
        from loongforge.training.train_logging import TrainingLogger, log_effective_config, setup_logging, log_stage
        from loongforge.engines.torch.checkpointing import (
            get_latest_checkpoint, resume_training_state,
        )
        from loongforge.engines.torch.distributed.parallel import wrap_model
        from loongforge.engines.torch.optimizer import build_optimizer, build_scheduler
        import torch

        training_args = self.training_args
        opt = self.optimization

        # Apply optimization-level args tweak
        if opt.prepare_args is not None:
            training_args = opt.prepare_args(training_args)
            self.training_args = training_args

        # 1. Distributed context
        self.ctx = DistributedContext()
        self.ctx.init()

        # 2. Seed
        set_seed(training_args.seed, training_args.set_seed_by_rank)
        if training_args.deterministic_mode:
            set_deterministic()
        if training_args.cudnn_benchmark:
            if training_args.deterministic_mode:
                raise ValueError(
                    "--cudnn-benchmark and --deterministic-mode conflict: autotuning "
                    "picks algorithms per shape and is not reproducible."
                )
            torch.backends.cudnn.benchmark = True
        if training_args.disable_tf32:
            set_precision(allow_tf32=False)

        # 3. Output directories + logging
        self.output_dir = training_args.output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        if self.ctx.is_main:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.ctx.barrier()
        setup_logging(self.output_dir, self.ctx.rank)
        set_backend_precision(self.model_cfg)

        log_effective_config(training_args, self.model_cfg, self.data_cfg)

        # GC configuration
        if opt.configure_gc is not None:
            opt.configure_gc(training_args, self.ctx)
        else:
            self._configure_manual_gc()

        # 4. TrainingLogger
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

        # 5. Build model
        with log_stage(
            "model",
            start_msg=(
                f"start building model: model_type={self.model_cfg.model_type}, "
                f"class={self.model_cfg.__class__.__name__}"
            ),
            end_msg="model built in {elapsed}",
        ):
            self.model = self.method.build_model(self.model_cfg)

        # 6. Pretrained weights / Resume
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

        # 7. Freeze modules
        self._freeze_modules(training_args.freeze_modules)

        with log_stage(
            "wrap_model",
            start_msg=f"wrap_model: strategy={training_args.distributed_strategy}, dtype={training_args.dtype}",
            end_msg="done in {elapsed}",
        ):
            if opt.wrap_model is not None:
                self.model = opt.wrap_model(self.model, training_args, self.ctx)
            else:
                self.model = wrap_model(self.model, training_args, self.ctx)

        # 7.5 Deferred materialize + load_pretrained
        if training_args.init_on_meta and not training_args.resume:
            with log_stage(
                "materialize",
                start_msg=f"materializing meta tensors on {self.ctx.device}",
                end_msg="materialized in {elapsed}",
            ):
                self.model.materialize(self.ctx.device, training_args.dtype)
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
            if opt.build_optimizer is not None:
                self.optimizer = opt.build_optimizer(self.model, training_args, self.ctx)
            else:
                self.optimizer = build_optimizer(self.model, training_args)
            self.lr_scheduler = build_scheduler(self.optimizer, training_args)

        # 9. Resume optimizer/scheduler/RNG state
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
                if saved_epoch is not None:
                    self.current_epoch = saved_epoch
                self._resume_dataloader_state = dataloader_state or {}
                self._resume_rng_per_rank = rng_per_rank

        # 10. Data
        with log_stage("data", start_msg="building dataloaders"):
            self.dataloaders = self.method.build_dataloaders(
                self.model_cfg, self.data_cfg, training_args, self.ctx
            )
            self._restore_dataloader_states()

        # 11. Print stats
        self.logger.log_param_stats(self.model)

        # Hook
        self.method.on_train_begin(self.model, self.ctx)

        # Optimization after_setup hook
        if opt.after_setup is not None:
            opt.after_setup(training_args)

    # ═══════════════════════════════════════════════
    # Training loop
    # ═══════════════════════════════════════════════

    def _training_loop(self):
        from loongforge.engines.torch.profiling import Profiler
        from loongforge.training.train_logging import StageTimers

        training_args = self.training_args
        opt = self.optimization
        log_interval = training_args.log_interval
        detail_log_interval = training_args.detail_log_interval
        save_interval = training_args.save_interval

        prof = Profiler(training_args, self.ctx, self.output_dir)
        prof.start()

        self._stage_timers = StageTimers()

        for name in self.dataloaders:
            self._init_data_iterator(name)
        self.method.on_after_data_iterators_initialized(
            self.model,
            training_args=training_args,
            completed_steps=self.completed_steps,
            optimizer=self.optimizer,
            ctx=self.ctx,
        )

        step_ctx = StepContext(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            training_args=training_args,
            model_cfg=self.model_cfg,
            ctx=self.ctx,
            timers=self._stage_timers,
            method=self.method,
            optimization=opt,
            fetch_cpu_batch=partial(self._fetch_batch_cpu, "vla"),
            log_loss_spike=self.logger.log_loss_spike,
        )

        while self.completed_steps < self.train_iters:

            prof.step(self.completed_steps)
            enable_detail = (
                detail_log_interval > 0
                and (self.completed_steps + 1) % detail_log_interval == 0
            )
            self._stage_timers.set_enabled(enable_detail)

            t0 = time.perf_counter()

            step_ctx.iteration = self.completed_steps
            result = train_step(step_ctx)

            self.completed_steps += 1

            if result.nan:
                self.nan_iterations += 1
            if result.skipped:
                self.skipped_iterations += 1

            # Cross-rank loss aggregation
            loss_log_ranks = training_args.loss_log_rank
            log_dict = result.log_dict
            if any(r < 0 for r in loss_log_ranks):
                for key in list(log_dict.keys()):
                    if "loss" in key:
                        log_dict[key] = self.ctx.all_reduce_mean(log_dict[key])
            elif (self.ctx.rank in loss_log_ranks
                  and self.completed_steps % log_interval == 0):
                self._log_local_loss(log_dict)

            # Metrics
            step_time = time.perf_counter() - t0
            local_batch_size = training_args.gradient_accumulation_steps * training_args.per_device_batch_size
            global_batch_size = local_batch_size * self.ctx.world_size
            consumed_samples = self.completed_steps * global_batch_size
            metrics = self.logger.collect_metrics(
                log_dict, step_time,
                self.completed_steps, self.lr_scheduler,
                consumed_samples,
                self.model, local_batch_size, result.grad_norm,
            )
            metrics["nan_iterations"] = self.nan_iterations
            metrics["skipped_iterations"] = self.skipped_iterations

            # Step-end hook
            if opt.on_step_end is not None:
                opt.on_step_end(metrics, self.completed_steps, self.model)
            self._maybe_collect_manual_gc()

            # Profiler stop
            if prof.should_stop(self.completed_steps):
                prof.stop()

            # Logging
            if self.completed_steps % log_interval == 0:
                self.logger.log_metrics(
                    metrics, self.completed_steps, self.train_iters,
                    training_args.per_device_batch_size, self.ctx.world_size, self.ctx.is_distributed,
                    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                )

            # Per-stage timing log
            if enable_detail:
                self.logger.log_stage_times(
                    self._stage_timers, self.ctx, log_level=training_args.timing_log_level
                )
                self._stage_timers.reset()

            # Commit the availability marker of an in-flight async save as soon
            # as its write lands, instead of waiting for the next save to start
            # (which would make resume fall back a whole save interval).
            from loongforge.engines.torch.checkpointing import maybe_finalize_pending_save
            maybe_finalize_pending_save(self.ctx, self.completed_steps)

            # Checkpoint
            if save_interval and self.completed_steps % save_interval == 0:
                self._save_checkpoint()

        prof.stop()

    def _log_local_loss(self, log_dict: dict):
        """Print this rank's own local loss (no cross-rank communication)."""
        loss_str = " ".join(
            f"{k}={v:.6f}" for k, v in log_dict.items()
            if "loss" in k and isinstance(v, (int, float))
        )
        logger.warning(
            "[rank %d][step %d] %s", self.ctx.rank, self.completed_steps, loss_str
        )

    # ═══════════════════════════════════════════════
    # Infrastructure helpers
    # ═══════════════════════════════════════════════

    def _load_pretrained(self, path: str):
        """Load pretrained weights, preferring model.load_pretrained if available."""
        from loongforge.engines.torch.checkpointing import load_pretrained
        if hasattr(self.model, "load_pretrained"):
            self.model.load_pretrained(path, device=self.ctx.device)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "load_pretrained"):
            self.model.model.load_pretrained(path, device=self.ctx.device)
        else:
            load_pretrained(self.model, path, self.ctx)
        self.logger.log_pretrained_loaded(path)

    def _handle_resume(self, path: str, step: int, epoch: int):
        """Resume model weights from a discovered checkpoint."""
        from loongforge.engines.torch.checkpointing import (
            detect_checkpoint_format,
            is_lora_adapter_checkpoint,
            load_pretrained,
            read_adapter_meta,
        )
        if is_lora_adapter_checkpoint(path):
            if not self.training_args.use_lora:
                raise ValueError(
                    "Resuming a LoRA adapter checkpoint requires --use-lora."
                )
            meta = read_adapter_meta(path) or {}
            base_checkpoint = meta.get("base_checkpoint")
            if base_checkpoint:
                if self.ctx.is_main:
                    logger.info(
                        "LoRA resume: loading base weights from %s",
                        base_checkpoint,
                    )
                self._load_pretrained(base_checkpoint)
            elif self.ctx.is_main:
                logger.info(
                    "LoRA resume: model provider supplied base weights; "
                    "adapter weights will load before distributed wrapping."
                )
            self._lora_resume_adapter_path = path
            self.completed_steps = step
            self.current_epoch = epoch
            self.logger.log_resume(step)
            return

        fmt = detect_checkpoint_format(path)
        if fmt == "dcp":
            if self.ctx.is_main:
                logger.info(
                    "resume: detected DCP checkpoint at %s \u2014 deferring weight "
                    "load until after wrap_model.", path,
                )
        else:
            load_pretrained(self.model, path, self.ctx)
        self.completed_steps = step
        self.current_epoch = epoch
        self.logger.log_resume(step)

    def _freeze_modules(self, freeze_str: str):
        """Freeze specified modules by dot-path."""
        if not freeze_str:
            freeze_func = getattr(self.model, "freeze_modules", None)
            if callable(freeze_func):
                freeze_func()
            return
        for dot_path in [p.strip() for p in freeze_str.split(",") if p.strip()]:
            current_module = self.model
            successfully_traversed = []
            try:
                for attr_name in dot_path.split("."):
                    current_module = getattr(current_module, attr_name)
                    successfully_traversed.append(attr_name)
                for param in current_module.parameters():
                    param.requires_grad = False
                self.logger.log_frozen_module(dot_path)
            except AttributeError:
                resolved_prefix = ".".join(successfully_traversed) if successfully_traversed else "<root>"
                missing_attr = dot_path.split(".")[len(successfully_traversed)]
                self.logger.log_freeze_not_found(dot_path, resolved_prefix, missing_attr)

    def _save_checkpoint(self):
        from loongforge.engines.torch.checkpointing import save_checkpoint
        save_checkpoint(
            self.model, self.optimizer, self.lr_scheduler,
            self.completed_steps, self.checkpoint_dir, self.ctx, self.training_args,
            epoch=self.current_epoch,
            dataloader_state=self._get_dataloader_state(),
            model_cfg=self.model_cfg,
        )

    def _apply_lora_before_wrap(self, model):
        """Apply generic LoRA injection before distributed wrapping."""
        from loongforge.engines.torch.lora import apply_lora, is_lora_enabled, load_adapter_into_model
        from loongforge.training.train_logging import log_stage
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

    # ═══════════════════════════════════════════════
    # Data / state
    # ═══════════════════════════════════════════════

    def _init_data_iterator(self, name: str):
        """Initialize iterator for named dataloader."""
        dl = self.dataloaders[name]
        epoch = self._epochs.get(name, self.current_epoch if name == "vla" else 0)
        sampler = getattr(dl, "sampler", None)
        restored_from_state = name in self._resume_dataloader_state
        if (
            sampler is not None
            and hasattr(sampler, "set_epoch")
            and not restored_from_state
        ):
            sampler.set_epoch(epoch)
        self._epochs[name] = epoch
        self._data_iters[name] = iter(dl)
        self._maybe_restore_rng_once()
        if self.ctx.is_main:
            logger.info(f"Dataloader '{name}' positioned at epoch={epoch}")

    def _advance_epoch(self, name: str):
        """Move the named dataloader to the next epoch."""
        self._epochs[name] = self._epochs.get(name, 0) + 1
        if name == "vla":
            self.current_epoch = self._epochs[name]
        dl = self.dataloaders[name]
        if hasattr(dl, "sampler") and hasattr(dl.sampler, "set_epoch"):
            dl.sampler.set_epoch(self._epochs[name])
        self._data_iters[name] = iter(dl)

    def _fetch_batch_cpu(self, dl_name: str):
        """Fetch next CPU batch without moving it to the training device."""
        try:
            batch = next(self._data_iters[dl_name])
        except StopIteration:
            self._advance_epoch(dl_name)
            batch = next(self._data_iters[dl_name])
        return batch

    def _maybe_restore_rng_once(self):
        """Restore per-rank RNG state exactly once."""
        if self._resume_rng_per_rank is None:
            return
        from loongforge.engines.torch.checkpointing import restore_rank_rng_state
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
        """Return full dataloader states for exact checkpoint resume."""
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

    # ═══════════════════════════════════════════════
    # GC management
    # ═══════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════
    # Finalize
    # ═══════════════════════════════════════════════

    def _shutdown_dataloaders(self) -> None:
        """Explicitly stop persistent DataLoader workers before process teardown."""
        seen: set = set()

        def shutdown_iterator(name: str, iterator) -> None:
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
                except Exception as exc:
                    logger.warning("DataLoader iterator shutdown failed for %s: %s", name, exc)

        for name, iterator in list(self._data_iters.items()):
            shutdown_iterator(name, iterator)
        self._data_iters.clear()

        for name, dataloader in list(self.dataloaders.items()):
            iterator = getattr(dataloader, "_iterator", None)
            shutdown_iterator(name, iterator)
            if hasattr(dataloader, "_iterator"):
                dataloader._iterator = None

    def _finalize(self):
        """End of training: save final model, close W&B."""
        from loongforge.engines.torch.checkpointing import flush_pending_save
        opt = self.optimization

        # Optimization close hook (e.g. graph release, GC restore)
        if opt.close is not None:
            opt.close()

        save_interval = self.training_args.save_interval
        if save_interval and self.completed_steps % save_interval != 0:
            self._save_checkpoint()

        flush_pending_save(self.ctx)
        self.logger.finish()
        self._shutdown_dataloaders()

        if self.training_args.manual_gc:
            gc.enable()

        self.ctx.barrier()
        self.ctx.destroy()
