# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training logging utilities for metrics, W&B, and progress tracking."""

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from loongforge.embodied.optimizer import get_grad_norm


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Config formatting utilities
# ═══════════════════════════════════════════════════════════════════


def _format_block(title: str, items: dict) -> str:
    """Render a Megatron-style key/value block.

    Layout:
        ------------------------ {title} ------------------------
          key1 ...................................... value
          key2 ...................................... value
        -------------------- end of {title} ---------------------
    """
    lines = [f"------------------------ {title} ------------------------"]
    for key in sorted(items.keys(), key=str.lower):
        dots = "." * max(1, 48 - len(key))
        lines.append(f"  {key} {dots} {items[key]}")
    lines.append(f"-------------------- end of {title} ---------------------")
    return "\n".join(lines)


def _dataclass_items(obj, prefix: str = "") -> dict:
    """Flatten a (possibly nested) dataclass instance into dotted keys."""
    import dataclasses

    out = {}
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        key = f"{prefix}.{f.name}" if prefix else f.name
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            out.update(_dataclass_items(value, key))
        else:
            out[key] = value
    return out


def log_effective_config(training_args, model_cfg, data_cfg):
    """Log fully-resolved TrainingArgs + ModelConfig + DataConfig in Megatron-style.

    Output is gated by the logger level: rank 0 is set to INFO and others to
    WARNING (see setup_logging), so non-rank-0 logger.info calls are filtered
    automatically — no explicit rank check needed here.
    """
    logger.info("\n%s", _format_block("training training_args", _dataclass_items(training_args)))
    logger.info("\n%s", _format_block("model config", _dataclass_items(model_cfg)))
    logger.info("\n%s", _format_block("data config", _dataclass_items(data_cfg)))


# ═══════════════════════════════════════════════════════════════════
# Per-stage timing
# ═══════════════════════════════════════════════════════════════════


class StageTimers:
    """Lightweight per-stage timer for training loop profiling.

    Records wall-clock elapsed time for named stages (forward, backward, etc.)
    and renders a report aligned with the AIAK-Training-Omni / Megatron
    `max time across ranks (ms):` format.

    Unlike Megatron timers, there is no log_level. Timing is toggled globally via
    set_enabled(); the train loop enables it only on the step that will be logged,
    so the cuda.synchronize() inside the timers keeps steady-state overhead at zero.
    When disabled, the context manager returned by __call__ is a no-op, so call
    sites stay free of per-stage `if` checks:

        with stage_timers("forward-compute"):
            output = self._train_forward(batch)

    A cuda.synchronize() is issued on each start/stop so GPU async execution does
    not skew measurements. Stage order in the report matches the fixed ORDER list.
    """

    ORDER = [
        "forward-backward",
        "forward-compute",
        "backward-compute",
        "batch-generator",
        "nan-grad-cleanup",
        "grad-clip",
        "optimizer",
        "optimizer-inner-step",
        "optimizer-scheduler-step",
        "optimizer-zero-grad",
        "manual-gc",
    ]

    def __init__(self):
        self._elapsed: Dict[str, float] = {}
        self._start_time: Dict[str, float] = {}
        self._enabled: bool = False

    def set_enabled(self, enabled: bool):
        """Enable or disable timing. When disabled, __call__ is a no-op."""
        self._enabled = enabled

    @contextmanager
    def __call__(self, name: str):
        """Context manager that times the wrapped block when enabled."""
        if not self._enabled:
            yield
            return
        self._start(name)
        try:
            yield
        finally:
            self._stop(name)

    def _start(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_time[name] = time.perf_counter()

    def _stop(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if name not in self._start_time:
            return
        self._elapsed[name] = self._elapsed.get(name, 0.0) + (
            time.perf_counter() - self._start_time.pop(name)
        )

    def reset(self):
        """Clear all accumulated times."""
        self._elapsed.clear()
        self._start_time.clear()

    def get_max_time_string(
        self, ctx, normalizer: float = 1.0, log_level: int = 0
    ) -> Optional[str]:
        """Render `max time across ranks (ms):` string on rank 0, else None.

        Args:
            ctx: DistributedContext (provides rank/world_size/is_distributed).
            normalizer: Divide accumulated times by this (e.g. number of iters).
            log_level: 0 prints only the max across ranks. 1 additionally prints
                each rank's per-stage time.
        """
        names = [n for n in self.ORDER if n in self._elapsed]
        if not names:
            return None

        # ── Phase 1: collective gather (all ranks must participate) ──────────
        # Timing data is tiny; gather on CPU (gloo) to avoid occupying the GPU
        # stream and a GPU->host sync. The default process group is created with
        # backend "cpu:gloo,cuda:nccl", so CPU tensors are routed via gloo.
        local = torch.tensor(
            [self._elapsed[n] for n in names],
            dtype=torch.float,
            device="cpu",
        )

        if ctx.is_distributed and ctx.world_size > 1:
            # all_gather is a collective op — early return must NOT happen before
            # this point, otherwise non-rank-0 processes would skip it and hang.
            gathered = [torch.zeros_like(local) for _ in range(ctx.world_size)]
            torch.distributed.all_gather(gathered, local)
        else:
            # Normalise to a list so rank-0 formatting code is uniform.
            gathered = [local]

        # ── Phase 2: rank-0 only formatting ─────────────────────────────────
        if ctx.rank != 0:
            return None

        max_times = torch.stack(gathered, dim=0).max(dim=0).values.tolist()
        output_string = "max time across ranks (ms):"
        for name, t in zip(names, max_times):
            ms = (t / normalizer) * 1000.0
            output_string += "\n    {}: {:.2f}".format(name.ljust(48, "."), ms)

        if log_level >= 1:
            for r, g in enumerate(gathered):
                output_string += "\n  rank {} time (ms):".format(r)
                for name, t in zip(names, g.tolist()):
                    ms = (t / normalizer) * 1000.0
                    output_string += "\n    {}: {:.2f}".format(
                        name.ljust(48, "."), ms
                    )
        return output_string


# ═══════════════════════════════════════════════════════════════════
# TrainingLogger class
# ═══════════════════════════════════════════════════════════════════


class TrainingLogger:
    """Handles training metrics logging, W&B integration, and progress tracking.

    This class centralizes all logging-related functionality that was previously
    scattered in BaseTrainer. It provides:
      - Metrics collection and formatting
      - W&B integration
      - JSONL metrics export
      - Progress bar management
      - Training configuration logging
    """

    def __init__(
        self,
        output_dir: str,
        wandb_project: Optional[str] = None,
        wandb_mode: str = "disabled",
        is_main: bool = True,
        model_cfg: Optional[Any] = None,
        tensorboard_dir: Optional[str] = None,
        tensorboard_queue_size: int = 1000,
        run_name: Optional[str] = None,
    ):
        """Initialize the training logger.

        W&B and TensorBoard are independent and may be enabled together. Both
        default to off (W&B via ``wandb_mode="disabled"``, TensorBoard via
        ``tensorboard_dir=None``).

        Args:
            output_dir: Directory for output files (metrics.jsonl, etc.)
            wandb_project: W&B project name (None to disable)
            wandb_mode: W&B mode ("online", "offline", "disabled")
            is_main: Whether this is the main process (only main logs)
            model_cfg: Model config for image size estimation
            tensorboard_dir: Directory for TensorBoard event files. If a
                relative path is given, it is resolved under ``output_dir``.
                ``None`` disables TensorBoard.
            tensorboard_queue_size: Async event-queue size for SummaryWriter.
            run_name: W&B run name (defaults to output_dir basename).
        """
        self.output_dir = output_dir
        self.wandb_project = wandb_project
        self.wandb_mode = wandb_mode
        self.is_main = is_main
        self.model_cfg = model_cfg
        self._wandb_initialized = False
        self._tb_writer = None
        self._tb_dir: Optional[str] = None

        wandb_enabled = wandb_mode != "disabled"
        tb_enabled = bool(tensorboard_dir)

        if not self.is_main:
            return

        if tb_enabled:
            tb_dir = tensorboard_dir
            if not os.path.isabs(tb_dir):
                tb_dir = os.path.join(output_dir, tb_dir)
            try:
                os.makedirs(tb_dir, exist_ok=True)
                self._tb_writer = SummaryWriter(
                    log_dir=tb_dir, max_queue=tensorboard_queue_size
                )
                self._tb_dir = tb_dir
                logger.info(f"TensorBoard enabled: {tb_dir}")
            except Exception as e:
                logger.warning(f"TensorBoard init failed: {e}")
        if wandb_enabled:
            try:
                wandb.init(
                    project=self.wandb_project,
                    name=run_name or os.path.basename(self.output_dir),
                    mode=self.wandb_mode,
                )
                self._wandb_initialized = True
            except Exception as e:
                logger.warning(f"W&B init failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    # W&B Integration
    # ═══════════════════════════════════════════════════════════════

    def log_to_wandb(self, metrics: Dict[str, float], step: int):
        """Log metrics to W&B.

        Args:
            metrics: Dictionary of metric names and values
            step: Current training step
        """
        if not self.is_main or not self._wandb_initialized:
            return
        try:
            if wandb.run:
                wandb.log(metrics, step=step)
        except Exception as e:
            # debug: this is called every log step; a persistent failure (e.g.
            # network drop) would flood the log at warning level. Enable DEBUG
            # logging to diagnose W&B connectivity issues.
            logger.debug(f"W&B log failed at step {step}: {e}")

    def finish_wandb(self):
        """Finish W&B run."""
        if not self.is_main:
            return
        try:
            if wandb.run:
                wandb.finish()
        except Exception as e:
            # warning: finish() is called once at training end; failure may
            # cause run data loss in W&B, so the user should be notified.
            logger.warning(f"W&B finish failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    # TensorBoard Integration
    # ═══════════════════════════════════════════════════════════════

    def log_to_tensorboard(self, metrics: Dict[str, float], step: int):
        """Log scalar metrics to TensorBoard.

        Args:
            metrics: Dictionary of metric names and values
            step: Current training step
        """
        if not self.is_main or self._tb_writer is None:
            return
        for k, v in metrics.items():
            if k == "step":
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                try:
                    self._tb_writer.add_scalar(k, float(v), step)
                except Exception as e:
                    # debug: called every step per metric key; use debug to
                    # avoid log flooding on persistent write errors.
                    logger.debug(f"TensorBoard add_scalar failed for '{k}' at step {step}: {e}")

    def finish_tensorboard(self):
        """Flush and close the TensorBoard writer."""
        if not self.is_main or self._tb_writer is None:
            return
        try:
            self._tb_writer.flush()
            self._tb_writer.close()
        except Exception as e:
            # warning: flush/close is called once at training end; failure may
            # cause incomplete trace files, so the user should be notified.
            logger.warning(f"TensorBoard finish failed: {e}")
        self._tb_writer = None

    def finish(self):
        """Close all logging backends (W&B + TensorBoard)."""
        self.finish_wandb()
        self.finish_tensorboard()

    # ═══════════════════════════════════════════════════════════════
    # Metrics Collection
    # ═══════════════════════════════════════════════════════════════

    def count_batch_size(self, batch) -> int:
        """Count batch size from batch data.

        Supports:
          - list of samples
          - dict with tensor values
          - dataclass/obj with tensor attributes

        Args:
            batch: Batch data

        Returns:
            Number of samples in the batch.
        """
        # 1. List format: len(batch)
        if isinstance(batch, list):
            return len(batch)

        # 2. Dict format: check common keys for batch dimension
        if isinstance(batch, dict):
            for key in ("actions", "action", "input_ids", "attention_mask", "state"):
                if key in batch and batch[key] is not None:
                    return self._get_batch_dim(batch[key])
            return 0

        # 3. Dataclass/object format: check common attributes for batch dimension
        for key in ("actions", "action", "input_ids", "attention_mask", "state"):
            if hasattr(batch, key):
                val = getattr(batch, key)
                if val is not None:
                    return self._get_batch_dim(val)
        return 0

    def _get_batch_dim(self, value) -> int:
        """Get batch dimension from tensor/ndarray.

        Returns 0 if value is not a tensor/ndarray or has no batch dim.
        """
        if isinstance(value, torch.Tensor):
            return value.shape[0] if value.ndim > 0 else 0
        elif isinstance(value, np.ndarray):
            return value.shape[0] if value.ndim > 0 else 0
        elif isinstance(value, list) and len(value) > 0:
            # Handle list of tensors (e.g., images_list)
            first = value[0]
            if isinstance(first, torch.Tensor):
                return first.shape[0] if first.ndim > 0 else 0
            elif isinstance(first, np.ndarray):
                return first.shape[0] if first.ndim > 0 else 0
        return 0

    def collect_metrics(
        self,
        output: Dict[str, Any],
        step_time: float,
        completed_steps: int,
        lr_scheduler: Any,
        consumed_samples: int,
        model: torch.nn.Module,
        local_batch_size: int = 0,
        grad_norm: Optional[float] = None,
    ) -> Dict[str, float]:
        """Collect metrics, including loss, step time, and gradient norm.

        Args:
            output: Output dict from forward pass containing scalar metrics.
                Loss metrics are logged from this dict by key.
            step_time: Time taken for the step
            completed_steps: Current training step
            lr_scheduler: Learning rate scheduler
            consumed_samples: Total consumed samples so far
            model: Model for gradient norm calculation
            batch_size: Batch size (for throughput calculation)
            grad_norm: Pre-clip global gradient norm computed by the train loop.
                Passing it here avoids recomputing the (post-clip) norm.

        Returns:
            Dictionary of collected metrics
        """
        metrics = {
            "step_time": step_time,
            "step": completed_steps,
        }

        # Add output metrics (excluding action_loss which we already have)
        for k, v in output.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                metrics[k] = v.item()
            elif isinstance(v, (int, float)):
                metrics[k] = v

        metrics["lr"] = lr_scheduler.get_last_lr()[0]
        metrics["consumed_samples"] = consumed_samples

        # Calculate samples/second throughput
        if local_batch_size > 0 and step_time > 0:
            samples_per_sec = local_batch_size / step_time
            metrics["samples_per_sec"] = samples_per_sec
        else:
            metrics["samples_per_sec"] = 0.0

        # Gradient norm: prefer the pre-clip value passed by the train loop.
        # Fall back to recomputing only if it wasn't provided.
        if grad_norm is None:
            grad_norm = get_grad_norm(model)
        metrics["grad_norm"] = grad_norm

        return metrics

    # ═══════════════════════════════════════════════════════════════
    # Logging Output
    # ═══════════════════════════════════════════════════════════════

    def log_metrics(
        self,
        metrics: Dict[str, float],
        completed_steps: int,
        train_iters: int,
        per_device_batch_size: int,
        world_size: int = 1,
        is_distributed: bool = False,
        gradient_accumulation_steps: int = 1,
    ):
        """Log metrics to console, W&B, and JSONL file.

        Args:
            metrics: Dictionary of metrics
            completed_steps: Current training step
            train_iters: Total training iterations
            per_device_batch_size: Batch size per device
            world_size: Number of distributed workers
            is_distributed: Whether running in distributed mode
        """
        if not self.is_main:
            return

        lr = metrics.get("lr", 0)
        samples_per_sec = metrics.get("samples_per_sec", 0)
        step_time = metrics.get("step_time", 0)
        consumed_samples = metrics.get("consumed_samples", 0)
        grad_norm = metrics.get("grad_norm", 0)
        skipped_iters = int(metrics.get("skipped_iterations", 0))
        nan_iters = int(metrics.get("nan_iterations", 0))

        # Global batch size = per-device × world_size × gradient_accumulation_steps
        global_batch_size = per_device_batch_size
        if is_distributed:
            global_batch_size *= world_size
        global_batch_size *= gradient_accumulation_steps

        # Format aligned with main framework
        log_string = "iteration {:8d}/{:8d} |".format(completed_steps, train_iters)
        log_string += " consumed samples: {:12d} |".format(consumed_samples)
        log_string += " elapsed time per iteration (ms): {:.1f} |".format(step_time * 1000)
        log_string += " throughput (samples/sec/per_device): {:.3f} |".format(samples_per_sec)
        log_string += " learning rate: {:.6E} |".format(lr)
        log_string += " global batch size: {:5d} |".format(global_batch_size)
        for key, value in metrics.items():
            if "loss" in key and isinstance(value, (int, float)):
                log_string += " {}: {:.6E} |".format(key.replace("_", " "), value)
        log_string += " loss scale: 1.0 |"
        if grad_norm is not None and grad_norm > 0:
            log_string += " grad norm: {:.6f} |".format(grad_norm)
        log_string += " number of skipped iterations: {:3d} |".format(skipped_iters)
        log_string += " number of nan iterations: {:3d} |".format(nan_iters)

        logger.info(log_string)

        # W&B
        self.log_to_wandb(metrics, completed_steps)

        # TensorBoard
        self.log_to_tensorboard(metrics, completed_steps)

        # JSONL
        self._append_metrics_jsonl(metrics)

    def log_stage_times(
        self, timers: "StageTimers", ctx, normalizer: float = 1.0, log_level: int = 0
    ):
        """Log per-stage timing in `max time across ranks (ms):` format.

        All ranks must call this (all_gather is collective); only rank 0 emits.
        """
        output_string = timers.get_max_time_string(ctx, normalizer, log_level)
        if output_string is not None:
            logger.info(output_string)

    def _append_metrics_jsonl(self, metrics: Dict[str, float]):
        """Append metrics to JSONL file."""
        metrics_file = os.path.join(self.output_dir, "metrics.jsonl")
        try:
            with open(metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════
    # Logging Methods
    # ═══════════════════════════════════════════════════════════════

    def log_param_stats(self, model: torch.nn.Module):
        """Log parameter statistics.

        Args:
            model: The model to analyze
        """
        if not self.is_main:
            return

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Parameters: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable "
            f"({100 * trainable / max(total, 1):.1f}%)"
        )

    def log_pretrained_loaded(self, path: str):
        """Log pretrained weight loading."""
        if not self.is_main:
            return
        logger.info(f"Pretrained loaded: {path}")

    def log_resume(self, step: int):
        """Log resume from checkpoint.

        Args:
            step: Step number resumed from
        """
        if not self.is_main:
            return
        logger.info(f"Resumed model weights from step {step}")

    def log_resume_not_found(self):
        """Log that resume was requested but no checkpoint found."""
        if not self.is_main:
            return
        logger.warning("--resume set but no checkpoint found, starting from scratch")

    def log_frozen_module(self, dot_path: str):
        """Log frozen module.

        Args:
            dot_path: Dot-path of frozen module
        """
        if not self.is_main:
            return
        logger.info(f"Frozen: {dot_path}")

    def log_freeze_not_found(self, dot_path: str, resolved_prefix: str = "", missing_attr: str = ""):
        """Log that freeze target was not found.

        Args:
            dot_path: Full dot-path that was requested.
            resolved_prefix: How far traversal succeeded before the failure.
            missing_attr: The attribute name that caused the AttributeError.
        """
        if not self.is_main:
            return
        if resolved_prefix and missing_attr:
            logger.warning(
                f"Freeze target not found: '{dot_path}' "
                f"(resolved up to '{resolved_prefix}', then '{missing_attr}' does not exist)"
            )
        else:
            logger.warning(f"Freeze target not found: '{dot_path}'")

    def log_loss_spike(self, step: int, loss_val: float):
        """Log loss spike detection.

        Args:
            step: Current training step
            loss_val: Loss value that triggered the spike
        """
        if not self.is_main:
            return
        logger.warning(f"[step {step}] Loss spike: {loss_val:.4f}, zeroing")
