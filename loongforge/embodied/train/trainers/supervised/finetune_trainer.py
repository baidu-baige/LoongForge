# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FinetuneTrainer — standard single-stream finetune paradigm.

Migrated from the former BCTrainer plus the model-independent infrastructure /
data-state implementations that previously lived in BaseTrainer. BaseTrainer now
only declares these as abstract methods.
"""

import logging
from contextlib import nullcontext
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loongforge.embodied.data import build_dataloader
from loongforge.embodied.distributed.checkpoint import (
    detect_checkpoint_format,
    is_lora_adapter_checkpoint,
    load_pretrained,
    read_adapter_meta,
    save_checkpoint,
)
from loongforge.embodied.distributed.utils import unwrap_model
from loongforge.embodied.train.utils.utils import resolve_dtype
from loongforge.embodied.model import build_model
from loongforge.embodied.optimizer import (
    build_optimizer,
    build_scheduler,
    clean_nan_gradients,
    clip_gradients,
)
from ..base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class FinetuneTrainer(BaseTrainer):
    """
    Standard single-stream finetune trainer.

    forward(batch) → (loss, log_loss_dict) → backward → step, over the single "vla"
    dataloader. Behaviorally equivalent to the former BCTrainer, with optional
    model-owned hooks for pre-wrap preparation and training-time policy setup.
    """

    # ═══════════════════════════════════════════════
    # Compute — model / data / paradigm
    # ═══════════════════════════════════════════════

    def _build_model(self) -> nn.Module:
        return build_model(self.model_cfg)

    def _build_dataloaders(self) -> Dict[str, DataLoader]:
        dl = build_dataloader(self.model_cfg, self.data_cfg, self.training_args, self.ctx)
        return {"vla": dl}

    def _train_forward(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Single forward: call model(batch).

        Returns (loss, log_loss_dict). ``loss`` is the single scalar that needs
        backward; ``log_loss_dict`` carries extra scalars used only for
        printing/reporting.
        """
        dtype = getattr(self, "_compute_dtype", None)
        if dtype is None:
            dtype = resolve_dtype(self.training_args.dtype)
            self._compute_dtype = dtype

        autocast_ctx = (
            nullcontext()
            if self._cfg_bool("disable_train_autocast", False)
            else torch.autocast("cuda", dtype=dtype)
        )

        with autocast_ctx:
            loss, log_loss_dict = self.model(batch)

        return loss, log_loss_dict

    def _forward_backward(self) -> dict:
        """Single-stream gradient-accumulation loop (reuses base timed helpers).

        Returns log_dict — a per-step accumulator of model-provided reporting
        losses; the loop feeds it straight to collect_metrics.
        """
        st = self._stage_timers
        grad_accum = self.training_args.gradient_accumulation_steps
        log_dict: Dict[str, float] = {}
        with st("forward-backward"):
            for micro in range(grad_accum):
                with self._stage_timers("batch-generator"):
                    batch = self._fetch_batch("vla")
                self._on_after_train_batch_fetch(batch, micro)
                self._prepare_model_for_train_step()
                with st("forward-compute"):
                    loss, log_loss_dict = self._train_forward(batch)
                sync_grads = self._should_sync_grads(micro, grad_accum)
                self._backward_loss(loss, log_loss_dict, log_dict, grad_accum, sync_grads)
        return log_dict

    def _backward_loss(self, loss: torch.Tensor,
                       log_loss_dict: Dict[str, torch.Tensor],
                       log_dict: Dict[str, float],
                       grad_accum: int, sync_grads: bool) -> None:
        """Scale + spike-guard + backward routing, accumulating losses into log_dict.

        ``loss`` is the single scalar to backpropagate; it is scaled by
        1/grad_accum, spike-protected, and backwarded with cross-rank gradient
        sync gated so the all-reduce happens exactly once per optimizer step
        (only on the last accumulation step).

        ``log_loss_dict`` contains model-provided reporting losses, which are
        accumulated into ``log_dict`` across micro-steps.
        """
        threshold = self.training_args.loss_spike_threshold
        with self._stage_timers("backward-compute"):
            # Scale + loss spike protection (zero out to prevent NaN propagation).
            raw_loss = loss
            loss = raw_loss / grad_accum
            loss_val = loss.detach().item()
            is_nan = bool(torch.isnan(loss) or torch.isinf(loss))
            if is_nan or loss_val > threshold:
                self.logger.log_loss_spike(self.completed_steps, loss_val)
                if is_nan:
                    self._step_loss_is_nan = True
                self._step_loss_spiked = True
                loss = loss * 0.0

            # Backward; skip cross-rank gradient sync except on the final
            # backward (all-reduce exactly once per optimizer step).
            if self.ctx.is_distributed and not sync_grads:
                if hasattr(self.model, "no_sync"):
                    with self.model.no_sync():
                        loss.backward()
                else:
                    # FSDP2 (fully_shard): use set_requires_gradient_sync
                    self.model.set_requires_gradient_sync(False)
                    loss.backward()
                    self.model.set_requires_gradient_sync(True)
            else:
                loss.backward()

        # Print-only losses (summed across micro-steps).
        for key, value in log_loss_dict.items():
            v = value.detach().item() if isinstance(value, torch.Tensor) else float(value)
            log_dict[key] = log_dict.get(key, 0.0) + v / grad_accum

    def _cfg_bool(self, key: str, default: bool = False) -> bool:
        value = getattr(self.model_cfg, key, default)
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _configure_backend_precision(self) -> None:
        """Apply optional CUDA backend precision policy from model config."""
        if not self._cfg_bool("disable_reduced_precision_reduction", False):
            return

        matmul = torch.backends.cuda.matmul
        if hasattr(matmul, "allow_bf16_reduced_precision_reduction"):
            matmul.allow_bf16_reduced_precision_reduction = False
        if hasattr(matmul, "allow_fp16_reduced_precision_reduction"):
            matmul.allow_fp16_reduced_precision_reduction = False

    def _on_after_data_iterators_initialized(self):
        raw = unwrap_model(self.model)
        hook = getattr(raw, "on_after_data_iterators_initialized", None)
        if callable(hook):
            hook(
                args=self.training_args,
                completed_steps=self.completed_steps,
                optimizer=self.optimizer,
                ctx=self.ctx,
            )

    def _on_after_train_batch_fetch(self, batch, micro_step: int) -> None:
        raw = unwrap_model(self.model)
        hook = getattr(raw, "on_after_train_batch_fetch", None)
        if callable(hook):
            hook(
                args=self.training_args,
                completed_steps=self.completed_steps,
                micro_step=micro_step,
                batch=batch,
            )

    def _prepare_model_for_train_step(self):
        """Put model in train mode and let policies re-freeze eval-only modules."""
        self.model.train()
        raw = unwrap_model(self.model)
        if hasattr(raw, "set_frozen_modules_to_eval_mode"):
            raw.set_frozen_modules_to_eval_mode()

    def _on_train_begin(self):
        if self.ctx.is_main:
            model = unwrap_model(self.model)
            logger.info(f"Model: {model.__class__.__name__}")

    # ═══════════════════════════════════════════════
    # Infrastructure
    # ═══════════════════════════════════════════════

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW with per-module LR groups."""
        return build_optimizer(self.model, self.training_args)

    def _build_scheduler(self):
        """Build LR scheduler."""
        return build_scheduler(self.optimizer, self.training_args)

    def _clip_gradients(self, max_norm: float) -> float:
        """Gradient clipping. Returns the pre-clip global gradient norm."""
        if hasattr(self.optimizer, "clip_grad_norm"):
            return self.optimizer.clip_grad_norm(max_norm)
        return clip_gradients(self.model, max_norm)

    def _clean_nan_gradients(self):
        """Replace NaN/Inf gradients with 0."""
        clean_nan_gradients(self.model)

    def _load_pretrained(self, path: str):
        """Load pretrained weights, preferring model.load_pretrained if available."""
        if hasattr(self.model, "load_pretrained"):
            self.model.load_pretrained(path, device=self.ctx.device)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "load_pretrained"):
            self.model.model.load_pretrained(path, device=self.ctx.device)
        else:
            load_pretrained(self.model, path, self.ctx)
        self.logger.log_pretrained_loaded(path)

    def _handle_resume(self, path: str, step: int, epoch: int):
        """Resume model weights from a discovered checkpoint.

        For ``dcp`` checkpoints, weight loading is deferred until after
        ``wrap_model`` (handled by ``resume_training_state``), since DCP needs
        the FSDP-sharded DTensor layout to know how to reshard. Calling
        ``load_pretrained`` here would also fail because there is no
        consolidated single-file model in a DCP checkpoint dir.
        """
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
                    "resume: detected DCP checkpoint at %s — deferring weight "
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
        save_checkpoint(
            self.model, self.optimizer, self.lr_scheduler,
            self.completed_steps, self.checkpoint_dir, self.ctx, self.training_args,
            epoch=self.current_epoch,
            dataloader_state=self._get_dataloader_state(),
            model_cfg=self.model_cfg,
        )

    # ═══════════════════════════════════════════════
    # Data / state — per-loader epoch + one-shot RNG restore
    # ═══════════════════════════════════════════════

    def _init_data_iterator(self, name: str):
        """Initialize iterator for named dataloader and store in self._data_iters.

        Uses a per-loader epoch counter (self._epochs). The primary loader "vla"
        mirrors self.current_epoch (set by checkpoint resume) so its shuffle
        stream is aligned; other loaders start at 0 unless restored.

        SKIP `set_epoch` when this loader's state was just restored via
        `dl.load_state_dict()` — `StatefulDistributedSampler.set_epoch` clears
        the `_yielded` progress counter, which would re-emit the epoch from
        sample 0 and silently undo the in-epoch resume position.
        """
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
        # One-shot RNG restore (only the first loader to init triggers it).
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

    def _fetch_batch(self, dl_name: str):
        """Fetch next batch, handle epoch boundary by cycling the iterator."""
        batch = self._fetch_batch_cpu(dl_name)
        return self._move_batch_to_device(batch)

    def _fetch_batch_cpu(self, dl_name: str):
        """Fetch next CPU batch without moving it to the training device."""
        try:
            batch = next(self._data_iters[dl_name])
        except StopIteration:
            self._advance_epoch(dl_name)
            batch = next(self._data_iters[dl_name])
        return batch

    def _move_batch_to_device(self, batch):
        """Move a fetched batch to the current model device."""
        device = next(self.model.parameters()).device
        if hasattr(batch, "to"):
            batch = batch.to(device)
        return batch
