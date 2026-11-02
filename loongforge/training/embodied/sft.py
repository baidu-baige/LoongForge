# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Shared embodied SFT recipe."""

import inspect
import logging
from contextlib import nullcontext
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loongforge.data.embodied.dataloader import build_dataloader
from loongforge.data.embodied.registry import DataRuntime
from loongforge.engines.torch.base_trainer import BaseTrainer
from loongforge.engines.torch.distributed.utils import unwrap_model
from loongforge.engines.torch.utils.utils import resolve_dtype, set_seed
from loongforge.models.embodied import build_model

logger = logging.getLogger(__name__)


class FinetuneTrainer(BaseTrainer):
    """
    Standard single-stream finetune trainer.

    forward(batch) → (loss, log_loss_dict) → backward → step, over the single "vla"
    dataloader. Behaviorally equivalent to the former BCTrainer, with optional
    model-owned hooks for pre-wrap preparation and training-time policy setup.
    """

    def _build_model(self) -> nn.Module:
        return build_model(self.model_cfg, self.training_args)

    def _build_dataloaders(self) -> Dict[str, DataLoader]:
        runtime = DataRuntime(
            rank=self.ctx.rank,
            world_size=self.ctx.world_size,
            is_distributed=self.ctx.is_distributed,
            grad_accum_steps=self.training_args.gradient_accumulation_steps,
            deterministic=self.training_args.deterministic_mode,
            graph_enabled=self.training_args.cuda_graph_impl == "local",
            graph_pad_length=self.training_args.cuda_graph_pad_length,
        )
        dl = build_dataloader(
            self.model_cfg, self.data_cfg, self.training_args, runtime,
        )
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

        fwd_kwargs = self._select_forward_kwargs(
            iteration=self.completed_steps,
        )

        autocast_ctx = (
            nullcontext()
            if self._cfg_bool("disable_train_autocast", False)
            else torch.autocast("cuda", dtype=dtype)
        )

        with self._fp8_forward_ctx(), autocast_ctx:
            loss, log_loss_dict = self.model(batch, **fwd_kwargs)

        return loss, log_loss_dict

    def _select_forward_kwargs(self, **candidates) -> Dict[str, object]:
        """Keep the candidate kwargs the model ``forward`` can actually accept.

        The trainer offers optional training-context values (``iteration`` today,
        more later); a model opts in either by naming the parameter in its
        ``forward`` signature or by declaring ``**kwargs``. The unwrapped forward
        signature is introspected once and cached. ``*args`` is ignored -- these are
        always passed by keyword.

        Silently dropping a value is the dangerous outcome: ``iteration`` seeds the
        deterministic noise/timestep generators, so losing it downgrades a run to the
        global-RNG path without any error. Hence a ``**kwargs`` model gets everything.
        """
        if getattr(self, "_fwd_param_names", None) is None:
            core = self.model
            for attr in ("_orig_mod", "module"):
                core = getattr(core, attr, core)
            params = list(inspect.signature(core.forward).parameters.values())
            self._fwd_param_names = {
                param.name
                for param in params
                if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  inspect.Parameter.KEYWORD_ONLY)
            }
            self._fwd_accepts_var_kw = any(
                param.kind is inspect.Parameter.VAR_KEYWORD for param in params
            )
        if self._fwd_accepts_var_kw:
            # **kwargs absorbs whatever is not declared by name.
            return dict(candidates)
        return {
            name: value
            for name, value in candidates.items()
            if name in self._fwd_param_names
        }

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
        if not callable(hook):
            return
        rng_reset = hook(
            args=self.training_args,
            completed_steps=self.completed_steps,
            micro_step=micro_step,
            batch=batch,
        )
        if rng_reset is None:
            return

        seed, cpu_burn = rng_reset
        seed, cpu_burn = int(seed), int(cpu_burn)
        if cpu_burn < 0:
            raise ValueError("RNG reset cpu_burn must be >= 0")
        set_seed(seed)
        if cpu_burn:
            torch.rand(cpu_burn)
        logger.info(
            "%s reset training RNG after first batch fetch: seed=%d cpu_burn=%d",
            raw.__class__.__name__,
            seed,
            cpu_burn,
        )

    def _on_train_begin(self):
        model = unwrap_model(self.model)
        hook = getattr(model, "on_train_begin", None)
        if callable(hook):
            hook(ctx=self.ctx)
        if self.ctx.is_main:
            logger.info(f"Model: {model.__class__.__name__}")
