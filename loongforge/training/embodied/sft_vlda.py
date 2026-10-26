# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLDA model assembly, deterministic sampling, and teacher updates."""

from loongforge.training.embodied import sft
from loongforge.training.embodied.sft import (
    forward,
    on_after_train_batch_fetch,
    on_train_begin,
)

__all__ = [
    "build_model", "build_dataloaders", "forward", "on_train_begin",
    "on_after_data_iterators_initialized", "on_after_train_batch_fetch", "on_step_end",
]


def build_model(model_cfg, training_args):
    if training_args.distributed_strategy != "ddp" or training_args.init_on_meta:
        raise ValueError("VLDA currently supports DDP and DDP + ZeRO1; FSDP teacher EMA is not implemented")
    if training_args.use_lora or training_args.save_format not in ("safetensors", "pt"):
        raise ValueError("VLDA uses full-model safetensors/pt checkpoints")
    if training_args.dataset_strategy != "vlda":
        raise ValueError("VLDA requires --dataset-strategy vlda for future frames and negative actions")
    model = sft.build_model(model_cfg, training_args)
    model.model.initialize_training_modules(training_args.seed)
    return model


def build_dataloaders(model_cfg, data_cfg, training_args, ctx):
    return sft.build_dataloaders(model_cfg, data_cfg, training_args, ctx)


def on_after_data_iterators_initialized(model, *, training_args, completed_steps, optimizer, ctx):
    from loongforge.engines.torch.initialize import set_seed

    # Iterator startup must not shift the training RNG.
    if not training_args.resume:
        set_seed(training_args.seed, training_args.set_seed_by_rank)


def on_step_end(metrics, completed_steps, model, *, training_args):
    from loongforge.engines.torch.distributed.utils import unwrap_model

    raw = unwrap_model(model)
    fraction = (completed_steps - 1) / training_args.train_iters
    momentum = raw.config.ema_start + fraction * (raw.config.ema_end - raw.config.ema_start)
    raw.update_teacher_ema(momentum)
    metrics["teacher_ema"] = momentum
