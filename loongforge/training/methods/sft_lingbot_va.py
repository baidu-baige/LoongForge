# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SFT method for LingBot-VA: re-exports sft_embodied, overrides forward only."""

from __future__ import annotations

from loongforge.training.methods.sft_embodied import (  # noqa: F401
    build_model,
    build_dataloaders,
    on_train_begin,
    on_after_data_iterators_initialized,
    on_after_train_batch_fetch,
)
from loongforge.training.methods import sft_embodied


def _map_loss_log_dict(log_loss_dict, *, backward_loss, gradient_accumulation_steps: int):
    """Map LingBot loss names to the keys the common trainer logs."""
    metric_key_map = {
        "total loss": "lingbot_total_loss",
        "video loss": "lingbot_video_loss",
        "action loss": "lingbot_diffusion_action_loss",
    }
    mapped_log_dict = {}
    for key, value in log_loss_dict.items():
        metric_key = metric_key_map.get(key, f"lingbot_{key.replace(' ', '_')}")
        if metric_key == "action_loss":
            metric_key = "lingbot_logged_action_loss"
        mapped_log_dict[metric_key] = value
    mapped_log_dict["action_loss"] = backward_loss.detach() * float(
        max(1, int(gradient_accumulation_steps))
    )
    return mapped_log_dict


def forward(model, batch, *, iteration: int, training_args) -> tuple:
    """LingBot forward: delegates to sft_embodied, then maps loss names."""
    loss, logs = sft_embodied.forward(model, batch, iteration=iteration, training_args=training_args)
    return loss, _map_loss_log_dict(
        logs,
        backward_loss=loss,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    )
