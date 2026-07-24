# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-only precision and sample diagnostics for the Finetune adapter."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch

from .features import feature_enabled

logger = logging.getLogger(__name__)


class LingBotTrainingDiagnostics:
    """Keep LingBot's optional precision/sample output out of the trainer."""

    def __init__(self):
        self._precision_loss_accum = []
        self._precision_loss_iteration = 0
        self._sample_meta_seq = 0

    @staticmethod
    def use_baseline_style_loss_log() -> bool:
        """Return whether baseline-style loss logging is enabled."""
        return feature_enabled("LINGBOT_BASELINE_LOSS_LOG")

    def before_forward(
        self, batch, *, ctx, completed_steps: int, gradient_accumulation_steps: int
    ) -> None:
        """Export sample metadata before the forward pass when configured."""
        export_dir = os.environ.get("LINGBOT_SAMPLE_META_EXPORT_DIR", "")
        if not feature_enabled("LINGBOT_SAMPLE_META_EXPORT") or not export_dir:
            return
        sample_meta = getattr(batch, "sample_meta", None)
        if sample_meta is None and hasattr(batch, "as_dict"):
            sample_meta = batch.as_dict().get("_lingbot_sample_meta")
        if not sample_meta:
            return

        rank = ctx.rank if ctx is not None else int(os.environ.get("RANK", "0"))
        self._sample_meta_seq += 1
        path = Path(export_dir) / f"rank{rank:05d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "rank": int(rank),
            "seq": self._sample_meta_seq,
            "step": int(completed_steps + 1),
            "micro": int(
                (self._sample_meta_seq - 1) % max(1, int(gradient_accumulation_steps))
            ),
            "sample_meta": _jsonable_sample_meta(sample_meta),
        }
        with path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def after_forward(
        self, log_loss_dict, *, ctx, gradient_accumulation_steps: int
    ) -> None:
        """Aggregate and log baseline-style loss values after forward."""
        if not self.use_baseline_style_loss_log():
            return
        total = log_loss_dict.get("total loss")
        video = log_loss_dict.get("video loss")
        action = log_loss_dict.get("action loss")
        if total is None or video is None or action is None:
            return

        local_losses = torch.stack(
            [
                total.detach().float(),
                video.detach().float(),
                action.detach().float(),
            ]
        )
        self._precision_loss_accum.append(local_losses)
        num_microbatches = max(1, int(gradient_accumulation_steps))
        if len(self._precision_loss_accum) < num_microbatches:
            return
        stacked = torch.stack(self._precision_loss_accum[:num_microbatches]).sum(dim=0)
        del self._precision_loss_accum[:num_microbatches]
        stacked = stacked / float(num_microbatches)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stacked)
            stacked = stacked / torch.distributed.get_world_size()
        self._precision_loss_iteration += 1
        if ctx is not None and ctx.is_main:
            logger.info(
                "[lingbot-precision-lf-baseline-style-loss] "
                "iteration=%d total_loss=%.10f video_loss=%.10f action_loss=%.10f",
                self._precision_loss_iteration,
                stacked[0].item(),
                stacked[1].item(),
                stacked[2].item(),
            )

    @staticmethod
    def map_loss_log_dict(
        log_loss_dict, *, backward_loss, gradient_accumulation_steps: int
    ):
        """Map LingBot loss names to trainer logging keys."""
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


def _jsonable_sample_meta(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            item = value.detach().cpu().item()
            return (
                int(item)
                if isinstance(item, int)
                else float(item) if isinstance(item, float) else item
            )
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable_sample_meta(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_sample_meta(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
