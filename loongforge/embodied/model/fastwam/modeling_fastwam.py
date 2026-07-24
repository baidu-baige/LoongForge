# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FastWAM policy for LoongForge embodied training."""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from loongforge.embodied.model.registry import register_model

logger = logging.getLogger(__name__)


def _resolve_dtype_from_str(dtype_str: str) -> torch.dtype:
    """Resolve torch dtype from a string alias."""
    name = dtype_str.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported FastWAM dtype: {dtype_str!r}")


@register_model("fastwam")
class FastWAMPolicy(nn.Module):
    """LoongForge wrapper for the local FastWAM implementation.

    ``BCTrainer`` expects ``model(batch)`` to return a dictionary containing
    ``action_loss``. FastWAM returns ``(loss, metrics)`` from ``training_loss``;
    this wrapper adapts the return value and exposes common checkpoint methods.
    """

    def __init__(self, core: nn.Module):
        """Wrap a FastWAM core module for the embodied trainer interface."""
        super().__init__()
        self.core = core
        self.dit = getattr(core, "dit", None)


    @classmethod
    def from_pretrained(cls, cfg: Any) -> "FastWAMPolicy":
        """Build a FastWAM policy from a typed FastWAMConfig instance."""
        from loongforge.embodied.model.fastwam.modeling_configuration_fastwam import FastWAMModelConfig
        from loongforge.embodied.model.fastwam.mot.fastwam import FastWAM
        from loongforge.embodied.model.fastwam.mot.idm import FastWAMIDM
        from loongforge.embodied.model.fastwam.mot.joint import FastWAMJoint

        if not isinstance(cfg, FastWAMModelConfig):
            raise TypeError(
                "FastWAMPolicy.from_pretrained expects a typed FastWAMModelConfig instance; "
                f"got {type(cfg).__name__}. build_model now passes ModelConfig directly."
            )
        config = cfg

        variant_map = {
            "base": FastWAM,
            "uncond": FastWAM,
            "joint": FastWAMJoint,
            "idm": FastWAMIDM,
        }
        cls_map = variant_map[config.variant]  # variant already validated in __post_init__

        model_dtype = _resolve_dtype_from_str(config.dtype)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        core = cls_map.from_wan22_pretrained(
            device=device,
            torch_dtype=model_dtype,
            model_id=config.model_id,
            tokenizer_model_id=config.tokenizer_model_id,
            tokenizer_max_len=config.tokenizer_max_len,
            load_text_encoder=config.load_text_encoder,
            proprio_dim=config.proprio_dim,
            redirect_common_files=config.redirect_common_files,
            video_dit_config=config.video_dit_config,
            action_dit_config=config.action_dit_config,
            action_dit_pretrained_path=config.action_dit_pretrained_path,
            skip_dit_load_from_pretrain=config.skip_dit_load_from_pretrain,
            mot_checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
            video_train_shift=float(config.video_scheduler["train_shift"]),
            video_infer_shift=float(config.video_scheduler["infer_shift"]),
            video_num_train_timesteps=int(config.video_scheduler["num_train_timesteps"]),
            action_train_shift=float(config.action_scheduler["train_shift"]),
            action_infer_shift=float(config.action_scheduler["infer_shift"]),
            action_num_train_timesteps=int(config.action_scheduler["num_train_timesteps"]),
            loss_lambda_video=float(config.loss["lambda_video"]),
            loss_lambda_action=float(config.loss["lambda_action"]),
        )
        return cls(core)

    def forward(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Run FastWAM training loss and adapt metrics to trainer output dict."""
        sample = batch.to_sample() if hasattr(batch, "to_sample") else batch
        loss, metrics = self.core.training_loss(sample)
        output: Dict[str, torch.Tensor] = {"action_loss": loss}
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if torch.is_tensor(value):
                    output[key] = value.detach()
                else:
                    output[key] = torch.tensor(float(value), device=loss.device)

        return loss, {k: v for k, v in output.items() if k != "action_loss"}

    def load_pretrained(self, path: str, device=None):
        """Load a FastWAM checkpoint through the wrapped core module."""
        del device
        return self.core.load_checkpoint(path)

    def save_checkpoint(self, *args, **kwargs):
        """Save a FastWAM checkpoint through the wrapped core module."""
        return self.core.save_checkpoint(*args, **kwargs)

    def load_checkpoint(self, *args, **kwargs):
        """Load a FastWAM checkpoint through the wrapped core module."""
        return self.core.load_checkpoint(*args, **kwargs)
