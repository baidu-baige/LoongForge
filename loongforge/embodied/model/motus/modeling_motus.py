# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from https://github.com/thu-ml/Motus under the Apache-2.0 License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Motus policy wrapper for the LoongForge embodied trainer.

This module exposes ``MotusPolicy`` — a thin ``nn.Module`` that adapts the
transplanted Motus VLA model (WAN video + Qwen3-VL + Action Expert three-modal MoT)
to the framework's model contract:

    build_model(model_cfg) -> cls.from_pretrained(model_cfg)  # -> MotusPolicy
    loss, log_loss_dict = policy.forward(batch)               # (scalar, {"...": tensor})

Design notes
------------
- The heavy Motus implementation (``models/motus.py`` + the ``wan`` backbone) is
  vendored under this package as ``motus_impl`` in a later step. It is imported
  **lazily** inside ``__init__`` so that ``@register_model("motus")`` runs at import
  time regardless of whether the vendored implementation is present yet (the registry
  auto-importer swallows ``ModuleNotFoundError`` and would otherwise silently drop the
  registration).
- The VAE side-stream prefetch is intentionally NOT driven from ``forward`` here — it
  is owned by ``MotusTrainer`` so it runs against the trainer's real static I/O.
  ``forward`` provides the plain two-stage path (VAE encode -> training_step).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from loongforge.embodied.model.registry import register_model
from loongforge.embodied.model.motus.model_configuration_motus import MotusModelConfig


def _build_motus_config(cfg: MotusModelConfig, batch_size: int):
    """Translate the flat framework ``MotusModelConfig`` into the model's ``MotusConfig``.

    Mapping is 1:1 by name except:
      - ``state_dim``                -> ``action_state_dim``
      - ``action_expert_hidden_size``-> ``action_expert_dim``
    ``batch_size`` is a training-time value (``training_args.per_device_batch_size``);
    it only seeds the model's default sample shapes and is superseded by the trainer's
    real capture inputs, so it is passed in explicitly rather than stored on the config.
    """
    # Lazy import: the vendored implementation may be added after this module.
    from loongforge.embodied.model.motus.motus_impl.motus import MotusConfig

    return MotusConfig(
        wan_checkpoint_path=cfg.wan_checkpoint_path,
        vae_path=cfg.vae_path,
        wan_config_path=cfg.wan_config_path,
        vlm_checkpoint_path=cfg.vlm_checkpoint_path,
        video_precision=cfg.video_precision,
        action_state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        num_layers=cfg.num_layers,
        action_expert_dim=cfg.action_expert_hidden_size,
        action_expert_ffn_dim_multiplier=cfg.action_expert_ffn_dim_multiplier,
        action_expert_norm_eps=cfg.action_expert_norm_eps,
        und_expert_hidden_size=cfg.und_expert_hidden_size,
        und_expert_ffn_dim_multiplier=cfg.und_expert_ffn_dim_multiplier,
        und_expert_norm_eps=cfg.und_expert_norm_eps,
        vlm_adapter_input_dim=cfg.vlm_adapter_input_dim,
        vlm_adapter_projector_type=cfg.vlm_adapter_projector_type,
        global_downsample_rate=cfg.global_downsample_rate,
        video_action_freq_ratio=cfg.video_action_freq_ratio,
        num_video_frames=cfg.num_video_frames,
        video_height=cfg.video_height,
        video_width=cfg.video_width,
        batch_size=batch_size,
        video_loss_weight=cfg.video_loss_weight,
        action_loss_weight=cfg.action_loss_weight,
        training_mode=cfg.training_mode,
        load_pretrained_backbones=cfg.load_pretrained_backbones,
    )


@register_model("motus")
class MotusPolicy(nn.Module):
    """Framework policy wrapper around the transplanted Motus model."""

    def __init__(self, config: MotusModelConfig):
        """Build the wrapped Motus model from ``config`` and cast Conv3d to channels_last_3d."""
        super().__init__()
        self.config = config

        import os

        # batch_size is a training arg; env override keeps a single source of truth on
        # the launch side. The trainer re-captures the graph with real shapes anyway.
        batch_size = int(os.environ.get("PER_DEVICE_BATCH_SIZE", "4"))

        from loongforge.embodied.model.motus.motus_impl.motus import Motus

        self.model = Motus(_build_motus_config(config, batch_size))

        # Conv3d channels_last_3d cast (parity with original train.py:602-604).
        for m in self.model.modules():
            if isinstance(m, nn.Conv3d):
                m.to(memory_format=torch.channels_last_3d)

    @classmethod
    def from_pretrained(cls, model_cfg) -> "MotusPolicy":
        """Instantiate the policy from a typed Motus ModelConfig (or a mapping)."""
        return cls(MotusModelConfig.from_config(model_cfg))

    def forward(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Plain (ungraphed) training forward: VAE encode -> training_step.

        Expects a ``MotusPreparedBatch`` exposing:
          ``first_frame``, ``video_frames``, ``action_sequence``, ``initial_state``,
          ``language_embedding`` (T5), ``vlm_inputs`` (dict).
        Returns ``(total_loss, {"video_loss": ..., "action_loss": ...})``.
        """
        clean_full_latent, condition_frame_latent = self.model.encode_video_latents(
            first_frame=batch.first_frame,
            video_frames=batch.video_frames,
        )
        outputs = self.model.training_step(
            clean_full_latent=clean_full_latent,
            condition_frame_latent=condition_frame_latent,
            state=batch.initial_state,
            actions=batch.action_sequence,
            language_embeddings=batch.language_embedding,
            vlm_inputs=batch.vlm_inputs,
        )
        total_loss = outputs["total_loss"]
        log_loss_dict = {
            "video_loss": outputs["video_loss"].detach(),
            "action_loss": outputs["action_loss"].detach(),
        }
        return total_loss, log_loss_dict

    @property
    def device(self):
        """Device of the policy's first parameter."""
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        """Dtype of the policy's first parameter."""
        return next(iter(self.parameters())).dtype
