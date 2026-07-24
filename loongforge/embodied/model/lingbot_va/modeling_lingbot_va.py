# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LingBot-VA under the Apache-2.0 License.
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

"""Embodied training adapter for the native PyTorch LingBot-VA backend."""

import hashlib
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from loongforge.embodied.model.registry import register_model

from .checkpoint import load_sharded_safetensors
from .modules.flow_match import (
    LingBotVAFlowMatchScheduler,
    get_mesh_id,
    sample_timestep_id,
)
from .modules.wan_model import WanAttention, WanTransformer3DModel


def _reference_rotary(x: torch.Tensor, frequencies: torch.Tensor):
    complex_x = torch.view_as_complex(x.double().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(complex_x * frequencies).flatten(3).to(x.dtype)


def _cfg_get(cfg, key, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _fixed_rng_seed(microbatch: int) -> int:
    return 42 + microbatch


def _rng_diag_enabled() -> bool:
    return os.environ.get("LINGBOT_RNG_DIAG", "0") == "1"


def _rng_digest(state: torch.Tensor) -> str:
    return hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()[:16]


def _rng_diag(side: str, phase: str, microbatch: int) -> None:
    if not _rng_diag_enabled() or int(os.environ.get("RANK", "0")) != 0:
        return
    cpu = _rng_digest(torch.get_rng_state())
    cuda = "none"
    if torch.cuda.is_available():
        cuda = _rng_digest(torch.cuda.get_rng_state())
    print(
        f"[lingbot-rng] side={side} phase={phase} microbatch={microbatch} "
        f"initial_seed={torch.initial_seed()} cpu={cpu} cuda={cuda}",
        flush=True,
    )


def _build_scheduler(cfg, shift_key: str, default_shift: float):
    scheduler = LingBotVAFlowMatchScheduler(
        num_train_timesteps=int(_cfg_get(cfg, "lingbot_va_num_train_timesteps", 1000)),
        shift=float(_cfg_get(cfg, shift_key, default_shift)),
        sigma_min=float(_cfg_get(cfg, "lingbot_va_sigma_min", 0.0)),
        extra_one_step=bool(_cfg_get(cfg, "lingbot_va_extra_one_step", True)),
    )
    scheduler.set_timesteps(scheduler.num_train_timesteps, training=True)
    return scheduler


@register_model("lingbot_va")
class LingBotVAEmbodiedModel(nn.Module):
    """Prepare LingBot training inputs and compute baseline-style losses."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_scheduler = _build_scheduler(cfg, "lingbot_va_snr_shift", 5.0)
        self.action_scheduler = _build_scheduler(
            cfg, "lingbot_va_action_snr_shift", 1.0
        )
        use_flex = bool(_cfg_get(cfg, "lingbot_va_use_flex_attention", False))
        self.model = WanTransformer3DModel(
            patch_size=tuple(_cfg_get(cfg, "latent_patch_size", (1, 2, 2))),
            num_attention_heads=int(_cfg_get(cfg, "num_attention_heads", 24)),
            attention_head_dim=int(_cfg_get(cfg, "hidden_size", 3072))
            // int(_cfg_get(cfg, "num_attention_heads", 24)),
            in_channels=int(_cfg_get(cfg, "latent_in_channels", 48)),
            out_channels=int(_cfg_get(cfg, "latent_out_channels", 48)),
            action_dim=int(_cfg_get(cfg, "action_dim", 30)),
            text_dim=int(_cfg_get(cfg, "text_dim", 4096)),
            freq_dim=int(_cfg_get(cfg, "freq_dim", 256)),
            ffn_dim=int(_cfg_get(cfg, "ffn_hidden_size", 14336)),
            num_layers=int(_cfg_get(cfg, "num_layers", 30)),
            cross_attn_norm=bool(_cfg_get(cfg, "cross_attn_norm", True)),
            eps=float(_cfg_get(cfg, "norm_epsilon", 1e-6)),
            rope_max_seq_len=int(_cfg_get(cfg, "rope_max_seq_len", 1024)),
            attn_mode="flex" if use_flex else "torch",
            recompute_granularity=_cfg_get(cfg, "recompute_granularity", None),
        )

    @classmethod
    def from_pretrained(cls, cfg):
        """Create a model instance and load pretrained weights when configured."""
        model = cls(cfg)
        path = _cfg_get(cfg, "lingbot_va_diffusers_checkpoint_path", None)
        if path:
            model.load_pretrained(path)
        return model

    def load_pretrained(self, path: str, device=None):
        """Load sharded pretrained weights and optionally move the module."""
        report = load_sharded_safetensors(self.model, path)
        if device is not None:
            self.to(device)
        return report

    def forward(self, batch):
        """Prepare inputs, run the backend model, and return training losses."""
        batch_dict = batch.as_dict() if hasattr(batch, "as_dict") else batch
        input_dict = self._prepare_input_dict(batch_dict)
        return self._loss(input_dict, self.model(input_dict))

    def _prepare_input_dict(self, batch: Dict[str, torch.Tensor]):
        # Keep model-side stochastic inputs reproducible across launchers by
        # using the same per-microbatch RNG protocol as the baseline patch.
        microbatch = getattr(self, "_rng_diag_counter", 0)
        self._rng_diag_counter = microbatch + 1
        seed = _fixed_rng_seed(microbatch)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        _rng_diag("loongforge", "before_prepare", microbatch)
        latent_dict = self._add_noise(
            batch["latents"],
            self.latent_scheduler,
            False,
            noisy_cond_prob=float(
                _cfg_get(self.cfg, "lingbot_va_noisy_cond_prob", 0.5)
            ),
        )
        action_dict = self._add_noise(
            batch["actions"],
            self.action_scheduler,
            True,
            batch["actions_mask"],
            noisy_cond_prob=float(
                _cfg_get(self.cfg, "lingbot_va_action_noisy_cond_prob", 0.0)
            ),
        )
        latent_dict["text_emb"] = batch["text_emb"]
        action_dict["text_emb"] = batch["text_emb"]
        action_dict["actions_mask"] = batch["actions_mask"]
        chunk_high = int(_cfg_get(self.cfg, "lingbot_va_chunk_size", 4)) + 1
        window_high = int(_cfg_get(self.cfg, "lingbot_va_window_size", 64)) + 1
        chunk_size = torch.randint(1, chunk_high, (1,)).item()
        window_size = torch.randint(4, window_high, (1,)).item()
        result = {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": chunk_size,
            "window_size": window_size,
        }
        _rng_diag("loongforge", "after_prepare", microbatch)
        return result

    def _add_noise(
        self, latent, scheduler, action_mode, action_mask=None, noisy_cond_prob=0.0
    ):
        batch_size, _, frames, height, width = latent.shape
        timestep_ids = sample_timestep_id(frames, scheduler.num_train_timesteps)
        noise = torch.empty_like(latent).normal_()
        timesteps = scheduler.timesteps[timestep_ids].to(latent.device)
        noisy_latents = scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = scheduler.training_target(latent, noise, timesteps)
        patch = (
            (1, 1, 1)
            if action_mode
            else tuple(_cfg_get(self.cfg, "latent_patch_size", (1, 2, 2)))
        )
        grid_id = (
            get_mesh_id(
                frames // patch[0],
                height // patch[1],
                width // patch[2],
                token_type=1 if action_mode else 0,
                action=action_mode,
            )
            .to(latent.device)[None]
            .repeat(batch_size, 1, 1)
        )
        if torch.rand(1).item() < noisy_cond_prob:
            cond_ids = sample_timestep_id(
                frames, scheduler.num_train_timesteps, 0.5, 1.0
            )
            cond_timesteps = scheduler.timesteps[cond_ids].to(latent.device)
            latent = scheduler.add_noise(
                latent, torch.empty_like(latent).normal_(), cond_timesteps, t_dim=2
            )
        else:
            cond_timesteps = torch.zeros_like(timesteps)
        if action_mask is not None:
            mask = action_mask.to(latent.dtype)
            noisy_latents = noisy_latents * mask
            targets = targets * mask
            latent = latent * mask
        return {
            "timesteps": timesteps[None].repeat(batch_size, 1),
            "noisy_latents": noisy_latents,
            "targets": targets,
            "latent": latent,
            "cond_timesteps": cond_timesteps[None].repeat(batch_size, 1),
            "grid_id": grid_id,
        }

    def _loss(self, input_dict, prediction):
        latent_pred, action_pred = prediction
        action_target = input_dict["action_dict"]["targets"]
        action_pred = rearrange(
            action_pred, "b (f t) c -> b c f t 1", f=action_target.shape[-3]
        )
        latent_target = input_dict["latent_dict"]["targets"]
        latent_weights = self.latent_scheduler.training_weight(
            input_dict["latent_dict"]["timesteps"].flatten()
        ).reshape(input_dict["latent_dict"]["timesteps"].shape)
        action_weights = self.action_scheduler.training_weight(
            input_dict["action_dict"]["timesteps"].flatten()
        ).reshape(input_dict["action_dict"]["timesteps"].shape)
        latent_loss = F.mse_loss(
            latent_pred.float(), latent_target.float().detach(), reduction="none"
        )
        latent_loss = latent_loss * latent_weights[:, None, :, None, None]
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        latent_count = torch.ones_like(latent_loss).sum(dim=1)
        latent_loss = (latent_loss.sum(dim=1) / (latent_count + 1e-6)).mean()
        action_mask = input_dict["action_dict"]["actions_mask"].float()
        action_loss = F.mse_loss(
            action_pred.float(), action_target.float().detach(), reduction="none"
        )
        action_loss = action_loss * action_weights[:, None, :, None, None] * action_mask
        action_loss = action_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_mask = action_mask.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_loss = (action_loss.sum(dim=1) / (action_mask.sum(dim=1) + 1e-6)).mean()
        total_loss = (
            float(_cfg_get(self.cfg, "lingbot_va_video_loss_weight", 1.0)) * latent_loss
            + float(_cfg_get(self.cfg, "lingbot_va_action_loss_weight", 1.0))
            * action_loss
        )
        return total_loss, {
            "total loss": total_loss.detach(),
            "video loss": latent_loss.detach(),
            "action loss": action_loss.detach(),
        }
