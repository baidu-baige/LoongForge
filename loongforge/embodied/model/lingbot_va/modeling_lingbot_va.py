# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LingBot-VA (``wan_va/train.py``) under the Apache-2.0 License.
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

"""Embodied training adapter for the native PyTorch LingBot-VA backend."""

from collections import OrderedDict
import hashlib
import os
from typing import Any, Dict

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
from .modules.wan_model import WanTransformer3DModel


def _reference_rotary(x: torch.Tensor, frequencies: torch.Tensor):
    complex_x = torch.view_as_complex(x.double().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(complex_x * frequencies).flatten(3).to(x.dtype)


def _cfg_get(cfg, key, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _noise_and_target(sample, noise, sigma):
    return (1 - sigma) * sample + sigma * noise, noise - sample


def _apply_action_mask(noisy_latents, targets, latent, mask):
    return noisy_latents * mask, targets * mask, latent * mask


def _baseline_loss_reduction(
    latent_pred,
    latent_target,
    latent_weights,
    action_pred,
    action_target,
    action_weights,
    action_mask,
):
    latent_loss = F.mse_loss(
        latent_pred.float(), latent_target.float().detach(), reduction="none"
    )
    latent_loss = latent_loss * latent_weights[:, None, :, None, None]
    latent_loss = latent_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
    latent_count = torch.ones_like(latent_loss).sum(dim=1)
    latent_loss = (latent_loss.sum(dim=1) / (latent_count + 1e-6)).mean()

    action_mask = action_mask.float()
    action_loss = F.mse_loss(
        action_pred.float(), action_target.float().detach(), reduction="none"
    )
    action_loss = action_loss * action_weights[:, None, :, None, None] * action_mask
    action_loss = action_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
    action_mask = action_mask.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
    action_loss = (action_loss.sum(dim=1) / (action_mask.sum(dim=1) + 1e-6)).mean()
    return latent_loss, action_loss



_COMPILED_BASELINE_LOSS_REDUCTION = (
    torch.compile(_baseline_loss_reduction, dynamic=True)
    if getattr(torch, "compile", None) is not None
    else _baseline_loss_reduction
)

_COMPILED_NOISE_AND_TARGET = (
    torch.compile(_noise_and_target, dynamic=True)
    if getattr(torch, "compile", None) is not None
    else _noise_and_target
)
_COMPILED_ACTION_MASK = (
    torch.compile(_apply_action_mask, dynamic=True)
    if getattr(torch, "compile", None) is not None
    else _apply_action_mask
)


def _fixed_rng_seed(microbatch: int) -> int:
    return 42 + microbatch


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
        self._grid_id_cache = OrderedDict()
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

    @staticmethod
    def default_fp8_targets() -> Dict[str, Any]:
        """Convert only FFN projections inside Wan transformer blocks.

        Attention projections are intentionally kept in BF16. Quantization
        noise in Q/K/V and attention output projections is amplified by the
        softmax and then fed through the residual stream, which can produce a
        large loss drift over the full training run. FFN GEMMs are the largest
        remaining safe target and still provide most of the FP8 throughput
        benefit.
        """
        return {
            "module_patterns": ["model.blocks.*.ffn"],
            "skip_modules": [],
        }

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
        microbatch = getattr(self, "_microbatch_index", 0)
        self._microbatch_index = microbatch + 1
        seed = _fixed_rng_seed(microbatch)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Drive every random draw inside _prepare_input_dict/_add_noise through
        # explicit per-microbatch Generators so the sequence is byte-identical
        # to the baseline replay regardless of upstream global RNG consumption.
        cpu_gen = torch.Generator(device="cpu").manual_seed(seed)
        latent_tensor = batch["latents"]
        cuda_gen = None
        if latent_tensor.is_cuda:
            cuda_gen = torch.Generator(device=latent_tensor.device).manual_seed(seed)
        latent_dict = self._add_noise(
            batch["latents"],
            self.latent_scheduler,
            False,
            noisy_cond_prob=float(
                _cfg_get(self.cfg, "lingbot_va_noisy_cond_prob", 0.5)
            ),
            cpu_gen=cpu_gen,
            cuda_gen=cuda_gen,
        )
        action_dict = self._add_noise(
            batch["actions"],
            self.action_scheduler,
            True,
            batch["actions_mask"],
            noisy_cond_prob=float(
                _cfg_get(self.cfg, "lingbot_va_action_noisy_cond_prob", 0.0)
            ),
            cpu_gen=cpu_gen,
            cuda_gen=cuda_gen,
        )
        latent_dict["text_emb"] = batch["text_emb"]
        action_dict["text_emb"] = batch["text_emb"]
        action_dict["actions_mask"] = batch["actions_mask"]
        chunk_high = int(_cfg_get(self.cfg, "lingbot_va_chunk_size", 4)) + 1
        window_high = int(_cfg_get(self.cfg, "lingbot_va_window_size", 64)) + 1
        chunk_size = torch.randint(1, chunk_high, (1,), generator=cpu_gen).item()
        window_size = torch.randint(4, window_high, (1,), generator=cpu_gen).item()
        result = {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": chunk_size,
            "window_size": window_size,
        }
        return result

    def _add_noise(
        self,
        latent,
        scheduler,
        action_mode,
        action_mask=None,
        noisy_cond_prob=0.0,
        cpu_gen=None,
        cuda_gen=None,
    ):
        batch_size, _, frames, height, width = latent.shape
        timestep_ids = sample_timestep_id(
            frames, scheduler.num_train_timesteps, generator=cpu_gen
        )
        noise = torch.empty_like(latent).normal_(generator=cuda_gen)
        if latent.is_cuda:
            timestep_ids = timestep_ids.pin_memory()
        device_timestep_ids = timestep_ids.to(latent.device, non_blocking=True)
        timesteps = scheduler.timesteps_from_ids(device_timestep_ids)
        sigma = scheduler.sigma_from_ids(latent, device_timestep_ids, t_dim=2)
        noisy_latents, targets = _COMPILED_NOISE_AND_TARGET(latent, noise, sigma)
        patch = (
            (1, 1, 1)
            if action_mode
            else tuple(_cfg_get(self.cfg, "latent_patch_size", (1, 2, 2)))
        )
        grid_key = (
            frames // patch[0],
            height // patch[1],
            width // patch[2],
            1 if action_mode else 0,
            action_mode,
            str(latent.device),
        )
        grid_id = self._grid_id_cache.get(grid_key)
        if grid_id is None:
            grid_id = get_mesh_id(
                grid_key[0],
                grid_key[1],
                grid_key[2],
                token_type=grid_key[3],
                action=grid_key[4],
            ).to(latent.device)[None]
            if len(self._grid_id_cache) >= 64:
                self._grid_id_cache.popitem(last=False)
            self._grid_id_cache[grid_key] = grid_id
        else:
            self._grid_id_cache.move_to_end(grid_key)
        if batch_size > 1:
            grid_id = grid_id.repeat(batch_size, 1, 1)
        if torch.rand(1, generator=cpu_gen).item() < noisy_cond_prob:
            cond_ids = sample_timestep_id(
                frames,
                scheduler.num_train_timesteps,
                0.5,
                1.0,
                generator=cpu_gen,
            )
            if latent.is_cuda:
                cond_ids = cond_ids.pin_memory()
            cond_noise = torch.empty_like(latent).normal_(generator=cuda_gen)
            device_cond_ids = cond_ids.to(latent.device, non_blocking=True)
            cond_timesteps = scheduler.timesteps_from_ids(device_cond_ids)
            latent = scheduler.add_noise_from_ids(
                latent, cond_noise, device_cond_ids, t_dim=2
            )
        else:
            cond_timesteps = torch.zeros_like(timesteps)
        if action_mask is not None:
            mask = action_mask.to(latent.dtype)
            noisy_latents, targets, latent = _COMPILED_ACTION_MASK(
                noisy_latents, targets, latent, mask
            )
        return {
            "timesteps": timesteps[None].repeat(batch_size, 1),
            "timestep_ids": device_timestep_ids[None].repeat(batch_size, 1),
            "noisy_latents": noisy_latents,
            "targets": targets,
            "latent": latent,
            "cond_timesteps": cond_timesteps[None].repeat(batch_size, 1),
            "grid_id": grid_id,
            # The grid ids are a pure function of this key, so the rotary
            # frequency cache downstream can key on it instead of on the tensor
            # shape, which does not distinguish grid layouts of equal length.
            "grid_key": grid_key + (batch_size,),
        }

    def _loss(self, input_dict, prediction):
        latent_pred, action_pred = prediction
        action_target = input_dict["action_dict"]["targets"]
        action_pred = rearrange(
            action_pred, "b (f t) c -> b c f t 1", f=action_target.shape[-3]
        )
        latent_target = input_dict["latent_dict"]["targets"]
        latent_weights = self.latent_scheduler.training_weight_from_ids(
            input_dict["latent_dict"]["timestep_ids"].flatten()
        ).reshape(input_dict["latent_dict"]["timesteps"].shape)
        action_weights = self.action_scheduler.training_weight_from_ids(
            input_dict["action_dict"]["timestep_ids"].flatten()
        ).reshape(input_dict["action_dict"]["timesteps"].shape)
        loss_reduction = _COMPILED_BASELINE_LOSS_REDUCTION
        latent_loss, action_loss = loss_reduction(
            latent_pred,
            latent_target,
            latent_weights,
            action_pred,
            action_target,
            action_weights,
            input_dict["action_dict"]["actions_mask"],
        )
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
