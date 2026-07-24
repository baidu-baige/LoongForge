# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LingBot-VA under the Apache-2.0 License.
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

"""Baseline-compatible native PyTorch Wan transformer for LingBot-VA training."""

import math
import importlib
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from typing import ClassVar, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from ..features import (
    FLEX_MASK_CACHE_MAX_SIZE,
    SELF_FLEX_BWD_CONFIG,
    SELF_FLEX_FWD_CONFIG,
    feature_enabled,
)
from .rope import apply_triton_rope

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        and_masks,
        create_block_mask,
        flex_attention,
        or_masks,
    )
except ImportError:
    BlockMask = None
    flex_attention = None


class WanTimeTextImageEmbedding(nn.Module):
    """Diffusers-compatible timestep and text embedding stack."""

    def __init__(
        self, dim: int, time_freq_dim: int, time_proj_dim: int, text_embed_dim: int
    ):
        super().__init__()
        self.timesteps_proj = Timesteps(
            time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(time_freq_dim, dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh"
        )

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype):
        """Project timestep values into time embedding and modulation tensors."""
        batch, length = timestep.shape
        projected = self.timesteps_proj(timestep.reshape(-1))
        projected = projected.to(self.time_embedder.linear_1.weight.dtype)
        temb = self.time_embedder(projected).to(dtype=dtype)
        modulation = self.time_proj(self.act_fn(temb))
        return temb.reshape(batch, length, -1), modulation.reshape(batch, length, -1)


class WanRotaryPosEmbed(nn.Module):
    """Three-axis complex rotary position embedding used by Wan."""

    def __init__(
        self,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        del max_seq_len
        self.patch_size = tuple(patch_size)
        f_dim = attention_head_dim - 2 * (attention_head_dim // 3)
        h_dim = attention_head_dim // 3
        w_dim = attention_head_dim // 3
        self.register_buffer(
            "f_freqs_base", self._frequency_base(f_dim, theta), persistent=False
        )
        self.register_buffer(
            "h_freqs_base", self._frequency_base(h_dim, theta), persistent=False
        )
        self.register_buffer(
            "w_freqs_base", self._frequency_base(w_dim, theta), persistent=False
        )

    @staticmethod
    def _frequency_base(dim: int, theta: float):
        return 1.0 / theta ** (
            torch.arange(0, dim, 2, dtype=torch.float64)[: dim // 2] / dim
        )

    def forward(self, grid_ids: torch.Tensor):
        """Build complex rotary frequencies for latent and action grid ids."""
        with torch.no_grad():
            frequencies = torch.cat(
                (
                    grid_ids[:, 0, :, None] * self.f_freqs_base,
                    grid_ids[:, 1, :, None] * self.h_freqs_base,
                    grid_ids[:, 2, :, None] * self.w_freqs_base,
                ),
                dim=-1,
            ).float()
            return torch.polar(torch.ones_like(frequencies), frequencies)


_SELF_FLEX_BLOCK64_PATCH_ACTIVE = 0
_SELF_FLEX_BLOCK64_PATCHED = False
_SELF_FLEX_BLOCK64_PATCH_ERROR = None


def _self_flex_block64_install_patch():
    global _SELF_FLEX_BLOCK64_PATCHED, _SELF_FLEX_BLOCK64_PATCH_ERROR
    if _SELF_FLEX_BLOCK64_PATCHED:
        return True
    if _SELF_FLEX_BLOCK64_PATCH_ERROR is not None:
        raise RuntimeError(
            f"Failed to install the required LingBot Self Flex block64 kernel config: "
            f"{_SELF_FLEX_BLOCK64_PATCH_ERROR}"
        )
    try:
        import torch._inductor.lowering  # noqa: F401

        module = importlib.import_module("torch._inductor.kernel.flex_attention")
        if not getattr(module, "_lingbot_native_self_flex_block64_patched", False):
            original_fwd = module._get_default_config_fwd
            original_bwd = module._get_default_config_bwd
            fwd_config = SELF_FLEX_FWD_CONFIG
            bwd_config = SELF_FLEX_BWD_CONFIG

            def patched_fwd(query):
                return (
                    fwd_config
                    if _SELF_FLEX_BLOCK64_PATCH_ACTIVE
                    else original_fwd(query)
                )

            def patched_bwd(query):
                return (
                    bwd_config
                    if _SELF_FLEX_BLOCK64_PATCH_ACTIVE
                    else original_bwd(query)
                )

            module._get_default_config_fwd = patched_fwd
            module._get_default_config_bwd = patched_bwd
            module._lingbot_native_self_flex_block64_patched = True
        _SELF_FLEX_BLOCK64_PATCHED = True
        return True
    except Exception as error:
        _SELF_FLEX_BLOCK64_PATCH_ERROR = repr(error)
        raise RuntimeError(
            "Failed to install the required LingBot Self Flex block64 kernel config"
        ) from error


@contextmanager
def _self_flex_block64_patch_scope(active):
    global _SELF_FLEX_BLOCK64_PATCH_ACTIVE
    if not active:
        yield
        return
    _SELF_FLEX_BLOCK64_PATCH_ACTIVE += 1
    try:
        yield
    finally:
        _SELF_FLEX_BLOCK64_PATCH_ACTIVE -= 1


def _compiled_self_flex_block64_attention(query, key, value, block_mask):
    return flex_attention(query, key, value, block_mask=block_mask)


_SELF_FLEX_BLOCK64_COMPILED_SELF_FLEX = (
    torch.compile(_compiled_self_flex_block64_attention, dynamic=True)
    if flex_attention is not None
    else None
)


class FlexAttnFunc(nn.Module):
    """Flex attention with the original LingBot chunk/window mask semantics."""

    attention_mask: ClassVar[Optional["BlockMask"]] = None
    self_mask_cache: ClassVar[dict] = {}

    def __init__(self, is_cross: bool = False):
        super().__init__()
        if flex_attention is None:
            raise RuntimeError(
                "flex attention requires torch.nn.attention.flex_attention"
            )
        self.is_cross = is_cross

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """Apply cross attention or cached self flex attention to input states."""
        if self.is_cross:
            return F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            ).transpose(1, 2)
        mask = self.attention_mask
        if mask is None:
            raise RuntimeError("flex attention mask was not initialized")
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        _self_flex_block64_install_patch()
        with _self_flex_block64_patch_scope(True):
            output = _SELF_FLEX_BLOCK64_COMPILED_SELF_FLEX(q, k, v, mask)
        return output.transpose(1, 2)

    @classmethod
    @torch.no_grad()
    def init_mask(
        cls,
        latent_shape,
        action_shape,
        padded_length: int,
        chunk_size: int,
        window_size: int,
        patch_size,
        device,
    ) -> None:
        """Initialize or reuse the block mask for LingBot self attention."""
        batch, _, latent_frames, latent_height, latent_width = latent_shape
        _, _, action_frames, action_height, action_width = action_shape
        _self_flex_block64_install_patch()
        block_size = 64
        cache_key = (
            tuple(latent_shape),
            tuple(action_shape),
            int(padded_length),
            int(chunk_size),
            int(window_size),
            tuple(patch_size),
            str(device),
            block_size,
        )
        cached_self_mask = cls.self_mask_cache.get(cache_key)
        latent_tokens = (
            (latent_frames // patch_size[0])
            * (latent_height // patch_size[1])
            * (latent_width // patch_size[2])
        )
        action_tokens = action_frames * action_height * action_width
        sequence_ids = torch.cat(
            [torch.arange(batch).repeat_interleave(latent_tokens)] * 2
            + [torch.arange(batch).repeat_interleave(action_tokens)] * 2
        )
        latent_frame_ids = (
            torch.arange(latent_frames)
            .view(1, -1, 1, 1)
            .expand(
                batch, -1, latent_height // patch_size[1], latent_width // patch_size[2]
            )
            .flatten()
        )
        action_frame_ids = (
            torch.arange(action_frames)
            .view(1, -1, 1, 1)
            .expand(batch, -1, action_height, action_width)
            .flatten()
        )
        frame_ids = torch.cat(
            [latent_frame_ids.div(chunk_size, rounding_mode="floor") * 2] * 2
            + [action_frame_ids.div(chunk_size, rounding_mode="floor") * 2 + 1] * 2
        )
        noise_ids = torch.cat(
            [
                torch.zeros_like(latent_frame_ids),
                torch.ones_like(latent_frame_ids),
                torch.zeros_like(action_frame_ids),
                torch.ones_like(action_frame_ids),
            ]
        )
        sequence_ids = F.pad(sequence_ids, (0, padded_length), value=-1).to(device)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1).to(device)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1).to(device)

        def same_sequence(b, h, q_idx, kv_idx):
            del b, h
            return (sequence_ids[q_idx] == sequence_ids[kv_idx]) & (
                sequence_ids[q_idx] >= 0
            )

        def clean_causal(b, h, q_idx, kv_idx):
            del b, h
            return (
                (noise_ids[q_idx] == 1)
                & (noise_ids[kv_idx] == 1)
                & (frame_ids[kv_idx] <= frame_ids[q_idx])
            )

        def noise_to_clean(b, h, q_idx, kv_idx):
            del b, h
            return (
                (noise_ids[q_idx] == 0)
                & (noise_ids[kv_idx] == 1)
                & (frame_ids[kv_idx] < frame_ids[q_idx])
            )

        def noise_self(b, h, q_idx, kv_idx):
            del b, h
            return (
                (noise_ids[q_idx] == 0)
                & (noise_ids[kv_idx] == 0)
                & (frame_ids[kv_idx] == frame_ids[q_idx])
            )

        def in_window(b, h, q_idx, kv_idx, size):
            del b, h
            return (frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= size

        mask = and_masks(
            same_sequence,
            or_masks(clean_causal, noise_to_clean, noise_self),
            partial(in_window, size=window_size),
        )
        total_length = sequence_ids.numel()
        if cached_self_mask is None:
            cls.attention_mask = create_block_mask(
                mask,
                1,
                1,
                total_length,
                total_length,
                device=device,
                BLOCK_SIZE=block_size,
            )
            max_size = FLEX_MASK_CACHE_MAX_SIZE
            if len(cls.self_mask_cache) >= max_size:
                cls.self_mask_cache.pop(next(iter(cls.self_mask_cache)))
            cls.self_mask_cache[cache_key] = cls.attention_mask
        else:
            cls.attention_mask = cached_self_mask


def _build_qk_norm(hidden_size, eps):
    import transformer_engine.pytorch as te

    return te.RMSNorm(hidden_size, eps=eps)


class WanAttention(nn.Module):
    """Wan attention using native SDPA or flex attention."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        eps: float,
        dropout: float = 0.0,
        cross_attention_dim_head: Optional[int] = None,
        attn_mode: str = "torch",
    ):
        super().__init__()
        if attn_mode not in ("torch", "flex"):
            raise ValueError(f"Unsupported attention mode: {attn_mode}")
        self.inner_dim = heads * dim_head
        self.heads = heads
        self.is_cross = cross_attention_dim_head is not None
        kv_inner_dim = (
            self.inner_dim
            if cross_attention_dim_head is None
            else cross_attention_dim_head * heads
        )
        self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, kv_inner_dim, bias=True)
        self.to_v = nn.Linear(dim, kv_inner_dim, bias=True)
        self.to_out = nn.ModuleList(
            (nn.Linear(self.inner_dim, dim, bias=True), nn.Dropout(dropout))
        )
        self.norm_q = _build_qk_norm(self.inner_dim, eps)
        self.norm_k = _build_qk_norm(kv_inner_dim, eps)
        self.attn_op = FlexAttnFunc(self.is_cross) if attn_mode == "flex" else None

    @staticmethod
    def _apply_rotary(x: torch.Tensor, frequencies: torch.Tensor):
        return apply_triton_rope(x, frequencies)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rotary_emb=None
    ):
        """Project inputs, apply optional RoPE, and compute attention output."""
        query = self.norm_q(self.to_q(q)).unflatten(2, (self.heads, -1))
        key = self.norm_k(self.to_k(k)).unflatten(2, (self.heads, -1))
        value = self.to_v(v).unflatten(2, (self.heads, -1))
        if rotary_emb is not None:
            query = self._apply_rotary(query, rotary_emb)
            key = self._apply_rotary(key, rotary_emb)
        if self.attn_op is not None:
            hidden_states = self.attn_op(query, key, value)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            ).transpose(1, 2)
        return self.to_out[1](self.to_out[0](hidden_states.flatten(2)))


def _layerwise_norm_mod_no_affine(x, scale, shift, eps):
    norm = F.layer_norm(x.float(), (x.shape[-1],), None, None, eps)
    return (norm * (1.0 + scale) + shift).to(x.dtype)


def _layerwise_norm_affine_to_dtype(x, weight, bias, eps):
    norm = F.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps)
    return norm.to(x.dtype)


def _layerwise_residual_gate_to_dtype(residual, update, gate):
    return (residual.float() + update.float() * gate).to(residual.dtype)


def _layerwise_compile(function):
    compiler = getattr(torch, "compile", None)
    return compiler(function, dynamic=True) if compiler is not None else function


_LAYERWISE_NORM_MOD_NO_AFFINE = _layerwise_compile(_layerwise_norm_mod_no_affine)
_LAYERWISE_NORM_AFFINE_TO_DTYPE = _layerwise_compile(_layerwise_norm_affine_to_dtype)
_LAYERWISE_RESIDUAL_GATE_TO_DTYPE = _layerwise_compile(
    _layerwise_residual_gate_to_dtype
)


def _build_ffn(dim: int, ffn_dim: int):
    return FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")


class WanTransformerBlock(nn.Module):
    """Diffusers-compatible Wan transformer block and FSDP wrap boundary."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool,
        eps: float,
        attn_mode: str = "torch",
    ):
        super().__init__()
        head_dim = dim // num_heads
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim,
            num_heads,
            head_dim,
            eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )
        self.attn2 = WanAttention(
            dim,
            num_heads,
            head_dim,
            eps,
            cross_attention_dim_head=head_dim,
            attn_mode=attn_mode,
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.ffn = _build_ffn(dim, ffn_dim)
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
        """Run one Wan transformer block over hidden and encoder states."""
        modulation = self.scale_shift_table[None] + temb.float()
        shift, scale, gate, ff_shift, ff_scale, ff_gate = rearrange(
            modulation, "b l n c -> b n l c"
        ).chunk(6, dim=1)
        layerwise_enabled = feature_enabled("LINGBOT_LAYERWISE_COMPILE")
        if layerwise_enabled:
            normed = _LAYERWISE_NORM_MOD_NO_AFFINE(
                hidden_states, scale.squeeze(1), shift.squeeze(1), self.norm1.eps
            )
        else:
            normed = self.norm1(hidden_states.float())
            normed = (normed * (1 + scale.squeeze(1)) + shift.squeeze(1)).to(
                hidden_states.dtype
            )
        self_update = self.attn1(normed, normed, normed, rotary_emb)
        if layerwise_enabled:
            hidden_states = _LAYERWISE_RESIDUAL_GATE_TO_DTYPE(
                hidden_states, self_update, gate.squeeze(1)
            )
        else:
            hidden_states = (hidden_states.float() + self_update * gate.squeeze(1)).to(
                hidden_states.dtype
            )
        if layerwise_enabled and isinstance(self.norm2, FP32LayerNorm):
            cross_input = _LAYERWISE_NORM_AFFINE_TO_DTYPE(
                hidden_states, self.norm2.weight, self.norm2.bias, self.norm2.eps
            )
        else:
            cross_input = self.norm2(hidden_states.float()).to(hidden_states.dtype)
        cross_update = self.attn2(
            cross_input, encoder_hidden_states, encoder_hidden_states
        )
        hidden_states = hidden_states + cross_update
        if layerwise_enabled:
            normed = _LAYERWISE_NORM_MOD_NO_AFFINE(
                hidden_states, ff_scale.squeeze(1), ff_shift.squeeze(1), self.norm3.eps
            )
        else:
            normed = self.norm3(hidden_states.float())
            normed = (normed * (1 + ff_scale.squeeze(1)) + ff_shift.squeeze(1)).to(
                hidden_states.dtype
            )
        ffn_update = self.ffn(normed)
        if layerwise_enabled:
            output = _LAYERWISE_RESIDUAL_GATE_TO_DTYPE(
                hidden_states, ffn_update, ff_gate.squeeze(1)
            )
        else:
            output = (
                hidden_states.float() + ffn_update.float() * ff_gate.squeeze(1)
            ).to(hidden_states.dtype)
        return output


class WanTransformer3DModel(ModelMixin, ConfigMixin):
    """Native PyTorch LingBot Wan model implementing the training path only."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["WanTransformerBlock"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-6,
        rope_max_seq_len=1024,
        attn_mode="torch",
        recompute_granularity=None,
    ):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.recompute_granularity = recompute_granularity
        inner_dim = num_attention_heads * attention_head_dim
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(
            in_channels * math.prod(self.patch_size), inner_dim
        )
        self.action_embedder = nn.Linear(action_dim, inner_dim)
        self.condition_embedder = WanTimeTextImageEmbedding(
            inner_dim, freq_dim, inner_dim * 6, text_dim
        )
        self.condition_embedder_action = deepcopy(self.condition_embedder)
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    cross_attn_norm,
                    eps,
                    attn_mode,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(self.patch_size))
        self.action_proj_out = nn.Linear(inner_dim, action_dim)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

    def _input_embed(self, values: torch.Tensor, input_type: str):
        if input_type == "latent":
            values = rearrange(
                values,
                "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
            return self.patch_embedding_mlp(values)
        if input_type == "action":
            return self.action_embedder(rearrange(values, "b c f h w -> b (f h w) c"))
        if input_type == "text":
            return self.condition_embedder.text_embedder(values)
        raise ValueError(f"Unsupported input type: {input_type}")

    def _time_embed(self, timesteps, height, width, dtype, action_mode=False):
        patch_h, patch_w = (1, 1) if action_mode else self.patch_size[1:]
        timesteps = torch.repeat_interleave(
            timesteps, (height // patch_h) * (width // patch_w), dim=1
        )
        embedder = (
            self.condition_embedder_action if action_mode else self.condition_embedder
        )
        temb, modulation = embedder(timesteps, dtype)
        return temb, modulation.unflatten(2, (6, -1))

    def forward(self, input_dict):
        """Run the LingBot Wan model and return latent and action predictions."""
        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        noisy_latent = latent_dict["noisy_latents"]
        noisy_action = action_dict["noisy_latents"]
        batch_size, _, frames, height, width = noisy_latent.shape
        dtype = self.patch_embedding_mlp.weight.dtype

        latent_hidden = self._input_embed(noisy_latent.to(dtype), "latent").flatten(
            0, 1
        )[None]
        latent_condition = self._input_embed(
            latent_dict["latent"].to(dtype), "latent"
        ).flatten(0, 1)[None]
        action_hidden = self._input_embed(noisy_action.to(dtype), "action").flatten(
            0, 1
        )[None]
        action_condition = self._input_embed(
            action_dict["latent"].to(dtype), "action"
        ).flatten(0, 1)[None]
        text_hidden = self._input_embed(
            latent_dict["text_emb"].to(dtype), "text"
        ).flatten(0, 1)[None]
        hidden_states = torch.cat(
            (latent_hidden, latent_condition, action_hidden, action_condition), dim=1
        )

        latent_grid = latent_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        action_grid = action_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        rotary_emb = self.rope(
            torch.cat((latent_grid, latent_grid, action_grid, action_grid), dim=2)
        )[:, :, None]

        latent_steps = torch.cat(
            (
                latent_dict["timesteps"].flatten(0, 1),
                latent_dict["cond_timesteps"].flatten(0, 1),
            )
        )[None]
        action_steps = torch.cat(
            (
                action_dict["timesteps"].flatten(0, 1),
                action_dict["cond_timesteps"].flatten(0, 1),
            )
        )[None]
        latent_temb, latent_modulation = self._time_embed(
            latent_steps, height, width, hidden_states.dtype
        )
        action_temb, action_modulation = self._time_embed(
            action_steps,
            noisy_action.shape[-2],
            noisy_action.shape[-1],
            hidden_states.dtype,
            action_mode=True,
        )
        temb = torch.cat((latent_temb, action_temb), dim=1)
        modulation = torch.cat((latent_modulation, action_modulation), dim=1)

        total_length = hidden_states.shape[1]
        padded_length = (-total_length) % 128
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        modulation = F.pad(modulation, (0, 0, 0, 0, 0, padded_length))
        if self.config.attn_mode == "flex":
            FlexAttnFunc.init_mask(
                noisy_latent.shape,
                noisy_action.shape,
                padded_length,
                input_dict["chunk_size"],
                input_dict["window_size"],
                self.patch_size,
                hidden_states.device,
            )

        for block in self.blocks:
            if self.training and self.recompute_granularity == "full":
                hidden_states = checkpoint(
                    block,
                    hidden_states,
                    text_hidden,
                    modulation,
                    rotary_emb,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states, text_hidden, modulation, rotary_emb
                )

        output_modulation = self.scale_shift_table[None] + temb[:, :, None]
        shift, scale = rearrange(output_modulation, "b l n c -> b n l c").chunk(
            2, dim=1
        )
        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale.squeeze(1))
            + shift.squeeze(1)
        ).to(hidden_states.dtype)
        split_sizes = [
            latent_hidden.shape[1],
            latent_condition.shape[1],
            action_hidden.shape[1],
            action_condition.shape[1],
            padded_length,
        ]
        latent_hidden, _, action_hidden, _, _ = torch.split(
            hidden_states, split_sizes, dim=1
        )
        latent_hidden = self.proj_out(latent_hidden)
        latent_hidden = rearrange(
            latent_hidden,
            "1 (b f h w) (p1 p2 p3 c) -> b c (f p1) (h p2) (w p3)",
            b=batch_size,
            f=frames // self.patch_size[0],
            h=height // self.patch_size[1],
            w=width // self.patch_size[2],
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            p3=self.patch_size[2],
        )
        action_hidden = rearrange(
            self.action_proj_out(action_hidden), "1 (b l) c -> b l c", b=batch_size
        )
        return latent_hidden, action_hidden
