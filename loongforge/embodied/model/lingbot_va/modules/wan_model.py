# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LingBot-VA under the Apache-2.0 License.
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

"""Baseline-compatible native PyTorch Wan transformer for LingBot-VA training."""

import math
import importlib
import logging
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
    SELF_FLEX_BWD_CONFIG,
    SELF_FLEX_OPTIMIZED_FWD_CONFIG,
)
from .rope import apply_triton_rope_pair

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
    create_block_mask = None
    flex_attention = None


_COMPILED_CREATE_BLOCK_MASK = None
_COMPILED_CREATE_BLOCK_MASK_ERROR = None
# Frozen recipe value for the self-attention BlockMask cache.
SELF_MASK_CACHE_SIZE = 256

logger = logging.getLogger(__name__)


def _report_block_mask_fallback(error):
    """Record the compile failure and say so once, then fall back eagerly."""
    global _COMPILED_CREATE_BLOCK_MASK_ERROR
    _COMPILED_CREATE_BLOCK_MASK_ERROR = repr(error)
    logger.warning(
        "Compiled create_block_mask unavailable, using the eager path: %s",
        _COMPILED_CREATE_BLOCK_MASK_ERROR,
    )


def _create_lingbot_block_mask(mask, total_length, device, block_size):
    """Build LingBot's BlockMask through an optional compiled fast path."""
    global _COMPILED_CREATE_BLOCK_MASK, _COMPILED_CREATE_BLOCK_MASK_ERROR
    if _COMPILED_CREATE_BLOCK_MASK is None and _COMPILED_CREATE_BLOCK_MASK_ERROR is None:
        try:
            _COMPILED_CREATE_BLOCK_MASK = torch.compile(
                create_block_mask,
                dynamic=True,
            )
        except Exception as error:
            _report_block_mask_fallback(error)
    if _COMPILED_CREATE_BLOCK_MASK is not None:
        try:
            return _COMPILED_CREATE_BLOCK_MASK(
                mask,
                1,
                1,
                total_length,
                total_length,
                device=device,
                BLOCK_SIZE=block_size,
            )
        except Exception as error:
            _report_block_mask_fallback(error)
            _COMPILED_CREATE_BLOCK_MASK = None
    return create_block_mask(
        mask,
        1,
        1,
        total_length,
        total_length,
        device=device,
        BLOCK_SIZE=block_size,
    )


def _run_compiled_self_flex_attention(query, key, value, block_mask):
    if _SELF_FLEX_DIVISIBLE_COMPILED_SELF_FLEX is None:
        raise RuntimeError("compiled FlexAttention is unavailable")
    with _self_flex_patch_scope():
        return _SELF_FLEX_DIVISIBLE_COMPILED_SELF_FLEX(
            query, key, value, block_mask
        )


def _candidate_compile(function):
    compiler = getattr(torch, "compile", None)
    if compiler is None:
        return function
    # The accepted recipe compiles with dynamic shapes and the default mode.
    return compiler(function, dynamic=True)


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
        self._timestep_frequency_cache = {}

    def _project_timesteps(self, timestep: torch.Tensor):
        half_dim = self.timesteps_proj.num_channels // 2
        cache_key = (str(timestep.device), half_dim)
        frequency = self._timestep_frequency_cache.get(cache_key)
        if frequency is None:
            exponent = -math.log(10000) * torch.arange(
                half_dim, dtype=torch.float32, device=timestep.device
            )
            frequency = torch.exp(
                exponent / (half_dim - self.timesteps_proj.downscale_freq_shift)
            )
            self._timestep_frequency_cache[cache_key] = frequency
        angles = timestep[:, None].float() * frequency[None, :]
        return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype):
        """Project timestep values into time embedding and modulation tensors."""
        batch, length = timestep.shape
        projected = self._project_timesteps(timestep.reshape(-1))
        projected = projected.to(self.time_embedder.linear_1.weight.dtype)
        temb, modulation = _COMPILED_TIME_EMBED(
            projected,
            self.time_embedder.linear_1.weight,
            self.time_embedder.linear_1.bias,
            self.time_embedder.linear_2.weight,
            self.time_embedder.linear_2.bias,
            self.time_proj.weight,
            self.time_proj.bias,
            dtype,
        )
        return temb.reshape(batch, length, -1), modulation.reshape(batch, length, -1)


def _time_embed_from_projected(
    projected, linear1_weight, linear1_bias, linear2_weight, linear2_bias,
    projection_weight, projection_bias, dtype
):
    temb = F.linear(projected, linear1_weight, linear1_bias)
    temb = F.silu(temb)
    temb = F.linear(temb, linear2_weight, linear2_bias).to(dtype=dtype)
    modulation = F.linear(F.silu(temb), projection_weight, projection_bias)
    return temb, modulation


_COMPILED_TIME_EMBED = _candidate_compile(_time_embed_from_projected)


def _text_embed_compiled(values, linear1_weight, linear1_bias, linear2_weight, linear2_bias):
    hidden_states = F.linear(values, linear1_weight, linear1_bias)
    hidden_states = F.gelu(hidden_states, approximate="tanh")
    return F.linear(hidden_states, linear2_weight, linear2_bias)


_COMPILED_TEXT_EMBED = _candidate_compile(_text_embed_compiled)


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
        self._frequency_cache = {}
        self._compiled_frequency_forward = None

    @staticmethod
    def _frequency_base(dim: int, theta: float):
        return 1.0 / theta ** (
            torch.arange(0, dim, 2, dtype=torch.float64)[: dim // 2] / dim
        )

    def _frequency_forward(self, grid_ids: torch.Tensor):
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

    def _cache_result(self, cache_key, result):
        if len(self._frequency_cache) >= 16:
            self._frequency_cache.pop(next(iter(self._frequency_cache)))
        self._frequency_cache[cache_key] = result
        return result

    def forward_pair(
        self,
        latent_grid: torch.Tensor,
        action_grid: torch.Tensor,
        grid_keys=None,
    ) -> torch.Tensor:
        """Build or reuse frequencies without repeating the grid concatenation.

        ``grid_keys`` identifies the *content* of the two grids (the arguments
        ``get_mesh_id`` was called with). Caching on tensor shape alone is not
        sound: the grids are flattened, so two different (frames, height, width,
        token_type) layouts of equal length share a shape while holding
        different ids, and the cache would then hand back the wrong rotary
        frequencies. Without a key the frequencies are rebuilt every call --
        content hashing is deliberately not an option here, it would need a
        device-to-host copy on the critical path.
        """
        cache_key = None if grid_keys is None else ("pair", grid_keys)
        if cache_key is not None:
            cached = self._frequency_cache.get(cache_key)
            if cached is not None:
                return cached
        grid_ids = torch.cat(
            (latent_grid, latent_grid, action_grid, action_grid), dim=2
        )
        result = self._frequency_forward(grid_ids)
        if cache_key is None:
            return result
        return self._cache_result(cache_key, result)

    def forward(self, grid_ids: torch.Tensor, grid_key=None):
        """Build complex rotary frequencies for latent and action grid ids.

        Same contract as ``forward_pair``: reuse only happens when the caller
        supplies a key that determines the grid contents.
        """
        cache_key = None if grid_key is None else ("single", grid_key)
        if cache_key is not None:
            cached = self._frequency_cache.get(cache_key)
            if cached is not None:
                return cached
        result = self._frequency_forward(grid_ids)
        if cache_key is None:
            return result
        return self._cache_result(cache_key, result)


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

        module = None
        for module_name in (
            "torch._inductor.kernel.flex_attention",
            "torch._inductor.kernel.flex.flex_attention",
        ):
            try:
                module = importlib.import_module(module_name)
                break
            except ModuleNotFoundError:
                continue
        if module is None:
            raise ModuleNotFoundError("no supported Inductor FlexAttention module")
        has_legacy_defaults = hasattr(module, "_get_default_config_fwd") and hasattr(
            module, "_get_default_config_bwd"
        )
        if has_legacy_defaults and not getattr(
            module, "_lingbot_native_self_flex_block64_patched", False
        ):
            original_fwd = module._get_default_config_fwd
            original_bwd = module._get_default_config_bwd
            fwd_config = SELF_FLEX_OPTIMIZED_FWD_CONFIG
            bwd_config = SELF_FLEX_BWD_CONFIG

            def patched_fwd(query):
                if _SELF_FLEX_BLOCK64_PATCH_ACTIVE:
                    return fwd_config
                return original_fwd(query)

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
def _self_flex_patch_scope():
    global _SELF_FLEX_BLOCK64_PATCH_ACTIVE
    _SELF_FLEX_BLOCK64_PATCH_ACTIVE += 1
    try:
        yield
    finally:
        _SELF_FLEX_BLOCK64_PATCH_ACTIVE -= 1


def _self_flex_kernel_options():
    fwd = SELF_FLEX_OPTIMIZED_FWD_CONFIG
    bwd = SELF_FLEX_BWD_CONFIG
    options = {
        "fwd_BLOCK_M": fwd[0],
        "fwd_BLOCK_N": fwd[1],
        "fwd_num_warps": fwd[2],
        "fwd_num_stages": fwd[3],
        "bwd_BLOCK_M1": bwd[0],
        "bwd_BLOCK_N1": bwd[1],
        "bwd_BLOCK_M2": bwd[0],
        "bwd_BLOCK_N2": bwd[1],
        "bwd_num_warps": bwd[2],
        "bwd_num_stages": bwd[3],
    }
    options["IS_DIVISIBLE"] = True
    return options


def _compiled_self_flex_divisible_attention(query, key, value, block_mask):
    return flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
        kernel_options=_self_flex_kernel_options(),
    )


_SELF_FLEX_DIVISIBLE_COMPILED_SELF_FLEX = (
    torch.compile(_compiled_self_flex_divisible_attention, dynamic=True)
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
        if q.shape[-2] % 128 or k.shape[-2] % 128 or v.shape[-2] % 128:
            raise RuntimeError(
                "self FlexAttention requires q/k/v sequence lengths divisible by 128"
            )
        with _self_flex_patch_scope():
            output = _SELF_FLEX_DIVISIBLE_COMPILED_SELF_FLEX(q, k, v, mask)
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
        if cached_self_mask is not None:
            # A cache hit must avoid rebuilding CPU metadata and repeating H2D copies.
            cls.attention_mask = cached_self_mask
            return
        latent_tokens = (
            (latent_frames // patch_size[0])
            * (latent_height // patch_size[1])
            * (latent_width // patch_size[2])
        )
        action_tokens = action_frames * action_height * action_width
        metadata_device = None
        sequence_ids = torch.cat(
            [
                torch.arange(batch, device=metadata_device).repeat_interleave(
                    latent_tokens
                )
            ]
            * 2
            + [
                torch.arange(batch, device=metadata_device).repeat_interleave(
                    action_tokens
                )
            ]
            * 2
        )
        latent_frame_ids = (
            torch.arange(latent_frames, device=metadata_device)
            .view(1, -1, 1, 1)
            .expand(
                batch, -1, latent_height // patch_size[1], latent_width // patch_size[2]
            )
            .flatten()
        )
        action_frame_ids = (
            torch.arange(action_frames, device=metadata_device)
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
        sequence_ids = F.pad(sequence_ids, (0, padded_length), value=-1)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)
        if metadata_device is None:
            sequence_ids = sequence_ids.to(device)
            frame_ids = frame_ids.to(device)
            noise_ids = noise_ids.to(device)

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
        cls.attention_mask = _create_lingbot_block_mask(
            mask,
            total_length,
            device,
            block_size,
        )
        if len(cls.self_mask_cache) >= SELF_MASK_CACHE_SIZE:
            cls.self_mask_cache.pop(next(iter(cls.self_mask_cache)))
        cls.self_mask_cache[cache_key] = cls.attention_mask


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

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rotary_emb=None
    ):
        """Project inputs, apply optional RoPE, and compute attention output."""
        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query).unflatten(2, (self.heads, -1))
        key = self.norm_k(key).unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:
            query, key = apply_triton_rope_pair(query, key, rotary_emb)
        if self.attn_op is not None:
            hidden_states = self.attn_op(query, key, value)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            ).transpose(1, 2)
        return self.to_out[1](self.to_out[0](hidden_states.flatten(2)))


def _layerwise_residual_gate_to_dtype(residual, update, gate):
    return (residual.float() + update.float() * gate).to(residual.dtype)


def _layerwise_modulation_prologue_bf16(
    hidden_states, scale_shift_table, temb, eps
):
    dtype = hidden_states.dtype
    modulation = scale_shift_table.to(dtype)[None] + temb.to(dtype)
    shift, scale, gate, ff_shift, ff_scale, ff_gate = modulation.permute(
        0, 2, 1, 3
    ).chunk(6, dim=1)
    normed = F.layer_norm(
        hidden_states, (hidden_states.shape[-1],), None, None, eps
    )
    normed = normed * (1.0 + scale.squeeze(1)) + shift.squeeze(1)
    return normed, gate, ff_shift, ff_scale, ff_gate


def _layerwise_self_residual_cross_norm_bf16(
    residual, update, gate, weight, bias, eps
):
    dtype = residual.dtype
    hidden_states = residual + update * gate.to(dtype)
    cross_input = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight.to(dtype),
        bias.to(dtype),
        eps,
    )
    return hidden_states, cross_input


def _layerwise_cross_residual_ff_norm_bf16(
    residual, update, scale, shift, eps
):
    dtype = residual.dtype
    hidden_states = residual + update
    normed = F.layer_norm(
        hidden_states, (hidden_states.shape[-1],), None, None, eps
    )
    normed = normed * (1.0 + scale.to(dtype)) + shift.to(dtype)
    return hidden_states, normed


def _layerwise_residual_gate_bf16(residual, update, gate):
    return residual + update * gate.to(residual.dtype)


def _layerwise_output_modulation_norm_bf16(
    hidden_states, scale_shift_table, temb, eps
):
    dtype = hidden_states.dtype
    output_modulation = scale_shift_table.to(dtype)[None] + temb[:, :, None].to(dtype)
    shift, scale = output_modulation.permute(0, 2, 1, 3).chunk(2, dim=1)
    normed = F.layer_norm(
        hidden_states, (hidden_states.shape[-1],), None, None, eps
    )
    return normed * (1.0 + scale.squeeze(1)) + shift.squeeze(1)


def _layerwise_compile(function):
    return _candidate_compile(function)


_LAYERWISE_RESIDUAL_GATE_TO_DTYPE = _layerwise_compile(
    _layerwise_residual_gate_to_dtype
)
_LAYERWISE_MODULATION_PROLOGUE_BF16 = _layerwise_compile(
    _layerwise_modulation_prologue_bf16
)
_LAYERWISE_SELF_RESIDUAL_CROSS_NORM_BF16 = _layerwise_compile(
    _layerwise_self_residual_cross_norm_bf16
)
_LAYERWISE_CROSS_RESIDUAL_FF_NORM_BF16 = _layerwise_compile(
    _layerwise_cross_residual_ff_norm_bf16
)
_LAYERWISE_RESIDUAL_GATE_BF16 = _layerwise_compile(
    _layerwise_residual_gate_bf16
)
_LAYERWISE_OUTPUT_MODULATION_NORM_BF16 = _layerwise_compile(
    _layerwise_output_modulation_norm_bf16
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
        return self._forward_impl(
            hidden_states, encoder_hidden_states, temb, rotary_emb
        )

    def _forward_impl(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
        # Accepted recipe: layerwise compile, compiled block boundaries, compiled
        # modulation prologue and BF16 block math are all part of the frozen path.
        normed, gate, ff_shift, ff_scale, ff_gate = _LAYERWISE_MODULATION_PROLOGUE_BF16(
            hidden_states,
            self.scale_shift_table,
            temb,
            self.norm1.eps,
        )
        self_update = self.attn1(normed, normed, normed, rotary_emb)
        if isinstance(self.norm2, FP32LayerNorm):
            hidden_states, cross_input = _LAYERWISE_SELF_RESIDUAL_CROSS_NORM_BF16(
                hidden_states,
                self_update,
                gate.squeeze(1),
                self.norm2.weight,
                self.norm2.bias,
                self.norm2.eps,
            )
        else:
            hidden_states = _LAYERWISE_RESIDUAL_GATE_TO_DTYPE(
                hidden_states, self_update, gate.squeeze(1)
            )
            cross_input = self.norm2(hidden_states.float()).to(hidden_states.dtype)
        cross_update = self.attn2(
            cross_input, encoder_hidden_states, encoder_hidden_states
        )
        hidden_states, normed = _LAYERWISE_CROSS_RESIDUAL_FF_NORM_BF16(
            hidden_states,
            cross_update,
            ff_scale.squeeze(1),
            ff_shift.squeeze(1),
            self.norm3.eps,
        )
        ffn_update = self.ffn(normed)
        return _LAYERWISE_RESIDUAL_GATE_BF16(
            hidden_states, ffn_update, ff_gate.squeeze(1)
        )


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
        self._padding_cache = {}
        self._padded_rope_cache = {}

    def _padding_zeros(self, reference: torch.Tensor, shape):
        cache_key = (tuple(shape), str(reference.device), str(reference.dtype))
        cached = self._padding_cache.get(cache_key)
        if cached is None:
            cached = reference.new_zeros(*shape)
            if len(self._padding_cache) >= 16:
                self._padding_cache.pop(next(iter(self._padding_cache)))
            self._padding_cache[cache_key] = cached
        return cached

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
            embedder = self.condition_embedder.text_embedder
            return _COMPILED_TEXT_EMBED(
                values,
                embedder.linear_1.weight,
                embedder.linear_1.bias,
                embedder.linear_2.weight,
                embedder.linear_2.bias,
            )
        raise ValueError(f"Unsupported input type: {input_type}")

    def _time_embed(self, timesteps, height, width, dtype, action_mode=False):
        patch_h, patch_w = (1, 1) if action_mode else self.patch_size[1:]
        spatial_repeats = (height // patch_h) * (width // patch_w)
        embedder = (
            self.condition_embedder_action if action_mode else self.condition_embedder
        )
        # Embedding before the spatial repeat keeps the embedder batch small.
        temb, modulation = embedder(timesteps, dtype)
        temb = torch.repeat_interleave(temb, spatial_repeats, dim=1)
        modulation = torch.repeat_interleave(modulation, spatial_repeats, dim=1)
        return temb, modulation.unflatten(2, (6, -1))

    def forward(self, input_dict):
        """Run the LingBot Wan model and return latent and action predictions."""
        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        noisy_latent = latent_dict["noisy_latents"]
        noisy_action = action_dict["noisy_latents"]
        batch_size, _, frames, height, width = noisy_latent.shape
        dtype = self.patch_embedding_mlp.weight.dtype

        latent_pair = self._input_embed(
            torch.cat(
                (noisy_latent.to(dtype), latent_dict["latent"].to(dtype)), dim=0
            ),
            "latent",
        )
        latent_hidden, latent_condition = latent_pair.chunk(2, dim=0)
        latent_hidden = latent_hidden.flatten(0, 1)[None]
        latent_condition = latent_condition.flatten(0, 1)[None]
        action_pair = self._input_embed(
            torch.cat(
                (noisy_action.to(dtype), action_dict["latent"].to(dtype)), dim=0
            ),
            "action",
        )
        action_hidden, action_condition = action_pair.chunk(2, dim=0)
        action_hidden = action_hidden.flatten(0, 1)[None]
        action_condition = action_condition.flatten(0, 1)[None]
        text_hidden = self._input_embed(
            latent_dict["text_emb"].to(dtype), "text"
        ).flatten(0, 1)[None]
        hidden_parts = (
            latent_hidden,
            latent_condition,
            action_hidden,
            action_condition,
        )
        total_length = sum(part.shape[1] for part in hidden_parts)
        padded_length = (-total_length) % 128
        if padded_length:
            hidden_states = torch.cat(
                hidden_parts
                + (
                    self._padding_zeros(
                        latent_hidden,
                        (
                            latent_hidden.shape[0],
                            padded_length,
                            latent_hidden.shape[2],
                        ),
                    ),
                ),
                dim=1,
            )
        else:
            hidden_states = torch.cat(hidden_parts, dim=1)

        latent_grid = latent_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        action_grid = action_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        grid_keys = None
        latent_key = latent_dict.get("grid_key")
        action_key = action_dict.get("grid_key")
        if latent_key is not None and action_key is not None:
            grid_keys = (latent_key, action_key)
        rotary_emb = self.rope.forward_pair(
            latent_grid, action_grid, grid_keys=grid_keys
        )[:, :, None]
        if padded_length:
            # The padded tensor carries the frequency *content*, so it may only
            # be reused under a content-determining key, exactly as in the
            # frequency cache. With no key the concatenation is redone.
            rope_key = (
                None
                if grid_keys is None
                else (
                    grid_keys,
                    tuple(rotary_emb.shape),
                    int(padded_length),
                    str(rotary_emb.device),
                    str(rotary_emb.dtype),
                )
            )
            padded_rotary = (
                None if rope_key is None else self._padded_rope_cache.get(rope_key)
            )
            if padded_rotary is None:
                padded_rotary = torch.cat(
                    (
                        rotary_emb,
                        self._padding_zeros(
                            rotary_emb,
                            (
                                rotary_emb.shape[0],
                                padded_length,
                                rotary_emb.shape[2],
                                rotary_emb.shape[3],
                            ),
                        ),
                    ),
                    dim=1,
                )
                if rope_key is not None:
                    if len(self._padded_rope_cache) >= 64:
                        self._padded_rope_cache.pop(
                            next(iter(self._padded_rope_cache))
                        )
                    self._padded_rope_cache[rope_key] = padded_rotary
            rotary_emb = padded_rotary

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
            latent_steps, height, width, latent_hidden.dtype
        )
        action_temb, action_modulation = self._time_embed(
            action_steps,
            noisy_action.shape[-2],
            noisy_action.shape[-1],
            latent_hidden.dtype,
            action_mode=True,
        )
        if padded_length:
            temb = torch.cat(
                (
                    latent_temb,
                    action_temb,
                    self._padding_zeros(
                        latent_temb,
                        (
                            latent_temb.shape[0],
                            padded_length,
                            latent_temb.shape[2],
                        ),
                    ),
                ),
                dim=1,
            )
            modulation = torch.cat(
                (
                    latent_modulation,
                    action_modulation,
                    self._padding_zeros(
                        latent_modulation,
                        (
                            latent_modulation.shape[0],
                            padded_length,
                            latent_modulation.shape[2],
                            latent_modulation.shape[3],
                        ),
                    ),
                ),
                dim=1,
            )
        else:
            temb = torch.cat((latent_temb, action_temb), dim=1)
            modulation = torch.cat((latent_modulation, action_modulation), dim=1)
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

        hidden_states = _LAYERWISE_OUTPUT_MODULATION_NORM_BF16(
            hidden_states,
            self.scale_shift_table,
            temb,
            self.norm_out.eps,
        )
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
