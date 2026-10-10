# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash vision encoder and projector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from megatron.core.transformer.torch_norm import WrappedTorchNorm

try:
    from .glm5_next_config import Glm5NextVisionConfig
except ImportError:
    from glm5_next_config import Glm5NextVisionConfig


@dataclass
class Glm5NextVisionOutput:
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor | tuple[torch.Tensor, ...]


def _native_rmsnorm(config: Glm5NextVisionConfig, hidden_size: int):
    return WrappedTorchNorm(config, hidden_size=hidden_size, eps=config.rms_norm_eps)


def get_vision_position_ids(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    position_ids = []
    for temporal, height, width in grid_thw.tolist():
        height_ids, width_ids = torch.meshgrid(
            torch.arange(height, device=grid_thw.device),
            torch.arange(width, device=grid_thw.device),
            indexing="ij",
        )
        block_shape = (
            height // spatial_merge_size,
            spatial_merge_size,
            width // spatial_merge_size,
            spatial_merge_size,
        )
        height_ids = height_ids.reshape(block_shape).transpose(1, 2).flatten()
        width_ids = width_ids.reshape(block_shape).transpose(1, 2).flatten()
        position_ids.append(torch.stack([height_ids, width_ids], dim=-1).repeat(temporal, 1))
    return torch.cat(position_ids, dim=0)


def get_vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0), value=0)


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_dtype = query.dtype
    key_dtype = key.dtype
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    query = query.float() * cos + rotate_half(query.float()) * sin
    key = key.float() * cos + rotate_half(key.float()) * sin
    return query.to(query_dtype), key.to(key_dtype)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class VisionPatchEmbed(nn.Module):
    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            self.in_channels,
            self.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(pixel_values.to(self.proj.weight.dtype)).view(-1, self.hidden_size)


class VisionAttention(nn.Module):
    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.attention_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.q_norm = _native_rmsnorm(config, self.head_dim)
        self.k_norm = _native_rmsnorm(config, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(sequence_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        outputs = []
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            chunk_query = query[start:end].transpose(0, 1)
            chunk_key = key[start:end].transpose(0, 1)
            chunk_value = value[start:end].transpose(0, 1)
            output = F.scaled_dot_product_attention(
                chunk_query,
                chunk_key,
                chunk_value,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                scale=self.scaling,
            )
            outputs.append(output.transpose(0, 1))
        return self.proj(torch.cat(outputs, dim=0).reshape(sequence_length, -1))


class VisionMLP(nn.Module):
    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.attention_bias)
        self.limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states).clamp(max=self.limit)
        up = self.up_proj(hidden_states).clamp(min=-self.limit, max=self.limit)
        return self.down_proj(F.silu(gate) * up)


class VisionBlock(nn.Module):
    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.norm1 = _native_rmsnorm(config, config.hidden_size)
        self.norm2 = _native_rmsnorm(config, config.hidden_size)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, position_embeddings)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class VisionPatchMerger(nn.Module):
    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        hidden_size = config.out_hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.post_projection_norm = nn.LayerNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, config.projection_intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, config.projection_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.projection_intermediate_size, hidden_size, bias=False)
        self.limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.gelu(self.post_projection_norm(self.proj(hidden_states)))
        gate = self.gate_proj(hidden_states).clamp(max=self.limit)
        up = self.up_proj(hidden_states).clamp(min=-self.limit, max=self.limit)
        return self.down_proj(F.silu(gate) * up)


class Glm5NextVisionModel(nn.Module):
    def __init__(self, config: Glm5NextVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = VisionPatchEmbed(config)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.depth)])
        self.merger = VisionPatchMerger(config)
        self.downsample = nn.Conv2d(
            config.hidden_size,
            config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.post_layernorm = _native_rmsnorm(config, config.hidden_size)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> Glm5NextVisionOutput:
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw)
        hidden_states = self.patch_embed(pixel_values)
        rotary = self.rotary_pos_emb(position_ids)
        rotary = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = rotary.cos(), rotary.sin()
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, position_embeddings)
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = hidden_states.view(
            -1,
            self.spatial_merge_size,
            self.spatial_merge_size,
            hidden_states.shape[-1],
        ).permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.config.out_hidden_size)
        return Glm5NextVisionOutput(hidden_states, self.merger(hidden_states))
