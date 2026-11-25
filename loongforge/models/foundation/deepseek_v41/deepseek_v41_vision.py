# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Native DeepSeek-V4.1 vision tower and spatial aligner."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DeepseekV41VisionConfig:
    """Configuration copied from the nested released checkpoint metadata."""

    num_hidden_layers: int = 32
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 2816
    patch_size: int = 14
    rope_theta: float = 10000.0
    downsample_ratio: int = 3
    max_image_tokens: int = 1024
    min_pixels: int = 295936
    max_wh_ratio: float | None = None

    @classmethod
    def from_mapping(cls, value: dict | None) -> "DeepseekV41VisionConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in (value or {}).items() if key in fields})


def _vision_cos_sin(
    height: int,
    width: int,
    rotary_dim: int,
    theta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim))
    hpos = torch.arange(height, device=device).unsqueeze(1).expand(height, width)
    wpos = torch.arange(width, device=device).unsqueeze(0).expand(height, width)
    freqs = torch.stack((hpos, wpos), dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _apply_vision_rope(
    values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    first, second = values.float().chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(values.dtype)


class DeepseekV41VisionRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        value = hidden_states.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * self.weight.float()).to(hidden_states.dtype)


class _DeepseekV41VisionAttention(nn.Module):
    def __init__(self, config: DeepseekV41VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        lengths: tuple[int, ...],
    ) -> torch.Tensor:
        q, k, value = self.wqkv(hidden_states).chunk(3, dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        value = value.view(-1, self.num_heads, self.head_dim)
        q = _apply_vision_rope(q, cos, sin)
        k = _apply_vision_rope(k, cos, sin)
        outputs = []
        start = 0
        for length in lengths:
            end = start + length
            outputs.append(
                F.scaled_dot_product_attention(
                    q[start:end].transpose(0, 1),
                    k[start:end].transpose(0, 1),
                    value[start:end].transpose(0, 1),
                )
                .transpose(0, 1)
                .reshape(length, -1)
            )
            start = end
        return self.wo(torch.cat(outputs, dim=0))


class _DeepseekV41VisionBlock(nn.Module):
    def __init__(self, config: DeepseekV41VisionConfig):
        super().__init__()
        self.norm1 = DeepseekV41VisionRMSNorm(config.hidden_size)
        self.attn = _DeepseekV41VisionAttention(config)
        self.norm2 = DeepseekV41VisionRMSNorm(config.hidden_size)
        self.mlp = nn.Module()
        self.mlp.w1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.mlp.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        lengths: tuple[int, ...],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cos, sin, lengths)
        mlp_input = self.norm2(hidden_states)
        gate, up = self.mlp.w1(mlp_input).chunk(2, dim=-1)
        return hidden_states + self.mlp.w2(F.silu(gate) * up)


class DeepseekV41VisionTransformer(nn.Module):
    """Native full-attention 2D-RoPE vision encoder."""

    def __init__(self, config: DeepseekV41VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Linear(3 * config.patch_size**2, config.hidden_size, bias=True)
        self.blocks = nn.ModuleList(_DeepseekV41VisionBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = DeepseekV41VisionRMSNorm(config.hidden_size)

    def forward(
        self,
        patches: torch.Tensor,
        image_grid_hws: torch.Tensor,
    ) -> torch.Tensor:
        if patches.ndim != 4:
            raise ValueError("DeepSeek-V4.1 vision expects [patches, 3, patch, patch]")
        if image_grid_hws.ndim != 2 or image_grid_hws.shape[1] != 2:
            raise ValueError("image_grid_hws must have shape [images, 2]")
        grids = [(int(h), int(w)) for h, w in image_grid_hws.detach().cpu().tolist()]
        lengths = tuple(h * w for h, w in grids)
        if sum(lengths) != patches.shape[0]:
            raise ValueError("image_grid_hws does not account for all image patches")
        patch_projection = self.patch_embed.proj
        patches = patches.to(dtype=patch_projection.weight.dtype)
        hidden_states = patch_projection(patches.flatten(1))
        rotary = [
            _vision_cos_sin(
                h,
                w,
                self.config.hidden_size // self.config.num_attention_heads // 2,
                self.config.rope_theta,
                hidden_states.device,
            )
            for h, w in grids
        ]
        cos = torch.cat([item[0] for item in rotary], dim=0)
        sin = torch.cat([item[1] for item in rotary], dim=0)
        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin, lengths)
        return self.norm(hidden_states)


class DeepseekV41VisionAligner(nn.Module):
    """Pad, spatially downsample, and project vision features into text width."""

    def __init__(self, config: DeepseekV41VisionConfig, text_hidden_size: int):
        super().__init__()
        self.config = config
        merged_size = config.hidden_size * config.downsample_ratio**2
        self.w1 = nn.Linear(merged_size, text_hidden_size, bias=True)
        self.w2 = nn.Linear(text_hidden_size, text_hidden_size, bias=True)

    def forward(self, features: torch.Tensor, image_grid_hws: torch.Tensor) -> torch.Tensor:
        ratio = self.config.downsample_ratio
        merged = []
        offset = 0
        for height, width in image_grid_hws.detach().cpu().tolist():
            height, width = int(height), int(width)
            length = height * width
            image = features[offset : offset + length].view(height, width, -1).permute(2, 0, 1)
            image = F.pad(image, (0, -width % ratio, 0, -height % ratio))
            merged.append(F.unfold(image.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1))
            offset += length
        values = torch.cat(merged, dim=0)
        return self.w2(F.gelu(self.w1(values)))
