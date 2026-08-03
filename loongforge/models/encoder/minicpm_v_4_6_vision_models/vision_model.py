# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Hugging Face Transformers MiniCPM-V-4.6 under the Apache-2.0 License.
# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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

"""MiniCPM-V-4.6 vision tower."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from loongforge.models.common import BaseMegatronVisionModule

from .minicpm_v_4_6_config import MiniCPMV46VisionConfig


def _target_sizes_from_grid(image_grid_thw: torch.Tensor) -> torch.Tensor:
    """Convert LoongForge grid_thw to MiniCPM target_sizes."""
    if image_grid_thw is None:
        raise ValueError("MiniCPM-V-4.6 vision forward requires image_grid_thw.")
    if image_grid_thw.dim() != 2 or image_grid_thw.shape[-1] != 3:
        raise ValueError(f"Expected image_grid_thw with shape [n, 3], got {tuple(image_grid_thw.shape)}.")
    return image_grid_thw[:, 1:].to(dtype=torch.long)


def get_vision_nearest_position_ids(
    target_sizes: torch.Tensor,
    num_patches_per_side: int,
) -> torch.Tensor:
    """Nearest-neighbor ids into the learned square position table."""
    device = target_sizes.device
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side, device=device)
    pos_ids_list = []
    for height, width in target_sizes.tolist():
        height, width = int(height), int(width)
        h_coords = torch.arange(height, device=device) / height
        w_coords = torch.arange(width, device=device) / width
        bucket_h = torch.bucketize(h_coords, boundaries, right=True)
        bucket_w = torch.bucketize(w_coords, boundaries, right=True)
        pos_ids_list.append((bucket_h[:, None] * num_patches_per_side + bucket_w).flatten())
    return torch.cat(pos_ids_list)


def get_vision_window_index(
    target_sizes: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return MiniCPM window-attention reorder indices and cumulative sequence lengths."""
    window_index = []
    cu_window_seqlens = [0]
    window_index_id = 0

    for height, width in target_sizes.tolist():
        grid_t, grid_h, grid_w = 1, int(height), int(width)
        index = torch.arange(grid_t * grid_h * grid_w, device=target_sizes.device).reshape(grid_t, grid_h, grid_w)
        pad_h = window_size - grid_h % window_size
        pad_w = window_size - grid_w % window_size
        num_windows_h = (grid_h + pad_h) // window_size
        num_windows_w = (grid_w + pad_w) // window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(grid_t, num_windows_h, window_size, num_windows_w, window_size)
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, num_windows_h * num_windows_w, window_size, window_size
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_seqlens_tmp = seqlens.cumsum(0) + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += grid_t * grid_h * grid_w

    return (
        torch.cat(window_index, dim=0),
        torch.unique_consecutive(torch.tensor(cu_window_seqlens, device=target_sizes.device, dtype=torch.int32)),
    )


def _split_by_cu_seqlens(x: torch.Tensor, cu_seqlens: torch.Tensor) -> list[torch.Tensor]:
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    return list(torch.split(x, lengths, dim=2))


class MiniCPMV46VisionEmbeddings(nn.Module):
    """Conv2d patch embedding plus nearest-neighbor learned position ids."""

    def __init__(self, config: MiniCPMV46VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.num_patches_per_side = self.image_size // self.patch_size
        self.position_embedding = nn.Embedding(self.num_patches_per_side**2, self.embed_dim)

    def _unpack_pixel_values(self, pixel_values: torch.Tensor, target_sizes: torch.Tensor) -> torch.Tensor:
        """Convert packed patch rows to BCHW images when needed."""
        if pixel_values.dim() != 2:
            raise ValueError(f"Unsupported MiniCPM pixel_values shape: {tuple(pixel_values.shape)}.")
        patch_dim = 3 * self.patch_size * self.patch_size
        if pixel_values.shape[-1] != patch_dim:
            raise ValueError(f"Expected packed patch dim {patch_dim}, got {pixel_values.shape[-1]}.")
        images = []
        offset = 0
        for height, width in target_sizes.tolist():
            height, width = int(height), int(width)
            n_patches = height * width
            patches = pixel_values[offset: offset + n_patches]
            offset += n_patches
            image = patches.view(height, width, 3, self.patch_size, self.patch_size)
            image = image.permute(2, 0, 3, 1, 4).reshape(3, height * self.patch_size, width * self.patch_size)
            images.append(image)
        if offset != pixel_values.shape[0]:
            raise ValueError(
                f"MiniCPM packed pixel_values rows {pixel_values.shape[0]} "
                f"does not match target_sizes total patches {offset}."
            )
        if len({tuple(image.shape) for image in images}) != 1:
            raise ValueError("MiniCPM packed pixel_values must have uniform image sizes for batched Conv2d.")
        return torch.stack(images, dim=0)

    def forward(self, pixel_values: torch.Tensor, target_sizes: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() == 2:
            pixel_values = self._unpack_pixel_values(pixel_values, target_sizes)
        elif pixel_values.dim() != 4:
            raise ValueError(f"Unsupported MiniCPM pixel_values shape: {tuple(pixel_values.shape)}.")
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=self.patch_embedding.weight.dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        pos_ids = get_vision_nearest_position_ids(target_sizes, self.num_patches_per_side).to(
            self.position_embedding.weight.device
        )
        position_embeddings = self.position_embedding(pos_ids).unsqueeze(0)
        return embeddings + position_embeddings


class MiniCPMV46VisionMLP(nn.Module):
    """MiniCPM vision MLP."""

    def __init__(self, config: MiniCPMV46VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.ffn_hidden_size)
        self.fc2 = nn.Linear(config.ffn_hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states), approximate="tanh"))


class MiniCPMV46VisionAttention(nn.Module):
    """MiniCPM vision self-attention without RoPE."""

    def __init__(self, config: MiniCPMV46VisionConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        del max_seqlen
        batch_size, seq_len, _ = hidden_states.shape

        def project(module: nn.Linear) -> torch.Tensor:
            return module(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        query_states = project(self.q_proj)
        key_states = project(self.k_proj)
        value_states = project(self.v_proj)
        attn_outputs = []
        for q, k, v in zip(
            _split_by_cu_seqlens(query_states, cu_seqlens),
            _split_by_cu_seqlens(key_states, cu_seqlens),
            _split_by_cu_seqlens(value_states, cu_seqlens),
        ):
            attn_outputs.append(
                F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    scale=self.scaling,
                ).transpose(1, 2)
            )
        attn_output = torch.cat(attn_outputs, dim=1).reshape(batch_size, seq_len, self.dim).contiguous()
        return self.out_proj(attn_output)


class MiniCPMV46VisionEncoderLayer(nn.Module):
    """MiniCPM vision encoder layer."""

    def __init__(self, config: MiniCPMV46VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.self_attn = MiniCPMV46VisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.mlp = MiniCPMV46VisionMLP(config)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MiniCPMV46VisionEncoder(nn.Module):
    """MiniCPM vision encoder container."""

    def __init__(self, config: MiniCPMV46VisionConfig):
        super().__init__()
        self.layers = nn.ModuleList([MiniCPMV46VisionEncoderLayer(config) for _ in range(config.num_layers)])


class MiniCPMV46ViTWindowAttentionMerger(nn.Module):
    """Intermediate MiniCPM ViT window attention and 2x2 merger."""

    def __init__(self, config: MiniCPMV46VisionConfig):
        super().__init__()
        self.window_kernel_size = tuple(config.window_kernel_size)
        self.embed_dim = config.hidden_size
        self.self_attn = MiniCPMV46VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layernorm_epsilon)
        self.pre_norm = nn.LayerNorm(config.window_hidden_size, eps=config.layernorm_epsilon)
        self.linear_1 = nn.Linear(config.window_hidden_size, config.window_intermediate_size, bias=True)
        self.linear_2 = nn.Linear(config.window_intermediate_size, self.embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor, target_sizes: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        window_h, window_w = self.window_kernel_size
        if window_h != window_w:
            raise ValueError(f"window_kernel_size must be square, got {self.window_kernel_size}.")
        window_index, cu_seqlens = get_vision_window_index(target_sizes, window_h)
        window_index = window_index.to(hidden_states.device)
        cu_seqlens = cu_seqlens.to(hidden_states.device)

        hidden_states = hidden_states[:, window_index, :]
        hidden_states = self.self_attn(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=window_h * window_w)
        hidden_states = hidden_states[:, torch.argsort(window_index), :]
        hidden_states = residual + hidden_states

        embed_dim = hidden_states.shape[-1]
        outputs = []
        offset = 0
        for height, width in target_sizes.tolist():
            height, width = int(height), int(width)
            if height % window_h != 0 or width % window_w != 0:
                raise ValueError(
                    f"Patch grid ({height}, {width}) must be divisible by window kernel size "
                    f"{self.window_kernel_size}."
                )
            num_patches = height * width
            merged_h = height // window_h
            merged_w = width // window_w
            patch = hidden_states[:, offset : offset + num_patches, :]
            offset += num_patches
            patch_5d = patch.view(1, merged_h, window_h, merged_w, window_w, embed_dim).permute(0, 1, 3, 2, 4, 5)
            flat = patch_5d.reshape(merged_h * merged_w, window_h * window_w * embed_dim)
            residual = patch_5d.reshape(merged_h * merged_w, window_h * window_w, embed_dim).mean(dim=1)
            hidden_state = self.pre_norm(flat)
            hidden_state = self.linear_1(hidden_state)
            hidden_state = F.gelu(hidden_state, approximate="tanh")
            hidden_state = self.linear_2(hidden_state)
            outputs.append(hidden_state + residual)
        return torch.cat(outputs, dim=0).unsqueeze(0)


class MiniCPMV46VisionModel(BaseMegatronVisionModule):
    """MiniCPM-V-4.6 vision model."""

    config_class = MiniCPMV46VisionConfig

    def __init__(
        self,
        config: MiniCPMV46VisionConfig,
        vp_stage: Optional[int] = None,
        **kwargs,
    ) -> None:
        del vp_stage, kwargs
        super().__init__(config)
        self.embeddings = MiniCPMV46VisionEmbeddings(config)
        self.encoder = MiniCPMV46VisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.vit_merger = MiniCPMV46ViTWindowAttentionMerger(config)
        if getattr(config, "freeze", False):
            self.freeze()

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        self.input_tensor = input_tensor

    def get_dummy_input(self, device):
        patch_size = self.config.patch_size
        return (
            torch.randn((1, 3, patch_size * 2, patch_size * 2), dtype=torch.bfloat16, device=device),
            torch.tensor([[1, 2, 2]], dtype=torch.int32, device=device),
        )

    def get_downsampled_inputs(
        self,
        target_sizes: torch.Tensor,
        max_seqlen: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_sizes = target_sizes // 2
        max_seqlen = max_seqlen // 4
        cu_seqlens = F.pad(
            torch.cumsum(target_sizes[:, 0] * target_sizes[:, 1], dim=0, dtype=torch.int32).to(device),
            (1, 0),
        )
        return target_sizes, cu_seqlens, max_seqlen

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> tuple[torch.Tensor, None, list]:
        target_sizes = _target_sizes_from_grid(image_grid_thw)
        hidden_states = self.embeddings(pixel_values, target_sizes=target_sizes)
        cu_seqlens = F.pad(
            torch.cumsum(target_sizes[:, 0] * target_sizes[:, 1], dim=0, dtype=torch.int32).to(hidden_states.device),
            (1, 0),
        )
        max_seqlen = torch.max(cu_seqlens[1:] - cu_seqlens[:-1])
        use_vit_merger = self.config.downsample_mode != "4x"

        for layer_index, encoder_layer in enumerate(self.encoder.layers):
            hidden_states = encoder_layer(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=int(max_seqlen.item()))
            if use_vit_merger and layer_index == self.config.insert_layer_id:
                hidden_states = self.vit_merger(hidden_states, target_sizes)
                target_sizes, cu_seqlens, max_seqlen = self.get_downsampled_inputs(
                    target_sizes=target_sizes,
                    max_seqlen=max_seqlen,
                    device=hidden_states.device,
                )

        last_hidden_state = F.layer_norm(
            hidden_states.float(),
            self.post_layernorm.normalized_shape,
            self.post_layernorm.weight.float(),
            self.post_layernorm.bias.float(),
            self.post_layernorm.eps,
        )
        return last_hidden_state, target_sizes, []
