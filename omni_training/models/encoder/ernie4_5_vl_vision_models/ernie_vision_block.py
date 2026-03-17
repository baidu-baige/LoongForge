# Copyright 2026 The OmniTraining Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from ERNIE (https://github.com/PaddlePaddle/ERNIE/)
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Ernie-Vision Block """

import torch
import torch.nn as nn
import math
from typing import Optional

import logging

logger = logging.getLogger(__name__)


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Applies the GELU approximation function to the input tensor.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after applying the GELU approximation function.
        """
        return input * torch.sigmoid(1.702 * input)


class VisionMlp(nn.Module):
    """VisionMLP"""

    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = QuickGELUActivation()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: VisionMLP output tensor
        """
        return self.fc2(self.act(self.fc1(x)))


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)  # shape is the same as x


def apply_rotary_pos_emb_vision(
    tensor: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Applies Rotary Position Embedding to the input tensors.

    Args:
        tensor (torch.Tensor): The input tensor.
        freqs (torch.Tensor): The frequencies used for the rotation.
    Returns:
        output (torch.Tensor): the tensor rotated using the Rotary Position Embedding.
    """
    orig_dtype = tensor.dtype

    tensor = tensor.type(dtype=torch.float32)
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).tile(1, 1, 2).unsqueeze(0).type(dtype=torch.float32)
    sin = sin.unsqueeze(1).tile(1, 1, 2).unsqueeze(0).type(dtype=torch.float32)
    output = tensor * cos + rotate_half(tensor) * sin
    output = output.to(orig_dtype)
    return output


class VisionAttention(nn.Module):
    """VisionAttention"""

    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.head_dim = dim // num_heads  # must added

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """forward function for vision attention"""
        seq_length = hidden_states.shape[0]
        qkv = (
            self.qkv(hidden_states)
            .reshape([seq_length, 3, self.num_heads, -1])
            .permute(1, 0, 2, 3)
        )
        q, k, v = qkv.unbind(axis=0)

        q = apply_rotary_pos_emb_vision(q.unsqueeze(dim=0), rotary_pos_emb).squeeze(
            dim=0
        )
        k = apply_rotary_pos_emb_vision(k.unsqueeze(dim=0), rotary_pos_emb).squeeze(
            dim=0
        )
        
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=1) for tensor in (q, k, v)
        ]
        
        attn_output = []
        for q, k, v in zip(*splits):
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q.dtype)
            attn_output_splited = torch.matmul(attn_weights, v)
            attn_output_splited = attn_output_splited.transpose(0, 1)
            attn_output.append(attn_output_splited)
        attn_output = torch.cat(attn_output, dim=0)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class DFNRopeVisionBlock(nn.Module):
    """DFNRopeVisionBlock"""

    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        """
        Args:
            config (dict): model configuration.
            attn_implementation (str, optional): attention implementation. Defaults to "sdpa".
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = VisionAttention(config.embed_dim, num_heads=config.num_heads)
        self.mlp = VisionMlp(
            dim=config.embed_dim,
            hidden_dim=mlp_hidden_dim,
            hidden_act=config.hidden_act,
        )
        self.config = config

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        """
        Args:
            hidden_states(torch.Tensor): hidden states
            cu_seqlens (torch.Tensor): cumulative sequence lengths
            rotary_pos_emb: rotary position embedding

        Returns:
            torch.Tensor: output tensor
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

