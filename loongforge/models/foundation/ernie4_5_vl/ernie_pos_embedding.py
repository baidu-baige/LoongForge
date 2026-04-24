# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Ernie_pos_embedding"""

import torch
from torch import Tensor, nn
from megatron.core.transformer.transformer_config import TransformerConfig
from typing import Any, Optional, Dict, List


class ErnieRopeEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation for transformer models.

    RoPE encodes absolute positional information with rotation matrices and
    naturally incorporates relative position information in self-attention.

    Args:
        head_dim (int): Dimension size of each attention head
        compression_ratio (float, optional): Sequence length compression ratio. Defaults to 1.0.
        base (int, optional): Base value for frequency calculation. Defaults to 10000.

    Attributes:
        head_dim (int): Dimension size of each attention head
        compression_ratio (float): Sequence length compression factor
        base (int): Base value for frequency calculation
    """

    def __init__(self, head_dim, compression_ratio=1.0, base=10000, freq_allocation=0):
        """
        Initialize RoPE embedding layer.

        Args:
            head_dim: Dimension of each attention head
            compression_ratio: Scaling factor for position indices
            base: Base value for frequency calculation
        """
        super().__init__()
        self.head_dim = head_dim
        self.compression_ratio = compression_ratio
        self.base = base

        # num of freq allocated to time
        self.freq_allocation = freq_allocation

    def forward(self, position_ids):
        """
        Compute rotary position embeddings for given sequence length.

        Args:
            position_ids (Tensor, optional): Custom position indices. Defaults to None.

        Returns:
            Tensor: Rotary position embeddings of shape [1, 1, seq_length, head_dim]
        """
        seq_length = position_ids.max() + 1

        indices = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        indices = 1 / self.base ** (indices / self.head_dim)
        pos_ids = torch.arange(
            0, seq_length, 1, dtype=torch.float32
        ).unsqueeze(1)
        pos_ids = pos_ids / self.compression_ratio
        sinusoid_inp = pos_ids * indices.unsqueeze(0)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb.view(-1, 1, seq_length, self.head_dim)
        pos_emb = pos_emb.detach()
        # cal sin cos
        pos_emb = pos_emb.permute([0, 2, 1, 3])

        sin_cos = self.get_sin_cos(pos_emb, position_ids)
        return sin_cos

    def get_sin_cos(self, rp, position_ids):
        """Get sin cos from rp."""
        sin, cos = torch.chunk(rp, 2, axis=-1)
        # assert position_ids.shape[:1] == q.shape[:1]
        batch_indices = torch.arange(end=position_ids.shape[0])
        batch_indices = batch_indices[..., None]
        sin = sin.tile(position_ids.shape[0], 1, 1, 1).to(device=position_ids.device)
        cos = cos.tile(position_ids.shape[0], 1, 1, 1).to(device=position_ids.device)

        assert self.freq_allocation != 0
        sin_t = sin[batch_indices, position_ids[..., 0], :, -self.freq_allocation :]
        sin_h = sin[
            batch_indices,
            position_ids[..., 1],
            :,
            : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        sin_w = sin[
            batch_indices,
            position_ids[..., 2],
            :,
            1 : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        sin_hw = torch.stack([sin_h, sin_w], dim=-1).reshape(
            sin_h.shape[:-1] + (sin_h.shape[-1] * 2,)
        )
        sin_thw = torch.cat([sin_hw, sin_t], dim=-1)

        cos_t = cos[batch_indices, position_ids[..., 0], :, -self.freq_allocation :]
        cos_h = cos[
            batch_indices,
            position_ids[..., 1],
            :,
            : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        cos_w = cos[
            batch_indices,
            position_ids[..., 2],
            :,
            1 : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        cos_hw = torch.stack([cos_h, cos_w], dim=-1).reshape(
            cos_h.shape[:-1] + (cos_h.shape[-1] * 2,)
        )
        cos_thw = torch.cat([cos_hw, cos_t], dim=-1)

        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = (
            torch.stack([sin_thw, sin_thw], dim=-1)
            .reshape(sin_thw.shape[:3] + (sin_thw.shape[-1] * 2,))
            # .to(current_device)
        )
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = (
            torch.stack([cos_thw, cos_thw], dim=-1)
            .reshape(cos_thw.shape[:3] + (cos_thw.shape[-1] * 2,))
            # .to(current_device)
        )
        return torch.cat([sin_pos, cos_pos])
        


def apply_rotary_3d(
    q: Tensor,
    sin_cos: Tensor,
    config: TransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    **kwargs,
):
    """
    rope 3d rotary

    Supports both BSHD format (4D: [seq, batch, heads, dim]) and
    THD packed format (3D: [total_tokens, heads, dim]).
    """
    # Handle 3D (THD packed) by adding a batch dim
    squeezed = q.ndim == 3
    if squeezed:
        q = q.unsqueeze(1)  # (seq, 1, heads, dim)

    sin_cos = sin_cos.permute(1, 0, 2, 3).to(q.device)
    sin_pos, cos_pos = torch.split(sin_cos, 1, dim=1)
    # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
    rotate_half_q = torch.stack(
        [-q[:, :, :, 1::2], q[:, :, :, 0::2]], dim=-1
    ).reshape(q.shape)
    query = (q.to(torch.float32) * cos_pos) + (
        rotate_half_q.to(torch.float32) * sin_pos
    )
    query = query.to(q.dtype)

    if squeezed:
        query = query.squeeze(1)  # back to (seq, heads, dim)
    return query