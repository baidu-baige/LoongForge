# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""RoPE utilities for vision models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

from transformer_engine.pytorch.attention import apply_rotary_pos_emb as te_apply_rotary_pos_emb


def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_bshd(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    rot_dim = freqs.shape[-1]
    input_dtype = t.dtype
    t = t.to(freqs.dtype)
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if multi_latent_attention:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = (torch.cos(freqs) * mscale)
    sin_ = (torch.sin(freqs) * mscale)

    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1).to(input_dtype)


def apply_rotary_pos_emb(
    t: Tensor,
    freqs: Tensor,
    config: TransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    rotary_interleaved: bool = False,
    mscale: float = 1.0,
    cp_group: torch.distributed.ProcessGroup = None,
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels

    NOTE: the RoPE of vision model should not be applied like thd-format because all
    freqs of each token have been computed.
    """

    if config.apply_rope_fusion:
        return te_apply_rotary_pos_emb(
            t.unsqueeze(1),
            freqs,
            tensor_format="sbhd",
            interleaved=rotary_interleaved,
            fused=True,
            mscale=mscale,
        ).squeeze(1)

    return _apply_rotary_pos_emb_bshd(
        t.unsqueeze(1),
        freqs,
        rotary_interleaved=config.rotary_interleaved,
        multi_latent_attention=config.multi_latent_attention,
        mscale=mscale,
    ).squeeze(1)