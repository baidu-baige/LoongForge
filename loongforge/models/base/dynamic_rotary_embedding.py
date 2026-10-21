# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Rotary embedding with dynamic NTK scaling, shared by models that request it."""

from typing import Optional

import torch
from torch import Tensor
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding


class DynamicRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding that rescales the base with dynamic NTK beyond max_position_embeddings.

    ``inv_freq`` depends only on the current ``max_seq_len``, so the parent's
    ``lru_cache`` (keyed by ``max_seq_len``, ``offset`` and ``packed_seq``) always
    returns a result computed from the matching frequencies. Within
    ``max_position_embeddings`` the result equals a plain ``RotaryEmbedding``.

    Args:
        max_position_embeddings (int): Sequence length the pretrained model was built for.
            Scaling starts above this length. Required: a wrong value silently changes RoPE.
        Other arguments are forwarded to ``RotaryEmbedding``.
    """

    def __init__(
        self,
        max_position_embeddings: int,
        kv_channels: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        dtype: torch.dtype = torch.float32,
        use_cpu_initialization: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__(
            kv_channels=kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            dtype=dtype,
            use_cpu_initialization=use_cpu_initialization,
            cp_group=cp_group,
        )
        if max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings must be positive, got {max_position_embeddings}")
        self.max_position_embeddings = max_position_embeddings
        self.dim = kv_channels
        if rotary_percent < 1.0:
            self.dim = int(self.dim * rotary_percent)
        self.rotary_base = rotary_base
        self.scaling_factor = rope_scaling_factor
        # Frequencies as built by the parent, including any llama3 rope scaling.
        self._original_inv_freq = self.inv_freq

    def _inv_freq_for(self, max_seq_len: int) -> Tensor:
        """Return the frequencies for ``max_seq_len``: the parent's ones, or NTK-scaled ones."""
        original = self._original_inv_freq.to(device=torch.cuda.current_device())
        if max_seq_len <= self.max_position_embeddings:
            return original
        base = self.rotary_base * (
            self.scaling_factor * max_seq_len / self.max_position_embeddings - (self.scaling_factor - 1)
        ) ** (self.dim / (self.dim - 2))
        return 1.0 / base ** (
            torch.arange(0, self.dim, 2, dtype=torch.float32, device=original.device) / self.dim
        )

    def forward(self, max_seq_len: int, offset: int = 0, packed_seq: bool = False) -> Tensor:
        """Forward pass of RoPE embedding."""
        self.inv_freq = self._inv_freq_for(max_seq_len)
        return super().forward(max_seq_len, offset, packed_seq)
