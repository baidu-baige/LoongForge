""" dynamic_rotary_pos_embedding.py """

from __future__ import annotations

import torch
from torch import Tensor
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

class DynamicRotaryEmbedding(RotaryEmbedding):
    """Dynamic Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE for longer sequences.
        The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to 10000.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float = None,
        rotary_base: int = 10000,
        dtype: torch.dtype = torch.float32,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        use_cpu_initialization: bool = False,
        max_position_embeddings: int = 4096,
    ) -> None:
        super().__init__(kv_channels, rotary_percent, rotary_interleaved, seq_len_interpolation_factor, rotary_base,
                         dtype, rope_scaling, rope_scaling_factor, use_cpu_initialization)
        self.dim = kv_channels
        self.rotary_base = rotary_base
        self.scaling_factor = rope_scaling_factor
        if rotary_percent < 1.0:
            self.dim = int(self.dim * rotary_percent)
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len_cached = max_position_embeddings

    def forward(self, max_seq_len: int, offset: int = 0, packed_seq: bool = False) -> Tensor:
        """Forward pass of RoPE embedding """
        if max_seq_len > self.max_position_embeddings:
            base = self.rotary_base * (
                    (self.scaling_factor * max_seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (
                    base
                    ** (
                            torch.arange(0, self.dim, 2, dtype=torch.float32, device=torch.cuda.current_device())
                            / self.dim
                    )
            )

        return super().forward(max_seq_len, offset, packed_seq)

