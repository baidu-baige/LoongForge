"""Megatron local attention"""

import torch
from torch import Tensor
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from megatron.core.process_groups_config import ProcessGroupCollection
from einops import rearrange
from typing import Optional


class FlashAttentionCore(MegatronModule):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        cp_comm_type: str = None,
        softmax_scale: str = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        """__init__"""
        super().__init__(config=config)

        self.config: TransformerConfig = config

        assert (
            self.config.context_parallel_size == 1
        ), "Context parallelism is only supported by TEDotProductAttention!"

        assert (
            self.config.window_size is None
        ), "Sliding Window Attention is only supported by TEDotProductAttention!"

        self.attn_mask_type = attn_mask_type
        self.causal = True
        self.softmax_scale = self.config.softmax_scale
        self.dropout_p = (
            self.config.attention_dropout
            if attention_dropout is None
            else attention_dropout
        )

    def _flash_attention_wrapper(
        self,
        q,
        k,
        v,
        attention_mask=None,
        attn_mask_start_row_indices=None,
        seq_length=None,
    ):
        """Wrapper for flash attention implementation.
        Args:
            q (torch.Tensor): Query tensor
            k (torch.Tensor): Key tensor
            v (torch.Tensor): Value tensor
            attention_mask (Optional[torch.Tensor]): Attention mask
            attn_mask_start_row_indices (Optional[torch.Tensor]): Variable length indices
            seq_length (Optional[int]): Sequence length
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attention output and weights
        """
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        is_gqa =  k is not None and q.shape[1] != k.shape[1]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout_p,
                is_causal=q.shape[-2] == k.shape[-2],
                scale=self.softmax_scale,
                enable_gqa=is_gqa,
            )
        out = out.transpose(1, 2)
        return out, None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (S, B, H, D)
        """
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttention."
            "Please use TEDotProductAttention instead."
        )
        assert all(
            (i.dtype in [torch.float16, torch.bfloat16] for i in (query, key, value))
        )
        assert all((i.is_cuda for i in (query, key, value)))
        batch_size, seqlen_q = query.shape[1], query.shape[0]
        seqlen_k = key.shape[0]
        query, key, value = [
            rearrange(x, "s b ... -> b s ...") for x in [query, key, value]
        ]

        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            is_causal = self.causal

        output, _ = self._flash_attention_wrapper(
            query,
            key,
            value,
            attention_mask,
            attn_mask_start_row_indices=None,
            seq_length=seqlen_q,
        )
        output = rearrange(output, "b s h d -> s b (h d)", b=batch_size, s=seqlen_q)
        return output

