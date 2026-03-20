# Copyright 2026 The BaigeOmni Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Adjusted from megatron.core.transformer.experimental_attention_variant.dsa (MegatronLM).

Fused DSA (DeepSeek Sparse Attention) implementation. 
Avoids materializing full [sq, sk] tensors to reduce memory for long-sequence training.

Key differences from the original:
1. Fused indexer kernel (CUDA) replaces einsum-based [b, sq, sk] score materialization.
2. Fused attention kernel (CUDA + TileLang) replaces bmm-based [b, np, sq, skv] computation.
3. Packed sequence (thd format) support with cu_seqlens sharding.
4. Sparse KL loss on [s/TP, topk] subset via triton_attn_dist, with immediate backward().
5. MQA absorbed KV layout: query [sq, b, h, kv_lora_rank], key [skv, b, kv_lora_rank], no value.
6. RoPE splits [pe, nope] (PE-first) vs original's [nope, pe].
"""

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossLoggingHelper as DSAIndexerLossLoggingHelperFused,
)

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None
    
from .dsa_fused_utils import (
    all_to_all_hp2sp_with_padding,
    gather_sequence_and_scatter_heads,
    shard_packed_cu_seqlens_for_sp_rank
)
from .dsa_fused_kernels import (
    triton_attn_dist,
    DSADotProductAttention,
    DSAIndexerKernel,
)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.
    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16).

    Returns:
        Rotated tensor.
    """
    assert (
        x.dtype == torch.bfloat16
    ), f"rotate_activation only support bf16 input, but got {x.dtype}"
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed."
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for indexer loss.

    This custom autograd function attaches a KL divergence loss to the activation
    to train the indexer to predict attention scores without affecting the forward pass.
    """

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor):
        """Preserve the indexer_loss by storing it in the context to avoid garbage collection.

        Args:
            output: The output tensor (activation).
            indexer_loss: The indexer KL divergence loss tensor.

        Returns:
            torch.Tensor: The output tensor unchanged.
        """
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for indexer loss.

        Args:
            grad_output: The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss
                gradient.
        """
        (indexer_loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=indexer_loss.device
            )
        indexer_loss_backward_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
        scaled_indexer_loss_grad = torch.ones_like(indexer_loss) * indexer_loss_backward_scale
        return grad_output, scaled_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """Set the scale of the indexer loss.

        Args:
            scale: The scale value to set.
        """
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


@dataclass
class DSAIndexerFusedSubmodules:
    """
    Configuration class for specifying the submodules of an DSA Indexer.

    Args:
        linear_wq_b: Linear projection for query bottleneck expansion.
        linear_wk: Linear projection for key.
        k_norm: Layer normalization for key.
        linear_weights_proj: Linear projection for attention weights.
    """

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_wk: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None


class DSAIndexerFused(MegatronModule):
    """
    DSA Lightning Indexer for DeepSeek Sparse Attention.

    Computes index scores to identify the top-k most relevant key-value pairs for each query in
    sparse attention.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L431-L480
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAIndexerFusedSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            submodules (DSAIndexerFusedSubmodules): Indexer submodules specification.
            pg_collection (ProcessGroupCollection, optional): Process groups for the indexer.
        """
        super().__init__(config=config)
        self.hidden_size = self.config.hidden_size
        self.qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim
        self.q_lora_rank = (
            self.config.q_lora_rank
            if self.config.q_lora_rank is not None
            else self.config.hidden_size
        )

        self.index_n_heads = self.config.dsa_indexer_n_heads
        self.index_head_dim = self.config.dsa_indexer_head_dim
        self.index_topk = self.config.dsa_indexer_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        # Initialize Position Embedding.
        if self.config.rope_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f'Unsupported RoPE type: {self.config.rope_type}, supported types are "rope" and '
                f'"yarn"'
            )

        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.linear_wk = build_module(
            submodules.linear_wk,
            self.hidden_size,
            self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        k_norm_config = copy.copy(self.config)
        k_norm_config.normalization = "LayerNorm"
        self.k_norm = build_module(
            submodules.k_norm,
            config=k_norm_config,
            hidden_size=self.index_head_dim,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.indexer_kernel = DSAIndexerKernel()

    def _apply_rope(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        mscale: float,
        cu_seqlens: torch.Tensor,
        *,
        is_sp: bool = False
    ):
        """Apply RoPE to the input tensor."""
        # x_pe   [seqlen, *, qk_pos_emb_head_dim]
        # x_nope [seqlen, *, index_head_dim - qk_pos_emb_head_dim]
        x_pe, x_nope = torch.split(
            x, [self.qk_pos_emb_head_dim, self.index_head_dim - self.qk_pos_emb_head_dim], dim=-1
        )

        extra_kwargs = dict()
        if is_sp:
            cu_seqlens, offsets = shard_packed_cu_seqlens_for_sp_rank(
                cu_seqlens,
                sp_rank=self.pg_collection.tp.rank(),
                sp_world_size=self.pg_collection.tp.size()
            )
            extra_kwargs["offsets"] = offsets

        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=cu_seqlens,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
            **extra_kwargs,
        )
        # [seqlen, *, index_head_dim]
        x = torch.cat([x_pe, x_nope], dim=-1)
        return x

    def get_query_key_weight_tensors(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for DSA Indexer that returns both index scores and top-k indices.

        This is used when KL loss is enabled to compare indexer scores with true attention scores.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, (batch,) q_lora_rank].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            q: Index query tensor [batch, seqlen, index_head_num, index_head_dim].
            k: Index key tensor [batch, seqlen, index_head_dim].
            weights: Index weights tensor [batch, index_head_num].
        """
        # assert packed_seq_params is None, "Packed sequence is not supported for DSAttentionFused"

        # Detach x and qr to prevent gradients of indexer from flowing back to the main model.
        x = x.detach()  # [s/TP, (b,) d]
        qr = qr.detach()  # [s/TP, (b,) qr]

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, x, self.config, packed_seq_params
        )
        # rotary_pos_emb:[s, b, 1, 64]
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)

        if packed_seq_params is not None:
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # q linear and apply rope to q
        # =========================================
        # qr: [s / TP, (b,) qr]
        # q: [s / TP, (b,) h * d]
        q, _ = self.linear_wq_b(qr)
        # if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
        #     q = gather_from_sequence_parallel_region(q)  # [s, (b,) h * d]
        q = q.view(*q.size()[:-1], self.index_n_heads, self.index_head_dim)  # [..., h, d]
        if packed_seq_params is None:
            offset = self.pg_collection.tp.rank() * q.size(0)
            q = self._apply_rope(q, rotary_pos_emb[offset:offset + q.size(0)], mscale, cu_seqlens_q)
        else:
            q = self._apply_rope(q, rotary_pos_emb, mscale, cu_seqlens_q, is_sp=True)

        # =========================================
        # k linear and apply rope to k
        # =========================================
        # k: [s / TP, b, d]
        k, _ = self.linear_wk(x)
        k = self.k_norm(k)
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            k = gather_from_sequence_parallel_region(k)  # [s, b, d]
        if packed_seq_params is None:
            k = k.unsqueeze(-2)  # [s, b, 1, d]
            k = self._apply_rope(k, rotary_pos_emb, mscale, cu_seqlens_kv)
            k = k.squeeze(-2)  # [s, b, d]
        else:
            # Cause head and batchsize are both 1, omit the batch squeeze and head unsqueeze
            k = self._apply_rope(k, rotary_pos_emb, mscale, cu_seqlens_kv)  # [s, b, d]

        # =========================================
        # Rotate activation
        # =========================================
        q = rotate_activation(q)
        k = rotate_activation(k)

        # =========================================
        # weight linear
        # =========================================
        # weights: [s / TP, b, h]
        weights, _ = self.linear_weights_proj(x)  # [s / TP, b, h]
        # if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
        #     weights = gather_from_sequence_parallel_region(weights)  # [s, b, h]
        weights *= self.index_n_heads ** -0.5

        return q, k, weights

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Forward pass for DSA Indexer.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, (batch,) q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            topk_indices: Top-k indices for sparse attention [batch, seqlen, index_topk].
        """
        
        # =========================================
        # Indexer forward to get query/key/weights
        # =========================================
        (
            indexer_query,  # [s/TP, (b,) h, d]
            indexer_key,  # [s, b, d]
            indexer_weights,  # [s/TP, b, h]
        ) = self.get_query_key_weight_tensors(x, qr, packed_seq_params)

        # DSAIndexerKernel does not support batched input
        if packed_seq_params is None:
            indexer_query = indexer_query.squeeze(1).contiguous()  # [s/TP, h, d]
        indexer_key = indexer_key.squeeze(1).contiguous()  # [s, d]
        indexer_weights = indexer_weights.squeeze(1).contiguous()  # [s/TP, h]

        # Indexer forward to get indexer_topk_scores and topk_indices
        kv_offset = self.pg_collection.tp.rank() * indexer_query.size(0)
        (
            index_score_topk,  # [s/TP, topk]
            topk_indices  # [s/TP, topk]
        ) = self.indexer_kernel(
            indexer_query,
            indexer_key,
            indexer_weights,
            self.index_topk,
            chunk_offset=kv_offset,
            packed_seq_params=packed_seq_params,
        )

        return index_score_topk, topk_indices


@dataclass
class DSAttentionFusedSubmodules:
    """
    Configuration class for specifying the submodules of DSAttentionFused.

    Args:
        indexer: DSA Indexer module for computing sparse attention indices.
    """

    indexer: Union[ModuleSpec, type] = None


class DSAttentionFused(MegatronModule):
    """
    This module implements sparse attention mechanism using an DSA Indexer to compute top-k
    attention indices for reducing computational complexity.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L491-L597
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAttentionFusedSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)

        self.layer_number = layer_number
        self.pg_collection = pg_collection

        self.indexer = build_module(
            submodules.indexer, config=self.config, pg_collection=pg_collection
        )

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale
        
        self.sparse_attention = build_module(
            DSADotProductAttention,
            config=self.config,
            softmax_scale=self.softmax_scale,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """
        Forward pass for MQA Sparse Attention.

        Args:
            query: Query tensor with absorbed kv_up_proj weight [sq, b, h, kv_lora_rank].
            key: Key low-rank tensor [skv, b, kv_lora_rank].
            value: In MQA setup, this is None.
            x: Original hidden states [sq, b, hidden_size].
            qr: Low-rank query representation [sq, b, q_lora_rank].
            attention_mask: Attention mask tensor [b, 1, sq, sk].
            attn_mask_type: Type of attention mask.
            attention_bias: Optional attention bias.
            packed_seq_params: Packed sequence parameters.

        Returns:
            output: Output tensor [sq, b, hidden_size]
        """
        sq = query.size(0)
        skv = key.size(0)

        # Detach x and qr to prevent gradients of indexer from flowing back to the main model.
        x = x.detach()
        qr = qr.detach()

        # ===================================
        # Get index scores and top-k indices
        # ===================================
        (
            index_scores,  # [s / TP, topk]
            topk_indices  # [s / TP, topk]
        ) = self.indexer(
            x, qr, packed_seq_params=packed_seq_params
        )

        # ===================================
        # Run sparse attention kernel
        # ===================================
        # query: [s, b, h / TP, d]
        # chunk_query: [s / TP, b, h, d]
        chunk_query = all_to_all_hp2sp_with_padding(query)  # head & seq alltoall
        chunk_sq = chunk_query.size(0)
        offset = self.pg_collection.tp.rank() * chunk_sq
        output, p_out = self.sparse_attention(
            chunk_query,
            key,
            topk_indices.unsqueeze(0).int(),
            offset,
            attn_mask_type=attn_mask_type,
            packed_seq_params=packed_seq_params,
            return_p_out=True
        )

        # ===================================
        # Attach indexer loss
        # ===================================
        if self.training and torch.is_grad_enabled():
            # Compute KL divergence loss between indexer scores and true attention scores
            indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)

            with torch.no_grad():
                main_attn_probs = triton_attn_dist(p_out, self.softmax_scale)  # [seq / TP, topk]

            index_attn_probs = torch.softmax(index_scores, dim=-1)  # [seq / TP, topk]

            loss = F.kl_div(
                (index_attn_probs + 1e-10).log(),
                main_attn_probs + 1e-10,
                reduction="sum"
            )
            indexer_loss = indexer_loss_coeff * loss / sq

            indexer_loss = reduce_from_tensor_model_parallel_region(indexer_loss)
            
            # Save indexer loss for logging
            if indexer_loss_coeff > 0:
                DSAIndexerLossLoggingHelperFused.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=self.config.num_layers + getattr(self.config, "mtp_num_layers", 0),
                )

            # Attach loss to output
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        output = gather_sequence_and_scatter_heads(output)

        return output
