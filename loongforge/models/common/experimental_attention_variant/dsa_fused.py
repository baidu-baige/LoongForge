# Copyright 2026 The LoongForge Authors.
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
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper as DSAIndexerLossLoggingHelperFused,
)

from loongforge.utils import get_args

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None
    
from .dsa_fused_utils import (
    all_to_all_hp2sp_with_padding,
    all_to_all_sp2hp_with_padding,
    gather_sequence_and_scatter_heads,
    shard_packed_cu_seqlens_for_sp_rank
)
from .dsa_fused_kernels import (
    triton_attn_dist,
    DSADotProductAttention,
    DSAIndexerKernel,
    fused_apply_mla_rope,
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
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'pp'])
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

        # Initialize chunkpipe configuration if enabled
        if getattr(self.config, 'enable_chunkpipe', False):
            self.setup_chunkpipe_indexer_cache_config()
            self.init_chunk_indexer_key_cache()

        self.indexer_kernel = DSAIndexerKernel()

    def setup_chunkpipe_indexer_cache_config(self) -> None:
        """Configure chunkpipe-specific indexer key cache parameters."""
        self.num_chunks_per_seq = self.config.chunk_num_per_seq

        # Calculate cache chunk size based on pipeline parallelism configuration
        pipeline_world_size = self.pg_collection.pp.size()
        pipeline_rank = self.pg_collection.pp.rank()
        if self.config.virtual_pipeline_model_parallel_size is not None:
            self.indexer_key_cache_chunk_size = (pipeline_world_size - pipeline_rank - 1) * 2 \
                + self.num_chunks_per_seq * self.config.virtual_pipeline_model_parallel_size
        else:
            self.indexer_key_cache_chunk_size = (pipeline_world_size - pipeline_rank - 1) * 2 + self.num_chunks_per_seq

        # Initialize cache management data structures
        self.micro_batch_to_cache_chunk_map = {}
        self.empty_chunk_indices = list(range(self.indexer_key_cache_chunk_size))
    
    def is_enable_grad_chunkpipe(self) -> bool:
        """Determine if gradient is enabled in chunkpipe forward computation.
        
        This method is used for determining if tensor hooks should be registered 
        in the forward pass.
        
        Returns:
            bool: 
                - Returns True if gradient should be enabled
                - Returns False if gradient should be disabled

        Raises:
            RuntimeError: If chunkpipe is not enabled
        """
        # Validate chunkpipe configuration
        if not self.config.enable_chunkpipe:
            raise RuntimeError(
                "This method is valid only for Chunkpipe, "
                "please check config.enable_chunkpipe=True"
            )
        
        # In forward recomputation before backward pass, gradient should be enabled
        if not self.config.chunkpipe_forward:
            return True

        # During inference, gradient should always be disabled
        if not self.training:
            return False

        # During last 'keep_activations_chunks' chunks, gradient should be enabled
        current_chunk_idx = self.config.chunkpipe_forward_microbatch % self.num_chunks_per_seq
        return (current_chunk_idx + self.config.keep_activations_chunks >= self.num_chunks_per_seq)

    def init_chunk_indexer_key_cache(self) -> None:
        """Initialize the chunk indexer key cache memory allocations."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        total_cache_tokens = self.num_chunks_per_seq * self.config.chunksize
        cache_shape = (total_cache_tokens, self.config.micro_batch_size, self.index_head_dim)
        
        self.indexer_key_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        # Initialize gradient cache for backward pass
        self.indexer_key_cache_grad = {}

    def clear_chunk_indexer_key_cache(self) -> None:
        """Clear all chunk indexer key cache data and reset cache management state."""
        if not getattr(self.config, 'enable_chunkpipe', False):
            raise RuntimeError(
                "Chunk indexer key cache operations require chunkpipe to be enabled."
            )
        
        self.empty_chunk_indices = list(range(self.indexer_key_cache_chunk_size))
        self.micro_batch_to_cache_chunk_map.clear()

    def delete_chunk_indexer_key_cache(self, micro_batch_index: int) -> None:
        """Remove a specific micro-batch's indexer key cache entry.
        
        Args:
            micro_batch_index: Index of the micro-batch whose cache should be deleted
        """
        if not getattr(self.config, 'enable_chunkpipe', False):
            raise RuntimeError(
                "Chunk indexer key cache operations require chunkpipe to be enabled."
            )
        
        if micro_batch_index not in self.micro_batch_to_cache_chunk_map:
            return

        cache_chunk_index = self.micro_batch_to_cache_chunk_map.pop(micro_batch_index)
        self.empty_chunk_indices.append(cache_chunk_index)

    def append_chunk_indexer_key_cache(self, indexer_key: torch.Tensor) -> None:
        """Append indexer key for the current chunk to cache.
        
        Args:
            indexer_key: Tensor of shape [chunksize, batch, index_head_dim]
        """
        if not getattr(self.config, 'enable_chunkpipe', False):
            return
        
        # Only cache during forward pass
        if not self.config.chunkpipe_forward:
            return
        
        current_microbatch = self.config.chunkpipe_forward_microbatch
        
        # Skip caching for the last chunk in the sequence
        if (current_microbatch + 1) % self.num_chunks_per_seq == 0:
            return
        
        if not self.empty_chunk_indices:
            raise RuntimeError("No available cache chunks for indexer key.")
        
        available_chunk_id = self.empty_chunk_indices.pop(0)
        self.micro_batch_to_cache_chunk_map[current_microbatch] = available_chunk_id
        
        cache_indices = torch.arange(self.config.chunksize, device=indexer_key.device) + \
            (available_chunk_id * self.config.chunksize)
        if self.is_enable_grad_chunkpipe():
            # Detach to prevent gradient tracking of the same computational graph twice
            self.indexer_key_cache[cache_indices, :, :] = indexer_key.clone().detach()
        else:
            self.indexer_key_cache[cache_indices, :, :] = indexer_key

    def concat_cached_chunk_indexer_key(self, indexer_key: torch.Tensor) -> torch.Tensor:
        """Concatenate all cached indexer keys with current chunk's key.
        
        Args:
            indexer_key: Current chunk's indexer key [chunksize, batch, index_head_dim]
            
        Returns:
            Concatenated indexer key [total_seq_len, batch, index_head_dim]
        """
        if not getattr(self.config, 'enable_chunkpipe', False):
            return indexer_key
        
        is_forward = self.config.chunkpipe_forward
        current_microbatch = (
            self.config.chunkpipe_forward_microbatch if is_forward
            else self.config.chunkpipe_backward_microbatch
        )
        
        chunks_in_current_sequence = current_microbatch % self.num_chunks_per_seq
        starting_microbatch_idx = current_microbatch - chunks_in_current_sequence
        
        # Hook function to accumulate gradient from subsequent chunks during backward
        def indexer_key_cache_hook_fn(chunk_index):
            """
            Hook function to accumulate gradient of loss of current chunk 
            with respect to cached indexer key of previous chunk during backward pass.
            """
            def hook_fn(grad):
                if chunk_index not in self.indexer_key_cache_grad:
                    self.indexer_key_cache_grad[chunk_index] = grad
                else:
                    self.indexer_key_cache_grad[chunk_index] += grad
                return grad
            return hook_fn
        
        cached_keys = []
        
        # Retrieve all previous chunks from cache
        for chunk_offset in range(chunks_in_current_sequence):
            microbatch_idx = starting_microbatch_idx + chunk_offset
            cache_chunk_idx = self.micro_batch_to_cache_chunk_map[microbatch_idx]
            
            chunk_indices = torch.arange(
                self.config.chunksize, device=self.indexer_key_cache.device
            ) + (cache_chunk_idx * self.config.chunksize)
            
            cached_key = self.indexer_key_cache[chunk_indices, :, :]
            
            # Register gradient hook for backward pass
            if self.is_enable_grad_chunkpipe():
                cached_key.requires_grad = True
                cached_key.register_hook(indexer_key_cache_hook_fn(chunk_offset))
            
            cached_keys.append(cached_key)
        
        # Append current chunk's key
        cached_keys.append(indexer_key)
        
        # Concatenate along sequence dimension
        return torch.cat(cached_keys, dim=0)

    def _apply_rope(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        mscale: float,
        cu_seqlens: torch.Tensor,
        rotary_pos_cos: torch.Tensor = None,
        rotary_pos_sin: torch.Tensor = None,
        *,
        is_sp: bool = False
    ):
        """Apply RoPE to the input tensor."""
        if self.config.apply_rope_fusion:
            cp_rank = self.pg_collection.cp.rank()
            cp_size = self.pg_collection.cp.size()
            sp_offset = self.pg_collection.tp.rank() * x.shape[0] if is_sp else 0
            
            x = fused_apply_mla_rope(
                x,
                rotary_pos_cos,
                rotary_pos_sin,
                self.index_head_dim - self.qk_pos_emb_head_dim,
                self.qk_pos_emb_head_dim,
                cu_seqlens,
                cp_rank,
                cp_size,
                pe_first=True,
                sp_offset=sp_offset,
            )
        else:
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

            # DSA Indexer's q_pe/k_pe are non-interleaved (unlike MLA's interleaved layout),
            # so we must disable the MLA de-interleave preprocessing in apply_rotary_pos_emb.
            indexer_rope_config = copy.copy(self.config)
            indexer_rope_config.multi_latent_attention = False

            x_pe = apply_rotary_pos_emb(
                x_pe,
                rotary_pos_emb,
                config=indexer_rope_config,
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
        
        # Calculate position embedding offset for chunkpipe
        pos_emb_offset = 0
        if getattr(self.config, 'enable_chunkpipe', False):
            ck_fwd_mic = self.config.chunkpipe_forward_microbatch % self.num_chunks_per_seq
            if not self.config.chunkpipe_forward:
                ck_fwd_mic = self.config.chunkpipe_backward_microbatch % self.num_chunks_per_seq
            pos_emb_offset = ck_fwd_mic * self.config.chunksize
        
        # rotary_pos_emb:[s, b, 1, 64]
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, offset=pos_emb_offset, packed_seq=packed_seq)
            mscale = 1.0
        else:
            if self.config.apply_rope_fusion:
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                    rotary_seq_len, dtype=x.dtype, packed_seq=packed_seq
                )
                mscale = 1.0
                rotary_pos_emb = None
            else:
                rotary_pos_emb, mscale = self.rotary_pos_emb(
                    rotary_seq_len, offset=pos_emb_offset, packed_seq=packed_seq)

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
            if self.config.apply_rope_fusion:
                q = self._apply_rope(q, None, mscale, cu_seqlens_q, rotary_pos_cos[offset:offset + q.size(0)], 
                                        rotary_pos_sin[offset:offset + q.size(0)])
            else:
                q = self._apply_rope(q, rotary_pos_emb[offset:offset + q.size(0)], mscale, cu_seqlens_q)
        else:
            q = self._apply_rope(q, rotary_pos_emb, mscale, cu_seqlens_q, rotary_pos_cos, rotary_pos_sin, is_sp=True)

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
            k = self._apply_rope(k, rotary_pos_emb, mscale, cu_seqlens_kv, rotary_pos_cos, rotary_pos_sin)
            k = k.squeeze(-2)  # [s, b, d]
        else:
            # Cause head and batchsize are both 1, omit the batch squeeze and head unsqueeze
            k = self._apply_rope(k, rotary_pos_emb, mscale, cu_seqlens_kv, rotary_pos_cos, rotary_pos_sin)  # [s, b, d]

        # =========================================
        # Rotate activation
        # =========================================
        q = rotate_activation(q)
        k = rotate_activation(k)

        # =========================================
        # Chunkpipe: register hook to combine gradients from subsequent chunks
        # =========================================
        if getattr(self.config, 'enable_chunkpipe', False) and packed_seq_params is None:
            def indexer_key_hook_fn(grad):
                """
                Hook function to combine gradient of loss of subsequent chunk
                with respect to that of current chunk's indexer key.
                """
                chunks_in_current_sequence = self.config.chunkpipe_backward_microbatch % self.num_chunks_per_seq
                if chunks_in_current_sequence == self.num_chunks_per_seq - 1:
                    # Last chunk in sequence, no accumulated gradient to add
                    return grad
                else:
                    # Add accumulated gradient from subsequent chunks
                    grad_from_prev_chunk = self.indexer_key_cache_grad.pop(chunks_in_current_sequence)
                    return grad + grad_from_prev_chunk
            
            if self.is_enable_grad_chunkpipe():
                k.register_hook(indexer_key_hook_fn)

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

        # =========================================
        # Chunkpipe: cache and concatenate indexer_key
        # =========================================
        if getattr(self.config, 'enable_chunkpipe', False) and packed_seq_params is None:
            # Cache current chunk's indexer_key (only during forward pass, not last chunk)
            self.append_chunk_indexer_key_cache(indexer_key.unsqueeze(1))
            # Concatenate all cached chunks' indexer_key with current chunk
            indexer_key = self.concat_cached_chunk_indexer_key(indexer_key.unsqueeze(1)).squeeze(1)

        # Indexer forward to get indexer_topk_scores and topk_indices
        kv_offset = self.pg_collection.tp.rank() * indexer_query.size(0)
        # Calculate position embedding offset for chunkpipe
        if self.config.enable_chunkpipe:
            ck_fwd_mic = self.config.chunkpipe_forward_microbatch % self.num_chunks_per_seq
            if not self.config.chunkpipe_forward:
                ck_fwd_mic = self.config.chunkpipe_backward_microbatch % self.num_chunks_per_seq
            kv_offset += ck_fwd_mic * self.config.chunksize      
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
        args = get_args()
        self.use_dsa_sp_first = getattr(args, "use_dsa_sp_first", False) if args is not None else False

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

        #padding num_attention_heads to 128 
        _DSA_MIN_HEAD_DIM = 128  # Minimum head count required by DSA sparse attention kernel
        self.num_attention_heads_padded = max(config.num_attention_heads, _DSA_MIN_HEAD_DIM)
        self.num_attention_heads = config.num_attention_heads        

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
        if self.use_dsa_sp_first:
            # query is already chunked [S/TP, B, H, D]
            chunk_query = query
        else:
            # query: [S, B, H/TP, D] -> [S/TP, B, H, D] via All-to-All
            chunk_query = all_to_all_hp2sp_with_padding(query)

        #pad 0 for dim(2)
        if self.num_attention_heads < self.num_attention_heads_padded:
            pad_size = self.num_attention_heads_padded - self.num_attention_heads
            chunk_query = F.pad(chunk_query, [0, 0, 0, pad_size])  # [.., H, D] -> [.., 128, D]

        chunk_sq = chunk_query.size(0)
        offset = self.pg_collection.tp.rank() * chunk_sq
        if self.config.enable_chunkpipe:
            ck_fwd_mic = self.config.chunkpipe_forward_microbatch % self.config.chunk_num_per_seq
            if not self.config.chunkpipe_forward:
                ck_fwd_mic = self.config.chunkpipe_backward_microbatch % self.config.chunk_num_per_seq
            offset += ck_fwd_mic * self.config.chunksize      

        output, p_out = self.sparse_attention(
            chunk_query,
            key,
            topk_indices.unsqueeze(0).int(),
            offset,
            attn_mask_type=attn_mask_type,
            packed_seq_params=packed_seq_params,
            return_p_out=True
        )

        # Unpad output head dim back to original
        if self.num_attention_heads < self.num_attention_heads_padded:
           output = output[:, :, :self.num_attention_heads, :]   # [B, S/TP, H, d_v]
           p_out = p_out[:, :self.num_attention_heads, :]        # [S/TP, H, topk]

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
            if self.use_dsa_sp_first:
                loss_norm = sq * self.pg_collection.tp.size()
            elif getattr(self.config, 'enable_chunkpipe', False):
                loss_norm = sq * self.config.chunk_num_per_seq
            else:
                loss_norm = sq
            indexer_loss = indexer_loss_coeff * loss / loss_norm
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

        if not self.use_dsa_sp_first:
            is_thd = output.dim() == 3
            if is_thd:
                # thd format: [t, h, d] -> [t, 1, h, d] to get [s_local, b=1, h, d]
                # Use all_to_all_sp2hp_with_padding directly (seq-first layout)
                # instead of gather_sequence_and_scatter_heads (which expects batch-first
                # and does transpose(0,1), incorrectly swapping seq and batch dims)
                output = output.unsqueeze(1)  # [t, h, d] -> [t, 1, h, d]
                output = all_to_all_sp2hp_with_padding(output)  # [S, 1, h/TP, d]
                output = output.squeeze(1)  # [S, 1, h/TP, d] -> [S, h/TP, d]
            else:
                output = gather_sequence_and_scatter_heads(output)

        return output
