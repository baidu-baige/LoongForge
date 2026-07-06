# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compressed sparse attention modules for DeepSeek-v4 hybrid attention."""
import os
import copy
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace
except Exception:
    fused_mla_rope_inplace = None
from megatron.core.models.common.embeddings import RotaryEmbedding, apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import (
    all_to_all,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    FusedDSAIndexerLoss,
    fused_qk_topk_naive,
    rotate_activation,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide, get_pg_size, nvtx_range_pop, nvtx_range_push

import deep_gemm
import flashinfer
import lightning_indexer_bwd
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer

def _all_to_all_hp2sp(input_: torch.Tensor, tp_group) -> torch.Tensor:
    """All-to-All: head-parallel to seq-parallel.

    [S, B, H/TP, D] -> [S_padded/TP, B, H, D]

    Scatters heads and gathers sequence across TP ranks.
    """
    world_size = tp_group.size()
    if world_size == 1:
        return input_

    s, *bhd = input_.shape

    # Pad sequence to be divisible by world_size
    remainder = s % world_size
    padding_size = (world_size - remainder) % world_size
    if padding_size > 0:
        out = input_.new_zeros(s + padding_size, *input_.shape[1:])
        out[:s] = input_
        input_ = out

    s_padded = input_.size(0)
    input_ = input_.reshape(world_size, s_padded // world_size, *bhd)

    input_exchanged = all_to_all(tp_group, input_)  # [TP, s/TP, *bhd]
    output = input_exchanged.movedim(0, -3)  # [s/TP, (b,) TP, h/TP, d]
    output = output.flatten(-3, -2).contiguous()  # [s/TP, (b,) H, d]
    return output

def _all_to_all_sp2hp(input_: torch.Tensor, tp_group, ori_s: int = None) -> torch.Tensor:
    """All-to-All: seq-parallel to head-parallel.

    [S/TP, B, H, D] -> [S, B, H/TP, D]

    Gathers sequence and scatters heads across TP ranks.
    """
    world_size = tp_group.size()
    if world_size == 1:
        return input_

    s_local, b, h, d = input_.shape
    h_local = h // world_size

    input_ = input_.reshape(s_local, b, world_size, h_local, d)
    input_ = input_.movedim(2, 0).contiguous()  # [world_size, s_local, b, h_local, d]

    input_exchanged = all_to_all(tp_group, input_)
    output = input_exchanged.reshape(world_size * s_local, b, h_local, d)

    # Remove padding if original sequence length is provided
    if ori_s is not None:
        output = output[:ori_s].contiguous()

    return output

class _AllToAllHp2Sp(torch.autograd.Function):
    """Autograd-aware All-to-All: head-parallel -> seq-parallel."""

    @staticmethod
    def forward(ctx, input_, tp_group):
        """Forward function."""
        ctx.tp_group = tp_group
        ctx.orig_s = input_.size(0)
        return _all_to_all_hp2sp(input_, tp_group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _all_to_all_sp2hp(grad_output, ctx.tp_group, ctx.orig_s), None


class _AllToAllSp2Hp(torch.autograd.Function):
    """Autograd-aware All-to-All: seq-parallel -> head-parallel."""

    @staticmethod
    def forward(ctx, input_, tp_group):
        """Forward function."""
        ctx.tp_group = tp_group
        input_ = input_.transpose(0, 1).contiguous()
        output = _all_to_all_sp2hp(input_, tp_group)
        output = output.transpose(0, 1).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        grad_output = grad_output.transpose(0, 1).contiguous()
        output = _all_to_all_hp2sp(grad_output, ctx.tp_group)
        output = output.transpose(0, 1).contiguous()
        return output, None

# ---------------------------------------------------------------------------
# Helper functions for index computation
# ---------------------------------------------------------------------------


# TODO: the lru_cache may not work well with packed sequence
@lru_cache(maxsize=8)
def _get_window_topk_idxs_cached(window_size: int, seqlen: int, device_str: str) -> torch.Tensor:
    """Compute sliding-window indices for a single sequence (cached).

    Returns:
        indices: [seqlen, window_size] int tensor, -1 for invalid positions.
    """
    base = torch.arange(seqlen, device=device_str).unsqueeze(1)
    offsets = torch.arange(window_size, device=device_str)
    matrix = (base - window_size + 1).clamp(min=0) + offsets
    matrix = torch.where(matrix > base, -1, matrix)
    return matrix


def get_window_topk_idxs(
    window_size: int, batch_size: int, seqlen: int, device: torch.device
) -> torch.Tensor:
    """Sliding-window indices [batch, seqlen, window_size]."""
    matrix = _get_window_topk_idxs_cached(window_size, seqlen, str(device))
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


# TODO: the lru_cache may not work well with packed sequence
@lru_cache(maxsize=8)
def _get_compress_topk_idxs_cached(
    ratio: int, seqlen: int, offset: int, device_str: str
) -> torch.Tensor:
    """Compute all-compressed-positions indices for a single sequence (cached).

    Returns:
        indices: [seqlen, seqlen // ratio] int tensor, -1 for future positions.
    """
    n_compressed = seqlen // ratio
    matrix = torch.arange(n_compressed, device=device_str).repeat(seqlen, 1)
    mask = matrix >= torch.arange(1, seqlen + 1, device=device_str).unsqueeze(1) // ratio
    matrix = torch.where(mask, -1, matrix + offset)
    return matrix


def get_compress_topk_idxs(
    ratio: int, batch_size: int, seqlen: int, offset: int, device: torch.device
) -> torch.Tensor:
    """All-compressed-position indices [batch, seqlen, seqlen // ratio]."""
    matrix = _get_compress_topk_idxs_cached(ratio, seqlen, offset, str(device))
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


# ---------------------------------------------------------------------------
# Helper functions for RoPE
# ---------------------------------------------------------------------------


def _apply_rope(
    x: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    rotary_pos_emb_module: RotaryEmbedding,
    config: TransformerConfig,
    rotary_seq_len: int,
    ratio: int = 1,
    cp_group: torch.distributed.ProcessGroup = None,
) -> torch.Tensor:
    """Apply RoPE to the last ``qk_pos_emb_head_dim`` dims, leaving the rest unchanged.

    Accepts both 3-D ``[seq, batch, head_dim]`` and 4-D ``[seq, batch, heads, head_dim]``
    inputs.  When the input is 3-D a temporary head dimension is inserted for
    ``apply_rotary_pos_emb`` and removed before returning.
    """
    if ratio == 1:
        total_seq_len = rotary_seq_len
    else:
        total_seq_len = rotary_seq_len * ratio
    mscale = 1.0
    rotary_pos_cos = None
    rotary_pos_sin = None
    if config.rope_type == "rope":
        rotary_pos_emb = rotary_pos_emb_module(total_seq_len, packed_seq=False)
        mscale = 1.0
    else:
        if config.apply_rope_fusion:
            rotary_pos_cos, rotary_pos_sin = rotary_pos_emb_module.get_cached_cos_sin(
                total_seq_len, dtype=x.dtype, packed_seq=False
            )
            rotary_pos_emb = None
            assert (
                fused_mla_rope_inplace is not None
            ), "Fused MLA RoPE apply is not imported successfully"
        else:
            rotary_pos_emb, mscale = rotary_pos_emb_module(total_seq_len, packed_seq=False)
            # DSv4 reference (DS-Inf) RoPE is pure rotation (norm-preserving). Yarn's
            # concentration factor (mscale) is NOT part of the DSv4 model contract --
            # the model relies on Q/KV RMS-norm + unit-magnitude rotation. Force 1.0.
            mscale = 1.0
    if rotary_pos_emb is not None and ratio > 1:
        rotary_pos_emb = rotary_pos_emb[:total_seq_len:ratio][:rotary_seq_len]
    if rotary_pos_cos is not None and ratio > 1:
        rotary_pos_cos = rotary_pos_cos[:total_seq_len:ratio][:rotary_seq_len]
    if rotary_pos_sin is not None and ratio > 1:
        rotary_pos_sin = rotary_pos_sin[:total_seq_len:ratio][:rotary_seq_len]

    squeeze_head = x.dim() == 3
    if squeeze_head:
        x = x.unsqueeze(-2)
    if config.apply_rope_fusion:
        out = fused_mla_rope_inplace(
            x,
            rotary_pos_cos,
            rotary_pos_sin,
            nope_dim,
            pos_dim,
            None,
            cp_group.rank(),
            cp_group.size(),
            remove_interleaving=True,
        )
    else:
        x_nope, x_pe = torch.split(x, [nope_dim, pos_dim], dim=-1)
        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=cp_group,
            mla_rotary_interleaved=True,
            mla_output_remove_interleaving=True,
        )
        out = torch.cat([x_nope, x_pe], dim=-1)
    if squeeze_head:
        out = out.squeeze(-2)
    return out


# ---------------------------------------------------------------------------
# Sparse attention kernel (unfused, differentiable)
# ---------------------------------------------------------------------------


def unfused_compressed_sparse_attn(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Differentiable sparse attention with MQA and attention sink.

    Chunked over sq dimension to reduce peak memory from O(sq*topk*hn) to O(chunk*topk*hn).

    Args:
        query:        [sq, b, np, hn]   multi-head query.
        kv_full:      [n_kv, b, hn]     single-head KV (original + compressed).
        attn_sink:    [np]              per-head learnable bias.
        topk_indices: [b, sq, topk]     indices into kv_full (int32, -1 = invalid).
        softmax_scale: float
        chunk_size:   int, chunk size along sq for memory efficiency.

    Returns:
        output:       [sq, b, np * hn]
    """
    sq, b, np_, hn = query.size()

    kv_t = kv_full.permute(1, 0, 2)  # [b, n_kv, hn]
    sink = attn_sink.view(1, np_, 1, 1).float()

    output_chunks = []
    for start in range(0, sq, chunk_size):
        end = min(start + chunk_size, sq)
        chunk_len = end - start

        q_chunk = query[start:end].permute(1, 2, 0, 3).float()  # [b, np, chunk, hn]
        idx_chunk = topk_indices[:, start:end, :]  # [b, chunk, topk]

        safe_idx = idx_chunk.clamp(min=0).long().unsqueeze(-1).expand(-1, -1, -1, hn)
        kv_chunk = torch.gather(
            kv_t.unsqueeze(1).expand(-1, chunk_len, -1, -1), dim=2, index=safe_idx
        ).float()  # [b, chunk, topk, hn]

        scores = torch.einsum("bnsh,bskh->bnsk", q_chunk, kv_chunk) * softmax_scale
        invalid_mask = (idx_chunk < 0).unsqueeze(1)
        scores = scores.masked_fill(invalid_mask, float("-inf"))

        scores_max = torch.max(scores.max(dim=-1, keepdim=True).values, sink)
        exp_scores = torch.exp(scores - scores_max)
        exp_sink_val = torch.exp(sink - scores_max)
        attn_weights = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + exp_sink_val)

        out_chunk = torch.einsum("bnsk,bskh->bnsh", attn_weights, kv_chunk)
        out_chunk = out_chunk.to(query.dtype).permute(2, 0, 1, 3).contiguous()
        output_chunks.append(out_chunk.reshape(chunk_len, b, np_ * hn))

    return torch.cat(output_chunks, dim=0)


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


@dataclass
class CompressorSubmodules:
    """Submodule specs for CSA and HCA Compressor."""

    linear_wkv: Union[ModuleSpec, type] = None
    linear_wgate: Union[ModuleSpec, type] = None
    norm: Union[ModuleSpec, type] = None


class Compressor(MegatronModule):
    """Gated pooling compressor for CSA and HCA sparse attention.

    Compresses a sequence of tokens into a shorter sequence by pooling groups of
    ``compress_ratio`` tokens using learned gated weights.

    For ``compress_ratio == 4``, overlapping compression is used (``coff = 2``).
    For ``compress_ratio == 128``, non-overlapping compression is used (``coff = 1``).
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CompressorSubmodules,
        compress_ratio: int,
        head_dim: int,
        rotate: bool = False,
        rotary_pos_emb: nn.Module = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize the compressor submodules and compression parameters."""
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = compress_ratio == 4
        self.coff = 1 + int(self.overlap)
        self.rotate = rotate
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim

        self.rotary_pos_emb = rotary_pos_emb

        proj_out_dim = self.coff * head_dim

        self.linear_wkv = build_module(
            submodules.linear_wkv,
            config.hidden_size,
            proj_out_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        for param in self.linear_wkv.parameters():
            if config.tensor_model_parallel_size > 1:
                setattr(param, 'average_gradients_across_tp_domain', True)

        self.linear_wgate = build_module(
            submodules.linear_wgate,
            config.hidden_size,
            proj_out_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        for param in self.linear_wgate.parameters():
            if config.tensor_model_parallel_size > 1:
                setattr(param, 'average_gradients_across_tp_domain', True)

        # keep to high precision
        _ape = torch.empty(
            compress_ratio, proj_out_dim, device=torch.cuda.current_device(), dtype=torch.float32
        )
        config.init_method(_ape)
        self.ape = nn.Parameter(_ape)
        if config.tensor_model_parallel_size > 1:
            setattr(self.ape, "average_gradients_across_tp_domain", True)

        norm_config = copy.copy(config)
        norm_config.normalization = "RMSNorm"
        self.norm = build_module(
            submodules.norm, config=norm_config, hidden_size=head_dim, eps=config.layernorm_epsilon
        )
        # compressor.norm sees full-seq input (compressor always runs on gathered
        # sequence even under SP), so all ranks compute identical grads -> AVG
        # across TP (overrides standard SP=SUM via if/elif)
        for param in self.norm.parameters():
            if config.tensor_model_parallel_size > 1:
                setattr(param, 'average_gradients_across_tp_domain', True)

    def _overlap_transform(self, tensor: torch.Tensor, fill_value: float = 0) -> torch.Tensor:
        """Apply overlapping window transform for 4x compression.

        Input shape:  [n_groups, ratio, b, coff * head_dim]
        Output shape: [n_groups, 2 * ratio, b, head_dim]
        """
        n_groups, ratio, b_dim, _ = tensor.size()
        d = self.head_dim
        new_tensor = tensor.new_full((n_groups, 2 * ratio, b_dim, d), fill_value)
        new_tensor[:, ratio:] = tensor[:, :, :, d:]
        new_tensor[1:, :ratio] = tensor[:-1, :, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Compress hidden states into shorter KV sequence.

        Args:
            x: [sq, b, hidden_size]

        Returns:
            compressed_kv [sq // ratio, b, head_dim] or None if too short.
        """
        nvtx_range_push("compressor")

        sq, b, _ = x.size()
        ratio = self.compress_ratio

        if sq < ratio:
            nvtx_range_pop("compressor")
            return None

        # Pad sequence length to multiple of 128 for FP8 blockwise GEMM compatibility
        pad_len = (128 - sq % 128) % 128 if self.config.fp8 else 0
        if pad_len > 0:
            x_padded = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_len))
        else:
            x_padded = x
        kv, _ = self.linear_wkv(x_padded)  # [sq+pad, b, coff * head_dim]
        score, _ = self.linear_wgate(x_padded)  # [sq+pad, b, coff * head_dim]
        if pad_len > 0:
            kv = kv[:sq]
            score = score[:sq]

        cutoff = (sq // ratio) * ratio
        if cutoff < sq:
            kv = kv[:cutoff]
            score = score[:cutoff]

        n_compressed = cutoff // ratio

        # Reshape: [n_compressed, ratio, b, coff * head_dim]
        kv = kv.view(n_compressed, ratio, b, -1)
        score = score.view(n_compressed, ratio, b, -1)

        # APE: [ratio, coff * head_dim] -> [1, ratio, 1, coff * head_dim]
        score = score + self.ape.view(1, ratio, 1, -1)

        if self.overlap:
            kv = self._overlap_transform(kv, fill_value=0)
            score = self._overlap_transform(score, fill_value=float("-inf"))

        kv = (kv * torch.softmax(score, dim=1)).sum(dim=1)  # [n_compressed, b, head_dim]

        kv = self.norm(kv.to(x.dtype))

        kv = _apply_rope(
            kv,
            self.head_dim - self.qk_pos_emb_head_dim,
            self.qk_pos_emb_head_dim,
            self.rotary_pos_emb,
            self.config,
            n_compressed,
            ratio=ratio,
            cp_group=self.pg_collection.cp,
        )

        if self.rotate:
            kv = rotate_activation(kv)

        nvtx_range_pop("compressor")
        return kv  # [n_compressed, b, head_dim]


# ---------------------------------------------------------------------------
# CSAIndexer
# ---------------------------------------------------------------------------

class CSAIndexerKernelFunction(torch.autograd.Function):
    """Fused FP8 indexer kernel for CSA, avoiding full [sq, sk] materialization.

    Adapted from DSAIndexerKernel in AIAK-Training-Omni. Uses deep_gemm.fp8_mqa_logits
    to compute scores in a memory-efficient manner with FP8 quantization.
    """

    quantizer = None  # Lazily initialized

    @staticmethod
    def _get_quantizer():
        if CSAIndexerKernelFunction.quantizer is None:
            CSAIndexerKernelFunction.quantizer = Float8BlockQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,
                amax_epsilon=1e-12,
                force_pow_2_scales=True,
                block_scaling_dim=1,
            )
        return CSAIndexerKernelFunction.quantizer

    @staticmethod
    def forward(
        ctx,
        index_q: torch.Tensor,  # [sq_local, b, h, d]
        index_k: torch.Tensor,  # [sk, b, d]  (full sk, not SP-sliced)
        weights: torch.Tensor,  # [sq_local, b, h]
        index_topk: int,
        compress_ratio: int,
        sq_offset: int = 0,  # global sq position offset for this rank's slice
    ):
        """FP8 fused indexer forward for CSA compressed keys.

        When SP is enabled, ``index_q`` / ``weights`` are SP-sliced along sq
        and ``sq_offset`` gives the global starting position of the slice so
        the causal-mask range over compressed keys stays correct.
        """
        sq, bsz, head, dim = index_q.size()
        sk = index_k.size(0)
        assert bsz == 1, "CSAIndexerKernel only supports batch_size=1"
        assert dim == 128, "Only support dim=128"
        device = index_q.device

        softmax_scale = dim ** -0.5

        # Squeeze batch dim for deep_gemm: [sq, h, d] and [sk, d]
        q = index_q.squeeze(1).contiguous()  # [sq, h, d]
        k = index_k.squeeze(1).contiguous()  # [sk, d]
        w = weights.squeeze(1).contiguous()  # [sq, h]

        quantizer = CSAIndexerKernelFunction._get_quantizer()
        quantized_q = quantizer.quantize(q)
        quantized_k = quantizer.quantize(k)
        q_fp8 = quantized_q.get_data_tensors(rowwise_data=True, columnwise_data=False).view(torch.float8_e4m3fn)
        k_fp8 = quantized_k.get_data_tensors(rowwise_data=True, columnwise_data=False).view(torch.float8_e4m3fn)
        q_scale = quantized_q._rowwise_scale_inv.reshape(q.shape[:-1])  # [sq, h]
        k_scale = quantized_k._rowwise_scale_inv.reshape(k.shape[:-1])  # [sk]

        # Causal mask for compressed keys: k_end[i_global] = (i_global + 1) // compress_ratio.
        # When SP-sliced, query position i in this rank corresponds to global i + sq_offset.
        k_start = torch.zeros(sq, dtype=torch.int, device=device)
        k_end = (
            torch.arange(sq, dtype=torch.int, device=device) + sq_offset + 1
        ) // compress_ratio
        # Clamp k_end to at least 1 to avoid empty ranges for early positions
        k_end = k_end.clamp(min=1)

        weight_scaled = w * q_scale * softmax_scale
        index_score = deep_gemm.fp8_mqa_logits(
            q_fp8, (k_fp8, k_scale), weight_scaled, k_start, k_end
        )

        # Top-k
        effective_topk = min(index_topk, sk)
        if effective_topk <= index_score.size(-1):
            index_score_topk, topk_indices = flashinfer.top_k(
                index_score.contiguous(), effective_topk, sorted=True
            )
        else:
            index_score_topk, topk_indices = index_score.topk(effective_topk, dim=-1)

        ctx.softmax_scale = softmax_scale
        ctx.index_topk = effective_topk
        ctx.save_for_backward(q_fp8, k_fp8, q_scale, k_scale, weight_scaled, topk_indices, k_start, k_end)

        return index_score_topk, topk_indices

    @staticmethod
    def backward(ctx, grad_score, grad_topk):
        """FP8 fused indexer backward using lightning_indexer_bwd."""
        q_fp8, k_fp8, q_scale, k_scale, weight_scaled, topk_indices, ks, ke = ctx.saved_tensors

        d_q, d_k, d_weights = lightning_indexer_bwd.fp8_mqa_logits_bwd(
            grad_score.contiguous(),
            q_fp8,
            (k_fp8, k_scale),
            weight_scaled,
            ks,
            ke,
            topk_indices=topk_indices.int(),
            topk=ctx.index_topk,
        )

        d_weights = d_weights * q_scale * ctx.softmax_scale
        d_q = d_q / q_scale.unsqueeze(-1)
        d_k = d_k / k_scale.unsqueeze(-1)

        # Unsqueeze batch dim back: [sq, 1, h, d], [sk, 1, d], [sq, 1, h]
        # Last 3 None: index_topk, compress_ratio, sq_offset (no grad)
        return d_q.unsqueeze(1), d_k.unsqueeze(1), d_weights.unsqueeze(1), None, None, None

class CSAIndexerKernel(torch.nn.Module):
    """Wrapper for fused FP8 indexer kernel for CSA."""

    def __init__(self, compress_ratio: int):
        super().__init__()
        self.compress_ratio = compress_ratio

    def forward(self, index_q, index_k, weights, index_topk, sq_offset=0):
        """Forward function."""
        return CSAIndexerKernelFunction.apply(
            index_q, index_k, weights, index_topk, self.compress_ratio, sq_offset
        )

@dataclass
class CSAIndexerSubmodules:
    """Submodule specs for CSAIndexer."""

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None
    compressor: Union[ModuleSpec, type] = None


class CSAIndexer(MegatronModule):
    """Learned top-k retrieval over compressed positions for CSA sparse attention.

    Computes index scores to select the most relevant compressed KV positions for each
    query.  Reuses the scoring logic from ``DSAIndexer`` (einsum -> relu -> weight -> sum
    -> topk) and ``rotate_activation`` (Hadamard transform) from ``dsa.py``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CSAIndexerSubmodules,
        compress_ratio: int,
        rotary_pos_emb: nn.Module = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize CSA indexer projections and its rotated compressor."""
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        self.compress_ratio = compress_ratio
        self.hidden_size = config.hidden_size
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim
        self.q_lora_rank = (
            config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size
        )

        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        self.rotary_pos_emb = rotary_pos_emb

        # Q projection
        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        # FIX: TE Linear.reset_parameters unconditionally calls
        # set_tensor_model_parallel_attributes(is_parallel=True, dim=0) regardless of
        # parallel_mode="duplicated", which makes distrib_optimizer treat this duplicated
        # weight as a TP-shard and allocate only 1/tp_size of the main_grad buffer.
        # Backward writes full-size grads into a 1/8 buffer => address-wrap accumulation,
        # producing the indexer.linear_wq_b grad explosion. Force replicated attrs.
        for param in self.linear_wq_b.parameters():
            if config.tensor_model_parallel_size > 1:
                setattr(param, 'tensor_model_parallel', False)
                setattr(param, 'partition_dim', -1)
                setattr(param, 'partition_stride', 1)

        # Weights projection
        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        # FIX: same as linear_wq_b above.
        for param in self.linear_weights_proj.parameters():
            if config.tensor_model_parallel_size > 1:
                setattr(param, 'tensor_model_parallel', False)
                setattr(param, 'partition_dim', -1)
                setattr(param, 'partition_stride', 1)

        # Own compressor (smaller head_dim, with Hadamard rotation)
        self.compressor = build_module(
            submodules.compressor,
            config=config,
            compress_ratio=compress_ratio,
            head_dim=self.index_head_dim,
            rotate=True,
            rotary_pos_emb=rotary_pos_emb,
            pg_collection=pg_collection,
        )

        self.use_fused_indexer = getattr(config, 'use_fused_indexer', True)
        if self.use_fused_indexer:
            self.indexer_kernel = CSAIndexerKernel(compress_ratio)

    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q, compressed K, and weights before top-k selection."""
        nvtx_range_push("indexer_before_topk")

        sq, bsz, _ = x.size()


        # Pad seq dim to multiple of 8 for FP8 Linear compatibility (packing may produce unaligned seq)
        pad_len = (128 - sq % 128) % 128 if self.config.fp8 else 0
        if pad_len > 0:
            qr = torch.nn.functional.pad(qr, (0, 0, 0, 0, 0, pad_len))
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_len))

        # Q path
        q, _ = self.linear_wq_b(qr)  # [sq_padded, b, n_heads * head_dim]
        if pad_len > 0:
            q = q[:sq]
        q = q.reshape(sq, bsz, self.index_n_heads, self.index_head_dim)
        q = _apply_rope(
            q,
            self.index_head_dim - self.qk_pos_emb_head_dim,
            self.qk_pos_emb_head_dim,
            self.rotary_pos_emb,
            self.config,
            sq,
            ratio=1,
            cp_group=self.pg_collection.cp,
        )
        q = rotate_activation(q)

        # K path: own compressor
        k = self.compressor(x)  # [sq_padded//ratio, b, index_head_dim]
        # CSAIndexerKernelFunction always FP8-quantizes k; its scale has 4-element
        # granularity along sk for forward, but lightning_indexer_bwd.fp8_mqa_logits_bwd
        # asserts seq_len_kv % 128 == 0. Align k length to multiple of 128
        # to satisfy both. Out-of-range padded keys are masked by k_end causal mask.
        target_n = sq // self.compress_ratio
        target_n_aligned = ((target_n + 127) // 128) * 128
        if k.size(0) >= target_n_aligned:
            k = k[:target_n_aligned]
        else:
            _k_pad = target_n_aligned - k.size(0)
            k = torch.cat([k, k.new_zeros((_k_pad, *k.shape[1:]))], dim=0)

        weights, _ = self.linear_weights_proj(x)  # [sq_padded, b, n_heads]
        if pad_len > 0:
            weights = weights[:sq]
        weights = weights * (self.index_n_heads**-0.5)


        nvtx_range_pop("indexer_before_topk")
        return q, k, weights

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (index_scores, topk_indices)."""
        nvtx_range_push("indexer")
        assert packed_seq_params is None, "Packed sequence not supported for CSAIndexer"
        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)
        nvtx_range_push("indexer_qk_topk")
        effective_topk = min(self.index_topk, k.size(0))
        if self.use_fused_indexer:
            # SP slicing for fused indexer to keep TE blockwise FP8 quantize
            # within CUDA grid limits at long seqlen.
            tp_size = get_pg_size(self.pg_collection.tp)
            use_sp = self.config.sequence_parallel and tp_size > 1
            if use_sp:
                # PATCH 4 TEMP: call indexer twice on local halves (no NCCL collectives).
                # Each rank does full work; sq is split into 2 local chunks to stay within
                # CUDA grid limits in TE blockwise FP8 quantize. Used to test whether
                # Patch 4's TP scatter/gather collectives are the grad-sync hang trigger.
                sq_full = q.size(0)
                half = sq_full // 2
                # FIX: lightning_indexer_bwd requires sq % 128 == 0.
                # If splitting would produce non-aligned halves, use single call.
                if half % 128 != 0 or (sq_full - half) % 128 != 0:
                    index_scores_topk, topk_indices = self.indexer_kernel(
                        q, k, weights, effective_topk, 0
                    )
                else:
                    s1, t1 = self.indexer_kernel(q[:half], k, weights[:half], effective_topk, 0)
                    s2, t2 = self.indexer_kernel(q[half:], k, weights[half:], effective_topk, half)
                    index_scores_topk = torch.cat([s1, s2], dim=0)
                    topk_indices = torch.cat([t1, t2], dim=0)
            else:
                index_scores_topk, topk_indices = self.indexer_kernel(
                    q, k, weights, effective_topk
                )
            nvtx_range_pop("indexer_qk_topk")
            nvtx_range_pop("indexer")
            return index_scores_topk, topk_indices
        else:
            index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, effective_topk, mask)
            nvtx_range_pop("indexer_qk_topk")
            nvtx_range_pop("indexer")
            return index_scores, topk_indices


# ---------------------------------------------------------------------------
# CompressedSparseAttention (core attention)
# ---------------------------------------------------------------------------


@dataclass
class CompressedSparseAttentionSubmodules:
    """Submodule specs for CompressedSparseAttention."""

    compressor: Union[ModuleSpec, type] = None
    indexer: Union[ModuleSpec, type] = None


class CompressedSparseAttention(MegatronModule):
    """Sparse core attention for CompressedSparseAttention.

    Combines sliding window attention with compressed KV attention.  The spec always
    provides compressor and indexer submodule specs; this ``__init__`` inspects
    ``config.csa_compress_ratios[layer_idx]`` and conditionally builds them:

    * ``ratio == 0``:  window-only (compressor and indexer NOT built)
    * ``ratio == 4``:  window + 4x compressed + learned Indexer (both built)
    * ``ratio == 128``: window + 128x compressed, attend to all (compressor built only)
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CompressedSparseAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: Optional[ProcessGroupCollection] = None,
        rotary_pos_emb: nn.Module = None,
        compress_ratio: int = 0,
        is_mtp_layer: bool = False,
    ):
        """Initialize compressed sparse attention kernels and optional indexer."""
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        self.layer_number = layer_number
        
        self.compress_ratio = compress_ratio
        self.window_size = config.csa_window_size
        self.v_head_dim = config.v_head_dim

        self.n_local_heads = divide(config.num_attention_heads, get_pg_size(self.pg_collection.tp))

        if softmax_scale is None:
            softmax_scale = config.v_head_dim**-0.5
        self.softmax_scale = softmax_scale

        self.force_unfused_dsa = getattr(config, "force_unfused_dsa", False)

        # Learnable attention sink per head
        self.attn_sink = nn.Parameter(torch.zeros(self.n_local_heads, dtype=torch.float32))
        if config.tensor_model_parallel_size > 1:
            setattr(self.attn_sink, "tensor_model_parallel", True)

        # Conditionally build Compressor (ratio > 1)
        if self.compress_ratio > 1 and submodules.compressor is not None:
            self.compressor = build_module(
                submodules.compressor,
                config=config,
                compress_ratio=self.compress_ratio,
                head_dim=config.v_head_dim,
                rotate=False,
                rotary_pos_emb=rotary_pos_emb,
                pg_collection=pg_collection,
            )
        else:
            self.compressor = None

        # Conditionally build Indexer (ratio == 4)
        if (
            self.compress_ratio == 4
            and not config.csa_dense_mode
            and submodules.indexer is not None
        ):
            self.indexer = build_module(
                submodules.indexer,
                config=config,
                compress_ratio=self.compress_ratio,
                rotary_pos_emb=rotary_pos_emb,
                pg_collection=pg_collection,
            )
            # DIAG: tag layer_number on indexer for forward diag prints
            self.indexer.layer_number = self.layer_number
        else:
            self.indexer = None


        if not self.force_unfused_dsa:
            try:
                from loongforge.models.common.experimental_attention_variant.dsa_fused_kernels import (
                    DSADotProductAttention as _DSAKernel,
                )
            except ImportError:
                raise ImportError(
                    "Fused sparse attention requires DSADotProductAttention kernel. "
                    "Set force_unfused_dsa=True in config or install the required kernel."
                )
            self.sparse_attention = _DSAKernel(config=config, softmax_scale=self.softmax_scale)


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        x: torch.Tensor = None,
        qr: torch.Tensor = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ) -> torch.Tensor:
        """Forward pass for CompressedSparseAttention.

        Args:
            query:  [sq, b, np, v_head_dim]
            key:    [sq, b, 1, v_head_dim]  (single-head MQA; head dim squeezed internally)
            value:  unused (key == value in MQA)
            attention_mask: attention mask (may be None for causal).
            x:      [sq, b, hidden_size]  original hidden states.
            qr:     [sq, b, q_lora_rank]  compressed query representation.

        Returns:
            output: [sq, b, np * v_head_dim]
        """
        nvtx_range_push("compressed_sparse_attn")
        if packed_seq_params is not None:
            assert packed_seq_params.qkv_format == "thd"
            cu_seqlens = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
            # Indexer-loss fix for packed: collect per-sub-seq indexer_loss instead of
            # letting each sub-seq independently call DSAIndexerLossAutoScaler.apply.
            # Without this, N sub-seqs inject N distinct backward seeds (each with local
            # /sq_local normalization) -> grad_norm blows up by ~N*(packed_total/L).
            # We capture them here and re-attach once after cat() with mean-aggregation,
            # which is exactly equivalent to a single global /(N*L) = /packed_total norm.
            captured_indexer_losses = []
            self._capture_indexer_loss_list = captured_indexer_losses
            try:
                outputs = []
                for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
                    if end <= start:
                        continue
                    query_i = query[start:end]
                    key_i = key[start:end]
                    value_i = value[start:end]
                    x_i = x[start:end] if x is not None else None
                    qr_i = qr[start:end] if qr is not None else None

                    if query_i.dim() == 3:
                        query_i = query_i.unsqueeze(1)
                    if key_i.dim() == 3:
                        key_i = key_i.unsqueeze(1)
                    if value_i.dim() == 3:
                        value_i = value_i.unsqueeze(1)
                    if x_i is not None and x_i.dim() == 2:
                        x_i = x_i.unsqueeze(1)
                    if qr_i is not None and qr_i.dim() == 2:
                        qr_i = qr_i.unsqueeze(1)

                    output_i = self.forward(
                        query_i,
                        key_i,
                        value_i,
                        attention_mask=None,
                        x=x_i,
                        qr=qr_i,
                        attn_mask_type=attn_mask_type,
                        attention_bias=attention_bias,
                        packed_seq_params=None,
                    )
                    outputs.append(output_i.squeeze(1))
            finally:
                self._capture_indexer_loss_list = None

            output = torch.cat(outputs, dim=0)
            if captured_indexer_losses and self.training and torch.is_grad_enabled():
                # mean over sub-seqs:
                #   avg_i(coeff * KL_sum_i / sq_local_i)
                # = (coeff * sum_i KL_sum_i) / (N * sq_local_i_avg)
                # When sum_i sq_local_i == packed_total_sq, this is identical to the
                # unpacked normalization coeff * KL_total / packed_total_sq, modulo
                # already-done TP allreduce per sub-seq (which is fine, the reduce is linear).
                aggregated_indexer_loss = sum(captured_indexer_losses) / len(captured_indexer_losses)
                output = DSAIndexerLossAutoScaler.apply(output, aggregated_indexer_loss)
            nvtx_range_pop("compressed_sparse_attn")
            return output

        sq, b, np, hn = query.size()

        # --- Step 1: Prepare single-head KV (squeeze singleton head dim) ---
        kv = key.squeeze(-2)  # [sq, b, 1, v_head_dim] -> [sq, b, v_head_dim]

        # --- Step 2: Compression ---
        # Skip compression when sq < window_size: window already covers all positions
        # (full attention contained in window), compressed path would only dilute softmax
        # by splitting probability mass across full-res window and low-res compressed views.
        if (self.compressor is not None and self.compress_ratio > 1
                and sq >= self.window_size):
            compressed_kv = self.compressor(x)  # [n_compressed, b, v_head_dim]
            if compressed_kv is not None:
                kv_full = torch.cat([kv, compressed_kv], dim=0)
                n_compressed = compressed_kv.size(0)
            else:
                kv_full = kv
                n_compressed = 0
        else:
            kv_full = kv
            n_compressed = 0

        offset = sq  # compressed indices start after original positions

        # --- Step 3: Window indices ---
        window_idxs = get_window_topk_idxs(self.window_size, b, sq, query.device)

        # --- Step 4: Compressed indices ---
        indexer_loss = None

        if self.force_unfused_dsa:
            if self.compress_ratio > 1 and n_compressed > 0:
                nvtx_range_push("compressed_indices")
                if self.indexer is not None:
                    x_det = x.detach()
                    qr_det = qr.detach()

                    causal_mask = (
                        torch.arange(n_compressed, device=x.device).unsqueeze(0).expand(sq, -1)
                    )
                    positions = torch.arange(1, sq + 1, device=x.device).unsqueeze(1)
                    causal_mask = (
                        torch.where(
                            causal_mask >= positions // self.compress_ratio, float("-inf"), 0.0
                        )
                        .unsqueeze(0)
                        .expand(b, -1, -1)
                    )  # [b, sq, n_compressed]

                    indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', None)
                    if self.training and torch.is_grad_enabled() and indexer_loss_coeff is not None:
                        q_indexer, k_indexer, weights_indexer = self.indexer.forward_before_topk(
                            x_det, qr_det, packed_seq_params
                        )
                        # compressed_kv is [n, b, hn]; expand to [n, b, np, hn] for loss
                        key_for_loss = compressed_kv.unsqueeze(2).expand(-1, -1, np, -1)
                        # ``FusedDSAIndexerLoss`` does not accept a separate
                        # indexer_softmax_scale; apply it here via the
                        # weights-scaling trick so the effective weights match
                        # the pre-scale-split behaviour.
                        weights_for_unfused = weights_indexer * self.indexer.softmax_scale
                        topk_indices_compressed, indexer_loss = FusedDSAIndexerLoss.apply(
                            q_indexer,
                            weights_for_unfused,
                            k_indexer,
                            query.detach(),
                            key_for_loss.detach(),
                            self.softmax_scale,
                            min(self.indexer.index_topk, n_compressed),
                            indexer_loss_coeff,
                            causal_mask,
                            getattr(self.config, "dsa_indexer_use_sparse_loss", True),
                            self.indexer.pg_collection,
                        )
                        if indexer_loss_coeff > 0:
                            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                                loss=indexer_loss,
                                layer_number=self.layer_number,
                                num_layers=self.config.num_layers
                                + (self.config.mtp_num_layers or 0),
                            )
                    else:
                        _, topk_indices_compressed = self.indexer(
                            x_det, qr_det, mask=causal_mask, packed_seq_params=packed_seq_params
                        )

                    # Normalize topk_indices_compressed to 3D [b, sq, topk]:
                    # FusedDSAIndexerLoss / fused-indexer return 2D [sq, topk];
                    # naive indexer returns 3D [b, sq, topk]. Below cat() with
                    # 3D window_idxs requires 3D, so unsqueeze when needed.
                    if topk_indices_compressed.dim() == 2:
                        topk_indices_compressed = topk_indices_compressed.unsqueeze(0)

                    n_valid_per_pos = positions // self.compress_ratio  # [sq, 1]
                    valid = topk_indices_compressed < n_valid_per_pos
                    compress_topk_idxs = torch.where(
                        valid, topk_indices_compressed + offset, torch.tensor(-1, device=x.device)
                    )
                else:
                    compress_topk_idxs = get_compress_topk_idxs(
                        self.compress_ratio, b, sq, offset, query.device
                    )

                topk_idxs = torch.cat([window_idxs, compress_topk_idxs], dim=-1)
                nvtx_range_pop("compressed_indices")
            else:
                topk_idxs = window_idxs

            topk_idxs = topk_idxs.int()

            # --- Step 5: Sparse attention ---
            nvtx_range_push("sparse_attn_kernel")
            output = unfused_compressed_sparse_attn(
                query, kv_full, self.attn_sink.float(), topk_idxs, self.softmax_scale
            )
            nvtx_range_pop("sparse_attn_kernel")

        else:
            tp_group = self.pg_collection.tp
            tp_size = get_pg_size(tp_group)
            tp_rank = tp_group.rank() if tp_size > 1 else 0

            chunk_query = _AllToAllHp2Sp.apply(query, tp_group)

            chunk_sq = chunk_query.size(0)
            chunk_offset = kv_full.size(0)

            chunk_start = tp_rank * chunk_sq
            chunk_end = chunk_start + chunk_sq

            if self.compress_ratio > 1 and n_compressed > 0:
                if self.indexer is not None:
                    x_det = x.detach()
                    qr_det = qr.detach()

                    causal_mask = (
                        torch.arange(n_compressed, device=x.device).unsqueeze(0).expand(sq, -1)
                    )
                    positions = torch.arange(1, sq + 1, device=x.device).unsqueeze(1)
                    causal_mask = (
                        torch.where(
                            causal_mask >= positions // self.compress_ratio, float("-inf"), 0.0
                        )
                        .unsqueeze(0)
                        .expand(b, -1, -1)
                    )  # [b, sq, n_compressed]

                    index_scores, topk_indices_compressed = self.indexer(
                        x_det, qr_det, mask=causal_mask, packed_seq_params=packed_seq_params
                    )

                    # Fused kernel returns [sq, topk]; naive returns [b, sq, topk]
                    # Normalize to [b, sq, topk] for downstream code
                    if topk_indices_compressed.dim() == 2:
                        topk_indices_compressed = topk_indices_compressed.unsqueeze(0)
                        index_scores = index_scores.unsqueeze(0)

                    n_valid_per_pos = positions // self.compress_ratio  # [sq, 1]
                    valid = topk_indices_compressed < n_valid_per_pos
                    compress_topk_idxs = torch.where(
                        valid, topk_indices_compressed + offset, torch.tensor(-1, device=x.device)
                    )
                else:
                    compress_topk_idxs = get_compress_topk_idxs(
                        self.compress_ratio, b, sq, offset, query.device
                    )

                topk_idxs = torch.cat([window_idxs, compress_topk_idxs], dim=-1).int()
                chunk_topk_idxs = topk_idxs[:, chunk_start:chunk_end, :]
                _csa_pad = chunk_sq - chunk_topk_idxs.shape[1]
                if _csa_pad > 0:  # SP padding fix: pad truncated tail with -1 sentinel
                    chunk_topk_idxs = F.pad(chunk_topk_idxs, (0, 0, 0, _csa_pad), value=-1)
                chunk_topk_idxs = chunk_topk_idxs.contiguous()


                output, p_out = self.sparse_attention(
                    chunk_query,
                    kv_full,
                    chunk_topk_idxs,
                    chunk_offset,
                    attn_mask_type=AttnMaskType.causal,
                    return_p_out=True,
                    window_size=128,
                    attn_sink=self.attn_sink.float(),
                )

                indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', None)
                if (self.indexer is not None and self.training and torch.is_grad_enabled()
                        and indexer_loss_coeff is not None and indexer_loss_coeff > 0):
                    from loongforge.models.common.experimental_attention_variant.dsa_fused_kernels import (
                        triton_attn_dist,
                    )
                    import triton

                    row_valid = (positions.squeeze(-1) >= self.compress_ratio)  # [sq]
                    chunk_row_valid = row_valid[chunk_start:chunk_end]  # [chunk_sq] (may be short under SP-pad)
                    _csa_indexer_pad = chunk_sq - chunk_row_valid.size(0)
                    if _csa_indexer_pad > 0:
                        chunk_row_valid = F.pad(chunk_row_valid, (0, _csa_indexer_pad), value=False)

                    with torch.no_grad():
                        # p_out: [chunk_sq, h_q, p_out_topk] -> main_attn_probs: [chunk_sq, p_out_topk]
                        p_out_topk = p_out.size(-1)
                        p_out_for_dist = p_out
                        # triton_attn_dist requires topk to be power of 2; pad if needed
                        next_pow2 = triton.next_power_of_2(p_out_topk)
                        if next_pow2 != p_out_topk:
                            p_out_for_dist = F.pad(p_out, (0, next_pow2 - p_out_topk), value=float('-inf'))
                        main_attn_probs = triton_attn_dist(p_out_for_dist, self.softmax_scale)
                        if next_pow2 != p_out_topk:
                            main_attn_probs = main_attn_probs[:, :p_out_topk]
                        main_attn_probs = main_attn_probs.masked_fill(
                            ~chunk_row_valid.unsqueeze(-1), 0.0
                        )

                    index_scores_topk = (
                        index_scores
                        if self.indexer.use_fused_indexer
                        else index_scores.gather(-1, topk_indices_compressed)
                    )

                    if not self.indexer.use_fused_indexer:
                        index_scores_topk = index_scores_topk * self.indexer.softmax_scale
                    chunk_index_scores = index_scores_topk[:, chunk_start:chunk_end, :]
                    if _csa_indexer_pad > 0:  # SP padding fix: align sq dim with chunk_row_valid
                        chunk_index_scores = F.pad(chunk_index_scores, (0, 0, 0, _csa_indexer_pad), value=0.0)
                    chunk_index_scores = chunk_index_scores.squeeze(0)
                    chunk_index_scores = chunk_index_scores.masked_fill(
                        ~chunk_row_valid.unsqueeze(-1), 0.0
                    )
                    index_attn_probs = torch.softmax(chunk_index_scores, dim=-1)
                    index_attn_probs = index_attn_probs.masked_fill(
                        ~chunk_row_valid.unsqueeze(-1), 0.0
                    )

                    # flash_mla_sparse_fwd p_out: [s_q, h_q, topk_aligned64 - window_size].
                    # The last (topk_aligned64 - chunk_topk_idxs.size(-1)) entries are
                    # 64-alignment padding (-1), and the trailing portion of the
                    # compress-keys are dropped beyond effective_topk. So the first
                    # effective_topk entries of main_attn_probs correspond 1:1 with
                    # compress_topk_idxs == index_attn_probs's keyset.
                    if main_attn_probs.size(-1) > index_attn_probs.size(-1):
                        main_attn_probs = main_attn_probs[:, :index_attn_probs.size(-1)]

                    indexer_loss = F.kl_div(
                        (index_attn_probs + 1e-10).log(),
                        main_attn_probs + 1e-10,
                        reduction="sum",
                    )
                    indexer_loss = indexer_loss_coeff * indexer_loss / sq
                    indexer_loss = reduce_from_tensor_model_parallel_region(
                        indexer_loss, group=tp_group
                    )

                    DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                        loss=indexer_loss,
                        layer_number=self.layer_number,
                        num_layers=self.config.num_layers
                        + (self.config.mtp_num_layers or 0),
                    )

                nvtx_range_pop("compressed_indices")

            else:
                topk_idxs = window_idxs.int()
                chunk_topk_idxs = topk_idxs[:, chunk_start:chunk_end, :]
                _csa_pad = chunk_sq - chunk_topk_idxs.shape[1]
                if _csa_pad > 0:  # SP padding fix: pad truncated tail with -1 sentinel
                    chunk_topk_idxs = F.pad(chunk_topk_idxs, (0, 0, 0, _csa_pad), value=-1)
                chunk_topk_idxs = chunk_topk_idxs.contiguous()

                output = self.sparse_attention(
                    chunk_query,
                    kv_full,
                    chunk_topk_idxs,
                    chunk_offset,
                    attn_mask_type=AttnMaskType.causal,
                    return_p_out=False,
                    window_size=0,
                    attn_sink=self.attn_sink.float(),
                )
                output = output.unsqueeze(0)

            output = _AllToAllSp2Hp.apply(output, tp_group)
            output = output[:, :sq, :].contiguous()  # SP padding fix: trim padded seq dim before reshape
            output = output.reshape(sq, b, -1)

        # --- Step 6: Attach indexer loss ---
        if indexer_loss is not None and self.training and torch.is_grad_enabled():
            # When inside a packed-sub-seq recursion, defer the attach: append
            # to the parent-collected list and let the outer packed dispatcher
            # do a single aggregated attach. See packed branch above for why.
            capture_list = getattr(self, "_capture_indexer_loss_list", None)
            if capture_list is not None:
                capture_list.append(indexer_loss)
            else:
                output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        nvtx_range_pop("compressed_sparse_attn")
        return output
