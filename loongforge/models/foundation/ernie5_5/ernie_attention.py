# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Adapted from NVIDIA-NeMo/Megatron-Bridge at 8f580c64376877e9e2bc3140ce3476afc0f75d76.
# Sparse attention and indexer KL follow NVIDIA/Megatron-LM at f78809f6d9c88f8638ea7822966f35057a63fd5c.

"""HCA and latent MQA for ERNIE 5.5."""

from copy import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.models.common.embeddings import RotaryEmbedding, apply_rotary_pos_emb
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.dsa import rotate_activation
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    _compute_index_scores,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module

from loongforge.models.foundation.deepseek_v4.deepseek_v4_attention import DSv4HybridSelfAttention
from loongforge.models.foundation.deepseek_v4.deepseek_v4_csa import (
    CompressedSparseAttention,
    get_compress_topk_idxs,
    get_window_topk_idxs,
)
from loongforge.models.foundation.deepseek_v4.deepseek_v4_rope import apply_dsv4_rotary_pos_emb

from .ernie_ops import Ernie55Norm, vha_postmix


def _packed_position_ids(cu_seqlens, global_positions):
    """Return within-document positions for global packed row indices."""
    document_ids = torch.bucketize(global_positions, cu_seqlens[1:], right=True)
    document_ids = document_ids.clamp_max(cu_seqlens.numel() - 2)
    return global_positions - cu_seqlens.index_select(0, document_ids)


def _packed_freqs(rotary_pos_emb, cu_seqlens, global_positions):
    position_ids = _packed_position_ids(cu_seqlens, global_positions)
    return rotary_pos_emb.index_select(0, position_ids.long())


def _packed_window_indices(query_positions, cu_seqlens, window_size):
    """Build global causal window indices for a packed query slice."""
    document_ids = torch.bucketize(query_positions, cu_seqlens[1:], right=True)
    document_ids = document_ids.clamp_max(cu_seqlens.numel() - 2)
    starts = cu_seqlens.index_select(0, document_ids)
    offsets = torch.arange(window_size, device=query_positions.device)
    indices = (query_positions - window_size + 1).clamp_min(0).unsqueeze(1) + offsets
    valid = (indices >= starts.unsqueeze(1)) & (indices <= query_positions.unsqueeze(1))
    return indices.masked_fill(~valid, -1)


def ernie55_sparse_attention(query, key, sink, indices, scale):
    """Sparse MQA with a denominator-only sink and flat KV gradient accumulation."""
    seq, batch, heads, width = query.shape
    keys = key.shape[0]
    query_rows = query.permute(1, 0, 2, 3).reshape(batch * seq, heads, width)
    key_rows = key.permute(1, 0, 2).reshape(batch * keys, width)
    offsets = torch.arange(batch, device=query.device).view(batch, 1, 1) * keys
    indices = torch.where(indices >= 0, indices + offsets, indices).reshape(batch * seq, -1)
    selected = key_rows.index_select(0, indices.clamp(min=0).long().flatten())
    selected = selected.view(batch * seq, -1, width).float()
    scores = torch.einsum("inh,ikh->ink", query_rows.float(), selected) * scale
    scores = scores.masked_fill((indices < 0).unsqueeze(1), float("-inf"))
    sink = sink.view(1, heads, 1).float()
    maximum = torch.maximum(scores.max(-1, keepdim=True).values, sink)
    weights = (scores - maximum).exp()
    weights = weights / (weights.sum(-1, keepdim=True) + (sink - maximum).exp())
    output = torch.einsum("ink,ikh->inh", weights, selected).to(query.dtype)
    return output.reshape(batch, seq, heads * width).permute(1, 0, 2).contiguous()


def ernie55_indexer_loss(scores, topk, query, key, remote, scale, coefficient, per_token):
    """Compute sparse KL in log space, with zero loss for rows without remote keys."""
    seq, batch, heads, width = query.shape
    valid = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, topk, True)
    valid = valid & remote.unsqueeze(0)
    with torch.no_grad():
        q = query.permute(1, 2, 0, 3).reshape(batch * heads, seq, width).float()
        k = key.expand(-1, -1, heads, -1).permute(1, 2, 3, 0)
        k = k.reshape(batch * heads, width, key.shape[0]).float()
        teacher = (torch.bmm(q, k) * scale).view(batch, heads, seq, -1)
        teacher_valid = valid.unsqueeze(1).expand_as(teacher)
        teacher = teacher.masked_fill(~teacher_valid, torch.finfo(torch.float32).min)
        row_max = teacher.max(-1, keepdim=True).values
        row_max = torch.where(teacher_valid.any(-1, keepdim=True), row_max, torch.zeros_like(row_max))
        target = (teacher - row_max).exp().masked_fill(~teacher_valid, 0.0)
        target = target / target.sum(-1, keepdim=True).clamp_min(1e-10)
        target = target.masked_fill(~teacher_valid, 0.0).sum(1)
        target = target / target.sum(-1, keepdim=True).clamp_min(1e-10)
    logits = scores.float().masked_fill(~valid, torch.finfo(torch.float32).min)
    logits = torch.where(valid.any(-1, keepdim=True), logits, torch.zeros_like(logits))
    log_probs = logits.log_softmax(-1).masked_fill(~valid, 0.0)
    loss_per_row = (target * (target.clamp_min(1e-10).log() - log_probs)).sum(-1)
    loss = loss_per_row.sum()
    if not per_token:
        loss = loss / loss_per_row.numel()
    return loss * coefficient


def _indexer_loss_for_logging(loss, seq, batch, per_token):
    """Scale a raw per-token sum to the mean used in logs."""
    if not per_token:
        return loss
    return loss / max(seq * batch, 1)


class Ernie55HCore(CompressedSparseAttention):
    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        x=None,
        qr=None,
        packed_seq_params=None,
        boundary_hidden=None,
        boundary_kv=None,
        **kwargs,
    ):
        if packed_seq_params is not None:
            return super().forward(
                query,
                key,
                value,
                attention_mask,
                x=x,
                qr=qr,
                packed_seq_params=packed_seq_params,
                boundary_hidden=boundary_hidden,
                boundary_kv=boundary_kv,
                **kwargs,
            )
        seq, batch = query.shape[:2]
        keys = key.squeeze(2)
        indices = get_window_topk_idxs(self.window_size, batch, seq, query.device)
        if seq >= self.window_size:
            compressed = self.compressor(x)
            if compressed is not None:
                keys = torch.cat((keys, compressed), dim=0)
                compressed_indices = get_compress_topk_idxs(
                    self.compress_ratio,
                    batch,
                    seq,
                    seq,
                    query.device,
                )
                indices = torch.cat((indices, compressed_indices), dim=-1)
        return ernie55_sparse_attention(query, keys, self.attn_sink, indices, self.softmax_scale)


class Ernie55HAttention(DSv4HybridSelfAttention):
    def __init__(self, config, *args, **kwargs):
        h_config = copy(config)
        h_config.rope_type = config.hca_rope_type
        super().__init__(h_config, *args, **kwargs)
        self.linear_output_gate = TELinear(
            config.q_lora_rank,
            config.o_groups * config.o_lora_rank,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        core = self.core_attention
        core.vha_postmix_u = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, config.mha_vha_rank, dtype=torch.float32)
        )
        core.vha_postmix_v = torch.nn.Parameter(torch.zeros_like(core.vha_postmix_u))
        with torch.no_grad():
            config.init_method(core.vha_postmix_u)
            core.attn_sink.normal_(0.0, 0.1)
        for parameter in (core.attn_sink, core.vha_postmix_u, core.vha_postmix_v, core.compressor.ape):
            parameter.keep_in_fp32 = True

    def forward(self, hidden_states, attention_mask, packed_seq_params=None, **kwargs):
        boundary_hidden = None
        query, key, value, qr, kv, boundary_kv = self.get_query_key_value_tensors(
            hidden_states,
            packed_seq_params=packed_seq_params,
            boundary_hidden=boundary_hidden,
        )
        output = self.core_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            attention_mask,
            x=kv,
            qr=qr,
            packed_seq_params=packed_seq_params,
            boundary_hidden=boundary_hidden,
            boundary_kv=boundary_kv,
        )
        output = output.view(*hidden_states.shape[:2], self.config.num_attention_heads, self.config.v_head_dim)
        packed = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if packed:
            cu_seqlens = (
                packed_seq_params.cu_seqlens_kv_padded
                if packed_seq_params.cu_seqlens_kv_padded is not None
                else packed_seq_params.cu_seqlens_kv
            )
            rope_seqlen = (
                packed_seq_params.max_seqlen_kv
                if packed_seq_params.max_seqlen_kv is not None
                else int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            )
        else:
            cu_seqlens = None
            rope_seqlen = hidden_states.shape[0]
        if self._dsv4_effective_rope_type == "rope":
            rotary = self.rotary_pos_emb(rope_seqlen, packed_seq=packed)
        else:
            rotary, _ = self.rotary_pos_emb(rope_seqlen, packed_seq=packed)
        content, position = output.split(
            (self.config.v_head_dim - self.config.qk_pos_emb_head_dim, self.config.qk_pos_emb_head_dim),
            dim=-1,
        )
        if packed:
            position = position.squeeze(1)
            position = apply_dsv4_rotary_pos_emb(
                position,
                rotary,
                self.config,
                cu_seqlens=cu_seqlens,
                cp_group=self.pg_collection.cp,
                inverse=True,
            )
            position = position.unsqueeze(1)
        else:
            position = apply_dsv4_rotary_pos_emb(
                position,
                rotary,
                self.config,
                cp_group=self.pg_collection.cp,
                inverse=True,
            )
        output = vha_postmix(
            torch.cat((content, position), dim=-1),
            self.core_attention.vha_postmix_u,
            self.core_attention.vha_postmix_v,
        )
        output = output.reshape(*hidden_states.shape[:2], self.o_local_groups, -1)
        weight = self.linear_o_group_proj.view(self.o_local_groups, self.config.o_lora_rank, -1)
        output = torch.einsum("...gd,grd->...gr", output, weight).flatten(-2)
        gate, _ = self.linear_output_gate(qr)
        if gate.ndim == output.ndim - 1:
            gate = gate.unsqueeze(1)
        return self.linear_proj(output * gate.sigmoid())


class Ernie55IndexerNorm(torch.nn.LayerNorm):
    def __init__(self, config, hidden_size, eps):
        super().__init__(hidden_size, eps=eps, dtype=torch.float32)
        for parameter in self.parameters():
            parameter.keep_in_fp32 = True

    def forward(self, value):
        return F.layer_norm(value.float(), self.normalized_shape, self.weight, self.bias, self.eps).to(value.dtype)


@dataclass
class Ernie55DCoreSubmodules:
    indexer: object = None


class Ernie55DCore(MegatronModule):
    def __init__(
        self,
        config,
        submodules,
        layer_number,
        attn_mask_type=None,
        attention_type="self",
        attention_dropout=None,
        softmax_scale=None,
        k_channels=None,
        v_channels=None,
        cp_comm_type=None,
        pg_collection=None,
    ):
        super().__init__(config)
        self.layer_number = layer_number
        self.pg_collection = pg_collection
        self.softmax_scale = softmax_scale or (config.qk_head_dim + config.qk_pos_emb_head_dim) ** -0.5
        self.indexer = build_module(submodules.indexer, config=config, pg_collection=pg_collection)
        self.linear_output_gate = TELinear(
            config.q_lora_rank,
            config.num_attention_heads * config.v_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.attn_sink = torch.nn.Parameter(torch.zeros(config.num_attention_heads, dtype=torch.float32))
        self.vha_postmix_u = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, config.mha_vha_rank, dtype=torch.float32)
        )
        self.vha_postmix_v = torch.nn.Parameter(torch.zeros_like(self.vha_postmix_u))
        with torch.no_grad():
            config.init_method(self.vha_postmix_u)
        for parameter in (self.attn_sink, self.vha_postmix_u, self.vha_postmix_v):
            parameter.keep_in_fp32 = True

    def _indexer_tensors(self, hidden_states, qr, packed_seq_params):
        indexer = self.indexer
        if packed_seq_params is None:
            return indexer.forward_before_topk(hidden_states, qr)

        cu_seqlens = (
            packed_seq_params.cu_seqlens_q_padded
            if packed_seq_params.cu_seqlens_q_padded is not None
            else packed_seq_params.cu_seqlens_q
        )
        max_seqlen = (
            packed_seq_params.max_seqlen_q
            if packed_seq_params.max_seqlen_q is not None
            else int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        )
        rotary = indexer.rotary_pos_emb(max_seqlen, packed_seq=True)
        query_positions = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        query_freqs = _packed_freqs(rotary, cu_seqlens, query_positions.to(cu_seqlens.device))

        q, _ = indexer.linear_wq_b(qr)
        q = q.reshape(*hidden_states.shape[:2], indexer.index_n_heads, indexer.index_head_dim)
        q = indexer._apply_rope(q, query_freqs, 1.0)
        k, _ = indexer.linear_wk(hidden_states)
        k = indexer.k_norm(k)
        k = k.reshape(*hidden_states.shape[:2], 1, indexer.index_head_dim)
        k = indexer._apply_rope(k, query_freqs, 1.0).reshape(*hidden_states.shape[:2], indexer.index_head_dim)
        weights, _ = indexer.linear_weights_proj(hidden_states)
        weights = weights * (indexer.index_n_heads**-0.5) * indexer.softmax_scale
        if indexer.config.dsa_indexer_rotate_activation:
            q = rotate_activation(q)
            k = rotate_activation(k)
        return q, k, weights

    def forward(
        self,
        query,
        key,
        hidden_states,
        qr,
        *,
        packed_seq_params=None,
        query_positions=None,
        key_positions=None,
    ):
        seq, batch, heads, _ = query.shape
        cp = self.pg_collection.cp
        if packed_seq_params is None:
            query_positions = torch.arange(seq, device=query.device)
            key_positions = torch.arange(key.shape[0], device=key.device)
            remote = key_positions[None, :] <= query_positions[:, None] - self.config.csa_window_size
            window = get_window_topk_idxs(self.config.csa_window_size, batch, seq, query.device)
        else:
            cu_seqlens = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
            if query_positions is None or key_positions is None:
                query_positions = torch.arange(seq, device=query.device)
                key_positions = torch.arange(key.shape[0], device=key.device)
            document_ids = torch.bucketize(query_positions, cu_seqlens[1:], right=True)
            document_ids = document_ids.clamp_max(cu_seqlens.numel() - 2)
            starts = cu_seqlens.index_select(0, document_ids)
            ends = cu_seqlens.index_select(0, document_ids + 1)
            remote = (
                (key_positions.unsqueeze(0) >= starts.unsqueeze(1))
                & (key_positions.unsqueeze(0) < ends.unsqueeze(1))
                & (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1) - self.config.csa_window_size)
            )
            window = _packed_window_indices(query_positions, cu_seqlens, self.config.csa_window_size).unsqueeze(0)
            window = window.expand(batch, -1, -1)
        q_index, k_index, weights = self._indexer_tensors(
            hidden_states.detach(),
            qr.detach(),
            packed_seq_params,
        )
        weights = weights * self.config.dsa_indexer_head_dim**0.5
        scores = _compute_index_scores(q_index, weights, k_index)
        scores = scores.masked_fill(~remote.unsqueeze(0), float("-inf"))
        topk = scores.topk(min(key.shape[0], self.config.dsa_indexer_topk), dim=-1).indices
        valid = remote.unsqueeze(0).expand(batch, -1, -1).gather(-1, topk)
        remote_indices = topk.masked_fill(~valid, -1)
        remote_indices = F.pad(
            remote_indices,
            (0, self.config.dsa_indexer_topk - remote_indices.shape[-1]),
            value=-1,
        )
        indices = torch.cat((remote_indices, window), dim=-1)
        output = ernie55_sparse_attention(
            query,
            key.squeeze(2).float(),
            self.attn_sink,
            indices,
            self.softmax_scale,
        )
        output = output.view(seq, batch, heads, -1)[..., : self.config.kv_lora_rank]
        if self.training and self.config.dsa_indexer_loss_coeff:
            loss = ernie55_indexer_loss(
                scores * self.config.dsa_indexer_head_dim**-0.5,
                topk,
                query,
                key,
                remote,
                self.softmax_scale,
                self.config.dsa_indexer_loss_coeff,
                self.config.calculate_per_token_loss,
            )
            tracked_loss = _indexer_loss_for_logging(
                loss,
                seq,
                batch,
                self.config.calculate_per_token_loss,
            )
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                tracked_loss,
                self.layer_number,
                self.config.num_layers + (self.config.mtp_num_layers or 0),
                reduce_group=cp if self.config.calculate_per_token_loss else None,
                avg_group=cp if not self.config.calculate_per_token_loss else None,
            )
            output = DSAIndexerLossAutoScaler.apply(output, loss)
        return output


class Ernie55DAttention(MegatronModule):
    def __init__(self, config, submodules, layer_number, pg_collection=None, is_mtp_layer=False, **kwargs):
        d_config = copy(config)
        for name in (
            "num_attention_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_head_dim",
            "qk_pos_emb_head_dim",
            "v_head_dim",
        ):
            setattr(d_config, name, getattr(config, "d_" + name))
        d_config.num_query_groups = d_config.num_attention_heads
        d_config.rope_type = "rope"
        d_config.rotary_scaling_factor = 1.0
        d_config.mscale_all_dim = 0.0
        d_config.rotary_interleaved = False
        super().__init__(d_config)
        self.pg_collection = pg_collection or ProcessGroupCollection.use_mpu_process_groups()
        self.layer_number = layer_number
        for name, input_size, output_size in (
            ("linear_q_down_proj", d_config.hidden_size, d_config.q_lora_rank),
            (
                "linear_q_up_proj",
                d_config.q_lora_rank,
                d_config.num_attention_heads * (d_config.qk_head_dim + d_config.qk_pos_emb_head_dim),
            ),
            ("linear_kv_down_proj", d_config.hidden_size, d_config.kv_lora_rank + d_config.qk_pos_emb_head_dim),
            ("linear_proj", d_config.num_attention_heads * d_config.v_head_dim, d_config.hidden_size),
        ):
            setattr(
                self,
                name,
                TELinear(
                    input_size,
                    output_size,
                    config=d_config,
                    init_method=d_config.output_layer_init_method if name == "linear_proj" else d_config.init_method,
                    bias=False,
                    skip_bias_add=False,
                    skip_weight_param_allocation=False,
                    parallel_mode="duplicated",
                ),
            )
        self.k_b_proj = torch.nn.Parameter(
            torch.empty(
                d_config.num_attention_heads * d_config.kv_lora_rank,
                d_config.qk_head_dim,
                dtype=d_config.params_dtype,
            )
        )
        self.v_b_proj = torch.nn.Parameter(
            torch.empty(
                d_config.num_attention_heads * d_config.v_head_dim,
                d_config.kv_lora_rank,
                dtype=d_config.params_dtype,
            )
        )
        with torch.no_grad():
            d_config.init_method(self.k_b_proj)
            d_config.init_method(self.v_b_proj)
        self.q_layernorm = Ernie55Norm(d_config, d_config.q_lora_rank, d_config.layernorm_epsilon)
        self.kv_layernorm = Ernie55Norm(d_config, d_config.kv_lora_rank, d_config.layernorm_epsilon)
        self.rotary_pos_emb = RotaryEmbedding(
            d_config.qk_pos_emb_head_dim,
            rotary_percent=1.0,
            rotary_base=d_config.rotary_base,
            cp_group=self.pg_collection.cp,
        )
        self.core_attention = Ernie55DCore(
            config=d_config,
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=self.pg_collection,
        )

    def forward(self, hidden_states, attention_mask, packed_seq_params=None, **kwargs):
        config = self.config
        qr, _ = self.linear_q_down_proj(hidden_states)
        qr = self.q_layernorm(qr)
        query, _ = self.linear_q_up_proj(qr)
        query = query.view(*hidden_states.shape[:2], config.num_attention_heads, -1)
        content, position = query.split((config.qk_head_dim, config.qk_pos_emb_head_dim), dim=-1)
        k_weight = self.k_b_proj.view(config.num_attention_heads, config.kv_lora_rank, config.qk_head_dim)
        content = torch.einsum("...nd,ncd->...nc", content, k_weight)
        kv, _ = self.linear_kv_down_proj(hidden_states)
        latent, key_position = kv.split((config.kv_lora_rank, config.qk_pos_emb_head_dim), dim=-1)
        latent = self.kv_layernorm(latent).unsqueeze(2)
        packed = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if packed:
            cu_seqlens = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
            max_seqlen = (
                packed_seq_params.max_seqlen_q
                if packed_seq_params.max_seqlen_q is not None
                else int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            )
            rotary = self.rotary_pos_emb(max_seqlen, packed_seq=True)
            query_positions = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            position = apply_rotary_pos_emb(
                position,
                _packed_freqs(rotary, cu_seqlens, query_positions.to(cu_seqlens.device)),
                config=config,
            )
            key_position = apply_rotary_pos_emb(
                key_position.unsqueeze(2),
                _packed_freqs(rotary, cu_seqlens, query_positions.to(cu_seqlens.device)),
                config=config,
            )
        else:
            rotary = self.rotary_pos_emb(hidden_states.shape[0])
            position = apply_rotary_pos_emb(
                position,
                rotary,
                config=config,
                cp_group=self.pg_collection.cp,
                mla_rotary_interleaved=True,
            )
            key_position = apply_rotary_pos_emb(
                key_position.unsqueeze(2),
                rotary,
                config=config,
                cp_group=self.pg_collection.cp,
                mla_rotary_interleaved=True,
            )
        query = torch.cat((content, position), dim=-1)
        key = torch.cat((latent, key_position), dim=-1)
        key_positions = torch.arange(key.shape[0], device=key.device) if packed else None
        core = self.core_attention
        output = core(
            query,
            key,
            hidden_states,
            qr,
            packed_seq_params=packed_seq_params,
            query_positions=query_positions if packed else None,
            key_positions=key_positions if packed else None,
        )
        up = self.v_b_proj.view(
            self.config.num_attention_heads,
            self.config.v_head_dim,
            self.config.kv_lora_rank,
        )
        output = torch.einsum("...nc,ndc->...nd", output, up)
        output = vha_postmix(output, core.vha_postmix_u, core.vha_postmix_v).flatten(-2)
        gate, _ = core.linear_output_gate(qr)
        return self.linear_proj(output * gate.sigmoid())
