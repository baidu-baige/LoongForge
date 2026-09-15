# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash text attention components."""

from __future__ import annotations

import math
from copy import copy

import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAttention,
    get_dsa_index_share_topk_holder,
    unfused_dsa_fn,
)
from megatron.core.transformer.torch_norm import WrappedTorchNorm


def _native_rmsnorm(config, hidden_size: int, eps: float):
    norm_config = copy(config)
    norm_config.sequence_parallel = False
    return WrappedTorchNorm(norm_config, hidden_size=hidden_size, eps=eps)


class GatedRMSNorm(nn.Module):
    def __init__(self, config, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = _native_rmsnorm(config, hidden_size, eps)

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return (self.norm(hidden_states) * torch.sigmoid(gate.float())).to(hidden_states.dtype)


def _cp_reconstruct(value: torch.Tensor, cp_group, sequence_dim: int):
    if cp_group is None or cp_group.size() == 1:
        return value, None
    from torch.distributed.nn.functional import all_gather

    gathered = all_gather(value.contiguous(), group=cp_group)
    halves = [tensor.chunk(2, dim=sequence_dim) for tensor in gathered]
    global_value = torch.cat(
        [pair[0] for pair in halves] + [pair[1] for pair in reversed(halves)],
        dim=sequence_dim,
    )
    return global_value, (cp_group.rank(), cp_group.size())


def _cp_select_local(value: torch.Tensor, cp_context, sequence_dim: int):
    if cp_context is None:
        return value
    cp_rank, cp_size = cp_context
    chunks = value.chunk(2 * cp_size, dim=sequence_dim)
    return torch.cat((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]), dim=sequence_dim)


def _sp_reconstruct(value: torch.Tensor, tp_group):
    if tp_group is None or tp_group.size() == 1:
        return value, None
    from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

    return gather_from_sequence_parallel_region(value, group=tp_group), (
        tp_group.rank(),
        tp_group.size(),
    )


def _sp_select_local(value: torch.Tensor, sp_context):
    if sp_context is None:
        return value
    tp_rank, tp_size = sp_context
    return value.chunk(tp_size, dim=0)[tp_rank].contiguous()


def l2norm(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return tensor / torch.sqrt((tensor * tensor).sum(dim=dim, keepdim=True) + eps)


def chunk_kimi_delta_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    input_dtype = query.dtype
    query, key, value, beta, decay = [
        tensor.transpose(1, 2).contiguous().float()
        for tensor in (query, key, value, beta, decay)
    ]
    query = l2norm(query)
    key = l2norm(key)

    batch_size, num_heads, sequence_length, key_dim = key.shape
    value_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    total_length = sequence_length + pad_size
    query = F.pad(query, (0, 0, 0, pad_size)) * (query.shape[-1] ** -0.5)
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    decay = F.pad(decay, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    value_beta = value * beta.unsqueeze(-1)
    key_beta = key * beta.unsqueeze(-1)

    tensors = (query, key, value, decay, key_beta, value_beta)
    query, key, value, decay, key_beta, value_beta = [
        tensor.reshape(tensor.shape[0], tensor.shape[1], -1, chunk_size, tensor.shape[-1])
        for tensor in tensors
    ]
    beta = beta.reshape(beta.shape[0], beta.shape[1], -1, chunk_size)

    decay = decay.cumsum(dim=-2)
    diagonal_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )
    # Mask the strictly-upper-triangle differences BEFORE the exp: with real
    # weights the per-dim decay approaches lower_bound, so within-chunk cumsum
    # differences reach ~63 * |lower_bound| and exp() overflows to inf. The
    # entries are always masked downstream, but the backward pass computes
    # 0 (masked grad) * inf (exp result) = NaN and poisons every gradient.
    # Zeroing them here keeps the forward mathematically identical.
    decay_diff = decay.unsqueeze(-2) - decay.unsqueeze(-3)
    # Build the mask in 2-D first, then unsqueeze: triu on a (c, c, 1) tensor
    # would treat it as a batch of (c, 1) matrices and return all-False.
    upper_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=decay_diff.device),
        diagonal=1,
    ).unsqueeze(-1)
    decay_diff = decay_diff.masked_fill(upper_mask, float("-inf"))
    decay_mask = decay_diff.exp().float()
    attention = -(key_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1)
    attention = attention.masked_fill(diagonal_mask, 0)
    for index in range(1, chunk_size):
        row = attention[..., index, :index].clone()
        submatrix = attention[..., :index, :index].clone()
        attention[..., index, :index] = row + (row.unsqueeze(-1) * submatrix).sum(-2)

    attention = attention + torch.eye(chunk_size, dtype=attention.dtype, device=attention.device)
    value = attention @ value_beta
    cumulative_key = attention @ (key_beta * decay.exp())
    recurrent_state = torch.zeros(
        batch_size, num_heads, key_dim, value_dim, dtype=value.dtype, device=value.device
    )
    output = torch.zeros_like(value)
    causal_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )
    for index in range(total_length // chunk_size):
        query_chunk = query[:, :, index]
        key_chunk = key[:, :, index]
        value_chunk = value[:, :, index]
        decay_chunk = decay[:, :, index]
        inter = (query_chunk * decay_chunk.exp()) @ recurrent_state
        intra = (query_chunk.unsqueeze(-2) * key_chunk.unsqueeze(-3) * decay_mask[:, :, index]).sum(-1)
        intra = intra.masked_fill(causal_mask, 0)
        new_value = value_chunk - cumulative_key[:, :, index] @ recurrent_state
        output[:, :, index] = inter + intra @ new_value
        recurrent_state = (
            recurrent_state * decay_chunk[:, :, -1].exp().unsqueeze(-1)
            + (key_chunk * (decay_chunk[:, :, -1:] - decay_chunk).exp()).transpose(-1, -2) @ new_value
        )

    output = output.reshape(batch_size, num_heads, -1, value_dim)[:, :, :sequence_length]
    return output.transpose(1, 2).contiguous().to(input_dtype)


class Glm5NextTextLinearAttention(nn.Module):
    def __init__(self, config, layer_number=None, pg_collection=None, **kwargs) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.qkv_dim = self.num_heads * self.head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.lower_bound = config.linear_lower_bound
        self.cp_group = getattr(pg_collection, "cp", None)
        self.tp_group = getattr(pg_collection, "tp", None)

        self.q_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.q_conv1d = nn.Conv1d(self.qkv_dim, self.qkv_dim, self.conv_kernel_size, groups=self.qkv_dim, bias=False)
        self.k_conv1d = nn.Conv1d(self.qkv_dim, self.qkv_dim, self.conv_kernel_size, groups=self.qkv_dim, bias=False)
        self.v_conv1d = nn.Conv1d(self.qkv_dim, self.qkv_dim, self.conv_kernel_size, groups=self.qkv_dim, bias=False)

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.qkv_dim))
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = GatedRMSNorm(config, self.head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(self.qkv_dim, self.hidden_size, bias=False)

    def _conv(self, projection: nn.Linear, convolution: nn.Conv1d, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = projection(hidden_states).transpose(1, 2)
        projected = F.conv1d(
            projected.float(),
            convolution.weight.float(),
            padding=self.conv_kernel_size - 1,
            groups=self.qkv_dim,
        )[..., : hidden_states.shape[1]]
        return F.silu(projected).transpose(1, 2).to(hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        hidden_states, sp_context = _sp_reconstruct(hidden_states, self.tp_group)
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        hidden_states, cp_context = _cp_reconstruct(hidden_states, self.cp_group, 1)
        if attention_mask is not None and attention_mask.shape[1] != hidden_states.shape[1]:
            attention_mask, _ = _cp_reconstruct(attention_mask, self.cp_group, 1)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)
        batch_size, sequence_length = hidden_states.shape[:2]
        hidden_shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
        query = self._conv(self.q_proj, self.q_conv1d, hidden_states).view(hidden_shape)
        key = self._conv(self.k_proj, self.k_conv1d, hidden_states).view(hidden_shape)
        value = self._conv(self.v_proj, self.v_conv1d, hidden_states).view(hidden_shape)

        forget = self.f_b_proj(self.f_a_proj(hidden_states))
        forget = (forget.float() + self.dt_bias.float().view(1, 1, -1)).view(hidden_shape)
        decay_rate = self.A_log.float().exp().view(1, 1, self.num_heads, 1)
        if self.lower_bound is not None:
            decay = self.lower_bound * torch.sigmoid(decay_rate * forget)
        else:
            softplus = torch.where(forget > 20.0, forget, torch.log1p(torch.exp(forget)))
            decay = -decay_rate * softplus
        beta = torch.sigmoid(self.b_proj(hidden_states))
        output = chunk_kimi_delta_attention(query, key, value, decay, beta)
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
        output = self.o_norm(output, gate).reshape(batch_size, sequence_length, -1)
        output = self.o_proj(output)
        output = _cp_select_local(output, cp_context, 1).transpose(0, 1).contiguous()
        output = _sp_select_local(output, sp_context)
        return output, None


class KPoolDSAIndexer(DSAIndexer):
    def __init__(self, config, submodules, pg_collection=None) -> None:
        super().__init__(config=config, submodules=submodules, pg_collection=pg_collection)
        self.num_heads = self.index_n_heads
        self.head_dim = self.index_head_dim
        self.index_kpool = config.index_kpool
        self.always_select_tail = config.index_kpool_always_select_tail
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(self.index_kpool, self.head_dim))
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(self.head_dim, self.hidden_size))

    def _pooled_states(self, packed_states: torch.Tensor):
        keys, gate_scores, valid_keys = torch.split(
            packed_states, [self.head_dim, self.head_dim, 1], dim=-1
        )
        valid_keys = valid_keys.bool().squeeze(-1)
        batch_size, sequence_length = keys.shape[:2]
        pool_count = (sequence_length + self.index_kpool - 1) // self.index_kpool
        first_key = torch.where(
            valid_keys.any(-1),
            valid_keys.long().argmax(-1),
            torch.full((batch_size,), sequence_length, dtype=torch.long, device=keys.device),
        )
        offsets = torch.arange(pool_count * self.index_kpool, device=keys.device)
        pool_indices = first_key[:, None, None] + offsets.view(1, pool_count, self.index_kpool)
        batch_indices = torch.arange(batch_size, device=keys.device)[:, None, None]
        safe_indices = pool_indices.clamp(0, sequence_length - 1)
        grouped_keys = keys[batch_indices, safe_indices]
        grouped_scores = gate_scores[batch_indices, safe_indices]
        grouped_valid = valid_keys[batch_indices, safe_indices] & (pool_indices < sequence_length)
        pool_valid = grouped_valid.all(-1)
        pool_indices = pool_indices.masked_fill(~grouped_valid, -1)
        logits = grouped_scores.float() + self.index_kpool_compress_ape.float()[None, None]
        logits = logits.masked_fill(~grouped_valid[..., None], float("-inf"))
        probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(grouped_keys.dtype)
        pool_keys = (probabilities * grouped_keys).sum(dim=2)
        keep = pool_valid.any(0)
        return pool_keys[:, keep], pool_indices[:, keep], pool_valid[:, keep], valid_keys

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_seq_params=None,
    ) -> torch.Tensor:
        if packed_seq_params is not None:
            raise NotImplementedError("GLM-5.3 K-pool indexer does not support packed sequences")
        query, key, weights = self.forward_before_topk(x, qr, packed_seq_params)
        hidden_states = x
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.pg_collection.tp
            )
        query, _ = _cp_reconstruct(query, self.pg_collection.cp, 0)
        key, _ = _cp_reconstruct(key, self.pg_collection.cp, 0)
        weights, _ = _cp_reconstruct(weights, self.pg_collection.cp, 0)
        hidden_states, _ = _cp_reconstruct(hidden_states, self.pg_collection.cp, 0)
        query = query.permute(1, 0, 2, 3).contiguous()
        key = key.transpose(0, 1).contiguous()
        weights = weights.transpose(0, 1).contiguous()
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        if attention_mask.shape[1] != hidden_states.shape[1]:
            attention_mask, _ = _cp_reconstruct(attention_mask, self.pg_collection.cp, 1)
        batch_size, sequence_length = hidden_states.shape[:2]
        gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
        packed = torch.cat([key, gate_scores, attention_mask.to(key.dtype)[..., None]], dim=-1)
        pool_keys, pool_indices, pool_valid, valid_keys = self._pooled_states(packed)

        positions = torch.arange(sequence_length, device=hidden_states.device)
        visible = positions[None, None, :] <= positions[None, :, None]
        visible = visible & valid_keys[:, None, :]
        scores = torch.matmul(query.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores)
        index_scores = torch.matmul(weights.float().unsqueeze(-2), scores).squeeze(-2)
        pool_end = pool_indices[..., -1].clamp(0, sequence_length - 1)
        pool_visible = visible.gather(-1, pool_end[:, None, :].expand(batch_size, sequence_length, -1))
        valid_candidates = pool_visible & pool_valid[:, None]
        index_scores = index_scores.masked_fill(~valid_candidates, torch.finfo(index_scores.dtype).min)
        select_k = min(self.index_topk // self.index_kpool, index_scores.shape[-1])
        selected = index_scores.topk(select_k, dim=-1).indices
        batch_indices = torch.arange(batch_size, device=hidden_states.device)[:, None, None]
        selected_valid = valid_candidates.gather(-1, selected)
        selected_indices = pool_indices[batch_indices, selected]
        topk_indices = selected_indices.flatten(-2).masked_fill(
            ~selected_valid[..., None].expand_as(selected_indices).flatten(-2), -1
        )

        output_width = self.index_topk
        if self.always_select_tail and self.index_kpool > 1:
            max_tail = self.index_kpool - 1
            first_key = torch.where(
                valid_keys.any(-1),
                valid_keys.long().argmax(-1),
                torch.full((batch_size,), sequence_length, dtype=torch.long, device=hidden_states.device),
            )
            visible_count = visible.long().sum(-1)
            tail_count = visible_count.remainder(self.index_kpool)
            tail_offsets = torch.arange(max_tail, device=hidden_states.device)
            tail_start = first_key[:, None] + visible_count - tail_count
            tail_indices = tail_start[..., None] + tail_offsets
            tail_valid = (tail_offsets[None, None, :] < tail_count[..., None]) & tail_indices.lt(sequence_length)
            tail_visible = visible.gather(-1, tail_indices.clamp(0, sequence_length - 1))
            tail_indices = tail_indices.masked_fill(~(tail_valid & tail_visible), -1)
            topk_indices = torch.cat([topk_indices, tail_indices], dim=-1)
            output_width += max_tail

        topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)
        return topk_indices[..., :output_width].masked_fill(~attention_mask[..., None], -1).long()


class KPoolDSAttention(DSAttention):
    def __init__(
        self,
        config,
        submodules,
        layer_number,
        attn_mask_type,
        attention_type,
        attention_dropout=None,
        softmax_scale=None,
        k_channels=None,
        v_channels=None,
        cp_comm_type="p2p",
        pg_collection=None,
        is_mtp_layer=False,
        use_indexer=True,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            softmax_scale=softmax_scale,
            k_channels=k_channels,
            v_channels=v_channels,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        self.pg_collection = pg_collection
        self.skip_topk = not use_indexer
        self.index_share = any(kind == "shared" for kind in config.indexer_types)
        if self.skip_topk:
            self.indexer = None
            layer_index = self.layer_number - 1
            source_index = max(
                index
                for index, kind in enumerate(config.indexer_types[:layer_index])
                if kind == "full"
            )
            self.source_layer = source_index + 1

    @staticmethod
    def _replace_invalid_indices(topk: torch.Tensor, sequence_length: int) -> torch.Tensor:
        valid = topk.ge(0) & topk.lt(sequence_length)
        positions = torch.arange(topk.shape[1], device=topk.device).view(1, -1, 1)
        fallback = torch.where(
            valid,
            topk,
            torch.full_like(topk, sequence_length),
        ).amin(dim=-1, keepdim=True)
        fallback = torch.where(
            fallback.lt(sequence_length),
            fallback,
            positions.clamp_max(sequence_length - 1),
        )
        return torch.where(valid, topk, fallback).long()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        index_share_carrier=None,
    ) -> torch.Tensor:
        if attention_bias is not None:
            raise ValueError("GLM-5.3 sparse attention does not accept attention bias")
        query, cp_context = _cp_reconstruct(query, self.pg_collection.cp, 0)
        key, _ = _cp_reconstruct(key, self.pg_collection.cp, 0)
        value, _ = _cp_reconstruct(value, self.pg_collection.cp, 0)
        sequence_length = query.shape[0]
        holder = get_dsa_index_share_topk_holder(
            packed_seq_params,
            index_share_carrier,
            self.skip_topk,
            self.layer_number,
            self.source_layer,
        )
        if self.skip_topk:
            if self.source_layer not in holder:
                raise RuntimeError(
                    f"layer {self.layer_number} requires K-pool indices from layer {self.source_layer}"
                )
            topk = holder[self.source_layer]
        else:
            topk = self.indexer(x, qr, attention_mask, packed_seq_params)
            if holder is not None:
                holder[self.layer_number] = topk
        safe_topk = self._replace_invalid_indices(topk, sequence_length)
        output = unfused_dsa_fn(query, key, value, safe_topk, self.softmax_scale)
        return _cp_select_local(output, cp_context, 0)

def initialize_kda(module: Glm5NextTextLinearAttention) -> None:
    nn.init.zeros_(module.A_log)
    nn.init.uniform_(module.dt_bias, a=math.log(1e-3), b=math.log(1e-1))
    decay = module.dt_bias.exp().clamp_min(1e-4)
    with torch.no_grad():
        module.dt_bias.copy_(decay + torch.log(-torch.expm1(-decay)))
