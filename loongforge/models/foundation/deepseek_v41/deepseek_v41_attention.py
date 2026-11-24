# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Reference eager CSA2 attention for DeepSeek-V4.1-Flash."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
from megatron.core.utils import get_pg_size
from torch import nn

from .deepseek_v41_linear import build_deepseek_v41_linear


@dataclass(frozen=True)
class DeepseekV41AttentionState:
    """Immutable state shared by all layers in one model forward."""

    compressed_kv: torch.Tensor | None = None
    compressed_latent: torch.Tensor | None = None
    compressed_positions: torch.Tensor | None = None
    compressed_valid: torch.Tensor | None = None
    compressed_sequence_ids: torch.Tensor | None = None
    compression_ratio: int = 0
    index_keys: torch.Tensor | None = None
    topk_indices: torch.Tensor | None = None
    candidates: torch.Tensor | None = None


def _linear_output(value):
    return value[0] if isinstance(value, tuple) else value


def _power_of_two_ceiling(value: torch.Tensor) -> torch.Tensor:
    """Round positive FP32 values to the next exactly represented power of two."""
    bits = value.contiguous().float().view(torch.int32)
    exponent = (bits >> 23) & 255
    has_mantissa = (bits & ((1 << 23) - 1)) != 0
    return ((exponent + has_mantissa.to(torch.int32)) << 23).view(torch.float32)


def _fake_quantize(values: torch.Tensor, *, format: str, block_size: int) -> torch.Tensor:
    """Apply the released cache quantization boundary with a STE gradient."""
    padding = -values.shape[-1] % block_size
    if padding:
        padded = torch.nn.functional.pad(values, (0, padding))
        return _fake_quantize(padded, format=format, block_size=block_size)[..., : values.shape[-1]]

    grouped = values.float().unflatten(-1, (-1, block_size))
    maximum = grouped.abs().amax(-1, keepdim=True)
    if format == "fp8":
        scale = _power_of_two_ceiling(maximum.clamp_min(1e-4) / 448.0)
        rounded = (grouped / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    else:
        if format == "nvfp4":
            scale = (maximum.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
        elif format == "mxfp4":
            scale = _power_of_two_ceiling(maximum.clamp_min(6 * 2**-126) / 6)
        else:
            raise ValueError(f"Unsupported DeepSeek-V4.1 cache format: {format}")
        normalized = (grouped / scale).clamp(-6.0, 6.0)
        magnitude = normalized.abs()
        midpoints = magnitude.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        levels = magnitude.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        lower = torch.bucketize(magnitude, midpoints)
        ties = magnitude == midpoints[lower.clamp_max(6)]
        selected = lower + (ties & (lower % 2 == 1)).to(lower.dtype)
        rounded = torch.copysign(levels[selected], normalized)
    dequantized = rounded.mul(scale).flatten(-2).to(values.dtype)
    return values + (dequantized - values).detach()


class _V41RMSNorm(nn.Module):
    """Small local norm used for compressor and indexer-owned tensors."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        value = hidden_states.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * self.weight.float()).to(hidden_states.dtype)


class _CompressorLinear(nn.Linear):
    """Keep pooling weights in FP32 and compute in the input dtype."""

    def __init__(self, input_size, output_size, init_method):
        super().__init__(input_size, output_size, bias=False, dtype=torch.float32)
        self.weight._keep_in_float32 = True
        init_method(self.weight)

    def forward(self, value):
        with torch.autocast(device_type=value.device.type, enabled=False):
            return torch.nn.functional.linear(value, self.weight.to(value.dtype))


def _apply_rope(values: torch.Tensor, angles: torch.Tensor, *, inverse: bool = False):
    rotary_dim = angles.shape[-1] * 2
    pairs = torch.view_as_complex(values[..., -rotary_dim:].float().unflatten(-1, (-1, 2)).contiguous())
    frequencies = torch.polar(torch.ones_like(angles), angles)
    if values.ndim == 4:
        frequencies = frequencies.unsqueeze(2)
    if inverse:
        frequencies = frequencies.conj()
    rotated = torch.view_as_real(pairs * frequencies).flatten(-2).to(values.dtype)
    output = values.clone(memory_format=torch.contiguous_format)
    output[..., -rotary_dim:] = rotated
    return output


class DeepseekV41Attention(nn.Module):
    """Full/Reuse/Reindex CSA2 with the released causal selection contract."""

    def __init__(
        self,
        config,
        layer_idx: int,
        backend,
        *,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.pg_collection = pg_collection
        self.compress_ratio = config.csa_compress_ratios[layer_idx]
        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids
        self.num_heads = config.num_attention_heads
        self.tp_size = get_pg_size(pg_collection.tp) if pg_collection is not None else 1
        if self.num_heads % self.tp_size:
            raise ValueError("DeepSeek-V4.1 attention heads must divide tensor parallel size")
        if config.o_groups % self.tp_size:
            raise ValueError("DeepSeek-V4.1 output groups must divide tensor parallel size")
        if config.index_n_heads % self.tp_size:
            raise ValueError("DeepSeek-V4.1 index heads must divide tensor parallel size")
        self.num_local_heads = self.num_heads // self.tp_size
        self.num_local_groups = config.o_groups // self.tp_size
        self.index_local_heads = config.index_n_heads // self.tp_size
        self.head_dim = config.v_head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank

        self.linear_q_down_proj = build_deepseek_v41_linear(
            config,
            config.hidden_size,
            config.q_lora_rank,
            mode="duplicated",
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            tp_group=pg_collection.tp if self.tp_size > 1 else None,
        )
        self.linear_q_up_proj = build_deepseek_v41_linear(
            config,
            config.q_lora_rank,
            config.num_attention_heads * config.v_head_dim,
            mode="column",
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            tp_group=pg_collection.tp,
        )
        self.linear_kv_proj = build_deepseek_v41_linear(
            config,
            config.hidden_size,
            config.v_head_dim,
            mode="duplicated",
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            tp_group=pg_collection.tp if self.tp_size > 1 else None,
        )
        self.linear_proj = build_deepseek_v41_linear(
            config,
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            mode="row",
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            tp_group=pg_collection.tp,
        )
        self.q_layernorm = _V41RMSNorm(config.q_lora_rank, config.layernorm_epsilon, config.params_dtype)
        # The column projection already sums the full-sequence input gradient.
        self.q_layernorm.weight.average_gradients_across_tp_domain = True
        self.kv_layernorm = _V41RMSNorm(config.v_head_dim, config.layernorm_epsilon, config.params_dtype)
        self.attn_sink = nn.Parameter(torch.zeros(self.num_local_heads, dtype=torch.float32))
        self.attn_sink._keep_in_float32 = True
        set_tensor_model_parallel_attributes(
            self.attn_sink,
            is_parallel=True,
            dim=0,
            stride=1,
        )

        group_input = self.num_local_heads * config.v_head_dim // self.num_local_groups
        self.linear_o_group_proj = nn.Parameter(
            torch.empty(
                self.num_local_groups * config.o_lora_rank,
                group_input,
                dtype=config.params_dtype,
            )
        )
        set_tensor_model_parallel_attributes(
            self.linear_o_group_proj,
            is_parallel=True,
            dim=0,
            stride=1,
        )
        config.init_method(self.linear_o_group_proj)

        self.compressor_wkv = None
        self.compressor_wgate = None
        self.compressor_norm = None
        if self.is_kv_source and self.compress_ratio:
            self.compressor_wkv = _CompressorLinear(config.hidden_size, config.v_head_dim, config.init_method)
            self.compressor_norm = _V41RMSNorm(config.v_head_dim, config.layernorm_epsilon, config.params_dtype)
            if self.compress_ratio > 1:
                self.compressor_wgate = _CompressorLinear(config.hidden_size, config.v_head_dim, config.init_method)

        self.indexer_wq_b = None
        self.indexer_wk = None
        self.indexer_k_norm = None
        self.indexer_weights_proj = None
        if self.is_index_source:
            self.indexer_wq_b = build_deepseek_v41_linear(
                config,
                config.q_lora_rank,
                config.index_n_heads * config.index_head_dim,
                mode="column",
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                tp_group=pg_collection.tp,
            )
            self.indexer_weights_proj = build_deepseek_v41_linear(
                config,
                config.hidden_size,
                config.index_n_heads,
                mode="column",
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                tp_group=pg_collection.tp,
            )
            if self.is_kv_source and self.compress_ratio:
                self.indexer_wk = build_deepseek_v41_linear(
                    config,
                    config.v_head_dim,
                    config.index_head_dim,
                    mode="duplicated",
                    init_method=config.init_method,
                    bias=False,
                    skip_bias_add=False,
                )
                self.indexer_k_norm = _V41RMSNorm(config.index_head_dim, config.layernorm_epsilon, config.params_dtype)
            for module in (
                self.indexer_wq_b,
                self.indexer_wk,
                self.indexer_k_norm,
                self.indexer_weights_proj,
            ):
                if module is not None:
                    module.requires_grad_(False)

    def _angles(self, positions: torch.Tensor, *, compressed: bool) -> torch.Tensor:
        dim = self.config.qk_rope_head_dim
        theta = self.config.compress_rope_theta if compressed else self.config.rotary_base
        frequencies = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim))
        if compressed and self.config.rope_type == "yarn":
            original = self.config.original_max_position_embeddings
            low = max(
                math.floor(dim * math.log(original / (self.config.beta_fast * 2 * math.pi)) / (2 * math.log(theta))),
                0,
            )
            high = min(
                math.ceil(dim * math.log(original / (self.config.beta_slow * 2 * math.pi)) / (2 * math.log(theta))),
                dim - 1,
            )
            ramp = (
                (torch.arange(dim // 2, device=positions.device, dtype=torch.float32) - low) / max(high - low, 1e-3)
            ).clamp(0, 1)
            frequencies = frequencies / self.config.rotary_scaling_factor * ramp + frequencies * (1 - ramp)
        return positions.float().unsqueeze(-1) * frequencies

    def _compress(self, hidden_states: torch.Tensor):
        ratio = self.compress_ratio
        length = hidden_states.shape[1] // ratio * ratio
        values = hidden_states[:, :length]
        if ratio == 1:
            projected = self.compressor_wkv(values)
            return _linear_output(self.compressor_norm(projected)), values[:, ::ratio]
        projected = self.compressor_wkv(values.float())
        scores = self.compressor_wgate(values.float())
        projected = projected.float().unflatten(1, (-1, ratio))
        scores = scores.float().unflatten(1, (-1, ratio))
        pooled = (projected * scores.softmax(dim=2)).sum(dim=2).to(hidden_states.dtype)
        return _linear_output(self.compressor_norm(pooled)), values[:, ::ratio]

    @torch.no_grad()
    def _update_index(
        self,
        state: DeepseekV41AttentionState,
        q_latent: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> DeepseekV41AttentionState:
        if not self.is_index_source:
            return state
        if state.compressed_latent is None or state.compressed_positions is None:
            raise ValueError("DeepSeek-V4.1 indexer requires a preceding Full CSA2 source")
        query = _linear_output(self.indexer_wq_b(q_latent)).view(
            q_latent.shape[0],
            q_latent.shape[1],
            self.index_local_heads,
            self.config.index_head_dim,
        )
        query = _apply_rope(query, self._angles(position_ids, compressed=bool(self.compress_ratio)))
        query = _fake_quantize(query, format="mxfp4", block_size=32)
        keys = state.index_keys
        if keys is None or self.is_kv_source:
            keys = _linear_output(self.indexer_wk(state.compressed_latent))
            keys = _linear_output(self.indexer_k_norm(keys))
            keys = _apply_rope(keys, self._angles(state.compressed_positions, compressed=True))
            keys = _fake_quantize(keys, format="mxfp4", block_size=32)
        weights = _linear_output(self.indexer_weights_proj(hidden_states))
        weights = weights * (self.config.index_head_dim**-0.5 * self.config.index_n_heads**-0.5)
        scores = torch.einsum("bshd,bkd->bshk", query, keys)
        scores = scores.relu()
        scores = (scores * weights.unsqueeze(-1)).sum(dim=2)
        if self.tp_size > 1:
            torch.distributed.all_reduce(scores, group=self.pg_collection.tp)
        visible = state.compressed_positions.unsqueeze(1) + state.compression_ratio - 1 <= position_ids.unsqueeze(-1)
        visible = visible & state.compressed_valid.unsqueeze(1)
        scores = scores.masked_fill(~visible, float("-inf"))

        candidates = state.candidates
        if self.layer_idx == self.config.candidate_source_layer_id:
            block_size = self.config.candidate_block_size
            blocks = (scores.shape[-1] + block_size - 1) // block_size
            padded = scores.new_full((*scores.shape[:2], blocks * block_size), float("-inf"))
            padded[..., : scores.shape[-1]] = scores
            block_scores = padded.view(*scores.shape[:2], blocks, block_size).amax(-1)
            latest = ((position_ids + 1) // state.compression_ratio - 1) // block_size
            block_ids = torch.arange(blocks, device=scores.device)
            block_scores = block_scores.masked_fill(block_ids == latest.unsqueeze(-1), float("inf"))
            keep = block_scores.topk(min(self.config.candidate_topk_blocks, blocks), dim=-1)
            candidates = torch.zeros_like(block_scores, dtype=torch.bool)
            candidates.scatter_(-1, keep.indices, keep.values > float("-inf"))
        elif self.config.candidate_source_layer_id >= 0 and candidates is not None:
            block_ids = torch.arange(scores.shape[-1], device=scores.device) // self.config.candidate_block_size
            scores = scores.masked_fill(~candidates.gather(-1, block_ids.expand_as(scores)), float("-inf"))

        topk = min(self.config.index_topk, scores.shape[-1])
        selected = scores.topk(topk, dim=-1, sorted=False).indices.sort(-1).values
        valid = torch.isfinite(scores.gather(-1, selected))
        selected = torch.where(valid, selected, torch.full_like(selected, -1))
        return replace(state, index_keys=keys, topk_indices=selected, candidates=candidates)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        state: DeepseekV41AttentionState,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        sequence_ids: torch.Tensor | None = None,
        vision_token_types: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DeepseekV41AttentionState]:
        if hidden_states.ndim != 3:
            raise ValueError("DeepSeek-V4.1 attention expects [batch, sequence, hidden]")
        batch, sequence, _ = hidden_states.shape
        valid = (
            torch.ones(batch, sequence, dtype=torch.bool, device=hidden_states.device)
            if attention_mask is None
            else attention_mask.bool()
        )
        q_latent = self.q_layernorm(_linear_output(self.linear_q_down_proj(hidden_states)))
        query = _linear_output(self.linear_q_up_proj(q_latent)).view(
            batch, sequence, self.num_local_heads, self.head_dim
        )
        current_kv = self.kv_layernorm(_linear_output(self.linear_kv_proj(hidden_states)))
        query_angles = self._angles(position_ids, compressed=bool(self.compress_ratio))
        query = _apply_rope(query, query_angles)
        local_kv = _apply_rope(current_kv, query_angles)
        local_kv = _fake_quantize(local_kv, format="fp8", block_size=32)

        next_state = state
        if self.compress_ratio and self.is_kv_source:
            compressed_latent, compressed_positions = self._compress(hidden_states)
            compressed_angles = self._angles(
                position_ids[:, : compressed_positions.shape[1] * self.compress_ratio : self.compress_ratio],
                compressed=True,
            )
            compressed_kv = _apply_rope(compressed_latent, compressed_angles)
            compressed_kv = _fake_quantize(compressed_kv, format="nvfp4", block_size=16)
            compressed_valid = (
                valid[:, : compressed_positions.shape[1] * self.compress_ratio]
                .view(batch, -1, self.compress_ratio)
                .all(-1)
            )
            next_state = DeepseekV41AttentionState(
                compressed_kv=compressed_kv,
                compressed_latent=compressed_latent,
                compressed_positions=position_ids[
                    :, : compressed_positions.shape[1] * self.compress_ratio : self.compress_ratio
                ],
                compressed_valid=compressed_valid,
                compressed_sequence_ids=(
                    None
                    if sequence_ids is None
                    else sequence_ids[:, : compressed_positions.shape[1] * self.compress_ratio : self.compress_ratio]
                ),
                compression_ratio=self.compress_ratio,
            )
        if self.compress_ratio:
            if next_state.compressed_kv is None or next_state.compression_ratio != self.compress_ratio:
                raise ValueError(f"DeepSeek-V4.1 layer {self.layer_idx} has no matching CSA2 source state")
            next_state = self._update_index(next_state, q_latent, hidden_states, position_ids)

        kv = local_kv
        if self.compress_ratio:
            kv = torch.cat((kv, next_state.compressed_kv), dim=1)
        # The shared KV projection is replicated on every TP rank. Preserve
        # that forward value while reducing its gradient across TP ranks.
        kv = copy_to_tensor_model_parallel_region(
            kv,
            group=self.pg_collection.tp,
        )
        keys = torch.cat((kv, kv.new_zeros(batch, 1, self.head_dim)), dim=1)
        values = keys

        scores = torch.einsum("bshd,bkd->bhsk", query.float(), keys.float()) * (self.head_dim**-0.5)
        query_pos = position_ids.unsqueeze(-1)
        local_pos = position_ids.unsqueeze(1)
        allowed = (local_pos <= query_pos) & (local_pos > query_pos - self.config.csa_window_size)
        allowed = allowed & valid.unsqueeze(1)
        if sequence_ids is not None:
            allowed = allowed & (sequence_ids.unsqueeze(1) == sequence_ids.unsqueeze(-1))
        if self.compress_ratio:
            compressed_pos = next_state.compressed_positions
            compressed_allowed = compressed_pos.unsqueeze(1) + self.compress_ratio - 1 <= query_pos
            compressed_allowed = compressed_allowed & next_state.compressed_valid.unsqueeze(1)
            if sequence_ids is not None and next_state.compressed_sequence_ids is not None:
                compressed_allowed = compressed_allowed & (
                    next_state.compressed_sequence_ids.unsqueeze(1) == sequence_ids.unsqueeze(-1)
                )
            if next_state.topk_indices is not None:
                width = compressed_pos.shape[1]
                safe = next_state.topk_indices.clamp_min(0)
                selected = torch.zeros(batch, sequence, width, dtype=torch.int32, device=hidden_states.device)
                selected.scatter_add_(2, safe, (next_state.topk_indices >= 0).int())
                compressed_allowed = compressed_allowed & (selected > 0)
            allowed = torch.cat((allowed, compressed_allowed), dim=-1)

        bias = torch.zeros(
            batch,
            1,
            sequence,
            allowed.shape[-1],
            device=hidden_states.device,
            dtype=torch.float32,
        ).masked_fill(~allowed.unsqueeze(1), float("-inf"))
        bias = bias.expand(-1, self.num_local_heads, -1, -1)
        sink = self.attn_sink.view(1, self.num_local_heads, 1, 1).expand(batch, -1, sequence, -1)
        bias = torch.cat((bias, sink), dim=-1)
        probabilities = (scores + bias).softmax(-1)
        attended = torch.einsum("bhsk,bkd->bshd", probabilities, values.float()).to(query.dtype)
        attended = _apply_rope(attended, query_angles, inverse=True)
        attended = attended.reshape(batch, sequence, self.num_local_groups, -1)
        group_proj = self.linear_o_group_proj.view(self.num_local_groups, self.o_lora_rank, -1)
        attended = torch.einsum("bsgd,god->bsgo", attended, group_proj).flatten(2)
        output, bias = self.linear_proj(attended.contiguous())
        if bias is not None:
            output = output + bias
        return output.masked_fill(~valid.unsqueeze(-1), 0), next_state
