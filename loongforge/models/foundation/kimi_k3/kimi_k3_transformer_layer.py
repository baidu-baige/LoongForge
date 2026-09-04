# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Megatron-Bridge under the Apache-2.0 License:
# https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/kimi/kimi_k3_layers.py

"""Kimi K3 transformer layer and AttnRes residual-bank handling."""

import torch
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    get_transformer_layer_offset,
)
from torch import nn

from .kimi_k3_ops import KimiRMSNorm, attn_res_aggregate, sum_grads_across_tp
from .kimi_k3_pipeline import bank_num_rows, pack_stage_boundary, unpack_stage_boundary


class KimiK3TransformerLayer(TransformerLayer):
    """Transformer layer implementing Kimi K3's AttnRes residual bank.

    K3 replaces the usual two residual adds with a bank of hidden-state
    snapshots taken every ``kimi_attn_res_block_size`` layers. Attention and MLP
    inputs are a learned mixture over that bank plus the running prefix sum,
    so the bank has to travel between layers; it rides MCore's ``context`` slot.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.config.hidden_dropout != 0.0:
            raise ValueError("Kimi K3 requires hidden_dropout=0")

        self.attn_res_block_size = self.config.kimi_attn_res_block_size
        self.self_attention_res_norm, self.self_attention_res_proj = self._score_pair()
        self.mlp_res_norm, self.mlp_res_proj = self._score_pair()
        self.is_last_layer = self.layer_number == self.config.num_layers
        if self.is_last_layer:
            self.output_attn_res_norm, self.output_attn_res_proj = self._score_pair()

        layer_idx = self.layer_number - 1
        stage_start = get_transformer_layer_offset(self.config)
        stage_end = stage_start + get_num_layers_to_build(self.config)
        self.is_stage_entry = layer_idx == stage_start and layer_idx > 0
        self.is_stage_exit = (
            layer_idx + 1 == stage_end and stage_end < self.config.num_layers
        )

    def _score_pair(self) -> tuple[KimiRMSNorm, nn.Linear]:
        """Build one AttnRes scoring head: an RMSNorm and a rank-1 projection."""
        hidden_size = self.config.hidden_size
        kwargs = {"device": torch.cuda.current_device(), "dtype": self.config.params_dtype}
        norm = KimiRMSNorm(hidden_size, self.config.layernorm_epsilon, **kwargs)
        proj = nn.Linear(hidden_size, 1, bias=False, **kwargs)
        sum_grads_across_tp(norm)
        sum_grads_across_tp(proj)
        return norm, proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply attention, MLP/MoE, and AttnRes state updates."""
        # MLP and MoE in Loong-Megatron reject padding_mask; the argument stays
        # on this layer only to satisfy the scheduler's calling contract.
        del context_mask, input_ids, padding_mask, kwargs
        layer_idx = self.layer_number - 1

        if context is not None:
            prefix_sum = hidden_states
            block_residual = context
        elif layer_idx == 0:
            prefix_sum = hidden_states
            block_residual = hidden_states.new_empty(
                *hidden_states.shape[:-1], 0, hidden_states.shape[-1]
            )
        else:
            if not self.is_stage_entry:
                raise ValueError("AttnRes snapshot bank is missing")
            prefix_sum, block_residual = unpack_stage_boundary(
                hidden_states,
                self.config.hidden_size,
                bank_num_rows(layer_idx, self.attn_res_block_size),
            )

        if block_residual.shape[-2] > 0:
            attention_input = attn_res_aggregate(
                prefix_sum,
                block_residual,
                self.self_attention_res_norm,
                self.self_attention_res_proj,
                self.input_layernorm,
            )
        else:
            attention_input = self.input_layernorm(prefix_sum)

        writes_snapshot = layer_idx % self.attn_res_block_size == 0
        if writes_snapshot:
            block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)

        attention_output = _add_bias(
            self.self_attention(
                attention_input,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        )
        # A snapshot layer restarts the prefix sum rather than extending it.
        prefix_sum = attention_output if writes_snapshot else prefix_sum + attention_output

        mlp_input = attn_res_aggregate(
            prefix_sum,
            block_residual,
            self.mlp_res_norm,
            self.mlp_res_proj,
            self.pre_mlp_layernorm,
        )
        prefix_sum = prefix_sum + _add_bias(self.mlp(mlp_input))

        if self.is_last_layer:
            prefix_sum = attn_res_aggregate(
                prefix_sum,
                block_residual,
                self.output_attn_res_norm,
                self.output_attn_res_proj,
            )
        if self.is_stage_exit:
            prefix_sum = pack_stage_boundary(prefix_sum, block_residual)
        return prefix_sum, block_residual


def _add_bias(output_with_bias: tuple[torch.Tensor, torch.Tensor | None]) -> torch.Tensor:
    output, bias = output_with_bias
    return output if bias is None else output + bias


__all__ = ["KimiK3TransformerLayer"]
