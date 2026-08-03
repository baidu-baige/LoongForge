# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-V-4.6 language model."""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
    get_pos_emb_on_this_cp_rank,
)
from megatron.core import InferenceParams
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection

from loongforge.models.foundation.base import BaseGPTModel
from loongforge.models.utils import import_module

from .minicpm_v_4_6_config import MiniCPMV46Config


class MiniCPMV46RotaryEmbedding(RotaryEmbedding):
    """Plain 1D RoPE indexed by per-token position_ids."""

    @torch.no_grad()
    def forward(self, position_ids: Tensor | int, offset: int = 0, packed_seq: bool = False) -> Tensor:
        if isinstance(position_ids, int):
            return super().forward(position_ids, offset=offset, packed_seq=packed_seq)

        if self.inv_freq.device.type == "cpu":
            self.inv_freq = self.inv_freq.to(device=position_ids.device)

        seq = position_ids.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        elif seq.dim() == 3 and seq.shape[0] == 1:
            seq = seq.squeeze(0)
        if seq.dim() != 2:
            raise ValueError(
                f"MiniCPMV46RotaryEmbedding expects position_ids with shape [batch, seq], got {tuple(position_ids.shape)}"
            )

        if self.seq_len_interpolation_factor is not None:
            seq = seq * (1 / self.seq_len_interpolation_factor)

        freqs = seq[..., None].float() * self.inv_freq[None, None, :].float()
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs, freqs), dim=-1).flatten(start_dim=-2)

        emb = emb.transpose(0, 1).unsqueeze(2).contiguous()
        if self.cp_group is not None and self.cp_group.size() > 1 and not packed_seq:
            emb = get_pos_emb_on_this_cp_rank(emb, 0, self.cp_group)
        return emb


class MiniCPMV46Model(BaseGPTModel):
    """MiniCPM-V-4.6 text model using plain RoPE."""

    config_class = MiniCPMV46Config

    def __init__(
        self,
        config: MiniCPMV46Config,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        scatter_embedding_sequence_parallel: bool = True,
        language_embedding: Optional[torch.nn.Module] = None,
        rotary_dtype: torch.dtype = torch.float32,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        **kwargs,
    ) -> None:
        del kwargs
        if config.model_spec is None:
            model_spec = [
                "loongforge.models.foundation.minicpm_v_4_6.minicpm_v_4_6_layer_spec",
                "get_minicpm_v_4_6_transformer_layer_spec",
            ]
        else:
            model_spec = config.model_spec

        transformer_layer_spec, mtp_layer_spec = import_module(
            model_spec, config, vp_stage=vp_stage
        )
        config.qwen35_hf_rope = True
        config.qwen35_hf_sdpa_attention = True
        rotary_pos_emb = MiniCPMV46RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=config.rotary_percent,
            rotary_interleaved=config.rotary_interleaved,
            seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
            rotary_base=config.rotary_base,
            rope_scaling=config.use_rope_scaling,
            rope_scaling_factor=config.rope_scaling_factor,
            use_cpu_initialization=config.use_cpu_initialization,
            cp_group=pg_collection.cp if pg_collection is not None else None,
        )

        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=config.padded_vocab_size,
            max_sequence_length=config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=config.fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=(
                not config.untie_embeddings_and_output_weights
            ),
            position_embedding_type=config.position_embedding_type,
            language_embedding=language_embedding,
            rotary_dtype=rotary_dtype,
            rotary_emb_func="MiniCPMV46RotaryEmbedding",
            rotary_pos_emb=rotary_pos_emb,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rotary_base,
            rope_scaling=config.use_rope_scaling,
            rope_scaling_factor=config.rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
            mtp_block_spec=mtp_layer_spec,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        if getattr(self, "mtp", None) is not None:
            for layer in self.mtp.layers:
                attention = layer.transformer_layer.self_attention
                attention.config = deepcopy(attention.config)
                attention.config.apply_rope_fusion = False

        if getattr(config, "freeze", False):
            self.freeze()

    def _preprocess(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        decoder_input: Tensor = None,
        inference_context: BaseInferenceContext = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        rotary_pos_emb: Tensor = None,
    ):
        del inference_params, rotary_pos_emb
        decoder_input, _, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = super()._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )
        if self.position_embedding_type == "rope" and not self.config.multi_latent_attention:
            rotary_pos_emb = self.rotary_pos_emb(
                position_ids,
                packed_seq=packed_seq_params is not None and packed_seq_params.qkv_format == "thd",
            )
        return decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset

    def compute_language_model_loss(self, labels: Tensor, logits: Tensor) -> Tensor:
        """Match the reference CE implementation when vocabulary parallelism is off."""
        if (
            not self.config.torch_cross_entropy_at_tp1
            or self.pg_collection.tp.size() != 1
        ):
            return super().compute_language_model_loss(labels, logits)

        transposed_labels = labels.transpose(0, 1).contiguous()
        losses = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            transposed_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        return losses.view_as(transposed_labels).transpose(0, 1).contiguous()
