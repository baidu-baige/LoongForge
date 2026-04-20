# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3_5 Model"""

from typing import Optional
from copy import deepcopy

import torch
from torch import Tensor

from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.models.common.embeddings.rotary_pos_embedding import MultimodalRotaryEmbedding

from loongforge.models.utils import import_module
from loongforge.models.foundation.base import BaseGPTModel
from .qwen3_5_config import Qwen35Config


class Qwen35RotaryEmbedding(MultimodalRotaryEmbedding):
    """Qwen3.5 Rotary Position Embedding."""

    def __init__(self, config: Qwen35Config):
        """Initialize Qwen3.5 rotary position embedding.

        Args:
            config: Qwen3.5 model configuration containing rotary embedding parameters.

        Returns:
            None
        """
        super().__init__(
            kv_channels=config.kv_channels,
            rotary_percent=config.rotary_percent,
            rotary_interleaved=config.rotary_interleaved,
            seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
            rotary_base=config.rotary_base,
        )

        self.config = config
        self.mrope_section = self.config.mrope_section
        self.mrope_interleaved = self.config.mrope_interleaved
        assert self.mrope_interleaved, "For Qwen3.5, mrope_interleaved should be True."

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.

        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.

        Args:
            freqs: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)

        Returns:
            freqs_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, position_ids, packed_seq) -> torch.Tensor:
        """Forward pass of Qwen3.5 rotary position embedding.

        Args:
            position_ids: Tensor containing position IDs.
            packed_seq: Packed sequence information.

        Returns:
            Tensor containing the rotary position embedding.
        """
        seq = position_ids.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        # shape (3, bs, dim, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1)
        # shape (3, bs, 1, seq_length)
        seq_expanded = seq[:, :, None, :].float()
        # shape (3, bs, seq_length, dim)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)

        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # shape (seq_length, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        return emb


class Qwen35Model(BaseGPTModel):
    """Qwen3_5 language model.

    Args:
        config (TransformerConfig):
            Transformer config
        transformer_layer_spec (ModuleSpec):
            Specifies module to use for transformer layers
        vocab_size (int):
            Vocabulary size
        max_sequence_length (int):
            maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional):
            Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional):
            Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional):
            Defaults to False.
        parallel_output (bool, optional):
            Do not gather the outputs, keep them split across tensor
            parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional):
            When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):
            Position embedding type. Defaults to 'learned_absolute'.
        rotary_percent (float, optional):
            Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional):
            Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'. Defaults to 10000.
        rope_scaling (bool, optional): Toggle RoPE scaling.
        rope_scaling_factor (float): RoPE scaling factor. Default 8.
        scatter_embedding_sequence_parallel (bool, optional):
            Whether embeddings should be scattered across sequence parallel
            region or not. Defaults to True.
        seq_len_interpolation_factor (Optional[float], optional):
            scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
        pg_collection (ProcessGroupCollection): Model communication process groups
    """

    config_class = Qwen35Config

    def __init__(
        self,
        config: Qwen35Config,
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

        if config.model_spec is None:
            model_spec = [
                "loongforge.models.foundation.qwen3_5.qwen3_5_layer_spec",
                "get_qwen3_5_transformer_layer_spec",
            ]
        else:
            model_spec = config.model_spec

        transformer_layer_spec, mtp_layer_spec = import_module(
            model_spec, config, vp_stage=vp_stage
        )

        rotary_pos_emb = Qwen35RotaryEmbedding(config)

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
            rotary_emb_func=config.rotary_emb_func,
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

        if getattr(self, 'mtp', None) is not None:
            for layer in self.mtp.layers:
                attention = layer.transformer_layer.self_attention
                attention.config = deepcopy(attention.config)
                attention.config.apply_rope_fusion = False

        if hasattr(config, 'freeze') and config.freeze:
            self.freeze()

    def _preprocess(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        decoder_input: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        rotary_pos_emb: Tensor = None,
    ):
        """Preprocesses inputs for the transformer decoder.

        Applies embeddings to input tokens, or uses `decoder_input` from a previous
        pipeline stage. Also sets up rotary positional embeddings.
        """
        # Decoder embedding.
        if decoder_input is None:
            if self.pre_process:
                decoder_input = self.embedding(
                    input_ids=input_ids, position_ids=position_ids
                )
            else:
                # intermediate stage of pipeline
                # decoder will get hidden_states from encoder.input_tensor
                decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = self.rotary_pos_emb(
            position_ids,
            packed_seq=packed_seq_params,
        )
            
        return decoder_input, rotary_pos_emb

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        rotary_pos_emb: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        loss_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Forward function of the GPT Model. This function passes the input tensors
        through the embedding layer, and then the decoder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given or the final hidden units.

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """
        decoder_input, rotary_pos_emb = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            rotary_pos_emb=rotary_pos_emb,
        )

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
        )
