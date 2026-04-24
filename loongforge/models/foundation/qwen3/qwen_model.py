# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""qwen model"""

from typing import Optional

import torch
from torch import Tensor
from megatron.core import InferenceParams, parallel_state
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection

from loongforge.models.foundation.base import BaseGPTModel
from loongforge.models.utils import import_module
from loongforge.models.omni_models.utils import get_pos_emb_on_this_cp_rank
from loongforge.models.foundation.qwen2.qwen_model import (
    Qwen2VLRotaryEmbedding,
    DynamicRotaryEmbedding,
)
from .qwen_config import Qwen3Config


def _load_state_dict_hook_ignore_extra_state(module, incompatible_keys):
    """Hook to ignore Transformer Engine _extra_state used for FP8.

    This is for backwards-compatibility. Newer TE versions add _extra_state keys to the state dict,
    while older models might not have those keys. Those keys can be ignored when not using FP8.

    Args:
        module (torch.nn.Module): The torch module this hook applies to. Required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys,
            which collect the missing and unexpected keys, respectively.
    """
    keys_to_remove = [
        key
        for key in incompatible_keys.missing_keys
        if "input_layernorm._extra_state" in key
        or "pre_mlp_layernorm._extra_state" in key
        or "enorm._extra_state" in key
        or "hnorm._extra_state" in key
        or "eh_proj._extra_state" in key
        or "output_layernorm._extra_state" in key
        or "self_attention.q_layernorm._extra_state" in key
        or "self_attention.k_layernorm._extra_state" in key
        or "linear_fc1._extra_state" in key
        or "linear_fc2._extra_state" in key
        or "final_layernorm._extra_state" in key
    ]

    for key in keys_to_remove:
        if key in incompatible_keys.missing_keys:
            incompatible_keys.missing_keys.remove(key)


class Qwen3VLRotaryEmbedding(Qwen2VLRotaryEmbedding):
    """Implements multimodal rotation"""
    def __init__(self, dim, theta=1000000, mrope_section=[24, 20, 20]):
        super().__init__(dim, theta)
        self.mrope_section = mrope_section
    
    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, position_ids, packed_seq):
        """Returns the frequency"""
        # Core RoPE block. In contrast to other models, Qwen2_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        ) 
        position_ids_expanded = position_ids[
            :, :, None, :
        ].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            2, 3
        )
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section) # shape (bs, seq_length, dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        # shape (seq_length, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()

        return emb


class Qwen3Model(BaseGPTModel):
    """Qwen3 Transformer language model.

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to True.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks.
            Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights
            are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional): Position embedding type, Defaults to 'rope'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type
            is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer
            sequences. The value must be a float larger than 1.0. Defaults to None.
    """

    config_class = Qwen3Config

    def __init__(
        self,
        config: Qwen3Config,
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
                "loongforge.models.foundation.qwen3.qwen_layer_spec",
                "get_qwen3_layer_with_te_spec",
            ]
        else:
            model_spec = config.model_spec
        transformer_layer_spec = import_module(model_spec, config)
        rotary_pos_emb = None
        if (
            config.rotary_emb_func == "Qwen2VLRotaryEmbedding"
            and config.position_embedding_type == "rope"
            and not config.multi_latent_attention
        ):
            rotary_pos_emb = Qwen2VLRotaryEmbedding(
                dim=config.hidden_size // config.num_attention_heads,
                theta=config.rotary_base,
            )
        elif (
            config.rotary_emb_func == "DynamicRotaryEmbedding"
            and config.position_embedding_type == "rope"
            and not config.multi_latent_attention
        ):
            rotary_pos_emb = DynamicRotaryEmbedding(
                kv_channels=config.kv_channels,
                rotary_percent=config.rotary_percent,
                rotary_interleaved=config.rotary_interleaved,
                seq_len_interpolation_factor=config.rotary_seq_len_interpolation_factor,
                rotary_base=config.rotary_base,
                dtype=rotary_dtype,
                rope_scaling=config.use_rope_scaling,
                rope_scaling_factor=config.rope_scaling_factor,
                use_cpu_initialization=config.use_cpu_initialization,
                cp_group=pg_collection.cp,
            )
        elif (
            config.rotary_emb_func == "Qwen3VLRotaryEmbedding"
            and config.position_embedding_type == "rope"
            and not config.multi_latent_attention
        ):
            rotary_pos_emb = Qwen3VLRotaryEmbedding(
                dim=config.kv_channels,
                theta=config.rotary_base,
                mrope_section=config.mrope_section
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
                not config.untie_embeddings_and_output_weights),
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
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        self.register_load_state_dict_post_hook(
            _load_state_dict_hook_ignore_extra_state
        )
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
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

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

        rotary_pos_cos = None
        rotary_pos_sin = None
        sequence_len_offset = None        
        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        if (
            rotary_pos_emb is None
            and self.position_embedding_type == "rope"
            and not self.config.multi_latent_attention
            and self.config.rotary_emb_func not in ["Qwen2VLRotaryEmbedding", "Qwen3VLRotaryEmbedding"]
        ):
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params,
                self.decoder,
                decoder_input,
                self.config,
                packed_seq_params,
            )
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd",
            )
        else:
            rotary_pos_emb = self.rotary_pos_emb(
                position_ids,
                packed_seq=packed_seq_params,
            )

        preproc_output = (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )

        return preproc_output
    
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
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """

        preproc_output = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            rotary_pos_emb=rotary_pos_emb,
        )

        (decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset) = (
            preproc_output[:5]
        )

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
        )

        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            runtime_gather_output=runtime_gather_output,
            sequence_len_offset=sequence_len_offset,
            extra_block_kwargs=extra_block_kwargs,
        )