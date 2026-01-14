"""DeepSeek model"""

from contextlib import nullcontext
import logging
from collections import OrderedDict
from typing import Dict, Literal, Optional, Any

import torch
import json
from torch import Tensor

from megatron.core import InferenceParams, tensor_parallel, parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType, AttnMaskType
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    MultiTokenPredictionBlock,
    roll_tensor,
    tie_output_layer_state_dict,
    tie_word_embeddings_state_dict,
)
from aiak_training_omni.models.foundation import DeepseekConfig
from aiak_training_omni.models.utils import import_module
from aiak_training_omni.models.common.base_model_mixins import (
    BaseMegatronLanguageModule,
)


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
        or "linear_fc1._extra_state" in key
        or "linear_fc2._extra_state" in key
    ]

    for key in keys_to_remove:
        if key in incompatible_keys.missing_keys:
            incompatible_keys.missing_keys.remove(key)


class DeepseekModelWithMTP(BaseMegatronLanguageModule):
    """DeepSeek Transformer language model with MTP module supported..

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks.
            Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights
            are shared. Defaults to False.
        share_mtp_embeddings_and_output_weights (bool, optional): When True, MTP layers' embeddings and output logit
            weights are shared with main model. Default to True.
        position_embedding_type (Literal[learned_absolute,rope], optional):  Position embedding type.. Defaults to
            'learned_absolute'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for
            longer sequences. The value must be a float larger than 1.0. Defaults to None.th
    """

    config_class = DeepseekConfig

    def __init__(
        self,
        config: DeepseekConfig,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        scatter_embedding_sequence_parallel: bool = True,
        language_embedding: Optional[torch.nn.Module] = None,
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        **kwargs,
    ) -> None:
        """Initialize pre-process, transformer block and post-process modules."""
        super().__init__(config=config, pg_collection=pg_collection, **kwargs)

        if has_config_logger_enabled(self.config):
            log_config_to_disk(self.config, locals(), prefix=type(self).__name__)

        if self.config.model_spec is None:
            model_spec = [
                "aiak_training_omni.models.foundation.deepseek.deepseek_layer_spec",
                "get_deepseek_decoder_block_and_mtp_spec",
            ]
        else:
            model_spec = self.config.model_spec

        # TODO: how to pass this param?
        self.scatter_embedding_sequence_parallel = scatter_embedding_sequence_parallel
        
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        self.vocab_size = self.config.padded_vocab_size
        # self.vocab_size = args.padded_vocab_size
        self.fp16_lm_cross_entropy = self.config.fp16_lm_cross_entropy
        self.seq_len_interpolation_factor = self.config.rotary_seq_len_interpolation_factor


        self.transformer_layer_spec, self.mtp_layer_spec = import_module(model_spec, self.config)
        self.mtp_process = self.mtp_layer_spec is not None
        self.vp_stage = vp_stage

        self.max_sequence_length = self.config.max_position_embeddings
        self.share_embeddings_and_output_weights = not self.config.untie_embeddings_and_output_weights
        self.position_embedding_type = self.config.position_embedding_type

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # These 4 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rotary_percent = self.config.rotary_percent
        self.rotary_base = self.config.rotary_base
        self.rope_scaling = self.config.use_rope_scaling
        self.rope_scaling_factor = self.config.rope_scaling_factor

        if self.pre_process or self.mtp_process:
            if language_embedding is None:
                self.embedding = LanguageModelEmbedding(
                    config=self.config,
                    vocab_size=self.vocab_size,
                    max_sequence_length=self.max_sequence_length,
                    position_embedding_type=self.position_embedding_type,
                    scatter_to_sequence_parallel=self.scatter_embedding_sequence_parallel,
                    tp_group=self.pg_collection.tp,
                )
            else:
                self.embedding = language_embedding

        if (
            self.position_embedding_type == "rope"
            and not self.config.multi_latent_attention
        ):
            # unused for deepseek
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=self.rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=self.seq_len_interpolation_factor,
                rotary_base=self.rotary_base,
                rope_scaling=self.rope_scaling,
                rope_scaling_factor=self.rope_scaling_factor,
                use_cpu_initialization=self.config.use_cpu_initialization,
                cp_group=self.pg_collection.cp,
            )

        # Cache for RoPE tensors which do not change between iterations.
        self.rotary_pos_emb_cache = {}

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=self.pg_collection,
            vp_stage=vp_stage,
        )

        if self.mtp_process:
            self.mtp = MultiTokenPredictionBlock(
                config=self.config, spec=self.mtp_block_spec, vp_stage=vp_stage
            )

        # Output
        if post_process:
            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs stored
                # in gradient buffer to calculate the weight gradients for the embedding final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                self.config.hidden_size,
                self.vocab_size,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
                tp_group=self.pg_collection.tp,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

        if has_config_logger_enabled(self.config):
            log_config_to_disk(
                self.config,
                self.state_dict(),
                prefix=f"{type(self).__name__}_init_ckpt",
            )

        self.register_load_state_dict_post_hook(
            _load_state_dict_hook_ignore_extra_state
        )

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert (
            len(input_tensor) == 1
        ), "input_tensor should only be length 1 for gpt/bert"
        self.decoder.set_input_tensor(input_tensor[0])

    def _preprocess(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        decoder_input: Tensor = None,
        #inference_context: BaseInferenceContext = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Preprocesses inputs for the transformer decoder.

        Applies embeddings to input tokens, or uses `decoder_input` from a previous
        pipeline stage. Also sets up rotary positional embeddings.
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config, packed_seq_params
            )
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None and packed_seq_params.qkv_format == 'thd',
            )

        preproc_output = (
            decoder_input,
            rotary_pos_emb,
        )

        return preproc_output
    
    
    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        attn_mask_type: Optional[AttnMaskType] = None,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        loss_mask: Optional[Tensor] = None,
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
        )

        (decoder_input, rotary_pos_emb) = (
            preproc_output[:2]
        )

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
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

    def _postprocess(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        mtp_in_postprocess=None,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
    ):
        """Postprocesses decoder hidden states to generate logits or compute loss.

        Applies Multi-Token Prediction if enabled, generates output logits through
        the output layer, and computes language model loss when labels are provided.
        """

        # logits and loss
        output_weight = None
        mtp_loss = None

        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if mtp_in_postprocess:
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
                embedding=self.embedding,
                **(extra_block_kwargs or {}),
            )

        if not self.post_process:
            return hidden_states
        
        if self.mtp_process:
            mtp_labels = labels.clone()
            hidden_states_list = torch.chunk(hidden_states, 1 + self.config.mtp_num_layers, dim=0)
            hidden_states = hidden_states_list[0]
            if loss_mask is None:
                # if loss_mask is not provided, use all ones as loss_mask
                loss_mask = torch.ones_like(mtp_labels)
            for mtp_layer_number in range(self.config.mtp_num_layers):
                # output
                mtp_logits, _ = self.output_layer(
                    hidden_states_list[mtp_layer_number + 1],
                    weight=output_weight,
                    runtime_gather_output=runtime_gather_output,
                )
                # Calc loss for the current Multi-Token Prediction (MTP) layers.
                mtp_labels, _ = roll_tensor(mtp_labels, shifts=-1, dims=-1, cp_group=self.cp_group)
                loss_mask, num_tokens = roll_tensor(
                    loss_mask, shifts=-1, dims=-1, cp_group=self.cp_group
                )
                mtp_loss = self.compute_language_model_loss(mtp_labels, mtp_logits)
                mtp_loss = loss_mask * mtp_loss
                if self.training:
                    MTPLossLoggingHelper.save_loss_to_tracker(
                        torch.sum(mtp_loss) / num_tokens,
                        mtp_layer_number,
                        self.config.mtp_num_layers,
                        avg_group=parallel_state.get_data_parallel_group(
                            with_context_parallel=True
                        ),
                    )
                mtp_loss_scale = self.config.mtp_loss_scaling_factor / self.config.mtp_num_layers
                if self.config.calculate_per_token_loss:
                    hidden_states = MTPLossAutoScaler.apply(
                        hidden_states, mtp_loss_scale * mtp_loss
                    )
                else:
                    hidden_states = MTPLossAutoScaler.apply(
                        hidden_states, mtp_loss_scale * mtp_loss / num_tokens
                    )

        logits, _ = self.output_layer(
            hidden_states,
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        )

        if has_config_logger_enabled(self.config):
            payload = OrderedDict(
                {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "decoder_input": decoder_input,
                    "logits": logits,
                }
            )
            log_config_to_disk(self.config, payload, prefix="input_and_logits")

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        # compute main model loss
        loss = self.compute_language_model_loss(labels, logits)

        return loss

    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the embedding weight or output logit weights when share input embedding and
        output weights set to True or when use Multi-Token Prediction (MTP) feature.

        Returns:
            Tensor: During pre processing or MTP process it returns the input embeddings weight.
            Otherwise, during post processing it returns the final output layers weight.
        """
        if self.pre_process or self.mtp_process:
            # Multi-Token Prediction (MTP) need both embedding layer and output layer.
            # So there will be both embedding layer and output layer in the mtp process stage.
            # In this case, if share_embeddings_and_output_weights is True, the shared weights
            # will be stored in embedding layer, and output layer will not have any weight.
            assert hasattr(
                self, 'embedding'
            ), f"embedding is needed in this pipeline stage, but it is not initialized."
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None
    
    
    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[Dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility.

        Removing extra state.
        Tie word embeddings and output layer in mtp process stage.

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        output_layer_extra_state_key = f'{prefix}output_layer._extra_state'

        # Old GPT checkpoints only stored the output layer weight key. So we remove the
        # _extra_state key but check that it doesn't contain any data anyway
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        assert not (
            output_extra_state and output_extra_state.data
        ), f'Expected output layer extra state to be empty, got: {output_extra_state}'

        # Multi-Token Prediction (MTP) need both embedding layer and output layer in
        # mtp process stage.
        # If MTP is not placed in the pre processing stage, we need to maintain a copy of
        # embedding layer in the mtp process stage and tie it to the embedding in the pre
        # processing stage.
        # Also, if MTP is not placed in the post processing stage, we need to maintain a copy
        # of output layer in the mtp process stage and tie it to the output layer in the post
        # processing stage.
        if self.mtp_process and not self.pre_process:
            emb_weight_key = f'{prefix}embedding.word_embeddings.weight'
            emb_weight = self.embedding.word_embeddings.weight
            tie_word_embeddings_state_dict(sharded_state_dict, emb_weight, emb_weight_key)
        if self.mtp_process and not self.post_process:
            # We only need to tie the output layer weight if share_embeddings_and_output_weights
            # is False. Because if share_embeddings_and_output_weights is True, the shared weight
            # will be stored in embedding layer, and output layer will not have any weight.
            if not self.share_embeddings_and_output_weights:
                output_layer_weight_key = f'{prefix}output_layer.weight'
                output_layer_weight = self.output_layer.weight
                tie_output_layer_state_dict(
                    sharded_state_dict, output_layer_weight, output_layer_weight_key
                )

        return sharded_state_dict

    def build_schedule_plan(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attn_mask_type: Optional[AttnMaskType] = None,
        decode_input: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        **kwargs: Any,
    ):
        """Build the schedule plan for the model."""
        from aiak_training_omni.models.common.fine_grained_schedule import build_model_chunk_schedule_plan

        return build_model_chunk_schedule_plan(
            self,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            labels=labels,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **kwargs
        )
