"""DeepSeek model"""

from contextlib import nullcontext
import logging
from collections import OrderedDict
from typing import Dict, Literal, Optional

import torch
import json
from torch import Tensor

from megatron.core import InferenceParams, tensor_parallel, parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType, AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock

from aiak_training_omni.models.deepseek.transformer import DeepSeekTransformerConfig
from aiak_training_omni.models.deepseek.transformer.mtp_transformer_layer import MultiTokenPredLayerDeepSeek


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
        key for key in incompatible_keys.missing_keys
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


class DeepseekModelWithMTP(LanguageModule):
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

    def __init__(
        self,
        config: DeepSeekTransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        share_mtp_embeddings_and_output_weights: bool = True,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_layer_spec: ModuleSpec = None,
    ) -> None:
        """Initialize pre-process, transformer block and post-process modules."""
        super().__init__(config=config)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # These 4 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent
        self.rotary_base = rotary_base
        self.rope_scaling = rope_scaling

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
            )

        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            # unused for deepseek
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                rope_scaling=rope_scaling,
                rope_scaling_factor=rope_scaling_factor,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )

        # Cache for RoPE tensors which do not change between iterations.
        self.rotary_pos_emb_cache = {}

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
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
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

        # MTP
        if self.config.num_nextn_predict_layers > 0:
            self.share_mtp_embeddings_and_output_weights = share_mtp_embeddings_and_output_weights
        else:
            self.share_mtp_embeddings_and_output_weights = False
            logging.getLogger(__name__).warning(
                f"`config.num_nextn_predict_layers` is {self.config.num_nextn_predict_layers}, "
                "`share_mtp_embeddings_and_output_weights` will not take effect, "
                "fall back to default value of False."
            )

        # Initialize the MTP layers
        self.mtp_layers = None
        if self.config.num_nextn_predict_layers > 0:
            if post_process and self.training:
                self.mtp_layers = torch.nn.ModuleList(
                    [
                        MultiTokenPredLayerDeepSeek(
                            config,
                            submodules=mtp_layer_spec.submodules,
                            # Params for MTP embedding layer
                            vocab_size=vocab_size,
                            max_sequence_length=max_sequence_length,
                            position_embedding_type=position_embedding_type,
                            rotary_percent=rotary_percent,
                            rotary_base=rotary_base, 
                            seq_len_interpolation_factor=seq_len_interpolation_factor,
                            # Params for MTP layer
                            share_mtp_embeddings_and_output_weights=True,
                            # Params for TransformerLayer
                            layer_number=len(self.decoder.submodules.layer_specs) + 1 + i,
                            # Params for parallel
                            pre_process=pre_process,
                            post_process=post_process,
                        )
                        for i in range(self.config.num_nextn_predict_layers)
                    ]
                )
            # handle the extra MTP logic.
            if self.pre_process or self.post_process:
                self.setup_mtp_embeddings_layer()

        if has_config_logger_enabled(self.config):
            log_config_to_disk(self.config, self.state_dict(), prefix=f'{type(self).__name__}_init_ckpt')

        self.register_load_state_dict_post_hook(_load_state_dict_hook_ignore_extra_state)

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

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def setup_mtp_embeddings_layer(self) -> None:
        """Sets up embedding layer in first stage and MTP module's embedding layer(s) in last stage.

        This function initalizes word embeddings in the final stage when we are
        using pipeline parallelism and sharing word embeddings, and sets up param
        attributes on the embedding layers on first and last stage.
        """
        # Similar to setup_embeddings_and_output_layer(), setup `is_embedding_or_output_parameter` attribute
        if self.post_process:
            for _mtp_layer in self.mtp_layers:
                # ...weight is not None check is for `skip_weight_param_allocation` is True.
                if _mtp_layer.embedding.word_embeddings.weight is not None:
                    _mtp_layer.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
        
        # If the MTP's embeddings and output weights are not shared with the main model.
        if not self.share_mtp_embeddings_and_output_weights:
            return
        
        if all([
            self.pre_process,
            self.post_process
        ]):
            # Zero out wgrad if sharing embeddings between two layers on same
            # pipeline stage to make sure grad accumulation into main_grad is
            # correct and does not include garbage values (e.g., from torch.empty).
            self.shared_embedding_weight().zero_out_wgrad = True
            return
        
        if all([
            parallel_state.is_pipeline_first_stage(),
            self.pre_process,
            not self.post_process
        ]):
            self.shared_embedding_weight().shared_embedding = True

        if self.post_process and not self.pre_process:
            assert not parallel_state.is_pipeline_first_stage()
            for _mtp_layer in self.mtp_layers:
                # set word_embeddings weights to 0 here, then copy first
                # stage's weights using all_reduce below.
                _mtp_layer.embedding.word_embeddings.weight.data.fill_(0)
                _mtp_layer.embedding.word_embeddings.weight.shared = True
                _mtp_layer.embedding.word_embeddings.weight.shared_embedding = True

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.

        # Ensure that first and last stages have the same initial parameter
        # values.
        if torch.distributed.is_initialized():
            if parallel_state.is_rank_in_embedding_group():
                weight = self.shared_embedding_weight()
                weight.data = weight.data.cuda()
                torch.distributed.all_reduce(
                    weight.data, group=parallel_state.get_embedding_group()
                )

        elif not getattr(LanguageModule, "embedding_warning_printed", False):
            logging.getLogger(__name__).warning(
                "Distributed processes aren't initialized, so the output layer "
                "is not initialized with weights from the word embeddings. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )
            LanguageModule.embedding_warning_printed = True

    def shared_embedding_weight(self) -> Optional[Tensor]:
        """
        For MTP involved only. Gets the embedding weight when `share_mtp_embeddings_and_output_weights`
        is True. Which means the embedding weight is shared between all MTP layers.

        Returns:
            Tensor: Returns the input embedding weight if in first stage of pipeline(pre_process),
            or returns the MTP embedding weight if in last stage of pipeline(post_process).
        """
        assert self.config.num_nextn_predict_layers > 0
        if self.pre_process:
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.mtp_layers[0].embedding.word_embeddings.weight
        else:
            return None

    def _mtp_forward(
        self,
        decoder_input: Tensor,
        ori_input_ids: Tensor,
        ori_labels: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        attn_mask_type: Optional[AttnMaskType] = None,
        rotary_pos_emb: Tensor = None,
        loss: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
    ) -> Tensor:
        """compute the forward path of each MTP layer"""
        # Prepare the embeddings and output weights in MTP
        # If embeddings and output weights are not tied, and MTP's output weights
        # are shared with main model.
        if not self.share_embeddings_and_output_weights and self.share_mtp_embeddings_and_output_weights:
            output_weight = self.output_layer.weight.detach()
            output_weight.zero_out_wgrad = True

        # If MTP's embedding weights are shared with main model.
        if self.share_mtp_embeddings_and_output_weights:
            embed_weight = self.shared_embedding_weight()
        else:
            embed_weight = None

        for mtp_i, mtp_layer in enumerate(self.mtp_layers):

            # Shift right by `mtp_depth` and pad back to regular length
            mtp_input_ids = torch.nn.functional.pad(
                ori_input_ids[:, mtp_i + 1:],  # [b, s-mtp_depth]
                (0, mtp_i + 1), "constant", 0,  # [b, s]
            ).contiguous()

            mtp_labels = torch.nn.functional.pad(
                ori_labels[:, mtp_i + 1:],  # [b, s-mtp_depth]
                (0, mtp_i + 1), "constant", 0,  # [b, s]
            ).contiguous()

            if self.pre_process and self.post_process:
                decoder_input = torch.nn.functional.pad(
                    decoder_input[mtp_i + 1:, ...],  # [s-mtp_depth, b, h]
                    (0, 0, 0, 0, 0, mtp_i + 1), "constant", 0,  # [s, b, h]
                ).contiguous()

            hidden_states, mtp_loss = mtp_layer(
                hidden_states=hidden_states,
                input_ids=mtp_input_ids,
                decoder_input=decoder_input,
                labels=mtp_labels,
                attention_mask=attention_mask,
                attn_mask_type=attn_mask_type,
                rotary_pos_emb=rotary_pos_emb,
                position_ids=position_ids,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                embed_weight=embed_weight,
                output_weight=output_weight,
            )

            mtp_loss[:, -(mtp_i + 1):] = 0.0
            loss += mtp_loss * self.config.mtp_loss_coef / self.config.num_nextn_predict_layers  # [b, s]

        return loss

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
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """
        if self.mtp_layers is not None:
            ori_input_ids = input_ids.detach()  # [b, s]
            ori_labels = labels.detach()  # [b, s]
            # TODO: Truncate the input_ids and labels to support MTP.
            # input_ids = input_ids[:, :-self.config.num_nextn_predict_layers]
            # labels = labels[:, :-self.config.num_nextn_predict_layers]

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

        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )

        if has_config_logger_enabled(self.config):
            payload = OrderedDict(
                {
                    'input_ids': input_ids,
                    'position_ids': position_ids,
                    'attention_mask': attention_mask,
                    'decoder_input': decoder_input,
                    'logits': logits,
                }
            )
            log_config_to_disk(self.config, payload, prefix='input_and_logits')

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        # compute main model loss
        loss = self.compute_language_model_loss(labels, logits)
        
        # compute mtp loss
        if self.mtp_layers is not None and self.training:
            loss = self._mtp_forward(
                decoder_input=decoder_input,
                ori_input_ids=ori_input_ids,
                ori_labels=ori_labels,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                attn_mask_type=attn_mask_type,
                rotary_pos_emb=rotary_pos_emb,
                loss=loss,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
            )

        return loss

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility
        (removing extra state).

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

        return sharded_state_dict

    def build_schedule_plan(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        attn_mask_type=None,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
    ):
        """ Build the schedule plan for the model."""
        from .fine_grained_schedule import build_model_chunk_schedule_plan

        return build_model_chunk_schedule_plan(
            self,
            input_ids,
            position_ids,
            attention_mask,
            attn_mask_type=attn_mask_type,
            decoder_input=decoder_input,
            labels=labels,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            num_nextn_predict_layers=self.config.num_nextn_predict_layers
        )
