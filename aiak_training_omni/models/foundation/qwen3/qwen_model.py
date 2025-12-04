"""qwen model"""

from collections import OrderedDict
from typing import Dict, Literal, Optional

from torch import Tensor
from .qwen_config import Qwen3Config
from megatron.core import InferenceParams, tensor_parallel, parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from aiak_training_omni.models.common.base_model_mixins import (
    BaseMegatronLanuageModule,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType, AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
import torch

from aiak_training_omni.models.utils import import_module
from aiak_training_omni.models.omni_models.utils import get_pos_emb_on_this_cp_rank


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
    ]

    for key in keys_to_remove:
        if key in incompatible_keys.missing_keys:
            incompatible_keys.missing_keys.remove(key)


class DynamicRotaryEmbedding(RotaryEmbedding):
    """Dynamic Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE for longer sequences.
        The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to 10000.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float = None,
        rotary_base: int = 10000,
        dtype: torch.dtype = torch.float32,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        use_cpu_initialization: bool = False,
        max_position_embeddings: int = 4096,
    ) -> None:
        super().__init__(
            kv_channels,
            rotary_percent,
            rotary_interleaved,
            seq_len_interpolation_factor,
            rotary_base,
            dtype,
            rope_scaling,
            rope_scaling_factor,
            use_cpu_initialization,
        )
        self.dim = kv_channels
        self.rotary_base = rotary_base
        self.scaling_factor = rope_scaling_factor
        if rotary_percent < 1.0:
            self.dim = int(self.dim * rotary_percent)
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len_cached = max_position_embeddings

    def forward(
        self, max_seq_len: int, offset: int = 0, packed_seq: bool = False
    ) -> Tensor:
        """Forward pass of RoPE embedding"""
        if max_seq_len > self.max_position_embeddings:
            base = self.rotary_base * (
                (self.scaling_factor * max_seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(
                        0,
                        self.dim,
                        2,
                        dtype=torch.float32,
                        device=torch.cuda.current_device(),
                    )
                    / self.dim
                )
            )

        return super().forward(max_seq_len, offset, packed_seq)


class Qwen2VLRotaryEmbedding(torch.nn.Module):
    """Implements multimodal rotation"""

    def __init__(self, dim, theta=1000000):
        super().__init__()
        self.inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64)
                .float()
                .to(torch.cuda.current_device())
                / dim
            )
        )

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
        ].float()  # shape (3, bs, 1, positions)

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            2, 3
        )
        emb = torch.cat((freqs, freqs), dim=-1)

        if parallel_state.get_context_parallel_world_size() > 1:
            emb = get_pos_emb_on_this_cp_rank(emb, 2, packed_seq)

        return emb


class Qwen3Model(BaseMegatronLanuageModule):
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
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)

        if has_config_logger_enabled(self.config):
            log_config_to_disk(self.config, locals(), prefix=type(self).__name__)
        if self.config.model_spec is None:
            model_spec = [
                "aiak_training_omni.models.foundation.qwen3.qwen_layer_spec",
                "get_qwen3_layer_with_te_spec",
            ]
        else:
            model_spec = self.config.model_spec
        self.transformer_layer_spec = import_module(model_spec, self.config)
        self.vocab_size = self.config.padded_vocab_size
        self.max_sequence_length = self.config.max_position_embeddings
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = self.config.fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = not self.config.untie_embeddings_and_output_weights
        self.position_embedding_type = self.config.position_embedding_type

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder
        self.rotary_dtype = rotary_dtype

        # These 4 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rotary_percent = self.config.rotary_percent
        self.rotary_base = self.config.rotary_base
        self.rotary_scaling = self.config.use_rope_scaling
        self.rope_scaling = self.config.use_rope_scaling
        self.rope_scaling_factor = self.config.rope_scaling_factor
        self.seq_len_interpolation_factor = self.config.rotary_seq_len_interpolation_factor


        # TODO: implement learned absolute position embedding
        if self.pre_process:
            if language_embedding is None:
                self.embedding = LanguageModelEmbedding(
                    config=self.config,
                    vocab_size=self.vocab_size,
                    max_sequence_length=self.max_sequence_length,
                    position_embedding_type=self.position_embedding_type,
                    scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                )
            else:
                self.embedding = language_embedding

        if (
            self.config.rotary_emb_func == "Qwen2VLRotaryEmbedding"
            and self.position_embedding_type == "rope"
            and not self.config.multi_latent_attention
        ):
            self.rotary_pos_emb = Qwen2VLRotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_attention_heads,
                theta=self.rotary_base,
            )
        elif (
            self.config.rotary_emb_func == "DynamicRotaryEmbedding"
            and self.position_embedding_type == "rope"
            and not self.config.multi_latent_attention
        ):
            self.rotary_pos_emb = DynamicRotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=self.rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=self.seq_len_interpolation_factor,
                rotary_base=self.rotary_base,
                dtype=self.rotary_dtype,
                rope_scaling=self.rope_scaling,
                rope_scaling_factor=self.rope_scaling_factor,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )
        elif (
            self.config.rotary_emb_func == "RotaryEmbedding"
            and self.position_embedding_type == "rope"
            and not self.config.multi_latent_attention
        ):
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=self.rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=self.seq_len_interpolation_factor,
                rotary_base=self.rotary_base,
                rope_scaling=self.rope_scaling,
                rope_scaling_factor=self.rope_scaling_factor,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )
        else:
            raise NotImplementedError(
                f"Rotarty embedding type {self.config.rotary_emb_func} not implemented."
            )
        # Cache for RoPE tensors which do not change between iterations.
        self.rotary_pos_emb_cache = {}

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_layer_spec,
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
        if hasattr(config, 'freeze') and config.freeze:
            self.freeze()

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

        assert len(input_tensor) == 1, "input_tensor should only be length 1"
        self.decoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        attn_mask_type: Optional[AttnMaskType] = None,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        rotary_pos_emb: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
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

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        if (
            rotary_pos_emb is None
            and self.position_embedding_type == "rope"
            and not self.config.multi_latent_attention
            and self.config.rotary_emb_func != "Qwen2VLRotaryEmbedding"
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
            rotary_pos_emb = (
                self.rotary_pos_emb(
                    position_ids,
                    packed_seq=packed_seq_params,
                )
                .transpose(0, 2)
                .contiguous()
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

        loss = self.compute_language_model_loss(labels, logits)

        return loss

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[Dict] = None,
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
        sharded_state_dict = super().sharded_state_dict(
            prefix, sharded_offsets, metadata
        )
        output_layer_extra_state_key = f"{prefix}output_layer._extra_state"

        # Old GPT checkpoints only stored the output layer weight key. So we remove the
        # _extra_state key but check that it doesn't contain any data anyway
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        assert not (
            output_extra_state and output_extra_state.data
        ), f"Expected output layer extra state to be empty, got: {output_extra_state}"

        return sharded_state_dict
