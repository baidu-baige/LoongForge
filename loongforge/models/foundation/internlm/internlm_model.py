# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""internlm model"""
from typing import Optional

import torch
from torch import Tensor
from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

from loongforge.models.utils import import_module
from loongforge.models.foundation.base import BaseGPTModel
from .internlm_config import InternLMConfig


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
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__(
            kv_channels=kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            dtype=dtype,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            use_cpu_initialization=use_cpu_initialization,
            cp_group=cp_group,
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


class InternLMModel(BaseGPTModel):
    """InternLM Transformer language model.

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

    config_class = InternLMConfig

    def __init__(
        self,
        config: InternLMConfig,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        rotary_dtype: torch.dtype = torch.float32,
        scatter_embedding_sequence_parallel: bool = True,
        language_embedding: Optional[torch.nn.Module] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        **kwargs,
    ) -> None:

        if config.model_spec is None:
            model_spec = [
                "loongforge.models.foundation.internlm.internlm_layer_spec",
                "get_internlm_layer_with_te_spec",
            ]
        else:
            model_spec = config.model_spec
        transformer_layer_spec = import_module(model_spec, config)
        rotary_pos_emb = None
        if (
            config.position_embedding_type == "rope"
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
                cp_group=pg_collection.cp if pg_collection else None,
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
            share_embeddings_and_output_weights=(not config.untie_embeddings_and_output_weights),
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

        if hasattr(config, 'freeze') and config.freeze:
            self.freeze()

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        loss_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Forward function of the InternLM Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """
        
        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            loss_mask=loss_mask,
            **kwargs,
        )