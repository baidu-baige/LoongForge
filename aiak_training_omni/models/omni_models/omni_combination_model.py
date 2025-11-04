"""
编排 encoder -> foundation -> decoder 的执行流程
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .configuration import OmniModelConfig
from .omni_encoder_model import OmniEncoderModel
from .omni_foundation_model import OmniFoundationModel
from .omni_decoder_model import OmniDecoderModel
from transformers.models.auto.modeling_auto import AutoModel
from aiak_training_llm.models.omni.base_mixins import BaseMegatronModuler
from megatron.core.transformer.enums import AttnMaskType
from megatron.core import InferenceParams
from megatron.core.transformer.module import MegatronModule
from aiak_training_llm.models.omni.base_mixins import core_transformer_config_from_hf_config


class OmniCombinationModel(BaseMegatronModuler):
    """Omni 多模态组合模型"""
    def __init__(self, config: OmniModelConfig,
                        train_args,
                        language_vocab_size: int,
                        language_max_sequence_length: int,
                        allow_missing_adapter_checkpoint: bool=False,
                        parallel_output: bool=True,
                        language_position_embedding_type: str='rope',
                        language_rotary_percent: float=1.0,
                        pre_process: bool=True,
                        post_process: bool=True,
                        add_encoder: bool=True,
                        add_decoder: bool=True,
                        language_rotary_base: int=1000000,
                        language_rope_scaling: bool = False,
                        language_rope_scaling_factor: float = 8.0,
                        language_rotary_dtype=torch.float32,
                        fp16_lm_cross_entropy: bool=False,
                        share_embeddings_and_output_weights: bool=True,
                        seq_len_interpolation_factor: float=None,
                        scatter_embedding_sequence_parallel=False
                 ) -> None:
        super().__init__(config.foundation_config, train_args)
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        if config.foundation is not None and add_decoder:
            # self.rotary_emb = Qwen2VLRotaryEmbedding(
            #     dim=config.foundation_config.hidden_size // config.foundation_config.num_attention_heads,
            #     theta=language_rotary_base
            # )
            self.foundation_model = AutoModel.from_config(
                config.foundation, 
                train_args=train_args,                
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                parallel_output=parallel_output,
                position_embedding_type=language_position_embedding_type,
                rotary_percent=language_rotary_percent,
                pre_process=False,
                post_process=self.post_process,
                rotary_base=language_rotary_base,
                rotary_dtype=language_rotary_dtype,
                rope_scaling=language_rope_scaling,
                rope_scaling_factor=language_rope_scaling_factor,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights)
            self.config = self.foundation_model.config
        else:
            raise ValueError("OmniCombinationModel requires a foundation_config to initialize foundation_model.")

        # 构建子模型
        if config.encoder_config is not None and add_encoder:
            self.encoder_model = OmniEncoderModel(
                config.encoder_config, 
                self.foundation_model.embedding,
                self.foundation_model.config,
                train_args,
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                position_embedding_type=language_position_embedding_type,
                scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
                allow_missing_adapter_checkpoint=allow_missing_adapter_checkpoint)
        else:
            self.encoder_model = None

        if config.decoder_config is not None:
            self.decoder_model = OmniDecoderModel(config.decoder_config)
        else:
            self.decoder_model = None
        self.share_embeddings_and_output_weights = (
                        self.foundation_model.share_embeddings_and_output_weights
                    )


    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.foundation_model.shared_embedding_or_output_weight()
        return None
    def set_input_embeddings(self, inputs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """跳过encoder,使用输入中的值用于后续 foundation + decoder 模型。"""
        input_embeds = inputs.get("inputs_embeds")
        decoder_inputs = inputs.get("decoder_inputs", {})
        
        if input_embeds is None:
            raise ValueError("离线模式下必须提供 inputs_embeds")
            
        return input_embeds, decoder_inputs


    def set_input_tensor(self, input_tensor) -> None:
        """set input tensor"""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for llava'

        if self.add_encoder and self.add_decoder:
            self.encoder_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.encoder_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0] # TOOD
        else:
            self.foundation_model.set_input_tensor(input_tensor[0])

    def set_output_embeddings(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """跳过foundation model,使用输入中的值用于decoder模型。"""
        output_embeddings = inputs.get("output_embeddings")
        
        if output_embeddings is None:
            raise ValueError("离线模式下必须提供 output_embeddings")
            
        return output_embeddings

    def get_modality(self):
        """获取当前模型的输入输出类型"""
        input_modality = self.encoder.modality
        output_modality = self.decoder.modality
        return {"input": input_modality, "output": output_modality}

    
    def prepare_inputs_for_generation(self, input_ids: Optional[torch.LongTensor]=None):
        """准备生成过程所需的输入"""
        pass

    def generate_multimodal(self, hidden_states):
        """生成多模态数据"""
        pass
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None,
        attn_mask_type: Optional[AttnMaskType]=None,
        packed_seq_params=None,
        labels: Optional[torch.LongTensor]=None,
        inference_params: InferenceParams=None,
        image_inputs: Optional[Dict[str, torch.Tensor]]=None,
        video_inputs: Optional[Dict[str, torch.Tensor]]=None,
        audio_inputs: Optional[Dict[str, torch.Tensor]]=None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """前向传播，支持多种执行路径。

        执行路径：
        1. 完整路径:encoder -> foundation -> decoder
        2. 离线编码器：使用预处理的 inputs_embeds
        3. 离线foundation model:使用预处理的 output_embeddings
        4. 仅解码器：冻结 encoder 和 foundation
        """
        use_inference_kv_cache = (
                    inference_params is not None
                    and "image_tokens_count" in inference_params.key_value_memory_dict
                )

        if use_inference_kv_cache:
            vision_embeddings = None # 使用默认值
        elif self.add_encoder:
            combined_embeddings, decode_input = self.encoder_model(
                input_ids=input_ids,
                image_inputs=image_inputs,
                video_inputs=video_inputs,
                inference_params=inference_params
            )
        else:
            combined_embeddings = None
        
        # rotary_pos_emb = self.rotary_emb(
        #     position_ids,
        #     packed_seq=packed_seq_params,
        # ).transpose(0, 2).contiguous()

        output = self.foundation_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            decoder_input=combined_embeddings,
            labels=labels,
            # rotary_pos_emb=rotary_pos_emb,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs={},
        )

        return output
