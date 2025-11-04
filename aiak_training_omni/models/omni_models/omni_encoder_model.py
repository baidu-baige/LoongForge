"""OmniEncoderModel:组合多模态编码器以产生输入嵌入。

此模块将模态特征映射到文本嵌入空间并返回：
    - 供基础模型使用的 input_embeds
    - 供可选解码器使用的 decoder_inputs 
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Literal

import torch
import torch.nn as nn
from collections import namedtuple
from functools import partial
from .configuration import OmniEncoderConfig
from ..common.base_model_mixins import BaseMegatronModuler
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from transformers.models.auto.modeling_auto import AutoModel


class OmniEncoderModel(torch.nn.Module):
    """Omni 多模态编码器模型。"""
    config_class = OmniEncoderConfig

    def __init__(self, config: OmniEncoderConfig, 
                 language_embedding,
                 foundation_config,
                 train_args,
                 vocab_size,
                 max_sequence_length,
                 position_embedding_type: Literal['learned_absolute', 'rope'] = 'rope',
                 scatter_embedding_sequence_parallel: Optional[float] = None,
                 allow_missing_adapter_checkpoint=False,
                 **kwargs) -> None:
        super().__init__()
        self.config = config
        self.modality: List[str] = []

        # 文本encoder
        self.text_encoder = language_embedding
        # mutli-modal encoder
        if self.config.image_encoder_config is not None:
            self.image_encoder: BaseMegatronModuler = AutoModel.from_config(self.config.image_encoder_config, 
                                                                            train_args=train_args, 
                                                                            **kwargs)
            self.modality.append("image")

        if self.config.video_encoder_config is not None:
            self.video_encoder: BaseMegatronModuler = AutoModel.from_config(self.config.video_encoder_config,
                                                                            train_args=train_args, 
                                                                            **kwargs)
            self.modality.append("video")

        if self.config.audio_encoder_config is not None:
            self.audio_encoder: BaseMegatronModuler = AutoModel.from_config(self.config.audio_encoder_config,  
                                                                            train_args=train_args, 
                                                                            **kwargs)
            self.modality.append("audio")

        if self.config.image_projector_config is not None:
            self.image_projector: BaseMegatronModuler = \
                                        AutoModel.from_config(self.config.image_projector_config,  
                                                            train_args=train_args,
                                                            input_size=self.config.image_encoder_config.hidden_size,
                                                            output_size=foundation_config.hidden_size,
                                                            **kwargs)
            if allow_missing_adapter_checkpoint:
                adapter_param_names = [
                    f"image_projector.{name}"
                    for name in self.image_projector.state_dict().keys()
                ]
                self.image_projector.register_load_state_dict_post_hook(
                    partial(_load_state_dict_hook_ignore_param_names, adapter_param_names)
                )
        if self.config.video_projector_config is not None:
            self.video_projector: BaseMegatronModuler = AutoModel.from_config(self.config.video_projector_config,
                                                                                train_args=train_args,
                                                                                **kwargs)
            if allow_missing_adapter_checkpoint:
                adapter_param_names = [
                    f"video_projector.{name}"
                    for name in self.video_projector.state_dict().keys()
                ]
                self.video_projector.register_load_state_dict_post_hook(
                    partial(_load_state_dict_hook_ignore_param_names, adapter_param_names)
                )
        if self.config.audio_projector_config is not None:
            self.audio_projector: BaseMegatronModuler = AutoModel.from_config(self.config.audio_projector_config,  
                                                                              train_args=train_args, 
                                                                              **kwargs)
            if allow_missing_adapter_checkpoint:
                adapter_param_names = [
                    f"audio_projector.{name}"
                    for name in self.audio_projector.state_dict().keys()
                ]
                self.audio_projector.register_load_state_dict_post_hook(
                    partial(_load_state_dict_hook_ignore_param_names, adapter_param_names)
                )

    def set_input_tensor(self, input_tensor) -> None:
        """set input tensor"""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for llava'

        if self.image_encoder is not None:
            self.image_encoder.set_input_tensor(input_tensor[0])
        else:
            raise ValueError("image_encoder is None, cannot set input tensor.")


    def set_projector_trainable_only(self):
        """pipeline parallel schedule interact"""
        for module in self.children():
            if isinstance(module, BaseMegatronModuler):
                module.set_projector_trainable_only()

    # def freeze(self, modality: Optional[str] = None, freeze_text_encoder: bool = True) -> None:
    #     """冻结编码器模型参数。
        
    #     参数:
    #     ----------
    #     modality : str, 可选
    #         要冻结的特定模态 ('image', 'video', 'audio')。如果为 None,则冻结所有模态编码器。
    #     freeze_text_encoder : bool, 默认 True
    #         是否同时冻结文本编码器。
    #     """
    #     if modality is not None:
    #         # 冻结特定模态的编码器
    #         if modality not in self.modality:
    #             raise ValueError(f"模态 '{modality}' 不存在。可用模态: {self.modality}")
            
    #         encoder_name = f"{modality}_encoder"
    #         if hasattr(self, encoder_name):
    #             encoder = getattr(self, encoder_name)
    #             for param in encoder.parameters():
    #                 param.requires_grad = False
    #             print(f"已冻结 {modality} 编码器")
    #     else:
    #         # 冻结所有模态编码器
    #         for mod in self.modality:
    #             encoder_name = f"{mod}_encoder"
    #             if hasattr(self, encoder_name):
    #                 encoder = getattr(self, encoder_name)
    #                 for param in encoder.parameters():
    #                     param.requires_grad = False
    #                 print(f"已冻结 {mod} 编码器")
        
    #     # 冻结文本编码器
    #     if freeze_text_encoder and hasattr(self, 'text_encoder'):
    #         for param in self.text_encoder.parameters():
    #             param.requires_grad = False
    #         print("已冻结文本编码器")

    def image_forward(self, input_ids: torch.Tensor,
                      input_embeds: torch.Tensor,
                      images: Optional[Dict[str, torch.Tensor]] = None,
                      image_grid_thw: Optional[torch.Tensor] = None,
                      inference_params: Optional[Dict] =None,
                      **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward function for image encoding."""
        image_embeddings, window_index = self.image_encoder(images, \
                                    grid_thw=image_grid_thw)
        image_embeddings = self.image_projector(image_embeddings, window_index)
        image_token_id = self.config.image_encoder_config.image_token_id
        n_image_tokens = (input_ids == self.config.image_encoder_config.image_token_id).sum().item()
        n_image_features = image_embeddings.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features {n_image_features} != image tokens {n_image_tokens}"
            )
        if inference_params is not None:
                    inference_params.key_value_memory_dict["image_tokens_count"] = (
                        image_embeddings.shape[0]
                    )

        images_mask = (
            (input_ids == image_token_id).transpose(0, 1)
            .unsqueeze(-1)
            .expand_as(input_embeds)
            .to(input_embeds.device)
        )
        image_embeddings = image_embeddings.to(input_embeds.device, input_embeds.dtype)
        combined_embeddings = input_embeds.masked_scatter(images_mask, image_embeddings)
        return combined_embeddings


    def video_forward(self, inputs_embeds: torch.Tensor, decoder_inputs, **kwargs):
        """Forward function for video encoding."""
        pass

    def audio_forward(self, inputs_embeds: torch.Tensor, decoder_inputs, **kwargs):
        """Forward function for audio encoding."""
        pass

    def text_forward(self, input_ids: torch.Tensor, position_ids, **kwargs) -> torch.Tensor:
        """Forward function for text encoding."""
        input_embeds = self.text_encoder(input_ids=input_ids, position_ids=position_ids)
        return input_embeds
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_inputs: Optional[Dict[str, torch.Tensor]] = None,
        video_inputs: Optional[Dict[str, torch.Tensor]] = None,
        audio_inputs: Optional[Dict[str, torch.Tensor]] = None,
        inference_params: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward function for OmniEncoderModel."""
        # 文本嵌入或外部 inputs_embeds
        if self.text_encoder is not None and input_ids is not None:
            input_embeds = self.text_forward(input_ids, position_ids)
        else:
            input_embeds = kwargs.get("inputs_embeds")

        decoder_inputs: Dict[str, torch.Tensor] = {}

        # 图像
        if "image" in self.modality:
            if image_inputs is None:
                input_embeds = self.lm_dummy_encode()
            else:
                input_embeds = self.image_forward(input_ids=input_ids,
                                                 input_embeds=input_embeds,
                                                 inference_params=inference_params,
                                                 **image_inputs)
        # 视频
        if "video" in self.modality:
            if video_inputs is None:
                input_embeds = self.lm_dummy_encode()
            else:
                input_embeds = self.video_forward(input_ids=input_ids,
                                                 input_embeds=input_embeds,
                                                 inference_params=inference_params,
                                                 **video_inputs)
        # 音频
        return input_embeds, decoder_inputs




def _load_state_dict_hook_ignore_param_names(
    param_names: List[str], module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore missing keys during checkpoint loading.

    By default, this should not be used to avoid accidentally missing weights in checkpoint loading.

    Example use case: Use this for the vision projection if you want to load a checkpoint that contains vision and
    language model weights but not the vision projection weights.

    Args:
        param_names (list of str): Parameter names allowed to be missing when calling load_state_dict.
        module (torch.nn.Module): The torch module this hook applies to. Unused here but required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys, which collect the
            missing and unexpected keys when calling load_state_dict on this torch module, respectively.
    """
    for param_name in param_names:
        if param_name in incompatible_keys.missing_keys:
            logging.getLogger(__name__).warning(
                f"{param_name} being removed from incompatible_keys.missing_keys in LlavaModel"
            )
            incompatible_keys.missing_keys.remove(param_name)
