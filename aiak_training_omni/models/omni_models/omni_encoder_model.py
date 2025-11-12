"""OmniEncoderModel: A multimodal encoder that produces input embeddings.

This model integrates various modality encoders (text, image, video, audio)
    and projects their features into a unified embedding space and returns:
    - input_embeds: For the base model
    - decoder_inputs: Optional inputs for the decoder
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Literal

import torch
import torch.nn as nn
from collections import namedtuple
from functools import partial
from ..common import BaseMegatronModule, BaseModelConfig
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from transformers.models.auto.modeling_auto import AutoModel


class OmniEncoderModel(torch.nn.Module):
    def __init__(
        self,
        config: BaseModelConfig,
        language_embedding,
        allow_missing_adapter_checkpoint=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.modality: List[str] = []
        self.text_encoder = language_embedding
        if  hasattr(self.config, "image_encoder") and self.config.image_encoder is not None:
            self.image_encoder: BaseMegatronModule = AutoModel.from_config(
                config.image_encoder, **kwargs
            )
            self.modality.append("image")

        if hasattr(self.config, "video_encoder") and self.config.video_encoder is not None:
            self.video_encoder: BaseMegatronModule = AutoModel.from_config(
                self.config.video_encoder, **kwargs)
            self.modality.append("video")

        if hasattr(self.config, "audio_encoder") and self.config.audio_encoder is not None:
            self.audio_encoder: BaseMegatronModule = AutoModel.from_config(
                self.config.audio_encoder, **kwargs)
            self.modality.append("audio")

        if hasattr(self.config, "image_projector")  and self.config.image_projector is not None:
            self.image_projector: BaseMegatronModule = AutoModel.from_config(
                config.image_projector,
                input_size=config.image_encoder.hidden_size,
                output_size=config.foundation.hidden_size,
                **kwargs,
            )
            if allow_missing_adapter_checkpoint:
                adapter_param_names = [
                    f"image_projector.{name}"
                    for name in self.image_projector.state_dict().keys()
                ]
                self.image_projector.register_load_state_dict_post_hook(
                    partial(
                        _load_state_dict_hook_ignore_param_names, adapter_param_names
                    )
                )

        if hasattr(self.config, "video_projector") and self.config.video_projector is not None:
            self.video_projector: BaseMegatronModule = AutoModel.from_config(
                self.config.video_projector, **kwargs)
            if allow_missing_adapter_checkpoint:
                adapter_param_names = [
                    f"video_projector.{name}"
                    for name in self.video_projector.state_dict().keys()
                ]
                self.video_projector.register_load_state_dict_post_hook(
                    partial(_load_state_dict_hook_ignore_param_names, adapter_param_names)
                )

        if hasattr(self.config, "audio_projector") and self.config.audio_projector is not None:
            self.audio_projector: BaseMegatronModule = AutoModel.from_config(
                self.config.audio_projector, **kwargs)
            if allow_missing_adapter_checkpoint:
                adapter_param_names = [
                    f"audio_projector.{name}"
                    for name in self.audio_projector.state_dict().keys()
                ]
                self.audio_projector.register_load_state_dict_post_hook(
                    partial(_load_state_dict_hook_ignore_param_names, adapter_param_names)
                )

    def set_input_tensor(self, input_tensor) -> None:
        """Set the input tensor for the encoder.
        Args:
            input_tensor: Input tensor to process. Will be converted to list if not already.
            
        Raises:
            ValueError: If image_encoder is not initialized.
        """
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for llava"

        if self.image_encoder is not None:
            self.image_encoder.set_input_tensor(input_tensor[0])
        else:
            raise ValueError("image_encoder is None, cannot set input tensor.")

    def set_projector_trainable_only(self):
        """Configure only projector layers to be trainable.
        Used for pipeline parallel training schedules.
        """
        for module in self.children():
            if isinstance(module, BaseMegatronModule):
                module.set_projector_trainable_only()

    def image_forward(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        images: Optional[Dict[str, torch.Tensor]] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        inference_params: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward function for image encoding."""
        image_embeddings, window_index = self.image_encoder(
            images, 
            image_grid_thw=image_grid_thw
        )
        image_embeddings = self.image_projector(image_embeddings, window_index)
        
        image_token_id = self.config.image_encoder.image_token_id
        n_image_tokens = (
            (input_ids == self.config.image_encoder.image_token_id).sum().item()
        )
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
            (input_ids == image_token_id)
            .transpose(0, 1)
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

    def text_forward(
        self, input_ids: torch.Tensor, position_ids, **kwargs
    ) -> torch.Tensor:
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
        """Forward pass for the OmniEncoderModel.

        Args:
            input_ids: Token IDs for text input
            position_ids: Position IDs for text tokens
            attention_mask: Attention mask
            image_inputs: Dictionary of image inputs
            video_inputs: Dictionary of video inputs
            audio_inputs: Dictionary of audio inputs
            inference_params: Additional parameters for inference
            kwargs: Additional keyword arguments
            
        Returns:
            Tuple containing:
                - Combined embeddings tensor
                - Dictionary of decoder inputs
        """
        if self.text_encoder is not None and input_ids is not None:
            input_embeds = self.text_forward(input_ids, position_ids)
        else:
            input_embeds = kwargs.get("inputs_embeds")

        decoder_inputs: Dict[str, torch.Tensor] = {}

        # Process image modality
        if "image" in self.modality:
            if image_inputs is None:
                input_embeds = self.lm_dummy_encode()
            else:
                input_embeds = self.image_forward(
                    input_ids=input_ids,
                    input_embeds=input_embeds,
                    inference_params=inference_params,
                    **image_inputs,
                )

        # Process audio modality
        if "audio" in self.modality:
            if audio_inputs is None:
                input_embeds = self.lm_dummy_encode()
            else:
                input_embeds = self.audio_forward(
                    input_ids=input_ids,
                    input_embeds=input_embeds,
                    inference_params=inference_params,
                    **audio_inputs,
                )

        # Process video modality
        if "video" in self.modality:
            if video_inputs is None:
                input_embeds = self.lm_dummy_encode()
            else:
                input_embeds = self.video_forward(
                    input_ids=input_ids,
                    input_embeds=input_embeds,
                    inference_params=inference_params,
                    **video_inputs,
                )
                
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
