# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
OmniEncoderModel: A multimodal encoder that produces input embeddings.

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
from loongforge.train.initialize import change_parallel_state
from loongforge.data.dp_balance.vit_balance import dp_balance_vit_encoder
from loongforge.utils import get_args

def make_encoder_forward_pre_hook(module_name):
    """
    Create a forward pre-hook function that switches the tensor-parallel
    context to a specific encoder module before executing its forward pass.

    Args:
        module_name (str): The name of the encoder module whose parallel
            state should be activated before the forward computation.

    Returns:
        Callable: A forward pre-hook function that accepts (module, input)
        and switches the tensor-parallel state by calling
        `change_parallel_state(module_name)` before returning the input
        unchanged.
    """

    def encoder_forward_pre_hook(module, input):
        change_parallel_state(module_name)
        return input

    return encoder_forward_pre_hook


def make_encoder_forward_hook(module_name):
    """
    Create a forward hook function that adjusts the tensor-parallel state
    before returning the module's output.

    Args:
        module_name : str
            The name of the module whose forward pass should trigger a
            parallel-state change.

    Returns:
        callable:
            A forward hook function with signature `(module, input, output)`,
            which updates the parallel state and returns the original output.
    """

    def encoder_forward_hook(module, input, output):
        change_parallel_state(module_name)
        return output

    return encoder_forward_hook


def make_encoder_backward_pre_hook(module_name):
    """
    Create a backward pre-hook that adjusts the tensor-parallel state
    before the backward pass of a module.

    This factory function generates a backward pre-hook intended for use
    with `module.register_full_backward_pre_hook(...)`. The returned hook
    is invoked before gradients are computed for the module, allowing the
    runtime to switch tensor-parallel or pipeline-parallel states
    dynamically based on the module name.

    Args:
        module_name : str
            The name of the module whose backward pass should trigger a
            parallel-state change.

    Returns:
        callable:
            A backward pre-hook function with signature `(module, input)`,
            which updates the parallel state and returns the original input
            tuple unchanged.
    """

    def encoder_backward_pre_hook(module, input):
        change_parallel_state(module_name)
        return input

    return encoder_backward_pre_hook


class OmniEncoderModel(torch.nn.Module):
    def __init__(
        self,
        config: BaseModelConfig,
        vocab_size: int,
        max_sequence_length: int,
        allow_missing_adapter_checkpoint=False,
        position_embedding_type: Literal["learned_absolute", "rope"] = "rope",
        scatter_embedding_sequence_parallel: bool = True,
        vp_stage: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.mix_used_vision_encoder = getattr(config, "mix_used_vision_encoder", True)
        self.mix_used_vision_projector = getattr(
            config, "mix_used_vision_projector", True
        )
        self.text_encoder = LanguageModelEmbedding(
            config=self.config.foundation,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            position_embedding_type=position_embedding_type,
            scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
        )
        self.encoder_modality = {}
        if (
            hasattr(self.config, "image_encoder")
            and self.config.image_encoder is not None
        ):
            change_parallel_state("image_encoder")
            self.image_encoder: BaseMegatronModule = AutoModel.from_config(
                config.image_encoder, vp_stage=vp_stage, **kwargs
            )
            self.encoder_modality["image"] = True
            self.image_encoder.register_forward_pre_hook(
                make_encoder_forward_pre_hook("image_encoder")
            )
            self.image_encoder.register_forward_hook(
                make_encoder_forward_hook("text_decoder")
            )

        if (
            hasattr(self.config, "video_encoder")
            and self.config.video_encoder is not None
        ):
            change_parallel_state("video_encoder")
            self.video_encoder: BaseMegatronModule = AutoModel.from_config(
                self.config.video_encoder, vp_stage=vp_stage, **kwargs
            )
            self.encoder_modality["video"] = True
            self.video_encoder.register_forward_pre_hook(
                make_encoder_forward_pre_hook("video_encoder")
            )
            self.video_encoder.register_forward_hook(
                make_encoder_forward_hook("text_decoder")
            )
        elif self.mix_used_vision_encoder:
            self.encoder_modality["video"] = True

        if (
            hasattr(self.config, "audio_encoder")
            and self.config.audio_encoder is not None
        ):
            change_parallel_state("audio_encoder")
            self.audio_encoder: BaseMegatronModule = AutoModel.from_config(
                self.config.audio_encoder, vp_stage=vp_stage, **kwargs
            )
            self.encoder_modality["audio"] = True
            self.audio_encoder.register_forward_pre_hook(
                make_encoder_forward_pre_hook("audio_encoder")
            )
            self.audio_encoder.register_forward_hook(
                make_encoder_forward_hook("text_decoder")
            )

        change_parallel_state("text_decoder")

        if (
            hasattr(self.config, "image_projector")
            and self.config.image_projector is not None
        ):
            self.image_projector: BaseMegatronModule = AutoModel.from_config(
                config.image_projector,
                input_size=config.image_encoder.hidden_size,
                output_size=config.foundation.hidden_size,
                **kwargs,
            )
            if allow_missing_adapter_checkpoint:
                adapter_param_names = [
                    f"encoder_model.image_projector.{name}"
                    for name in self.image_projector.state_dict().keys()
                ]
                self.image_projector.register_load_state_dict_post_hook(
                    partial(
                        _load_state_dict_hook_ignore_param_names, adapter_param_names
                    )
                )
        else:
            self.image_projector = None

        if (
            hasattr(self.config, "video_projector")
            and self.config.video_projector is not None
        ):
            self.video_projector: BaseMegatronModule = AutoModel.from_config(
                self.config.video_projector, **kwargs
            )
            if allow_missing_adapter_checkpoint:
                adapter_param_names = [
                    f"encoder_model.video_projector.{name}"
                    for name in self.video_projector.state_dict().keys()
                ]
                self.video_projector.register_load_state_dict_post_hook(
                    partial(
                        _load_state_dict_hook_ignore_param_names, adapter_param_names
                    )
                )
        else:
            self.video_projector = None

        if (
            hasattr(self.config, "audio_projector")
            and self.config.audio_projector is not None
        ):
            self.audio_projector: BaseMegatronModule = AutoModel.from_config(
                self.config.audio_projector, **kwargs
            )
            if allow_missing_adapter_checkpoint:
                adapter_param_names = [
                    f"encoder_model.audio_projector.{name}"
                    for name in self.audio_projector.state_dict().keys()
                ]
                self.audio_projector.register_load_state_dict_post_hook(
                    partial(
                        _load_state_dict_hook_ignore_param_names, adapter_param_names
                    )
                )
        else:
            self.audio_projector = None

    def _aggregate_deepstack_embeds(
        self,
        images_mask: Optional[bool],
        videos_mask: Optional[bool],
        deepstack_image_embeds: Optional[List[torch.Tensor]],
        deepstack_video_embeds: Optional[List[torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """Aggregates deepstack embeddings and position masks from image and video modalities."""
        visual_pos_masks = None
        deepstack_visual_embeds = None
        
        if (
            len(deepstack_image_embeds) != 0 
            or len(deepstack_video_embeds) != 0
        ):
            if images_mask is not None and videos_mask is not None:
                images_mask = images_mask[..., 0]
                videos_mask = videos_mask[..., 0]
                visual_pos_masks = images_mask | videos_mask
                
                deepstack_visual_embeds = []
                images_mask_joint = images_mask[visual_pos_masks]
                videos_mask_joint = videos_mask[visual_pos_masks]
                
                for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                    # Create a zero tensor to hold joint embeddings, size is (N_visual_tokens, Hidden_size)
                    embed_joint = img_embed.new_zeros(
                        visual_pos_masks.sum().item(), 
                        img_embed.shape[-1]
                    ).to(img_embed.device)
                    embed_joint[images_mask_joint, :] = img_embed
                    embed_joint[videos_mask_joint, :] = vid_embed
                    deepstack_visual_embeds.append(embed_joint)
                    
            elif images_mask is not None:
                images_mask = images_mask[..., 0]
                visual_pos_masks = images_mask
                deepstack_visual_embeds = deepstack_image_embeds
                
            elif videos_mask is not None:
                videos_mask = videos_mask[..., 0]
                visual_pos_masks = videos_mask
                deepstack_visual_embeds = deepstack_video_embeds

        return visual_pos_masks, deepstack_visual_embeds

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
        args = get_args()
        if args.use_vit_dp_balance:
            if args.enable_encoder_hetero_dp or args.enable_full_hetero_dp:
                change_parallel_state("image_encoder")
            image_embeddings, window_index, deepstack_image_embeds = dp_balance_vit_encoder(
                self.image_encoder, images, image_grid_thw
            )
        else:
            image_embeddings, window_index, deepstack_image_embeds = self.image_encoder(
                images, image_grid_thw=image_grid_thw
            )
        if self.image_projector is not None:
            image_embeddings = self.image_projector(image_embeddings, window_index)

        if isinstance(image_embeddings, (list, tuple)):
            image_embeddings = torch.cat([e.reshape(-1, e.shape[-1]) for e in image_embeddings], dim=0)

        image_token_id = self.image_encoder.config.image_token_id
        n_image_tokens = (input_ids == image_token_id).sum().item()
        n_image_features = image_embeddings.shape[0]

        if n_image_tokens != n_image_features:
            logging.getLogger(__name__).warning(
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

        return combined_embeddings, images_mask, deepstack_image_embeds

    def video_forward(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        pixel_values_videos: Optional[Dict[str, torch.Tensor]] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        inference_params: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward function for video encoding."""
        if self.mix_used_vision_encoder:
            video_embeddings, window_index, deepstack_video_embeds = self.image_encoder(pixel_values_videos, 
                                                                 image_grid_thw=video_grid_thw)
            video_token_id = self.image_encoder.config.video_token_id
        else:
            video_embeddings, window_index, deepstack_video_embeds = self.video_encoder(
                pixel_values_videos, image_grid_thw=video_grid_thw
            )
            video_token_id = self.video_encoder.config.video_token_id
        if self.mix_used_vision_projector and self.image_projector is not None:
            video_embeddings = self.image_projector(video_embeddings, window_index)
        elif self.video_projector is not None:
            video_embeddings = self.video_projector(video_embeddings, window_index)
        
        n_video_tokens = (input_ids == video_token_id).sum().item()
        n_video_features = video_embeddings.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"video features {n_video_features} != video tokens {n_video_tokens}"
            )

        # If running inference, the language model KV cache will be updated for image token positions.
        # Here we store the image tokens sequence length, which can be used as an offset to the KV cache later.
        if inference_params is not None:
            inference_params.key_value_memory_dict["video_tokens_count"] = (
                video_embeddings.shape[0]
            )

        videos_mask = (
            (input_ids == video_token_id)
            .transpose(0, 1)
            .unsqueeze(-1)
            .expand_as(input_embeds)
            .to(input_embeds.device)
        )
        video_embeddings = video_embeddings.to(input_embeds.device, input_embeds.dtype)
        combined_embeddings = input_embeds.masked_scatter(videos_mask, video_embeddings)

        return combined_embeddings, videos_mask, deepstack_video_embeds

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
        enable_encoder_hetero_dp: Optional[bool] = False,
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
        if self.text_encoder is not None and input_ids is not None and not enable_encoder_hetero_dp:
            input_embeds = self.text_forward(input_ids, position_ids)
        else:
            input_embeds = kwargs.get("inputs_embeds")

        decoder_inputs: Dict[str, torch.Tensor] = {}
        for modality in self.encoder_modality:
            self.encoder_modality[modality] = False
        
        images_mask, videos_mask = None, None
        deepstack_image_embeds, deepstack_video_embeds = [], []
        # Process image modality
        if "image" in self.encoder_modality:
            if image_inputs is None and not self.encoder_modality["image"]:
                input_embeds = self.encoder_dummy_forward(
                    input_embeds, self.image_encoder, self.image_projector
                )
            else:
                input_embeds, images_mask, deepstack_image_embeds = self.image_forward(
                    input_ids=input_ids,
                    input_embeds=input_embeds,
                    inference_params=inference_params,
                    **image_inputs,
                )
            self.encoder_modality["image"] = True

        # Process audio modality
        if "audio" in self.encoder_modality:
            if audio_inputs is None and not self.encoder_modality["audio"]:
                input_embeds = self.encoder_dummy_forward(
                    input_embeds, self.audio_encoder, self.audio_projector
                )
            else:
                input_embeds = self.audio_forward(
                    input_ids=input_ids,
                    input_embeds=input_embeds,
                    inference_params=inference_params,
                    **audio_inputs,
                )
            self.encoder_modality["audio"] = True

        # Process video modality
        if "video" in self.encoder_modality:
            if video_inputs is None and not self.encoder_modality["video"]:
                if self.mix_used_vision_encoder and not self.encoder_modality["image"]:
                    input_embeds = self.encoder_dummy_forward(
                        input_embeds, self.video_encoder, self.video_projector
                    )
            else:
                input_embeds, videos_mask, deepstack_video_embeds = self.video_forward(
                    input_ids=input_ids,
                    input_embeds=input_embeds,
                    inference_params=inference_params,
                    **video_inputs,
                )
            self.encoder_modality["video"] = True
        
        visual_pos_masks, deepstack_visual_embeds = self._aggregate_deepstack_embeds(
            images_mask=images_mask,
            videos_mask=videos_mask,
            deepstack_image_embeds=deepstack_image_embeds,
            deepstack_video_embeds=deepstack_video_embeds,
        )
                 
        return input_embeds, decoder_inputs, visual_pos_masks, deepstack_visual_embeds

    def encoder_dummy_forward(self, input_embeds, encoder_model, projector_model):
        """Helper method to handle empty inputs"""
        dummy_input = encoder_model.get_dummy_input(input_embeds.device)
        encoder_ret = encoder_model(*dummy_input)
        # Different encoders do not share a strict return signature:
        # some return (features, window_index, ...), while others return (features, None, ...) or only features.
        # We normalize both cases to avoid unpacking failures in dummy forward.
        if isinstance(encoder_ret, (tuple, list)):
            encoder_output = encoder_ret[0]
            window_index = encoder_ret[1] if len(encoder_ret) > 1 else None
        else:
            encoder_output = encoder_ret
            window_index = None
        if projector_model is not None:
            encoder_output = projector_model(encoder_output, window_index)
        if isinstance(encoder_output, (tuple, list)):
            encoder_output = encoder_output[0]
        input_embeds = input_embeds + encoder_output.sum() * 0.0
        return input_embeds


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
