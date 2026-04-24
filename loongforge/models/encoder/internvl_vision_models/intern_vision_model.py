# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from InternVL.
# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License
# --------------------------------------------------------


"""Intern Vision model."""

from typing import Optional
import logging
import warnings
import torch
from torch import nn
import torch.nn.functional as F
from megatron.core.transformer.enums import ModelType
from loongforge.models.encoder.vision_transformer_block import TransformerBlock

from .internvl_config import InternVisionConfig
from loongforge.models.common import BaseMegatronVisionModule
from megatron.core import tensor_parallel
from loongforge.models.utils import import_module


class InternVisionEmbeddings(nn.Module):
    """ InternVisionEmbeddings """

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim), )

        self.patch_embedding = nn.Conv2d(in_channels=3,
                                         out_channels=self.embed_dim,
                                         kernel_size=self.patch_size,
                                         stride=self.patch_size,
                                         dtype=torch.bfloat16)

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(1, self.image_size // self.patch_size, self.image_size // self.patch_size,
                                              -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """ forward """
        # Force convert all inputs to bfloat16 at the very beginning
        target_dtype = torch.bfloat16
        pixel_values = pixel_values.to(target_dtype)
        self.patch_embedding = self.patch_embedding.to(target_dtype)
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [batch, channel, height, width]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # [b s h]
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if self.config.model_type == 'intern_vit_300m':
            position_embedding = torch.cat(
                [
                    self.position_embedding[:, :1, :],
                    self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
                ], dim=1)
        else:
            position_embedding = self.position_embedding
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternVisionModel(BaseMegatronVisionModule):
    """Intern Vision Model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        add_class_token (bool, optional): Include a class token. Defaults to True.
        class_token_len (int): Class token length. Defaults to 1 but 8 may be faster.
        patch_dim (int): Image patch size.
        img_h (int): Input image height.
        img_w (int): Input image width.
    """

    config_class = InternVisionConfig

    def __init__(
        self,
        config: InternVisionConfig,
        vp_stage: Optional[int] = None,
        # transformer_layer_spec: ModuleSpec,
        **kwargs,
    ) -> None:
        super().__init__(config=config)
        self.config = config
        if self.config.model_spec is None:
            model_spec = [
                "loongforge.models.encoder.internvl_vision_models.internvl_layer_spec",
                "get_vision_layer_with_te_spec",
            ]
        else:
            model_spec = self.config.model_spec

        self.transformer_layer_spec = import_module(model_spec, self.config)
        self.embeddings = InternVisionEmbeddings(config)

        # megatron core pipelining currently depends on model type
        self.model_type = ModelType.encoder_or_decoder

        self.encoder = TransformerBlock(  # TODO add support for return intermediate hidden states
            config=self.config,
            spec=self.transformer_layer_spec,
            pre_process=True,
            post_process=False,  # no final layer norm
            vp_stage=vp_stage,
        )
        self.select_layer = self.config.select_layer
        self.ps_version = self.config.ps_version
        # self.img_context_token_id = img_context_token_id
        self.downsample_ratio = self.config.downsample_ratio
        if hasattr(config, 'freeze') and config.freeze:
            self.freeze()

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.encoder.set_input_tensor(input_tensor)

    def pixel_shuffle(self, x, scale_factor=0.5):
        """ pixel shuffle """
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,  # TODO add support for return intermediate hidden states
        return_dict: Optional[bool] = None,
        pixel_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,  # Accept additional keyword arguments to handle unexpected parameters
    ) -> torch.Tensor:
        """Forward pass function for the Intern Vision model.

        Processes input tensors through embedding layers and transformer layers.

        Args:
            pixel_values: Optional input tensor of pixel values, shape should be (batch_size, channels, height, width)
            output_hidden_states: Whether to output intermediate hidden states (not implemented yet)
            return_dict: Whether to return output as a dictionary
            pixel_embeds: Optional precomputed pixel embedding tensor

        Returns:
            When return_dict is False, returns a tuple (last_hidden_state, pooled_output, None)
            When return_dict is True, returns BaseModelOutputWithPooling object containing:
                - last_hidden_state: Final hidden state for each token in the sequence
                - pooler_output: Final hidden state of the first token in the sequence
                - hidden_states: None (not implemented yet)
                - attentions: None

        Raises:
            ValueError: When both pixel_values and pixel_embeds are None
            ValueError: When shape of pixel_values is not 4-dimensional
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds.to(torch.bfloat16)
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values.to(torch.bfloat16))
            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')

        # contiguous() call required as `permute` can sparsify the tensor and this breaks pipelining
        if not self.config.sequence_parallel:
            hidden_states = hidden_states.transpose(0, 1).contiguous()  # [b, s, h] -> [s, b, h]

        attention_mask = None

        if self.config.sequence_parallel:
            hidden_states = tensor_parallel.scatter_to_sequence_parallel_region(hidden_states)
            hidden_states, _ = self.encoder(hidden_states, attention_mask=attention_mask)
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(hidden_states)
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        else:
            hidden_states, _ = self.encoder(hidden_states, attention_mask=attention_mask)
        hidden_states = hidden_states.transpose(0, 1).contiguous()  # [s, b, h] -> [b, s, h]

        last_hidden_state = hidden_states
        pooled_output = last_hidden_state[:, 0, :]

        last_hidden_state = last_hidden_state[:, 1:, :]
        h = w = int(last_hidden_state.shape[1] ** 0.5)
        last_hidden_state = last_hidden_state.reshape(last_hidden_state.shape[0], h, w, -1)
        last_hidden_state = self.pixel_shuffle(last_hidden_state, scale_factor=self.config.downsample_ratio)
        output = last_hidden_state.reshape(last_hidden_state.shape[0], -1, last_hidden_state.shape[-1])
        # TODO add support for return intermediate hidden states
        # if not return_dict:
        #     return (last_hidden_state, pooled_output, None)
        
        return output, None, []

        # TODO add support for return intermediate hidden states
        # if not return_dict:
        #     return (last_hidden_state, pooled_output, None)

        # return BaseModelOutputWithPooling(
        #     last_hidden_state=last_hidden_state,
        #     pooler_output=pooled_output,
        #     hidden_states=None,
        #     attentions=None,
        # )

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        """ resize position embeddings """
        pos_emb = self.embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size
        logging.info('Resized position embeddings from {} to {}'.format(old_size, new_size))
