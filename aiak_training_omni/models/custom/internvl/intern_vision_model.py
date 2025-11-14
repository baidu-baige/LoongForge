# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
"""Intern Vision model."""
from typing import Optional
import logging
import torch
from torch import nn
import torch.nn.functional as F
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.enums import ModelType, AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock

from .internvl_config import InternVisionConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from megatron.core import parallel_state, tensor_parallel


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
                                         stride=self.patch_size)

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
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [batch, channel, height, width]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # [b s h]
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if self.config.vision_type == 'vit_300m':
            position_embedding = torch.cat([
                self.position_embedding[:, :1, :],
                self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
            ],
                                           dim=1)
        else:
            position_embedding = self.position_embedding
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternVisionModel(VisionModule):
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

    def __init__(
        self,
        config: InternVisionConfig,
        transformer_layer_spec: ModuleSpec,
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.embeddings = InternVisionEmbeddings(config)

        # megatron core pipelining currently depends on model type
        self.model_type = ModelType.encoder_or_decoder

        self.encoder = TransformerBlock(  # TODO add support for return intermediate hidden states
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,  # no final layer norm
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.encoder.set_input_tensor(input_tensor)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,  # TODO add support for return intermediate hidden states
        return_dict: Optional[bool] = None,
        pixel_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """Forward function of the Intern Vision Model. This function passes the input tensors
        through the embedding layer and then the transformer.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')

        # contiguous() call required as `permute` can sparsify the tensor and this breaks pipelining
        if not self.config.sequence_parallel:
            hidden_states = hidden_states.transpose(0, 1).contiguous()  # [b, s, h] -> [s, b, h]

        attention_mask = None
        attn_mask_type = AttnMaskType.no_mask

        if self.config.sequence_parallel:
            hidden_states = tensor_parallel.scatter_to_sequence_parallel_region(hidden_states)
            #hidden_states = hidden_states.transpose(0, 1).contiguous()  # [b, s, h] -> [s, b, h]
            hidden_states = self.encoder(hidden_states, attention_mask=attention_mask, attn_mask_type=attn_mask_type)
            #hidden_states = hidden_states.transpose(0, 1).contiguous()
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(hidden_states)
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        else:
            hidden_states = self.encoder(hidden_states, attention_mask=attention_mask, attn_mask_type=attn_mask_type)
        hidden_states = hidden_states.transpose(0, 1).contiguous()  # [s, b, h] -> [b, s, h]

        last_hidden_state = hidden_states
        pooled_output = last_hidden_state[:, 0, :]

        # TODO add support for return intermediate hidden states
        if not return_dict:
            return (last_hidden_state, pooled_output, None)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=None,
            attentions=None,
        )

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
