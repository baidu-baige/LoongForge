# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from transformers.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VisionTransformer module"""

import torch
import torch.nn.functional as F
from typing import Optional

from megatron.core.transformer.enums import ModelType

from loongforge.models.encoder.vision_transformer_block import TransformerBlock
from loongforge.models.encoder.qwen2_vl_vision_models.qwen2_vl_config import (
    Qwen2VisionModelConfig
) 
from loongforge.models.common import BaseMegatronVisionModule
from loongforge.models.utils import import_module


# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py # pylint: disable=line-too-long
class PatchEmbed(torch.nn.Module):
    """ Patch Embedding"""

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = torch.nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ " Forward pass"""
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


class VisionRotaryEmbedding(torch.nn.Module):
    """ Rotary Position Embedding"""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.inv_freq = inv_freq.to(torch.cuda.current_device())

    def forward(self, seqlen: int) -> torch.Tensor:
        """Forward Pass"""
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class BaseVisionModel(BaseMegatronVisionModule):
    """VisionTransformer model."""
    # NOTE: This model class is adapted from Qwen2VisionModel
    config_class = Qwen2VisionModelConfig

    def __init__(
        self, config: Qwen2VisionModelConfig, spatial_merge_size: int = 2, vp_stage: Optional[int] = None, **kwargs
    ) -> None:
        super().__init__(config)
        if self.config.model_spec is None:
            model_spec = [
                "loongforge.models.encoder.qwen2_vl_vision_models.qwen2_vl_layer_spec",
                "get_qwen2_vl_vision_model_layer_with_te_spec",
            ]
        else:
            model_spec = self.config.model_spec
        self.transformer_layer_spec = import_module(model_spec, self.config)
        self.model_type = ModelType.encoder_or_decoder
        self.spatial_merge_size = spatial_merge_size

        self.rotary_pos_emb = VisionRotaryEmbedding(self.config.kv_channels // 2)

        self.patch_embed = PatchEmbed(
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.config.hidden_size,
        )
    
        self.decoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_layer_spec,
            pre_process=True,
            post_process=False,
            vp_stage=vp_stage,
        )
        if hasattr(config, 'freeze') and config.freeze:
            self.freeze()

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.decoder.set_input_tensor(input_tensor)

    def rot_pos_emb(self, grid_thw):
        """rotation position embedding"""
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self, images: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """forward function"""
        rotary_pos_emb = self.rot_pos_emb(image_grid_thw)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(1).unsqueeze(2).float()

        x = self.patch_embed(images)
        x = x[:, None, :].contiguous()  # [s, h] -> [s, 1, h]
        x, _ = self.decoder(
            x,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=None,
        )
        x = x[:, 0, :].contiguous()  # [s, 1, h] -> [s, h]
        return x, None, []