# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LLaVA-OneVision-1.5 under the Apache-2.0 License.

"""VisionTransformer module"""

import torch
import torch.nn.functional as F
from typing import Optional
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from .llavaov_1_5_config import RiceVisionConfig
from loongforge.models.encoder.base_vision_models.base_vision_model import (
    BaseVisionModel,
)


class PatchEmbed(torch.nn.Module):
    """" Patch Embedding """
    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = torch.nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """" Forward pass """
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        
        return hidden_states


class RiceViTModel(BaseVisionModel):
    """Rice VIT Model"""

    config_class = RiceVisionConfig

    def __init__(
        self, config: RiceVisionConfig, spatial_merge_size: int = 2, vp_stage: Optional[int] = None, **kwargs
    ) -> None:
        if config.model_spec is None:
            config.model_spec = [
                "loongforge.models.encoder.llavaov1_5_vision_models.llavaov_1_5_layer_spec",
                "get_vision_layer_with_spec",
            ]
        super().__init__(config, spatial_merge_size, vp_stage=vp_stage)
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = list(range(config.num_layers))
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        self.register_buffer("class_embedding", torch.randn(config.hidden_size))
        self.register_buffer("class_pos_emb", torch.randn(1, config.kv_channels // 2))

        self.pre_layernorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-4)

    def forward(self, x: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """ Forward of the vision model"""
        x = self.patch_embed(x)

        batch_size = image_grid_thw.size(0)

        rotary_pos_emb = self.rot_pos_emb(image_grid_thw)

        class_embedding = self.class_embedding.view(1, -1)
        class_pos_emb = self.class_pos_emb.view(1, -1)
        class_tokens = class_embedding.expand(batch_size, -1)
        class_pos_embs = class_pos_emb.expand(batch_size, -1)

        tokens_per_sample = []

        for i in range(batch_size):
            t, h, w = image_grid_thw[i]
            tokens_per_sample.append((t * h * w).item())

        new_x = []
        start_idx = 0
        for i in range(batch_size):
            new_x.append(class_tokens[i : i + 1])
            new_x.append(x[start_idx : start_idx + tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]

        x = torch.cat(new_x, dim=0)

        new_rotary_pos_emb = []
        start_idx = 0
        for i in range(batch_size):
            new_rotary_pos_emb.append(class_pos_embs[i : i + 1])
            new_rotary_pos_emb.append(
                rotary_pos_emb[start_idx : start_idx + tokens_per_sample[i]]
            )
            start_idx += tokens_per_sample[i]

        rotary_pos_emb = torch.cat(new_rotary_pos_emb, dim=0)

        cu_seqlens = []
        cumulative_length = 0
        cu_seqlens.append(cumulative_length)  # Starts from 0
        for length in tokens_per_sample:

            cumulative_length += int(length + 1)
            cu_seqlens.append(cumulative_length)

        cu_seqlens = torch.tensor(
            cu_seqlens,
            device=x.device,
            dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        max_seqlen = max(tokens_per_sample) + 1  # +1 for class token

        x = x[:, None, :].contiguous()  # [s, h] -> [s, 1, h]

        x = self.pre_layernorm(x)

        x, _ = self.decoder(
            x,
            packed_seq_params=[
                PackedSeqParams(
                    qkv_format="thd",
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_kv=max_seqlen
                )
                for i in range(self.config.num_layers)
            ],
            rotary_pos_emb=rotary_pos_emb.unsqueeze(1).unsqueeze(2),
            attention_mask=None,
        )
        x = x[:, 0, :].contiguous()  # [s, 1, h] -> [s, h]

        patch_output = []
        start_idx = 0
        for i in range(batch_size):
            start_idx += 1
            patch_output.append(x[start_idx : start_idx + tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]
        patch_output = torch.cat(patch_output, dim=0)  # [original_seq_len, hidden_size]
        return patch_output, None, []
