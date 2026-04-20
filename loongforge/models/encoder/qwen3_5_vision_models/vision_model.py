# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

""" Qwen3.5 Vision Model """

from typing import Optional

import torch
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig

from loongforge.models.encoder.base_vision_models.base_vision_model import (
    BaseVisionModel,
    PatchEmbed,
)

from .qwen3_5_vision_config import Qwen35VisionConfig


class Qwen35VisionModel(BaseVisionModel):
    """ VisionModel with Qwen3.5 architecture """

    config_class = Qwen35VisionConfig

    def __init__(self,
        config: TransformerConfig,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config, vp_stage=vp_stage)
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
            bias=True,
        )
        self.pos_embed = torch.nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        
        if hasattr(config, 'freeze') and config.freeze:
            self.freeze()

    def fast_pos_embed_interpolate(self, image_grid_thw):
        """Interpolate position embeddings for a given grid."""
        grid_ts, grid_hs, grid_ws = image_grid_thw[:, 0], image_grid_thw[:, 1], image_grid_thw[:, 2]
        device = image_grid_thw.device

        idx_parts = [[] for _ in range(4)]
        weight_parts = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h.item(), device=device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w.item(), device=device)

            h_floor = h_idxs.to(torch.int64)
            w_floor = w_idxs.to(torch.int64)
            h_ceil = (h_floor + 1).clamp(max=self.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clamp(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_floor.float()
            dw = w_idxs - w_floor.float()

            idx_parts[0].append((h_floor[:, None] * self.num_grid_per_side + w_floor[None, :]).flatten())
            idx_parts[1].append((h_floor[:, None] * self.num_grid_per_side + w_ceil[None, :]).flatten())
            idx_parts[2].append((h_ceil[:, None] * self.num_grid_per_side + w_floor[None, :]).flatten())
            idx_parts[3].append((h_ceil[:, None] * self.num_grid_per_side + w_ceil[None, :]).flatten())

            weight_parts[0].append(((1 - dh)[:, None] * (1 - dw)[None, :]).flatten())
            weight_parts[1].append(((1 - dh)[:, None] * dw[None, :]).flatten())
            weight_parts[2].append((dh[:, None] * (1 - dw)[None, :]).flatten())
            weight_parts[3].append((dh[:, None] * dw[None, :]).flatten())

        idx_tensor = torch.stack([torch.cat(parts) for parts in idx_parts])      # (4, total)
        weight_tensor = torch.stack(
            [torch.cat(parts) for parts in weight_parts]
        ).to(dtype=self.pos_embed.weight.dtype) # (4, total)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute rotary positional embedding based on frame size."""
        merge_size = self.spatial_merge_size

        # Single GPU->CPU transfer, then all loop logic uses Python ints
        grid_thw_cpu = grid_thw.cpu().tolist()

        max_hw = max(max(h, w) for _, h, w in grid_thw_cpu)
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = sum(t * h * w for t, h, w in grid_thw_cpu)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_cpu:
            merged_h = height // merge_size
            merged_w = width // merge_size

            rows = torch.arange(height, device=device).reshape(merged_h, merge_size)
            cols = torch.arange(width, device=device).reshape(merged_w, merge_size)

            row_idx = rows[:, None, :, None].expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = cols[None, :, None, :].expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            if num_frames > 1:
                row_idx = row_idx.unsqueeze(0).expand(num_frames, -1).reshape(-1)
                col_idx = col_idx.unsqueeze(0).expand(num_frames, -1).reshape(-1)

            num_tokens = num_frames * height * width
            pos_ids[offset:offset + num_tokens, 0] = row_idx
            pos_ids[offset:offset + num_tokens, 1] = col_idx
            offset += num_tokens

        embeddings = freq_table[pos_ids].flatten(1)
        return embeddings

    def forward(self, x: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = self.patch_embed(x)

        pos_embeds = self.fast_pos_embed_interpolate(image_grid_thw)
        x = x + pos_embeds

        seq_len, _ = x.size()
        x = x.reshape(seq_len, -1)

        rotary_pos_emb = self.rot_pos_emb(image_grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, 1, 1, -1).repeat(1, 1, 1, 2)

        cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as image_grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        
        x = x[:, None, :].contiguous()  # [s, h] -> [s, 1, h]
        x, deepstack_feature_lists = self.decoder(
            x,
            packed_seq_params=[PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
            ) for _ in range(self.config.num_layers)],
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=None,
        )

        x = x[:, 0, :].contiguous()  # [s, 1, h] -> [s, h]
        return x, None, deepstack_feature_lists