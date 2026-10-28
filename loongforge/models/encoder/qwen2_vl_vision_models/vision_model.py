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
from megatron.core.packed_seq_params import PackedSeqParams
from loongforge.models.encoder.base_vision_models.base_vision_model import BaseVisionModel
from .qwen2_vl_config import Qwen2VisionRMSNormConfig


class Qwen2VisionModelWithRMSNorm(BaseVisionModel):
    """VisionModel With RMSNorm"""

    config_class = Qwen2VisionRMSNormConfig

    def __init__(
        self,
        config: Qwen2VisionRMSNormConfig,
        spatial_merge_size: int = 2,
        fullatt_block_indexes: list = [7, 15, 23, 31],
        window_size: int = 112,
        vp_stage: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(config, spatial_merge_size=spatial_merge_size, vp_stage=vp_stage)
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.window_size = window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

    def get_window_index(self, grid_thw):
        """Get window index for each token"""

        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(self, x: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        rotary_pos_emb = self.rot_pos_emb(image_grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(image_grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=x.device,
            dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = x.size()
        x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        cu_seqlens = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
        ).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as image_grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        x = x[:, None, :].contiguous()  # [s, h] -> [s, 1, h]
        x, _ = self.decoder(
            x,
            packed_seq_params=[
                PackedSeqParams(
                    qkv_format="thd",
                    cu_seqlens_q=(
                        cu_seqlens
                        if i in self.fullatt_block_indexes
                        else cu_window_seqlens
                    ),
                    cu_seqlens_kv=(
                        cu_seqlens
                        if i in self.fullatt_block_indexes
                        else cu_window_seqlens
                    ),
                )
                for i in range(self.config.num_layers)
            ],
            rotary_pos_emb=rotary_pos_emb.unsqueeze(1).unsqueeze(2),
            attention_mask=None,
        )
        x = x[:, 0, :].contiguous()  # [s, 1, h] -> [s, h]

        return x, window_index, []
