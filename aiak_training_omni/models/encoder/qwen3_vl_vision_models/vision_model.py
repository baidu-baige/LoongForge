"""VisionTransformer module"""

import torch
import torch.nn.functional as F
from typing import Optional

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.enums import ModelType, AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec

from aiak_training_omni.models.common import BaseMegatronVisionModule
from aiak_training_omni.models.utils import import_module
from aiak_training_omni.utils import get_model_config

from aiak_training_omni.models.encoder.vision_transformer_block import TransformerBlock
from aiak_training_omni.models.encoder.base_vision_models.base_vision_model import (
    BaseVisionModel,
    PatchEmbed
)
from .qwen3_vl_config import Qwen3VisionModelConfig  
from ..qwen2_vl_vision_models.adapter import Adapter


class Qwen3VisionModel(BaseVisionModel):
    """ VisionModel With LayerNorm (for Qwen3-VL) """

    config_class = Qwen3VisionModelConfig

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
        
        # DeepStack configuration for Qwen3-VL
        if hasattr(config, 'deepstack_visual_indexes'):
            self.deepstack_visual_indexes = config.deepstack_visual_indexes
        else:
            self.deepstack_visual_indexes = [8, 16, 24]  # Default Qwen3-VL layers
        
        # 在vision_model中创建deepstack_merger_list
        model_config = get_model_config()
        self.deepstack_merger_list = torch.nn.ModuleList(
            [
                Adapter(
                    model_config.image_projector,
                    input_size=self.config.hidden_size,
                    output_size=model_config.foundation.hidden_size,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )

        if hasattr(config, 'freeze') and config.freeze:
            self.freeze()

    def fast_pos_embed_interpolate(self, image_grid_thw):
        """Interpolate position embeddings for a given grid."""
        grid_ts, grid_hs, grid_ws = image_grid_thw[:, 0], image_grid_thw[:, 1], image_grid_thw[:, 2]
        device = image_grid_thw.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
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

    def forward(self, x: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = self.patch_embed(x)

        pos_embeds = self.fast_pos_embed_interpolate(image_grid_thw)
        x = x + pos_embeds

        seq_len, _ = x.size()
        x = x.reshape(seq_len, -1)

        pad_len = 0
        if (self.config.fp8) and self.config.fp8_recipe == 'blockwise':
            pad_len = (16 - seq_len % 16) % 16
        # Pad x to multiple of 16 when using fp8 blockwise
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        rotary_pos_emb = self.rot_pos_emb(image_grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        if pad_len > 0:
            rotary_pos_emb = F.pad(rotary_pos_emb, (0, 0, 0, pad_len))

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as image_grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Update cumulative length to include padding
        if pad_len > 0:
            cu_seqlens[-1] += pad_len
        
        x = x[:, None, :].contiguous()  # [s, h] -> [s, 1, h]
        x, deepstack_feature_lists = self.decoder(
            x,
            packed_seq_params=[PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
            ) for i in range(self.config.num_layers)],
            rotary_pos_emb=rotary_pos_emb.unsqueeze(1).unsqueeze(2),
            attention_mask=None,
            deepstack_visual_indexes=self.deepstack_visual_indexes,
            deepstack_merger_list=self.deepstack_merger_list
        )
        # x = x[:, 0, :].contiguous()  # [s, 1, h] -> [s, h]
        x = x[:-pad_len if pad_len > 0 else None, 0, :].contiguous()  # [s, 1, h] -> [s, h]
        if pad_len > 0:
            for i in range(len(deepstack_feature_lists)):
                deepstack_feature_lists[i] = deepstack_feature_lists[i][:-(pad_len // 4)]

        return x, None, deepstack_feature_lists