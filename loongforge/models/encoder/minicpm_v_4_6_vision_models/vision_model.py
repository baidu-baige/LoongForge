# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Hugging Face Transformers MiniCPM-V-4.6 under the Apache-2.0 License.
# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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

"""MiniCPM-V-4.6 vision tower."""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import build_module

from loongforge.models.common import BaseMegatronVisionModule
from loongforge.models.dispatch import multiacc_modules
from loongforge.models.encoder.vision_transformer_block import TransformerBlock
from loongforge.models.utils import import_module

from .minicpm_v_4_6_config import MiniCPMV46VisionConfig
from .minicpm_v_4_6_layer_spec import MiniCPMV46TEDotProductAttention


def _load_state_dict_hook_ignore_extra_state(module, incompatible_keys):
    """Ignore Transformer Engine FP8 metadata absent from HF-derived checkpoints."""
    del module
    for keys in incompatible_keys._asdict().values():
        for key in keys[::-1]:
            if key.endswith("._extra_state"):
                keys.remove(key)


def _target_sizes_from_grid(image_grid_thw: torch.Tensor) -> torch.Tensor:
    """Convert LoongForge grid_thw to MiniCPM target_sizes."""
    if image_grid_thw is None:
        raise ValueError("MiniCPM-V-4.6 vision forward requires image_grid_thw.")
    if image_grid_thw.dim() != 2 or image_grid_thw.shape[-1] != 3:
        raise ValueError(f"Expected image_grid_thw with shape [n, 3], got {tuple(image_grid_thw.shape)}.")
    return image_grid_thw[:, 1:].to(dtype=torch.long)


def get_vision_nearest_position_ids(
    target_sizes: torch.Tensor,
    num_patches_per_side: int,
) -> torch.Tensor:
    """Nearest-neighbor ids into the learned square position table."""
    device = target_sizes.device
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side, device=device)
    pos_ids_list = []
    for height, width in target_sizes.tolist():
        height, width = int(height), int(width)
        h_coords = torch.arange(height, device=device) / height
        w_coords = torch.arange(width, device=device) / width
        bucket_h = torch.bucketize(h_coords, boundaries, right=True)
        bucket_w = torch.bucketize(w_coords, boundaries, right=True)
        pos_ids_list.append((bucket_h[:, None] * num_patches_per_side + bucket_w).flatten())
    return torch.cat(pos_ids_list)


def get_vision_window_index(
    target_sizes: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return MiniCPM window-attention reorder indices and cumulative sequence lengths."""
    window_index = []
    cu_window_seqlens = [0]
    window_index_id = 0

    for height, width in target_sizes.tolist():
        grid_t, grid_h, grid_w = 1, int(height), int(width)
        index = torch.arange(grid_t * grid_h * grid_w, device=target_sizes.device).reshape(grid_t, grid_h, grid_w)
        pad_h = window_size - grid_h % window_size
        pad_w = window_size - grid_w % window_size
        num_windows_h = (grid_h + pad_h) // window_size
        num_windows_w = (grid_w + pad_w) // window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(grid_t, num_windows_h, window_size, num_windows_w, window_size)
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, num_windows_h * num_windows_w, window_size, window_size
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_seqlens_tmp = seqlens.cumsum(0) + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += grid_t * grid_h * grid_w

    return (
        torch.cat(window_index, dim=0),
        torch.unique_consecutive(torch.tensor(cu_window_seqlens, device=target_sizes.device, dtype=torch.int32)),
    )


class MiniCPMV46VisionEmbeddings(nn.Module):
    """Conv2d patch embedding plus nearest-neighbor learned position ids."""

    def __init__(self, config: MiniCPMV46VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.num_patches_per_side = self.image_size // self.patch_size
        self.position_embedding = nn.Embedding(self.num_patches_per_side**2, self.embed_dim)

    def forward(self, pixel_values: torch.Tensor, target_sizes: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() != 4:
            raise ValueError(
                "MiniCPM pixel_values must be a processor-packed or BCHW tensor, "
                f"got {tuple(pixel_values.shape)}."
            )
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=self.patch_embedding.weight.dtype)
        )
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        pos_ids = get_vision_nearest_position_ids(target_sizes, self.num_patches_per_side).to(
            self.position_embedding.weight.device
        )
        if embeddings.shape[1] != pos_ids.numel():
            raise ValueError(
                f"MiniCPM pixel_values contains {embeddings.shape[1]} patches, but target_sizes "
                f"describes {pos_ids.numel()}."
            )
        position_embeddings = self.position_embedding(pos_ids).unsqueeze(0)
        return embeddings + position_embeddings


class MiniCPMV46VisionAttention(nn.Module):
    """TE window attention retaining MiniCPM's separate Q/K/V parameters."""

    def __init__(self, config: MiniCPMV46VisionConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        linear_kwargs = {
            "config": config,
            "init_method": config.init_method,
            "bias": config.add_qkv_bias,
            "skip_bias_add": False,
            "parallel_mode": None,
            "skip_weight_param_allocation": False,
        }
        self.q_proj = build_module(multiacc_modules.TELinear, self.dim, self.dim, **linear_kwargs)
        self.k_proj = build_module(multiacc_modules.TELinear, self.dim, self.dim, **linear_kwargs)
        self.v_proj = build_module(multiacc_modules.TELinear, self.dim, self.dim, **linear_kwargs)
        self.core_attention = MiniCPMV46TEDotProductAttention(
            config=config,
            layer_number=config.insert_layer_id + 1,
            attn_mask_type=AttnMaskType.no_mask,
            attention_type="self",
        )
        self.out_proj = build_module(
            multiacc_modules.TELinear,
            self.dim,
            self.dim,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            skip_bias_add=False,
            parallel_mode=None,
            skip_weight_param_allocation=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        if batch_size != 1:
            raise ValueError("MiniCPM packed TE window attention requires batch size 1")

        def project(module: nn.Module) -> torch.Tensor:
            projected, _ = module(hidden_states)
            return projected.view(seq_len, self.num_heads, self.head_dim)

        query_states = project(self.q_proj)
        key_states = project(self.k_proj)
        value_states = project(self.v_proj)
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )
        attn_output = self.core_attention(
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            attn_mask_type=AttnMaskType.no_mask,
            packed_seq_params=packed_seq_params,
        )
        attn_output = attn_output.reshape(batch_size, seq_len, self.dim).contiguous()
        output, _ = self.out_proj(attn_output)
        return output


class MiniCPMV46ViTWindowAttentionMerger(nn.Module):
    """Intermediate MiniCPM ViT window attention and 2x2 merger."""

    def __init__(self, config: MiniCPMV46VisionConfig):
        super().__init__()
        self.window_kernel_size = tuple(config.window_kernel_size)
        self.embed_dim = config.hidden_size
        self.self_attn = MiniCPMV46VisionAttention(config)
        self.layer_norm1 = build_module(
            multiacc_modules.TENorm,
            config=config,
            hidden_size=self.embed_dim,
            eps=config.layernorm_epsilon,
        )
        self.pre_norm = build_module(
            multiacc_modules.TENorm,
            config=config,
            hidden_size=config.window_hidden_size,
            eps=config.layernorm_epsilon,
        )
        linear_kwargs = {
            "config": config,
            "bias": config.add_bias_linear,
            "skip_bias_add": False,
            "parallel_mode": None,
            "skip_weight_param_allocation": False,
        }
        self.linear_1 = build_module(
            multiacc_modules.TELinear,
            config.window_hidden_size,
            config.window_intermediate_size,
            init_method=config.init_method,
            **linear_kwargs,
        )
        self.linear_2 = build_module(
            multiacc_modules.TELinear,
            config.window_intermediate_size,
            self.embed_dim,
            init_method=config.output_layer_init_method,
            **linear_kwargs,
        )

    def forward(self, hidden_states: torch.Tensor, target_sizes: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        window_h, window_w = self.window_kernel_size
        if window_h != window_w:
            raise ValueError(f"window_kernel_size must be square, got {self.window_kernel_size}.")
        window_index, cu_seqlens = get_vision_window_index(target_sizes, window_h)
        window_index = window_index.to(hidden_states.device)
        cu_seqlens = cu_seqlens.to(hidden_states.device)

        hidden_states = hidden_states[:, window_index, :]
        hidden_states = self.self_attn(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=window_h * window_w)
        hidden_states = hidden_states[:, torch.argsort(window_index), :]
        hidden_states = residual + hidden_states

        embed_dim = hidden_states.shape[-1]
        outputs = []
        offset = 0
        for height, width in target_sizes.tolist():
            height, width = int(height), int(width)
            if height % window_h != 0 or width % window_w != 0:
                raise ValueError(
                    f"Patch grid ({height}, {width}) must be divisible by window kernel size "
                    f"{self.window_kernel_size}."
                )
            num_patches = height * width
            merged_h = height // window_h
            merged_w = width // window_w
            patch = hidden_states[:, offset : offset + num_patches, :]
            offset += num_patches
            patch_5d = patch.view(1, merged_h, window_h, merged_w, window_w, embed_dim).permute(0, 1, 3, 2, 4, 5)
            flat = patch_5d.reshape(merged_h * merged_w, window_h * window_w * embed_dim)
            residual = patch_5d.reshape(merged_h * merged_w, window_h * window_w, embed_dim).mean(dim=1)
            hidden_state = self.pre_norm(flat)
            hidden_state, _ = self.linear_1(hidden_state)
            hidden_state = F.gelu(hidden_state, approximate="tanh")
            hidden_state, _ = self.linear_2(hidden_state)
            outputs.append(hidden_state + residual)
        return torch.cat(outputs, dim=0).unsqueeze(0)


class MiniCPMV46VisionModel(BaseMegatronVisionModule):
    """MiniCPM-V-4.6 vision model."""

    config_class = MiniCPMV46VisionConfig

    def __init__(
        self,
        config: MiniCPMV46VisionConfig,
        vp_stage: Optional[int] = None,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__(config)
        self.embeddings = MiniCPMV46VisionEmbeddings(config)
        model_spec = config.model_spec or [
            "loongforge.models.encoder.minicpm_v_4_6_vision_models.minicpm_v_4_6_layer_spec",
            "get_minicpm_v_4_6_vision_layer_spec",
        ]
        transformer_layer_spec = import_module(model_spec, config)
        self.encoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
            post_layer_norm=False,
            vp_stage=vp_stage,
        )
        self.post_layernorm = build_module(
            multiacc_modules.TENorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        )
        self.vit_merger = MiniCPMV46ViTWindowAttentionMerger(config)
        self.register_load_state_dict_post_hook(_load_state_dict_hook_ignore_extra_state)
        if getattr(config, "freeze", False):
            self.freeze()

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        self.encoder.set_input_tensor(input_tensor)

    def get_dummy_input(self, device):
        patch_size = self.config.patch_size
        return (
            torch.randn((1, 3, patch_size * 2, patch_size * 2), dtype=torch.bfloat16, device=device),
            torch.tensor([[1, 2, 2]], dtype=torch.int32, device=device),
        )

    def get_downsampled_inputs(
        self,
        target_sizes: torch.Tensor,
        max_seqlen: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_sizes = target_sizes // 2
        max_seqlen = max_seqlen // 4
        cu_seqlens = F.pad(
            torch.cumsum(target_sizes[:, 0] * target_sizes[:, 1], dim=0, dtype=torch.int32).to(device),
            (1, 0),
        )
        return target_sizes, cu_seqlens, max_seqlen

    def _forward_encoder_segment(
        self,
        hidden_states: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        start_layer: int,
        end_layer: int,
    ) -> torch.Tensor:
        """Run a stable layer view through the standard TransformerBlock path.

        MiniCPM changes sequence length at its intermediate merger, so the
        layers before and after it need different packed-sequence metadata.
        The shallow block copy shares the original layer modules and parameters,
        while its private ``_modules`` mapping owns only the requested view.
        This is important for full activation recompute: the backward closure
        keeps this view instead of observing a restored or mutated module list.
        """
        if not 0 <= start_layer < end_layer <= len(self.encoder.layers):
            raise ValueError(
                f"Invalid MiniCPM vision layer range [{start_layer}, {end_layer}) "
                f"for {len(self.encoder.layers)} local layers."
            )

        segment = copy.copy(self.encoder)
        segment._modules = self.encoder._modules.copy()
        segment.layers = self.encoder.layers[start_layer:end_layer]
        segment.num_layers_per_pipeline_rank = len(segment.layers)
        hidden_states, _ = segment(
            hidden_states=hidden_states,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
        )
        return hidden_states

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> tuple[torch.Tensor, None, list]:
        if len(self.encoder.layers) != self.config.num_layers:
            raise RuntimeError(
                "MiniCPM's intermediate vision merger requires all vision encoder layers "
                "on the local encoder rank. LoongForge normally enforces PP=1 and VPP=None "
                f"for the image encoder, but this instance owns {len(self.encoder.layers)} "
                f"of {self.config.num_layers} layers."
            )
        target_sizes = _target_sizes_from_grid(image_grid_thw)
        hidden_states = self.embeddings(pixel_values, target_sizes=target_sizes)
        # Megatron TransformerLayer uses [sequence, batch, hidden].
        hidden_states = hidden_states.squeeze(0).unsqueeze(1).contiguous()
        cu_seqlens = F.pad(
            torch.cumsum(target_sizes[:, 0] * target_sizes[:, 1], dim=0, dtype=torch.int32).to(hidden_states.device),
            (1, 0),
        )
        max_seqlen = torch.max(cu_seqlens[1:] - cu_seqlens[:-1])
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=int(max_seqlen.item()),
            max_seqlen_kv=int(max_seqlen.item()),
        )
        use_vit_merger = self.config.downsample_mode != "4x"

        if use_vit_merger:
            split = self.config.insert_layer_id + 1
            hidden_states = self._forward_encoder_segment(
                hidden_states, packed_seq_params, 0, split
            )
            hidden_states = self.vit_merger(hidden_states.transpose(0, 1).contiguous(), target_sizes)
            hidden_states = hidden_states.squeeze(0).unsqueeze(1).contiguous()
            target_sizes, cu_seqlens, max_seqlen = self.get_downsampled_inputs(
                target_sizes=target_sizes,
                max_seqlen=max_seqlen,
                device=hidden_states.device,
            )
            packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=int(max_seqlen.item()),
                max_seqlen_kv=int(max_seqlen.item()),
            )
            hidden_states = self._forward_encoder_segment(
                hidden_states, packed_seq_params, split, self.config.num_layers
            )
        else:
            hidden_states = self._forward_encoder_segment(
                hidden_states, packed_seq_params, 0, self.config.num_layers
            )

        last_hidden_state = self.post_layernorm(hidden_states)
        return last_hidden_state[:, 0, :].contiguous(), target_sizes, []
