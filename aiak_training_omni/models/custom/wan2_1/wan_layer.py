# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""tranformer layer."""

from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Optional, Union
from einops import rearrange

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from megatron.core import parallel_state
from megatron.core.utils import make_viewless_tensor
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    get_transformer_layer_offset,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from aiak_training_omni.models.custom.transformer.vision.stdit_model_embedding import (
    t2i_modulate,
)
from aiak_training_omni.models.custom.transformer.vision.stdit_transformer_config import (
    StditTransformerConfig,
)
from megatron.core.parallel_state import (
    get_context_parallel_world_size,
)
import math


@dataclass
class WanLayerSubmodules:
    """Submodules for a transformer layer."""

    self_attention: Union[ModuleSpec, type] = IdentityOp
    wan_self_attention: Union[ModuleSpec, type] = IdentityOp

    cross_attention: Union[ModuleSpec, type] = IdentityOp
    wan_cross_attention: Union[ModuleSpec, type] = IdentityOp

    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class WanCrossAttentionSubmodules:
    """
    Configuration class for specifying the submodules of a cross-attention.
    """

    linear_q: Union[ModuleSpec, type] = None
    linear_kv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    apply_rotary_fn: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class WanLayer(MegatronModule, BaseTransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: StditTransformerConfig,
        submodules: WanLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__(config=config)

        self.layer_number = layer_number + get_transformer_layer_offset(self.config)
        self.hidden_dropout = (
            config.hidden_dropout if hidden_dropout is None else hidden_dropout
        )
        self.d_t = config.num_latent_frames // config.latent_patch_size[0]
        self.d_s = (
            config.max_latent_height
            // config.latent_patch_size[1]
            * config.max_latent_width
            // config.latent_patch_size[2]
        )
        if config.context_parallel_size > 1:
            self.d_t //= parallel_state.get_context_parallel_world_size()
        if config.sequence_parallel:
            self.d_t //= parallel_state.get_tensor_model_parallel_world_size()

        #######################################
        dim = config.hidden_size
        ffn_dim = config.ffn_hidden_size
        eps = config.layernorm_epsilon
        # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.modulation = nn.Parameter(torch.randn(6, 1, dim) / dim**0.5)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        # self.self_attn = build_module(
        #     submodules.self_attention, config=self.config, layer_number=layer_number)
        self.self_attn = build_module(
            submodules.wan_self_attention, config=self.config, layer_number=layer_number
        )
        # #标准cross attention
        # self.cross_attn = build_module(
        #     submodules.cross_attention, config=self.config, layer_number=layer_number)
        # 融了两个 cross attention 最后结果相加再做linear
        self.cross_attn = build_module(
            submodules.wan_cross_attention,
            config=self.config,
            layer_number=layer_number,
        )

    def modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
        """Apply modulation with shift and scale to input tensor."""
        return x * (1 + scale) + shift

    def gate(self, x, gate, residual):
        """x + gate * residual"""
        return x + gate * residual

    def forward(
        self,
        hidden_states,
        s_attn_mask=None,
        t_attn_mask=None,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        temporal_pos_embed=None,
        **kwargs,
    ):
        """hidden_states: [s, b, h]"""
        # hidden_states_shape: [32760/ 2 + 769 + 6 +1]
        cp = self.config.context_parallel_size
        image_len = math.ceil(self.config.max_image_length / cp) * cp
        text_legnth = math.ceil(self.config.max_text_length / cp) * cp
        conext_len = image_len + text_legnth
        video_len = self.config.max_video_length
        each_cp_hidden = video_len // cp
        each_cp_context = math.ceil(conext_len / cp)
        x = hidden_states[:each_cp_hidden, :, :].to(torch.bfloat16)
        context = hidden_states[each_cp_hidden : each_cp_hidden + each_cp_context, :, :]
        t_mod = hidden_states[-7:-1, :, :].to(torch.bfloat16)
        t_s = hidden_states[-1:, :, :].to(torch.bfloat16)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=0)
        norm1 = self.norm1(x)
        input_x = self.modulate(norm1, shift_msa, scale_msa)
        self_att_out, bias = self.self_attn(
            input_x, attention_mask=None, rotary_pos_emb=rotary_pos_emb.to(x.device)
        )
        self_att_out = self_att_out + bias
        self_att_out = self.gate(x, gate_msa, self_att_out)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            norm3 = self.norm3(self_att_out)
            cross_out, bias = self.cross_attn(
                norm3, attention_mask=context_mask, key_value_states=context
            )
            cross_out = cross_out + bias
        cross_out = self_att_out + cross_out
        input_x = self.modulate(self.norm2(cross_out), shift_mlp, scale_mlp)
        ffn_out = self.ffn(input_x)
        x = self.gate(cross_out, gate_mlp, ffn_out)
        output = torch.cat([x, context, t_mod, t_s], dim=0)
        output = make_viewless_tensor(
            inp=output, requires_grad=x.requires_grad, keep_graph=True
        )

        return output, context

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """sharded state dict"""
        sharded_state_dict = super().sharded_state_dict(
            prefix, sharded_offsets, metadata
        )
        prefixed_map = {
            f"{prefix}{k}": f"{prefix}{v}"
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict
