# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""tranformer layer."""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.utils import make_viewless_tensor
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_context_parallel_group,
    get_hierarchical_context_parallel_groups,
)
from megatron.core.process_groups_config import ProcessGroupCollection

from loongforge.models.common import BaseModelStditConfig

import math

import transformer_engine.pytorch as te


@dataclass
class WanLayerSubmodules(TransformerLayerSubmodules):
    """Submodules for a transformer layer."""

    wan_self_attention: Union[ModuleSpec, type] = IdentityOp
    wan_cross_attention: Union[ModuleSpec, type] = IdentityOp


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


class WanLayer(TransformerLayer):
    """A single transformer layer."""

    def __init__(
        self,
        config: BaseModelStditConfig,
        submodules: WanLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        self.d_t = ((config.num_latent_frames - 1) // config.vae_temporal_compress + 1) // config.latent_patch_size[0]
        self.d_s = (
            (config.max_latent_height // config.vae_spatial_compress // config.latent_patch_size[1])
            * (config.max_latent_width // config.vae_spatial_compress // config.latent_patch_size[2])
        )

        dim = config.hidden_size
        ffn_dim = config.ffn_hidden_size
        eps = config.layernorm_epsilon
        self.modulation = nn.Parameter(torch.randn(6, 1, dim) / dim**0.5)

        self.ffn = nn.Sequential(
            te.Linear(dim, ffn_dim, bias=True, return_bias=False),
            nn.GELU(approximate="tanh"),
            te.Linear(ffn_dim, dim, bias=True, return_bias=False),
        )

        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        # Determine cp_comm_type for this layer (mirrors TransformerLayer logic)
        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and getattr(config, 'cp_comm_type', None) is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[layer_number - 1]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type
            # Megatron's Attention.__init__ builds pg_collection with only ['tp','cp']
            # when None is passed, which fails TEDotProductAttention's hcp assertion
            # for cp_comm_type='a2a+p2p' (Ulysses+Ring hybrid). Supply a full pg_collection
            # including hcp ourselves.
            attention_optional_kwargs["pg_collection"] = ProcessGroupCollection(
                tp=get_tensor_model_parallel_group(check_initialized=False),
                cp=get_context_parallel_group(check_initialized=False),
                hcp=get_hierarchical_context_parallel_groups(check_initialized=False),
            )

        self.self_attention = build_module(
            submodules.wan_self_attention, config=self.config, layer_number=layer_number,
            **attention_optional_kwargs
        )

        self.cross_attn = build_module(
            submodules.wan_cross_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )

        _sel = getattr(self.config, 'recompute_granularity', None) == 'selective'
        self.recompute_ffn        = _sel
        self.recompute_cross_attn = _sel

        self.t_mod = None

    def modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
        return x * (1 + scale) + shift

    def gate(self, x, gate, residual):
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
        timestep_mod=None,
        **kwargs,
    ):
        x = hidden_states
        t_mod = timestep_mod if timestep_mod is not None else self.t_mod
        if t_mod is None:
            raise RuntimeError(
                "WanLayer.forward() requires timestep_mod. "
                "Ensure WanModel.forward() passes timestep_mod to the decoder."
            )

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype) + t_mod
        ).chunk(6, dim=0)

        norm1 = self.norm1(x)
        input_x = self.modulate(norm1, shift_msa, scale_msa)
        self_att_out, bias = self.self_attention(
            input_x,
            attention_mask=None,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
        )
        self_att_out = self_att_out + bias
        self_att_out = self.gate(x, gate_msa, self_att_out)

        norm3 = self.norm3(self_att_out)
        if self.recompute_cross_attn and self.training:
            def _cross_attn_fwd(norm3, context):
                out, bias = self.cross_attn(
                    norm3, attention_mask=context_mask, key_value_states=context
                )
                return out, bias
            cross_out, bias = torch.utils.checkpoint.checkpoint(
                _cross_attn_fwd, norm3, context, use_reentrant=False
            )
        else:
            cross_out, bias = self.cross_attn(
                norm3, attention_mask=context_mask, key_value_states=context
            )
        cross_out = cross_out + bias
        cross_out = self_att_out + cross_out

        input_x = self.modulate(self.norm2(cross_out), shift_mlp, scale_mlp)
        if self.recompute_ffn and self.training:
            ffn_out = torch.utils.checkpoint.checkpoint(
                self.ffn, input_x, use_reentrant=False
            )
        else:
            ffn_out = self.ffn(input_x)
        x = self.gate(cross_out, gate_mlp, ffn_out)
        output = make_viewless_tensor(
            inp=x, requires_grad=x.requires_grad, keep_graph=True
        )

        return output, context

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
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
