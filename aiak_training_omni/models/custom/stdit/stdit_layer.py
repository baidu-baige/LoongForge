# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""tranformer layer."""

from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Optional, Union
from einops import rearrange

import torch

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


@dataclass
class STDiTLayerSubmodules:
    """Submodules for a transformer layer."""

    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    spatial_self_attention: Union[ModuleSpec, type] = IdentityOp
    spatial_self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp
    temporal_self_attention: Union[ModuleSpec, type] = IdentityOp
    temporal_self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class STDiTLayer(MegatronModule, BaseTransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: StditTransformerConfig,
        submodules: STDiTLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__(config=config)
        self.submodules_config = submodules

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

        self.scale_shift_table = torch.nn.Parameter(
            torch.randn(6, config.hidden_size) / config.hidden_size**0.5
        )

        ## [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        ## [Module 2: Spatial SelfAttention]
        self.spatial_self_attention = build_module(
            submodules.spatial_self_attention,
            config=self.config,
            layer_number=layer_number,
        )
        self.spatial_self_attn_bda = build_module(submodules.spatial_self_attn_bda)

        ## [Module 3: Temporal SelfAttention]
        self.temporal_self_attention = build_module(
            submodules.temporal_self_attention,
            config=self.config,
            layer_number=layer_number,
        )
        self.temporal_self_attn_bda = build_module(submodules.temporal_self_attn_bda)

        ## [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        ## [Module 5: CrossAttention]
        self.cross_attention = build_module(
            submodules.cross_attention, config=self.config, layer_number=layer_number
        )

        ## [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(
            submodules.cross_attn_bda, config=self.config
        )

        ## [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        ## [Module 8: MLP block]
        # TODO how to set the gpt_layer_spec.py when we have moe_frequency > 1,
        #      where MLP and MoE layer both appear alternately?
        self.mlp = build_module(submodules.mlp, config=self.config)
        if hasattr(self.mlp, "set_layer_number"):
            self.mlp.set_layer_number(self.layer_number)

        ## [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def forward(
        self,
        hidden_states,
        s_attn_mask,
        t_attn_mask,
        attention_mask=None,
        context=None,
        context_mask=None,
        timestep=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        temporal_pos_embed=None,
    ):
        """hidden_states: [s, b, h]"""
        S, B, H = hidden_states.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[:, None] + timestep.reshape(6, B, -1)
        ).chunk(6)

        # Spatial Self Attention.
        residual = hidden_states
        hidden_states = t2i_modulate(
            self.input_layernorm(hidden_states), shift_msa, scale_msa
        )
        hidden_states = rearrange(
            hidden_states, "(T S) B C -> S (B T) C", T=self.d_t, S=self.d_s
        ).contiguous()
        spatial_attention_output, spatial_attention_bias = self.spatial_self_attention(
            hidden_states,
            attention_mask=s_attn_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
        )
        spatial_attention_output = rearrange(
            spatial_attention_output, "S (B T) C -> (T S) B C", T=self.d_t, S=self.d_s
        ).contiguous()

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.spatial_self_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(
                (spatial_attention_output, spatial_attention_bias),
                gate_msa,
                residual,
                self.hidden_dropout,
            )

        # Temporal Self attention.
        residual = hidden_states
        hidden_states = rearrange(
            hidden_states, "(T S) B C -> T (B S) C", T=self.d_t, S=self.d_s
        )
        if self.layer_number == 1 and temporal_pos_embed is not None:
            pos_embed_hidden_states = hidden_states + temporal_pos_embed.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        else:
            pos_embed_hidden_states = hidden_states

        temporal_attention_output, temporal_attention_bias = (
            self.temporal_self_attention(
                pos_embed_hidden_states,
                attention_mask=t_attn_mask,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
            )
        )
        temporal_attention_output = rearrange(
            temporal_attention_output, "T (B S) C -> (T S) B C", T=self.d_t, S=self.d_s
        ).contiguous()

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.temporal_self_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(
                (temporal_attention_output, temporal_attention_bias),
                gate_msa,
                residual,
                self.hidden_dropout,
            )

        # Cross attention.
        residual = hidden_states
        attention_output_with_bias = self.cross_attention(
            hidden_states,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if (
            isinstance(attention_output_with_bias, dict)
            and "context" in attention_output_with_bias
        ):
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, None, residual, self.hidden_dropout)

        # MLP.
        residual = hidden_states
        pre_mlp_layernorm_output = t2i_modulate(
            self.pre_mlp_layernorm(hidden_states), shift_mlp, scale_mlp
        )
        mlp_output = self.mlp(pre_mlp_layernorm_output)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(
                self.training, self.config.bias_dropout_fusion
            )(mlp_output, gate_mlp, residual, self.hidden_dropout)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.

        output = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
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
