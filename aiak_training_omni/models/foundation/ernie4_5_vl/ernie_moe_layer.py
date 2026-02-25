"""Moe layer for Ernie"""

from dataclasses import dataclass
from typing import Union, Optional
from copy import deepcopy
import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_layer import (
    MoESubmodules,
    MoELayer
)
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig
from megatron.core.extensions.transformer_engine import te_checkpoint
from aiak_training_omni.utils import get_args
from .ernie_router import TopKRouter


@dataclass
class  MultiTypeMoeSubmodules:
    """MoE Layer Submodule spec"""
    vision_moe_layer: Union[ModuleSpec, type] = None
    text_moe_layer: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class ErnieMoeLayer(MoELayer):
    """MoE Layer for Ernie"""
    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, pg_collection=pg_collection)
        # Add change: route dtype from fp16->fp32
        args = get_args()
        router_config = deepcopy(self.config)
        for mixed_precision in args.use_fp32_dtype_for_param_pattern:
            if "router" in mixed_precision:
                router_config.params_dtype = torch.float32

        self.router = TopKRouter(config=router_config, pg_collection=pg_collection)

    def forward(self, hidden_states: torch.Tensor, token_type_ids: torch.Tensor):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.
            token_type_ids (torch.Tensor): Token type IDs for routing.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        # if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
        #     raise ValueError(
        #         "During training, performance may degrade if MoE and tensor parallelism"
        #         "are enabled without also enabling sequence parallelism."
        #     )

        # MoE forward: route -> dispatch -> compute -> combine -> post-combine
        def custom_forward(hidden_states, token_type_ids):
            # Add Mask tokens from different modalities
            mask = token_type_ids == 0
            select_index = mask.squeeze(0).nonzero().squeeze(1)
            hidden_states = torch.index_select(input=hidden_states, dim=0, index=select_index)

            shared_expert_output = self.shared_experts_compute(hidden_states)
            # convert to fp32
            # hidden_states_fp32 = hidden_states.to(torch.float)
            # hidden_states_fp32, probs, residual = self.router_and_preprocess(hidden_states_fp32)
            # hidden_states = hidden_states_fp32.to(hidden_states.dtype)
            ori_dtype = hidden_states.dtype
            hidden_states, probs, residual = self.router_and_preprocess(hidden_states)
            probs = probs.to(ori_dtype)
            hidden_states = hidden_states.to(ori_dtype)

            dispatched_input, probs = self.dispatch(hidden_states, probs)
            dispatched_input, tokens_per_expert, permuted_probs = self.pre_routed_experts_compute(
                dispatched_input, probs)
            expert_output, mlp_bias = self.routed_experts_compute(dispatched_input, tokens_per_expert, permuted_probs)
            output = self.post_routed_experts_compute(expert_output)
            output = self.combine(output)
            output = self.post_combine(output, shared_expert_output)
            return output, mlp_bias

        def custom_forward_exclude_shared_experts(hidden_states, token_type_ids):
            # Add Mask tokens from different modalities
            mask = token_type_ids == 0
            select_index = mask.squeeze(0).nonzero().squeeze(1)
            hidden_states = torch.index_select(input=hidden_states, dim=0, index=select_index)

            hidden_states, probs, residual = self.router_and_preprocess(hidden_states)
            dispatched_input, probs = self.dispatch(hidden_states, probs)
            dispatched_input, tokens_per_expert, permuted_probs = self.pre_routed_experts_compute(
                dispatched_input, probs)
            expert_output, mlp_bias = self.routed_experts_compute(dispatched_input, tokens_per_expert, permuted_probs)
            output = self.post_routed_experts_compute(expert_output)
            output = self.token_dispatcher.token_combine(output)
            output = self.token_dispatcher.combine_postprocess(output)
            return output, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8:
                output, mlp_bias = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    token_type_ids
                )
            else:
                output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        elif self.routed_experts_recompute:
            if self.config.fp8:
                output, mlp_bias = te_checkpoint(
                    custom_forward_exclude_shared_experts,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    token_type_ids
                )
            else:
                output, mlp_bias = tensor_parallel.checkpoint(
                   custom_forward_exclude_shared_experts, 
                   False, 
                   hidden_states,
                   token_type_ids
                )
            if self.use_shared_expert and not self.shared_expert_overlap:
                output = output + self.shared_experts_compute(hidden_states)

        else:
            output, mlp_bias = custom_forward(hidden_states, token_type_ids)

        return output, mlp_bias


class ErnieMultiTypeMoE(MegatronModule):
    """Multimodal Mixture of Experts layer for text and vision modalities.

    Routes tokens to specialized experts based on modality type and merges 
    outputs. Part of ERNIE 4.5's multimodal MoE architecture.

    Args:
        config: ERNIE 4.5 multimodal MoE configuration.
        submodules: Expert network module configuration.
        layer_number: Layer index in the Transformer stack.
    """
    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.config = config
        self.layer_number = layer_number
        self.submodules = submodules
        self.text_config = deepcopy(self.config)
        self.vision_config = deepcopy(self.config)
        self.shared_config = deepcopy(self.config)
        self.text_config.moe_ffn_hidden_size = self.config.moe_intermediate_size[0]
        self.vision_config.moe_ffn_hidden_size = self.config.moe_intermediate_size[1]
        self.text_moe_layer = build_module(self.submodules.text_moe_layer, self.text_config)
        self.vision_moe_layer = build_module(self.submodules.vision_moe_layer, self.vision_config)
        self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)

    def forward(self, hidden_states, token_type_ids):
        """Forward pass for the ErnieMultiTypeMoE layer."""
        lm_output, bias_lm = self.text_moe_layer(hidden_states, token_type_ids)
        mm_output, bias_mm = self.vision_moe_layer(hidden_states, 1 - token_type_ids)
        assert bias_lm is None and bias_mm is None, "currently ernie model mlp_bias should be None"
        # output = torch.where(token_type_ids.unsqueeze(-1), lm_output, mm_output)

        shared_output = self.shared_experts(hidden_states)
        
        moe_output = torch.zeros_like(hidden_states)
        mask = token_type_ids.squeeze(0).bool()  # [1,839] → [839]
        moe_output[~mask] = lm_output  # [23, 2560]
        moe_output[mask] = mm_output  # [816, 2560]
        moe_output = moe_output + shared_output
        return moe_output, None

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.text_moe_layer.router.set_layer_number(layer_number)
        self.vision_moe_layer.router.set_layer_number(layer_number)