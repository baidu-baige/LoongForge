# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Vision Transformer Block"""

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    fine_grained_offloading_set_last_layer,
)
from megatron.core.pipeline_parallel.utils import is_vp_first_stage, is_vp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    get_transformer_layer_offset,
)
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    get_pg_rank,
    make_viewless_tensor,
)
from megatron.core.transformer.transformer_block import (
    TransformerBlock as MegatronTransformerBlock,
    TransformerBlockSubmodules,
)

try:
    import transformer_engine.pytorch as te  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import

    HAVE_APEX = True
except ImportError:
    HAVE_APEX = False

get_cpu_offload_context = None
te_checkpoint = None

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import (
        TENorm,
        get_cpu_offload_context,
        te_checkpoint,
    )

    LayerNormImpl = TENorm

elif HAVE_APEX:
    LayerNormImpl = FusedLayerNorm

else:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    LayerNormImpl = WrappedTorchNorm


logger = logging.getLogger(__name__)


def pad_sequence_for_alignment(
    hidden_states: Tensor,
    rotary_pos_emb: Optional[Tensor],
    packed_seq_params: Optional[List],
    target_multiple: int = 16,
) -> tuple[Tensor, Optional[Tensor], int]:
    """
    Pad hidden_states and related tensors to align with a target multiple.

    Args:
        hidden_states: Input tensor of shape [s, ...]
        rotary_pos_emb: Rotary positional embeddings
        packed_seq_params: List of PackedSeqParams for packed sequence processing
        target_multiple: The target multiple to align sequence length to

    Returns:
        A tuple of (padded_hidden_states, padded_rotary_pos_emb, pad_len)
    """
    seq_len = hidden_states.shape[0]
    pad_len = (target_multiple - seq_len % target_multiple) % target_multiple

    if pad_len > 0:
        hidden_states = F.pad(
            hidden_states,
            pad=[0, 0] * (hidden_states.dim() - 1) + [0, pad_len]
        )
        if rotary_pos_emb is not None:
            rotary_pos_emb = F.pad(
                rotary_pos_emb,
                pad=[0, 0] * (rotary_pos_emb.dim() - 1) + [0, pad_len]
            )
        if packed_seq_params is not None:
            for packed_seq_param in packed_seq_params:
                packed_seq_param.cu_seqlens_q = packed_seq_param.cu_seqlens_q.clone()
                packed_seq_param.cu_seqlens_q[-1] += pad_len
                packed_seq_param.cu_seqlens_kv = packed_seq_param.cu_seqlens_kv.clone()
                packed_seq_param.cu_seqlens_kv[-1] += pad_len

    return hidden_states, rotary_pos_emb, packed_seq_params, pad_len


def unpad_sequence(
    hidden_states: Tensor,
    deepstack_feature_lists: Optional[List[Tensor]] = None,
    pad_len: int = 0,
) -> tuple[Tensor, Optional[List[Tensor]]]:
    """
    Remove padding from hidden_states and deepstack_feature_lists.

    Args:
        hidden_states: Padded input tensor
        deepstack_feature_lists: Optional list of deepstack features that need unpadding
        pad_len: The number of padding tokens to remove from the end

    Returns:
        A tuple of (unpadded_hidden_states, unpadded_deepstack_feature_lists)
    """
    if pad_len > 0:
        hidden_states = hidden_states[:-pad_len].contiguous()
        if deepstack_feature_lists is not None:
            for i, deepstack_feature in enumerate(deepstack_feature_lists):
                deepstack_feature_lists[i] = deepstack_feature[:-(pad_len // 4)].contiguous()

    return hidden_states, deepstack_feature_lists


class TransformerBlock(MegatronTransformerBlock):
    """Vision Transformer Class."""
    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: ProcessGroupCollection = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=config,
                         spec=spec,
                         post_layer_norm=post_layer_norm,
                         pre_process=pre_process,
                         post_process=post_process,
                         pg_collection=pg_collection,
                         vp_stage=vp_stage)
        self.has_deepstack = getattr(config, 'deepstack_visual_indexes', False)

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        dynamic_inference_decode_only: Optional[bool] = None,
        **kwargs,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.
            dynamic_inference_decode_only: Optional[bool]: If true, indicates that the current
                inference context is for decode-only. This args is only used to uniquely
                identify decode and non-decode cuda graph runners in the cuda graph manager.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        deepstack_feature_lists = []

        inference_context = deprecate_inference_params(inference_context, inference_params)
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        # For FP4: NVFP4BlockScaling doesn't have delayed scaling, always uses inner context
        if self.config.fp8:
            use_outer_quantization_context = self.config.fp8_recipe == Fp8Recipe.delayed
            use_inner_quantization_context = self.config.fp8_recipe != Fp8Recipe.delayed
            outer_quantization_context = (
                get_fp8_context(self.config) if use_outer_quantization_context else nullcontext()
            )
        elif self.config.fp4:
            use_outer_quantization_context = False
            use_inner_quantization_context = True
            outer_quantization_context = nullcontext()
        else:
            # No quantization
            use_outer_quantization_context = False
            use_inner_quantization_context = False
            outer_quantization_context = nullcontext()

        if use_inner_quantization_context:
            hidden_states, rotary_pos_emb, packed_seq_params, pad_len = pad_sequence_for_alignment(
                hidden_states, rotary_pos_emb, packed_seq_params
            )

        with rng_context, outer_quantization_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states, deepstack_feature_lists = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                    **kwargs,
                )
            else:
                if self.has_deepstack:
                    deepstack_visual_indexes = kwargs.pop('deepstack_visual_indexes', None)
                    deepstack_merger_list = kwargs.pop('deepstack_merger_list', None)
                    assert deepstack_visual_indexes is not None and deepstack_merger_list is not None, \
                        "deepstack_visual_indexes and deepstack_merger_list must be passed when has_deepstack is True"
                for l_no, layer in enumerate(self.layers):
                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(
                                self.config, layer.layer_number - 1
                            )
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(
                                self.config, layer.layer_number - 1
                            )
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    if self.config.fine_grained_activation_offloading:
                        fine_grained_offloading_set_last_layer(
                            l_no == self.num_layers_per_pipeline_rank - 1
                        )

                    with self.offload_context, inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            rotary_pos_cos_sin=rotary_pos_cos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=(
                                packed_seq_params[l_no]
                                if isinstance(packed_seq_params, (list, tuple))
                                else packed_seq_params
                            ),
                            sequence_len_offset=sequence_len_offset,
                            **kwargs,
                        )

                    # vision deepstack
                    if self.has_deepstack and l_no in deepstack_visual_indexes:
                            merger = deepstack_merger_list[deepstack_visual_indexes.index(l_no)]
                            deepstack_feature = merger(hidden_states)
                            deepstack_feature_lists.append(deepstack_feature)

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        # If this TransformerBlock is empty, input and output hidden states will be the same node
        # on the computational graph and will lead to unexpected errors in pipeline schedules.
        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            hidden_states = hidden_states.clone()

        if use_inner_quantization_context:
            hidden_states, deepstack_feature_lists = unpad_sequence(
                hidden_states, deepstack_feature_lists, pad_len
            )

        return hidden_states, deepstack_feature_lists
        
    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_quantization_context: bool,
        **kwargs,
    ):
        """Forward method with activation checkpointing."""
        deepstack_feature_lists = []
        if self.has_deepstack:
            deepstack_visual_indexes = kwargs.pop('deepstack_visual_indexes', None)
            deepstack_merger_list = kwargs.pop('deepstack_merger_list', None)
            assert deepstack_visual_indexes is not None and deepstack_merger_list is not None, \
                "deepstack_visual_indexes and deepstack_merger_list must be passed when has_deepstack is True"
            
            # If DeepStack is enabled and using uniform strategy, recompute_num_layers must be 1, 
            # otherwise DeepStack may be skipped
            if self.config.recompute_method == 'uniform' and self.config.recompute_granularity == 'full':
                assert self._recompute_num_layers == 1, \
                    "If recompute_method is set to uniform, recompute_num_layers must be 1."

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states, attention_mask, context, context_mask, rotary_pos_emb
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)

                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(
                                self.config, layer.layer_number - 1
                            )
                        # TODO: check if fp4 is supported in this case
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(
                                self.config, layer.layer_number - 1
                            )
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=(
                                packed_seq_params[index]
                                if isinstance(packed_seq_params, (list, tuple))
                                else packed_seq_params
                            ),
                            **kwargs,
                        )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            # TODO: check if fp4 is supported in this case
            if self.config.fp8 or self.config.fp4:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    **kwargs,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    **kwargs,
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self._recompute_num_layers)
                )
                # Extract deepstack features
                if self.has_deepstack and layer_idx in deepstack_visual_indexes:
                    merger = deepstack_merger_list[deepstack_visual_indexes.index(layer_idx)]
                    deepstack_feature = merger(hidden_states)
                    deepstack_feature_lists.append(deepstack_feature)

                layer_idx += self._recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                # TODO: check if fp4 is supported in this case
                if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self._recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(
                        hidden_states, attention_mask, context, context_mask, rotary_pos_emb, **kwargs
                    )
                # Extract deepstack features
                if self.has_deepstack and layer_idx in deepstack_visual_indexes:
                    merger = deepstack_merger_list[deepstack_visual_indexes.index(layer_idx)]
                    deepstack_feature = merger(hidden_states)
                    deepstack_feature_lists.append(deepstack_feature)
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states, deepstack_feature_lists