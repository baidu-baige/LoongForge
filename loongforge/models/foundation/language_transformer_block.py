# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Language Transformer Block"""

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor

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


class TransformerBlock(MegatronTransformerBlock):
    """Language Transformer Class."""
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

        with rng_context, outer_quantization_context:
            # Forward pass.
            recompute_for_chunkpipe = False
            native_recompute = False
            if self.config.recompute_granularity == 'full':
                native_recompute = True
            if self.config.enable_chunkpipe:
                chunk_num = self.config.chunkpipe_forward_microbatch % self.config.chunk_num_per_seq
                if chunk_num + self.config.keep_activations_chunks < self.config.chunk_num_per_seq:
                    recompute_for_chunkpipe = True

            if (native_recompute or recompute_for_chunkpipe) and self.training:
                hidden_states = self._checkpointed_forward(
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
                deepstack_visual_embeds = kwargs.pop('deepstack_visual_embeds', None)
                visual_pos_masks = kwargs.pop('visual_pos_masks', None)
                has_deepstack = self._check_inputs_parameters(deepstack_visual_embeds, visual_pos_masks)
                
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
                        if isinstance(packed_seq_params, list) and len(packed_seq_params) > 0:
                            packed_seq_params_l_no = packed_seq_params[l_no]
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
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                            **kwargs,
                        )
                        
                        # vision deepstack features process
                        if (
                            has_deepstack and
                            l_no in range(len(deepstack_visual_embeds))
                        ):  
                            hidden_states = self._deepstack_process(
                                hidden_states,
                                visual_pos_masks,
                                deepstack_visual_embeds[l_no],
                            )

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

        return hidden_states
        
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
        deepstack_visual_embeds = kwargs.pop('deepstack_visual_embeds', None)
        visual_pos_masks = kwargs.pop('visual_pos_masks', None)
        has_deepstack = self._check_inputs_parameters(deepstack_visual_embeds, visual_pos_masks)
        if has_deepstack and (self.config.recompute_method == 'uniform'
            and self.config.recompute_granularity == 'full'):
            # If DeepStack is present and the uniform strategy is used, 
            # the value of recompute_num_layers must be set to 1; otherwise, DeepStack may be skipped
            assert self._recompute_num_layers == 1, \
                "If has_deepstack is true and recompute_method is set to uniform, recompute_num_layers must be 1."

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
                            packed_seq_params=packed_seq_params,
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

        if self.config.enable_chunkpipe:
            start_layer, end_layer = 0, self.num_layers_per_pipeline_rank
            for layer_idx in range(start_layer, end_layer):
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + 1)
                )
            return hidden_states

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self._recompute_num_layers)
                )
                if (
                    has_deepstack 
                    and layer_idx in range(len(deepstack_visual_embeds))
                ):  
                    hidden_states = self._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_visual_embeds[layer_idx],
                    )

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
                if (
                    has_deepstack
                    and layer_idx in range(len(deepstack_visual_embeds))
                ):  
                    hidden_states = self._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_visual_embeds[layer_idx],
                    )
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        if self.config.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(hidden_states)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        if self.config.sequence_parallel:
            hidden_states = tensor_parallel.scatter_to_sequence_parallel_region(hidden_states)  
        return hidden_states

    def _check_inputs_parameters(
        self,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
    ):
        """
        Check whether deepstack been used and set has_deepstack to True or False.
        """
        has_deepstack = False
        if deepstack_visual_embeds is not None and visual_pos_masks is not None:
            has_deepstack = True
        elif deepstack_visual_embeds is None and visual_pos_masks is None:
            has_deepstack = False
        else:
            raise ValueError(
                "deepstack_visual_embeds and visual_pos_masks must be provided together."
            )
        return has_deepstack