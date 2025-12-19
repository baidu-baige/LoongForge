# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""transformer block"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Union
import math
import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    get_transformer_layer_offset
)
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import is_te_min_version, make_viewless_tensor
from megatron.core.transformer.transformer_block import (
    TransformerBlock as MegatronTransformerBlock,
)

try:
    from megatron.core.extensions.transformer_engine import (
        TENorm,
        get_cpu_offload_context,
        te_checkpoint,
    )

    HAVE_TE = True
    LayerNormImpl = TENorm
except ImportError:
    HAVE_TE = False
    get_cpu_offload_context = None

    try:
        import apex
        LayerNormImpl = FusedLayerNorm

    except ImportError:
        from megatron.core.transformer.torch_norm import WrappedTorchNorm

        LayerNormImpl = WrappedTorchNorm


class TransformerBlock(MegatronTransformerBlock):
    """Transformer class."""

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_fp8_context: bool,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ):
        """Forward method with activation checkpointing."""
        if attn_mask_type is not None:
            attn_mask_type = torch.tensor([attn_mask_type.value], dtype=torch.int)

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states,
                attention_mask,
                attn_mask_type, 
                context, 
                context_mask, 
                rotary_pos_emb,
            ):
                if attn_mask_type is not None:
                    attn_mask_type = AttnMaskType(attn_mask_type.item())

                for index in range(start, end):
                    layer = self._get_layer(index)
                    inner_fp8_context = (
                        get_fp8_context(self.config, layer.layer_number - 1)
                        if use_inner_fp8_context
                        else nullcontext()
                    )
                    with inner_fp8_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            attn_mask_type=attn_mask_type,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_params=None,
                            packed_seq_params=packed_seq_params,
                            **kwargs,
                        )
                    
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            
            if self.config.fp8:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    attention_mask,
                    attn_mask_type,
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
                    attn_mask_type,
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
                    custom(layer_idx, layer_idx + self.config.recompute_num_layers),
                    layer_idx + self.config.recompute_num_layers
                )
                # only first pipeline stage, and if VP exists, only vp_rank == 0
                if (
                    deepstack_visual_embeds is not None and
                    layer_idx in range(len(deepstack_visual_embeds))
                ):  
                    hidden_states = self._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_visual_embeds[layer_idx],
                    )

                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if self.config.fp8 and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(
                        hidden_states,
                        attention_mask,
                        attn_mask_type,
                        context,
                        context_mask,
                        rotary_pos_emb,
                        visual_pos_masks,
                        *visual_embeds,
                        **kwargs,
                    )
                # only first pipeline stage, and if VP exists, only vp_rank == 0
                if (
                    deepstack_visual_embeds is not None and
                    layer_idx in range(len(deepstack_visual_embeds))
                ):  
                    hidden_states = self._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_visual_embeds[layer_idx],
                    )
        else:
            raise ValueError("Invalid activation recompute method.")
        
        return hidden_states

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        sequence_len_offset: Tensor = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_params (InferenceParams, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Update the inference parameters with the current batch size in case it is variable
        if inference_params and not self.training:
            inference_params.current_batch_size = hidden_states.size(1)

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
        use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
        use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

        with rng_context, outer_fp8_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    attn_mask_type=attn_mask_type,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_fp8_context=use_inner_fp8_context,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                    **kwargs,
                )
            else:
                for l_no, layer in enumerate(self.layers):
                    inner_fp8_context = (
                        get_fp8_context(self.config, layer.layer_number - 1)
                        if use_inner_fp8_context
                        else nullcontext()
                    )
                    
                    with self.offload_context, inner_fp8_context:
                        layer.use_cudagraph = True
                        if (len(self.cuda_graphs) == 0) or (not self.training):
                            hidden_states, context = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                attn_mask_type=attn_mask_type,
                                context=context,
                                context_mask=context_mask,
                                rotary_pos_emb=rotary_pos_emb,
                                rotary_pos_cos=rotary_pos_cos,
                                rotary_pos_sin=rotary_pos_sin,
                                attention_bias=attention_bias,
                                inference_params=inference_params,
                                packed_seq_params=packed_seq_params,
                                sequence_len_offset=sequence_len_offset,
                                **kwargs,
                            )
                        else:
                            # CUDA graph replay for layer `l_no` and microbatch
                            # `self.current_microbatch`. TransformerEngine versions>=1.10
                            # allow keyword arguments with CUDA graph. However, CUDA graph
                            # acccepts only Tensor inputs and Tensor outputs. Hence,
                            # `inference_params` and `packed_seq_params` are excluded from
                            # input list while output is limited to `hidden_states`.
                            cg_index = self.current_microbatch % len(self.cuda_graphs[l_no])
                            assert not any([inference_params, packed_seq_params]), \
                                "CUDA graph accepts only Tensor inputs."
                            
                            optional_inputs = self.get_cuda_graph_optional_args(
                                attention_mask,
                                context,
                                context_mask,
                                rotary_pos_emb,
                                attention_bias,
                                inference_params,
                                packed_seq_params,
                            )
                            hidden_states = self.cuda_graphs[l_no][cg_index](
                                hidden_states, **optional_inputs
                            )

                    if (
                        deepstack_visual_embeds is not None and
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

        return hidden_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the transformer block.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
                Defaults to an empty string.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (dict, optional): Additional metadata for sharding.
                Can specify if layers are non-homogeneous. Defaults to None.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the model.
        """
        assert not sharded_offsets, "Unexpected sharded offsets"
        non_homogeneous_layers = metadata is not None and metadata.get(
            'non_homogeneous_layers', False
        )
        if isinstance(self.config.moe_layer_freq, int):
            if self.config.moe_layer_freq > 1:
                non_homogeneous_layers = True
        elif isinstance(self.config.moe_layer_freq, list):
            non_homogeneous_layers = True

        sharded_state_dict = {}

        layer_prefix = f'{prefix}layers.'
        num_layers = self.config.num_layers
        for layer in self.layers:
            offset = get_transformer_layer_offset(self.config)

            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            # module list index in TransformerBlock
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'
            if non_homogeneous_layers:
                sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
                sharded_pp_offset = []
            else:
                sharded_prefix = layer_prefix
                sharded_pp_offset = [
                    (0, global_layer_offset, num_layers)
                ]  # PP sharding offset for ShardedTensors
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module, f'{prefix}{name}.', sharded_offsets, metadata
                    )
                )

        return sharded_state_dict
    
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