# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Transformer layer for Ernie."""

from dataclasses import dataclass
from typing import Optional

import torch
import logging
import torch.nn as nn
from copy import deepcopy
from megatron.core import parallel_state, tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.mlp import MLP
from megatron.core.utils import make_viewless_tensor
from megatron.core.utils import log_single_rank
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.utils import (
    nvtx_range_pop,
    nvtx_range_push,
)

from .ernie_experts import ErnieMLP


logger = logging.getLogger(__name__)


class TransformerLayerErnie(TransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """
    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=config, submodules=submodules,
            layer_number=layer_number, hidden_dropout=hidden_dropout,
            pg_collection=pg_collection, vp_stage=vp_stage )

        # [Module 8: MLP block]
        additional_mlp_kwargs = {}
        # import here to avoid circular import
        from megatron.core.extensions.transformer_engine import TEFusedMLP
        from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from .ernie_moe_layer import ErnieMoeLayer

        # MLP expects tp_group but MoELayer expects pg_collection to be passed in.
        # We can change MLP to accept pg_collection but it makes the logic implicit
        # The conditional below is to make the logic explicit
        # if submodules.mlp is not a ModuleSpec,we dont have to handle passing additional kwargs
        if isinstance(submodules.mlp, ModuleSpec):
            if submodules.mlp.module in (MoELayer, GroupedMLP, TEGroupedMLP, SequentialMLP, ErnieMoeLayer):
                additional_mlp_kwargs["pg_collection"] = pg_collection
            elif submodules.mlp.module in (ErnieMLP, MLP):
                assert hasattr(
                    pg_collection, 'tp'
                ), 'TP process group is required for MLP in TransformerLayer'
                additional_mlp_kwargs["tp_group"] = pg_collection.tp
            elif TEFusedMLP is not None and submodules.mlp.module == TEFusedMLP:
                assert hasattr(
                    pg_collection, 'tp'
                ), 'TP process group is required for TEFusedMLP in TransformerLayer'
                additional_mlp_kwargs["tp_group"] = pg_collection.tp
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unknown MLP type: {type(submodules.mlp)}. Using default kwargs.",
                )

        self.mlp = build_module(submodules.mlp, config=self.config, **additional_mlp_kwargs)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        # [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

        self.recompute_input_layernorm = False
        self.recompute_pre_mlp_layernorm = False
        self.recompute_mlp = False
        self.recompute_pre_mlp = False
        self.a2a_overlap_attn_recompute = False
        self.a2a_overlap_post_attn_recompute = False
        self.a2a_overlap_mlp_recompute = False

        if self.config.recompute_granularity == 'selective':
            if "layernorm" in self.config.recompute_modules:
                if (
                    not isinstance(self.input_layernorm, IdentityOp)
                    and self.config.cuda_graph_impl == "none"
                ):
                    self.recompute_input_layernorm = True
                    if self.config.fp8:
                        self.self_attention.set_for_recompute_input_layernorm()
                if not isinstance(self.pre_mlp_layernorm, IdentityOp):
                    self.recompute_pre_mlp_layernorm = True
                    if self.config.fp8:
                        if isinstance(self.mlp, (MoELayer, ErnieMoeLayer)):
                            self.mlp.set_for_recompute_pre_mlp_layernorm()
                        else:
                            from megatron.core.extensions.transformer_engine import (
                                set_save_original_input,
                            )

                            set_save_original_input(self.mlp.linear_fc1)
            if "mlp" in self.config.recompute_modules:
                if not isinstance(self.mlp, (MoELayer, ErnieMoeLayer)):
                    self.recompute_mlp = True
            if "pre_mlp" in self.config.recompute_modules:
                self.recompute_pre_mlp = True
                self.recompute_input_layernorm = False
            if "a2a_overlap_attn" in self.config.recompute_modules:
                self.a2a_overlap_attn_recompute = True
            if "a2a_overlap_post_attn" in self.config.recompute_modules:
                self.a2a_overlap_post_attn_recompute = True
            if "a2a_overlap_mlp" in self.config.recompute_modules:
                self.a2a_overlap_mlp_recompute = True
        self.offload_attn_norm = (
            self.config.fine_grained_activation_offloading
            and "attn_norm" in self.config.offload_modules
            and not isinstance(self.input_layernorm, IdentityOp)
        )
        self.offload_mlp_norm = (
            self.config.fine_grained_activation_offloading
            and "mlp_norm" in self.config.offload_modules
            and not isinstance(self.pre_mlp_layernorm, IdentityOp)
        )

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def forward(self, *args, **kwargs):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager
        kwargs.pop("dynamic_inference_decode_only", None)
        if self.recompute_pre_mlp:
            hidden_states, context = self._checkpoint_pre_mlp_forward(*args, **kwargs)
        else:
            hidden_states, context = self._forward_attention(*args, **kwargs)

        output = self._forward_mlp(hidden_states, kwargs["context_mask"], kwargs.get("inference_context", None))
        return output, context

    def _forward_mlp(self, hidden_states, context_mask, inference_context=None):
        """
        Perform a forward pass through the feed-forward layer.

        Args:
            hidden_states (Tensor): Transformed hidden states before the MLP layernorm.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            fine_grained_offloading_group_commit,
            fine_grained_offloading_group_start,
            get_fine_grained_offloading_context,
        )

        # Residual connection.
        residual = hidden_states

        if self.offload_mlp_norm:
            hidden_states = fine_grained_offloading_group_start(hidden_states, name="mlp_norm")
        # Optional Layer norm post the cross-attention.
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with get_fine_grained_offloading_context(self.offload_mlp_norm):
                pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                    self.pre_mlp_layernorm, hidden_states
                )
        else:
            with get_fine_grained_offloading_context(self.offload_mlp_norm):
                pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        nvtx_range_push(suffix="mlp")
        # Potentially chunk the MLP computation during prefill to minimize the peak activation size
        should_chunk_mlp_for_prefill = (
            self.config.mlp_chunks_for_prefill > 1
            and inference_context is not None
            and not inference_context.is_decode_only()
            and not isinstance(self.mlp, IdentityOp)
        )
        # ernie_add_at_here
        vision_mask = ~context_mask
        def ernie_mlp(pre_mlp_layernorm_output, vision_mask=None):
            if isinstance(self.mlp, ErnieMLP):
                mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)
            else:
                mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output, vision_mask)
            return mlp_output_with_bias

        if self.recompute_mlp:
            if self.config.fp8:
                # import here to avoid circular import
                from megatron.core.extensions.transformer_engine import te_checkpoint

                mlp_output_with_bias = te_checkpoint(
                    ernie_mlp,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    pre_mlp_layernorm_output,
                    vision_mask
                )
            else:
                mlp_output_with_bias = tensor_parallel.checkpoint(
                    ernie_mlp, False, pre_mlp_layernorm_output, vision_mask
                )
        elif should_chunk_mlp_for_prefill:
            # Chunk input along sequence dimension
            num_chunks = min(self.config.mlp_chunks_for_prefill, pre_mlp_layernorm_output.shape[0])
            chunks = pre_mlp_layernorm_output.chunk(num_chunks, dim=0)

            # Compute outputs for each chunk
                    # MLP, pass masks to mlp

            outputs = [ernie_mlp(chunk, vision_mask) for chunk in chunks]

            # Aggregate chunk outputs
            mlp_output = torch.cat([out for out, _ in outputs], dim=0)
            bias_chunks = [bias for _, bias in outputs if bias is not None]
            bias_output = torch.stack(bias_chunks, dim=0).sum(dim=0) if bias_chunks else None
            mlp_output_with_bias = (mlp_output, bias_output)

        else:
            mlp_output_with_bias = ernie_mlp(pre_mlp_layernorm_output, vision_mask)

        if self.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of mlp_output_with_bias[0]
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )
        nvtx_range_pop(suffix="mlp")

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        nvtx_range_push(suffix="mlp_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        nvtx_range_pop(suffix="mlp_bda")
        if self.offload_mlp_norm:
            (hidden_states,) = fine_grained_offloading_group_commit(
                hidden_states, name="mlp_norm", forced_released_tensors=[residual]
            )
            
        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output