# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""ErnieExperts"""

import logging
from dataclasses import dataclass
from typing import Optional, Union
import torch
import torch.nn.functional as F
from copy import deepcopy

from megatron.core.dist_checkpointing.mapping import (
    ShardedStateDict,
)
from megatron.core.fusions.fused_bias_geglu import (
    bias_geglu_impl,
    quick_gelu,
    weighted_bias_quick_geglu_impl,
)
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    nvtx_range_pop,
    nvtx_range_push,
)

from megatron.core import tensor_parallel
from megatron.core.transformer.mlp import MLP, apply_swiglu_sharded_factory
from megatron.core.transformer.moe.experts import SequentialMLP as MegaSequentialMLP
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP as MegaSharedExpertMLP
from megatron.core.process_groups_config import ProcessGroupCollection

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


logger = logging.getLogger(__name__)


# pylint: disable=missing-class-docstring
@dataclass
class MLPSubmodules:
    """
    The dataclass for ModuleSpecs of MLP submodules
    including  linear fc1, activation function, linear fc2.
    """

    linear_fc1: Union[ModuleSpec, type] = None
    activation_func: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class ErnieMLP(MLP):

    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: Optional[int] = None,
        ffn_hidden_size: int = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config, submodules=submodules, is_expert=is_expert,
            input_size=input_size, ffn_hidden_size=ffn_hidden_size, tp_group=tp_group)

        # If this is a gated linear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1_1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc1_1",
            tp_group=tp_group,
        )

    def forward(self, hidden_states, per_token_scale=None, idx=0):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="linear_fc1")
        nvtx_range_push(suffix="linear_fc1_1")

        intermediate_parallel_0, bias_parallel = self.linear_fc1(hidden_states)
        intermediate_parallel_1, bias_parallel_1 = self.linear_fc1_1(hidden_states)
        assert bias_parallel is None and bias_parallel_1 is None, "bias_parallel is not None in ernie4.5vl"

        nvtx_range_pop(suffix="linear_fc1")
        nvtx_range_pop(suffix="linear_fc1_1")

        def bias_act_func(intermediate_parallel, bias_parallel):
            nvtx_range_push(suffix="activation")
            if self.config.use_te_activation_func:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                intermediate_parallel = self.activation_func(intermediate_parallel)
                if per_token_scale is not None:
                    original_dtype = intermediate_parallel.dtype
                    intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                    intermediate_parallel = intermediate_parallel.to(original_dtype)
            elif self.config.bias_activation_fusion:
                if per_token_scale is not None:
                    if self.activation_func == F.silu and self.config.gated_linear_unit:
                        # dtype is handled inside the fused kernel
                        intermediate_parallel = weighted_bias_swiglu_impl(
                            intermediate_parallel,
                            bias_parallel,
                            per_token_scale.unsqueeze(-1),
                            self.config.activation_func_fp8_input_store,
                        )
                    elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                        intermediate_parallel = weighted_bias_quick_geglu_impl(
                            intermediate_parallel,
                            bias_parallel,
                            per_token_scale.unsqueeze(-1),
                            self.config.activation_func_fp8_input_store,
                            self.config.glu_linear_offset,
                            self.config.activation_func_clamp_value,
                        )
                    else:
                        raise ValueError(
                            "Only support fusion of swiglu and quick_gelu with per_token_scale in MLP."
                        )
                else:
                    if self.activation_func == F.gelu:
                        if self.config.gated_linear_unit:
                            intermediate_parallel = bias_geglu_impl(
                                intermediate_parallel, bias_parallel
                            )
                        else:
                            assert self.config.add_bias_linear is True
                            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                    elif self.activation_func == F.silu and self.config.gated_linear_unit:
                        intermediate_parallel = bias_swiglu_impl(
                            intermediate_parallel,
                            bias_parallel,
                            self.config.activation_func_fp8_input_store,
                            self.config.cpu_offloading
                            and self.config.cpu_offloading_activations
                            and HAVE_TE,
                        )
                    else:
                        raise ValueError("Only support fusion of gelu and swiglu")
            else:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                if self.config.gated_linear_unit:

                    def glu(x):
                        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                        if (val := self.config.activation_func_clamp_value) is not None:
                            x_glu = x_glu.clamp(min=None, max=val)
                            x_linear = x_linear.clamp(min=-val, max=val)
                        return self.config.activation_func(x_glu) * (
                            x_linear + self.config.glu_linear_offset
                        )

                    intermediate_parallel = glu(intermediate_parallel)
                else:
                    intermediate_parallel = self.activation_func(intermediate_parallel)
                # trick: no multiply at here, we combine with torch.matmul
                # if per_token_scale is not None:
                #     original_dtype = intermediate_parallel.dtype
                #     intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                #     intermediate_parallel = intermediate_parallel.to(original_dtype)
            nvtx_range_pop(suffix="activation")
            return intermediate_parallel

        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            intermediate_parallel_0 = self.activation_checkpoint.checkpoint(
                bias_act_func, intermediate_parallel_0, bias_parallel
            )
            # added_for ernie
            intermediate_parallel = intermediate_parallel_0 * intermediate_parallel_1
            # [s, b, h]
            nvtx_range_push(suffix="linear_fc2")
            output, output_bias = self.linear_fc2(intermediate_parallel)
            nvtx_range_pop(suffix="linear_fc2")
            self.activation_checkpoint.discard_output_and_register_recompute(output)
        else:
            intermediate_parallel_0 = bias_act_func(
                intermediate_parallel_0, bias_parallel
            )
            # added_for ernie
            intermediate_parallel = intermediate_parallel_0 * intermediate_parallel_1
            # [s, b, h]
            nvtx_range_push(suffix="linear_fc2")
            output, output_bias = self.linear_fc2(intermediate_parallel)
            nvtx_range_pop(suffix="linear_fc2")

            # if per_token_scale is not None:
            #     original_dtype = output.dtype
            #     output = output * per_token_scale.unsqueeze(-1)
            #     output = output.to(original_dtype)

        if per_token_scale is not None and output_bias is not None:
            # if this MLP is an expert, and bias is required, we add the bias to output directly
            # without doing bda later.
            output += output_bias.unsqueeze(0) * per_token_scale.unsqueeze(-1)
            output_bias = None
        return output, output_bias

    # pylint: disable=missing-function-docstring
    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Return the sharded state dictionary of the module."""
        sharded_state_dict = {}
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f"{prefix}{name}.", sharded_offsets, metadata)
            if self.config.gated_linear_unit and (name == "linear_fc1" or name == "linear_fc1_1"):
                for k, v in sub_sd.items():
                    if k in (f"{prefix}{name}.weight", f"{prefix}{name}.bias"):
                        sub_sd[k] = apply_swiglu_sharded_factory(
                            v, sharded_offsets, singleton_local_shards
                        )
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict

    def backward_dw(self):
        self.linear_fc2.backward_dw()
        self.linear_fc1_1.backward_dw()
        self.linear_fc1.backward_dw()


class SequentialMLP(MegaSequentialMLP):
    """An implementation of the Experts layer using a sequence of MLP layers.

    This class executes each expert sequentially.
    """

    def __init__(
        self,
        num_local_experts,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
    
        super().__init__(num_local_experts, config=config, submodules=submodules, pg_collection=pg_collection)
        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()
        # self.ep_group = pg_collection.ep
        # use pg_collection.expt_dp_group as data parallel group in this module.
        # TODO (Hepteract): expt_dp wont be needed here once distributed checkpoint is refactored
        self.dp_group = pg_collection.expt_dp

        for _ in range(self.num_local_experts):
            # expert use ErnieMLP instead of MLP
            expert = ErnieMLP(
                self.config,
                submodules,
                ffn_hidden_size=self.config.moe_ffn_hidden_size,
                is_expert=True,
                tp_group=pg_collection.expt_tp,
            )
            self.local_experts.append(expert)


class ErnieSharedExpertMLP(ErnieMLP):
    # This stream is used when '--moe-shared-expert-overlap' is set.
    # The shared experts are scheduled into this stream to be overlapped with the dispatcher.
    stream = None

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        gate: bool,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        config = deepcopy(config)
        assert config.add_bias_linear is False, "bias is not supported in the shared experts, "
        "please set '--disable-bias-linear' instead."
        #ernie shared_expert intermediate size may be different from normal_expert
        config.ffn_hidden_size = config.moe_intermediate_size[0] * config.moe_num_shared_experts
        # config.ffn_hidden_size = config.moe_shared_expert_intermediate_size
        # TODO(Hepteract): pass pg_collection to MLP after refactoring MLP
        super().__init__(config=config, submodules=submodules)

        self.use_shared_expert_gate = gate
        if self.use_shared_expert_gate:
            # TODO: Add support for GPU initialization, which requires updating the golden values.
            self.gate_weight = torch.nn.Parameter(torch.empty((1, self.config.hidden_size)))
            if config.perform_initialization:
                config.init_method(self.gate_weight)
            self.gate_weight.data = self.gate_weight.data.to(dtype=config.params_dtype)
            setattr(self.gate_weight, 'sequence_parallel', self.config.sequence_parallel)
        else:
            self.gate_weight = None

        if self.config.fp8 and is_te_min_version("2.6.0dev0"):
            # For fp8 training, the output of pre_mlp_layernorm is saved by router, and
            # the shared expert linear_fc1 also saves the quantized tensor of this output.
            # Here we set the linear_fc1 to save the original input tensors to avoid the extra
            # memory usage of the quantized tensor.
            shared_experts_recompute = (
                config.recompute_granularity == 'selective'
                and "shared_experts" in config.recompute_modules
            )
            if not shared_experts_recompute:
                try:
                    HAVE_TE = True
                    from megatron.core.extensions.transformer_engine import (
                        TELinear,
                        set_save_original_input,
                    )
                except ImportError:
                    HAVE_TE = False
                    TELinear, set_save_original_input = None, None

                if HAVE_TE and isinstance(self.linear_fc1, TELinear):
                    set_save_original_input(self.linear_fc1)

        if self.config.moe_shared_expert_overlap:
            # disable TP related AG/RS communications in the linear module
            for linear in [self.linear_fc1, self.linear_fc2]:
                if hasattr(linear, 'parallel_mode'):
                    # TELinear
                    linear.parallel_mode = None
                    linear.ub_overlap_rs_fprop = False
                    linear.ub_overlap_ag_dgrad = False
                    linear.ub_overlap_ag_fprop = False
                    linear.ub_overlap_rs_dgrad = False
                else:
                    # MCore legacy Linear
                    linear.explicit_expert_comm = True

            # The overlapped version is splitted into some separated functions and is put inside
            # the token dispatcher. These functions should be called in this order and no one can
            # be skipped:
            #     pre_forward_comm(input)
            #     linear_fc1_forward_and_act()
            #     linear_fc2_forward()
            #     post_forward_comm()
            #     output = get_output()
            #
            # We use cached intermediate results to avoid messy arg passing in the dispatcher.
            self.cached_fc1_input = None
            self.cached_fc2_input = None
            self.cached_fc2_output = None
            self.cached_output = None
            self.gate_score = None

            if self.stream is None:
                self.stream = torch.cuda.Stream()

    def forward(self, hidden_states):
        """Forward function"""
        output, _ = super().forward(hidden_states)
        if self.use_shared_expert_gate:
            logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
            gate_score = torch.nn.functional.sigmoid(logits)
            output = output * gate_score
        return output