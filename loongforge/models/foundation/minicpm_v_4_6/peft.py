# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-specific canonical LoRA implementation."""

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from torch import nn

from loongforge.models.common.peft.adapter_wrapper import AdapterWrapper
from loongforge.models.common.peft.canonical_lora import CanonicalLoRA
from loongforge.models.common.peft.lora_config import VLMLoraConfig
from loongforge.models.common.peft.lora_layers import (
    LinearAdapter as BaseLinearAdapter,
    LoRALinear,
    LoRATopKRouter,
    TELinearAdapter as BaseTELinearAdapter,
)
from loongforge.models.common.peft.utils import (
    ParallelLinearAdapter,
    get_adapter_attributes_from_linear,
    is_expert_linear,
)


logger = logging.getLogger(__name__)


def lora_linear_forward(
    linear_in: nn.Linear,
    linear_out: nn.Linear,
    inputs: torch.Tensor,
    scale: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Run MiniCPM FP32 LoRA weights at the base projection's output dtype."""
    inputs = inputs.to(dtype=linear_in.weight.dtype)
    if inputs.is_cuda and output_dtype in (torch.float16, torch.bfloat16):
        with torch.autocast(device_type="cuda", dtype=output_dtype):
            return linear_out(linear_in(inputs)) * scale
    return linear_out(linear_in(inputs)) * scale


class MiniCPMLinearAdapter(BaseLinearAdapter):
    """Dense linear adapter that preserves the MiniCPM model output dtype."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_forward = getattr(self, "super_fwd", None)
        if base_forward is not None:
            if base_forward == self.forward:
                raise RuntimeError("MiniCPM LoRA base forward cannot call itself")
            output = base_forward(inputs)
        else:
            output = F.linear(inputs, self.weight, self.bias)

        if not self._adapter_enabled:
            return output
        if self.dropout_position == "pre":
            inputs = self.dropout(inputs)
        adapter_output = lora_linear_forward(
            self.linear_in,
            self.linear_out,
            inputs,
            self.scale,
            output.dtype,
        )
        if self.dropout_position == "post":
            adapter_output = self.dropout(adapter_output)
        return output + adapter_output.to(dtype=output.dtype)


class MiniCPMTELinearAdapter(BaseTELinearAdapter):
    """Transformer Engine adapter with a MiniCPM-specific FP32 LoRA branch."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.linear_in.weight.dtype != self.weight.dtype:
            with torch.autocast(device_type="cuda", dtype=inputs.dtype):
                output = te.Linear.forward(self, inputs)
        else:
            output = te.Linear.forward(self, inputs)

        if not self._adapter_enabled:
            return output
        if self.dropout_position == "pre":
            inputs = self.dropout(inputs)
        adapter_output = lora_linear_forward(
            self.linear_in,
            self.linear_out,
            inputs,
            self.scale,
            output.dtype,
        )
        if self.dropout_position == "post":
            adapter_output = self.dropout(adapter_output)
        return output + adapter_output.to(dtype=output.dtype)


@dataclass
class MiniCPMV46VLMLoraConfig(VLMLoraConfig):
    """MiniCPM-only LoRA configuration fields."""

    adapter_backend: Literal["parallel", "auto"] = "auto"
    fc1_adapter_order: Literal["up_gate", "gate_up"] = "gate_up"


class DenseLinearAdapter(nn.Module):
    """Replicated LoRA branch used by the MiniCPM TP1 alignment path."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dim: int,
        alpha: int,
        dtype: torch.dtype,
        device: torch.device,
        a_init: str,
        b_init: str,
        dropout: float = 0.0,
        dropout_position: Literal["pre", "post"] = "pre",
    ) -> None:
        super().__init__()
        self.linear_in = nn.Linear(
            in_features, dim, bias=False, dtype=dtype, device=device
        )
        self.linear_out = nn.Linear(
            dim, out_features, bias=False, dtype=dtype, device=device
        )
        if a_init == "kaiming":
            nn.init.kaiming_uniform_(self.linear_in.weight, a=math.sqrt(5))
        elif a_init == "xavier":
            nn.init.xavier_normal_(self.linear_in.weight)
        else:
            raise ValueError(f"Unsupported canonical LoRA A initializer: {a_init}")
        if b_init != "zero":
            raise ValueError(f"Unsupported canonical LoRA B initializer: {b_init}")
        nn.init.zeros_(self.linear_out.weight)
        self.scale = alpha / dim
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        if dropout_position not in ("pre", "post"):
            raise ValueError(f"Unsupported LoRA dropout position: {dropout_position}")
        self.dropout_position = dropout_position

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output_dtype = inputs.dtype
        if self.dropout_position == "pre":
            inputs = self.dropout(inputs)
        outputs = lora_linear_forward(
            self.linear_in,
            self.linear_out,
            inputs,
            self.scale,
            output_dtype,
        )
        if self.dropout_position == "post":
            outputs = self.dropout(outputs)
        return outputs.to(dtype=output_dtype)

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        del metadata
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(keep_vars=True),
            prefix,
            sharded_offsets=sharded_offsets,
        )


class MiniCPMParallelLinearAdapter(ParallelLinearAdapter):
    """Tensor-parallel adapter with MiniCPM's explicit parameter dtype."""

    def __init__(self, *args, lora_dtype=None, **kwargs):
        super().__init__(*args, **kwargs)
        if lora_dtype is not None:
            self.to(dtype=lora_dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output_dtype = inputs.dtype
        outputs = super().forward(inputs.to(dtype=self.linear_in.weight.dtype))
        return outputs.to(dtype=output_dtype)


class MiniCPMModuleDict(nn.ModuleDict):
    """ModuleDict that skips disabled canonical adapter branches."""

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        state_dict = {}
        for key, layer in self.items():
            if layer is not None:
                state_dict.update(
                    layer.sharded_state_dict(
                        f"{prefix}{key}.", sharded_offsets, metadata
                    )
                )
        return state_dict


class MiniCPMLoRALinearSplitQKV(AdapterWrapper):
    """MiniCPM QKV adapter wrapper for its split-aware attention forward."""

    def adapter_output(self, component: str, inputs: torch.Tensor):
        adapter = getattr(self.adapter, f"adapter_{component}")
        return None if adapter is None else adapter(inputs)

    def forward(self, inputs: torch.Tensor, *args: Any, **kwargs: Any):
        linear_output, bias, layernorm_output = self.base_linear_forward(
            inputs, *args, **kwargs
        )
        if not self._adapter_enabled:
            return linear_output, bias

        outputs = []
        for component in ("q", "k", "v"):
            output = self.adapter_output(component, layernorm_output)
            if output is None:
                split_size = self.to_wrap.canonical_split_sizes["linear_q" if component == "q" else f"linear_{component}"]
                output = layernorm_output.new_zeros(*layernorm_output.shape[:-1], split_size)
            outputs.append(output)
        query, key, value = outputs

        config = self.to_wrap.config
        head_size = config.kv_channels
        head_num = query.size(-1) // head_size
        num_query_groups = key.size(-1) // head_size
        heads_per_group = head_num // num_query_groups
        leading_shape = query.shape[:-1]
        query = query.reshape(-1, head_num, head_size)
        key = key.reshape(-1, num_query_groups, head_size)
        value = value.reshape(-1, num_query_groups, head_size)
        chunks = []
        for index in range(num_query_groups):
            chunks.extend(
                [
                    query[:, index * heads_per_group : (index + 1) * heads_per_group],
                    key[:, index : index + 1],
                    value[:, index : index + 1],
                ]
            )
        adapter_output = torch.cat(chunks, dim=1).reshape(*leading_shape, -1)
        return linear_output + adapter_output, bias


class MiniCPMLoRALinearSplitFC1(AdapterWrapper):
    """MiniCPM gate/up adapter wrapper with a split torch fallback."""

    def forward(self, inputs: torch.Tensor, *args: Any, **kwargs: Any):
        linear_output, bias, layernorm_output = self.base_linear_forward(
            inputs, *args, **kwargs
        )
        if not self._adapter_enabled:
            return linear_output, bias
        gate = self.adapter.adapter_gate(layernorm_output)
        up = self.adapter.adapter_up(layernorm_output)
        return linear_output + torch.cat([gate, up], dim=-1), bias

    def forward_split(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        projection_input = inputs
        if self.to_wrap.config.sequence_parallel:
            projection_input = gather_from_sequence_parallel_region(inputs)

        weight_gate, weight_up = torch.chunk(self.to_wrap.weight, 2, dim=0)
        bias = getattr(self.to_wrap, "bias", None)
        if bias is not None and bias.numel() > 0:
            bias_gate, bias_up = torch.chunk(bias, 2, dim=0)
        else:
            bias_gate = bias_up = None

        gate = F.linear(projection_input, weight_gate, bias_gate)
        up = F.linear(projection_input, weight_up, bias_up)
        if self._adapter_enabled:
            for component, output in (("gate", gate), ("up", up)):
                adapter = getattr(self.adapter, f"adapter_{component}")
                if adapter is None:
                    continue
                adapter_input = (
                    projection_input
                    if getattr(adapter, "disable_sequence_parallel_comm", True)
                    else inputs
                )
                output.add_(adapter(adapter_input).reshape(output.shape))
        return gate, up


class MiniCPMLoRALinearSplitProjection(AdapterWrapper):
    """Wrapper for MiniCPM's fused QKVZ and BA projections."""

    def forward(self, inputs: torch.Tensor, *args: Any, **kwargs: Any):
        raise RuntimeError("MiniCPM split projection requires a split-aware forward")

    def adapter_output(self, component: str, inputs: torch.Tensor):
        adapter = getattr(self.adapter, f"adapter_{component}")
        return None if adapter is None else adapter(inputs)

    def backward_dw(self):
        return self.to_wrap.backward_dw()


@dataclass
class MiniCPMV46CanonicalLoRA(CanonicalLoRA):
    """Canonical LoRA policy for the MiniCPM-V-4.6 projection layout."""

    lora_dtype: Optional[torch.dtype] = None
    adapter_backend: Literal["parallel", "auto"] = "auto"
    fc1_adapter_order: Literal["up_gate", "gate_up"] = "gate_up"
    a2a_experimental: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.lora_dtype, str):
            dtype_map = {
                "float32": torch.float32,
                "float": torch.float32,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
            }
            try:
                self.lora_dtype = dtype_map[self.lora_dtype.lower()]
            except KeyError as error:
                raise ValueError(
                    f"Unsupported MiniCPM LoRA dtype: {self.lora_dtype}"
                ) from error
        if self.adapter_backend not in ("parallel", "auto"):
            raise ValueError("adapter_backend must be one of: parallel, auto")
        if self.fc1_adapter_order not in ("up_gate", "gate_up"):
            raise ValueError("fc1_adapter_order must be one of: up_gate, gate_up")

        for target in self.target_modules:
            if target.endswith("linear_qkv") or target.endswith("linear_fc1"):
                raise ValueError(f"Unsupported canonical MiniCPM target: {target}")
            mappings = (
                ("linear_q", "linear_qkv", "linear_q"),
                ("linear_k", "linear_qkv", "linear_k"),
                ("linear_v", "linear_qkv", "linear_v"),
                ("linear_fc1_up", "linear_fc1", "linear_fc1_up"),
                ("linear_fc1_gate", "linear_fc1", "linear_fc1_gate"),
                ("in_proj_qkv", "in_proj_qkvz", "in_proj_qkv"),
                ("in_proj_z", "in_proj_qkvz", "in_proj_z"),
                ("in_proj_b", "in_proj_ba", "in_proj_b"),
                ("in_proj_a", "in_proj_ba", "in_proj_a"),
            )
            for suffix, fused_suffix, canonical_name in mappings:
                if target.endswith(suffix):
                    fused_target = target[: -len(suffix)] + fused_suffix
                    self.canonical_mapping[fused_target].add(canonical_name)
                    break
            else:
                self.canonical_mapping[target].add(target)

    def _dense_adapter(self, module, in_features, out_features):
        return DenseLinearAdapter(
            in_features,
            out_features,
            dim=self.dim,
            alpha=self.alpha,
            dtype=self.lora_dtype or module.weight.dtype,
            device=module.weight.device,
            a_init=self.lora_A_init_method,
            b_init=self.lora_B_init_method,
            dropout=self.dropout,
            dropout_position=self.dropout_position,
        )

    def transform(
        self,
        module: nn.Module,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> nn.Module:
        adapter_types = (
            DenseLinearAdapter,
            BaseLinearAdapter,
            LoRALinear,
            LoRATopKRouter,
            BaseTELinearAdapter,
            MiniCPMLoRALinearSplitQKV,
            MiniCPMLoRALinearSplitFC1,
            MiniCPMLoRALinearSplitProjection,
        )
        if isinstance(module, adapter_types):
            return module

        match_result = self.match(module, name, prefix)
        if match_result is None:
            return module
        match, full_name = match_result
        canonical_submodules = self.canonical_mapping[match]
        split_sizes = getattr(module, "canonical_split_sizes", {})

        if name in ("in_proj_qkvz", "in_proj_ba"):
            if module.__class__ != te.Linear:
                raise TypeError(
                    f"MiniCPM split projection requires te.Linear, got {type(module)}"
                )
            components = (
                ("qkv", "in_proj_qkv"),
                ("z", "in_proj_z"),
            ) if name == "in_proj_qkvz" else (
                ("b", "in_proj_b"),
                ("a", "in_proj_a"),
            )
            adapters = {}
            for adapter_name, canonical_name in components:
                adapter = None
                if canonical_name in canonical_submodules:
                    if canonical_name not in split_sizes:
                        raise ValueError(
                            f"{full_name} does not declare a size for {canonical_name}"
                        )
                    adapter = self._dense_adapter(
                        module, module.in_features, split_sizes[canonical_name]
                    )
                adapters[f"adapter_{adapter_name}"] = adapter
            return MiniCPMLoRALinearSplitProjection(
                module, MiniCPMModuleDict(adapters)
            )

        if isinstance(module, nn.Linear):
            return MiniCPMLinearAdapter(
                module,
                dim=self.dim,
                alpha=self.alpha,
                dropout=self.dropout,
                lora_A_init_method=self.lora_A_init_method,
                lora_dtype=self.lora_dtype,
            )
        if module.__class__ == te.Linear:
            return MiniCPMTELinearAdapter(
                module,
                dim=self.dim,
                alpha=self.alpha,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                lora_A_init_method=self.lora_A_init_method,
                lora_dtype=self.lora_dtype,
            )

        is_expert = is_expert_linear(full_name)
        attrs = get_adapter_attributes_from_linear(module, is_expert=is_expert)
        adapter_kwargs = dict(
            dim=self.dim,
            base_linear_name=full_name,
            activation="identity",
            norm_type=None,
            column_init_method=self.lora_A_init_method,
            row_init_method=self.lora_B_init_method,
            gather_output=False,
            input_is_parallel=attrs.input_is_parallel,
            dropout=self.dropout,
            dropout_position=self.dropout_position,
            model_parallel_config=getattr(module, "config", None),
            alpha=self.alpha,
            is_expert=is_expert,
            disable_tensor_parallel_comm=attrs.disable_tensor_parallel_comm,
            disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
            base_linear_is_parallel=attrs.base_linear_is_parallel,
            lora_dtype=self.lora_dtype,
        )
        use_dense_adapter = (
            self.adapter_backend == "auto"
            and not is_expert
            and getattr(
                getattr(module, "config", None), "tensor_model_parallel_size", 1
            ) == 1
        )

        def make_adapter(in_features, out_features):
            if use_dense_adapter:
                return self._dense_adapter(module, in_features, out_features)
            return MiniCPMParallelLinearAdapter(
                in_features, out_features, **adapter_kwargs
            )

        logger.info("Adding MiniCPM LoRA to: %s (%s)", full_name, canonical_submodules)
        if name == "linear_qkv":
            q_out_features = split_sizes.get(
                "linear_q",
                module.config.kv_channels * module.config.num_attention_heads,
            )
            kv_out_features = (
                module.config.kv_channels * module.config.num_query_groups
            )
            adapters = MiniCPMModuleDict(
                {
                    "adapter_q": make_adapter(attrs.in_features, q_out_features)
                    if "linear_q" in canonical_submodules
                    else None,
                    "adapter_k": make_adapter(attrs.in_features, kv_out_features)
                    if "linear_k" in canonical_submodules
                    else None,
                    "adapter_v": make_adapter(attrs.in_features, kv_out_features)
                    if "linear_v" in canonical_submodules
                    else None,
                }
            )
            return MiniCPMLoRALinearSplitQKV(module, adapters)

        if name == "linear_fc1":
            adapters = {}
            for component in self.fc1_adapter_order.split("_"):
                canonical_name = f"linear_fc1_{component}"
                adapters[f"adapter_{component}"] = (
                    make_adapter(attrs.in_features, attrs.out_features // 2)
                    if canonical_name in canonical_submodules
                    else None
                )
            return MiniCPMLoRALinearSplitFC1(
                module, MiniCPMModuleDict(adapters)
            )

        adapter = make_adapter(attrs.in_features, attrs.out_features)
        if isinstance(module, TopKRouter):
            return LoRATopKRouter(module, adapter)
        return LoRALinear(module, adapter)

    def post_float16_wrap(self, model) -> None:
        """Restore MiniCPM adapter parameters after Float16Module casts the model."""
        if self.lora_dtype is None:
            return
        for model_module in model:
            for parameter in model_module.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.to(dtype=self.lora_dtype)
