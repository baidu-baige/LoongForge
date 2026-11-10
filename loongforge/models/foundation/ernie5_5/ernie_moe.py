# Copyright 2026 The LoongForge Authors.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA-NeMo/Megatron-Bridge at 8f580c64376877e9e2bc3140ce3476afc0f75d76.

"""ERNIE split-feature routing and latent MoE."""

import torch
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.transformer.moe.moe_layer import MoELayer, get_default_pg_collection
from megatron.core.transformer.moe.router import TopKRouter


class Ernie55Router(TopKRouter):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.weight.data = self.weight.data.float()
        self.weight_1 = torch.nn.Parameter(torch.empty_like(self.weight))
        self.routed_scaling_factor_param = torch.nn.Parameter(
            torch.full((config.num_moe_experts,), config.moe_router_topk_scaling_factor, dtype=torch.float32)
        )
        if config.perform_initialization:
            with torch.no_grad():
                config.init_method(self.weight_1)
        for parameter in self.parameters():
            parameter.keep_in_fp32 = True
        self.expert_bias.keep_in_fp32 = True
        self.local_tokens_per_expert = self.local_tokens_per_expert.to(torch.int64)

    def gating(self, value):
        with torch.autocast(device_type=value.device.type, enabled=False):
            value = value.float()
            return F.linear(value, self.weight).sigmoid() + F.linear(value, self.weight_1).sigmoid()

    def routing(self, logits, input_ids=None):
        scores = self.apply_z_loss(logits.reshape(-1, self.config.num_moe_experts))
        indices = torch.topk(scores + self.expert_bias, self.topk, dim=-1).indices
        probabilities = scores.gather(-1, indices)
        if self.config.norm_topk_prob and self.topk > 1:
            probabilities = probabilities / (probabilities.sum(-1, keepdim=True) + 1e-20)
        probabilities = probabilities * self.routed_scaling_factor_param[indices]
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, indices, True)
        if self.config.moe_router_bias_update_rate and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(0)
        return torch.zeros_like(scores).scatter(-1, indices, probabilities), routing_map


class Ernie55LatentProjection(torch.nn.Module):
    def __init__(self, config, input_size, output_size, init_method):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(output_size, input_size, dtype=torch.float32))
        self.weight.keep_in_fp32 = True
        if config.perform_initialization:
            with torch.no_grad():
                init_method(self.weight)

    def forward(self, value):
        return F.linear(value, self.weight.to(value.dtype))


class Ernie55MoELayer(MoELayer):
    def __init__(self, config, submodules, layer_number=None, pg_collection=None, is_mtp_layer=False):
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super().__init__(
            config,
            submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        self.router = Ernie55Router(
            config,
            pg_collection=pg_collection,
            layer_number=layer_number,
            is_mtp_layer=is_mtp_layer,
        )
        self.fc1_latent_proj = None
        self.fc2_latent_proj = None
        self.fc2_norm = None
        if config.moe_latent_size:
            self.fc1_latent_proj = Ernie55LatentProjection(
                config,
                config.hidden_size,
                config.moe_latent_size,
                config.init_method,
            )
            self.fc2_latent_proj = Ernie55LatentProjection(
                config,
                config.moe_latent_size,
                config.hidden_size,
                config.output_layer_init_method,
            )
            if config.moe_use_norm_before_up_proj:
                self.fc2_norm = TENorm(config, config.moe_latent_size, eps=config.layernorm_epsilon)
        ordered = {}
        for name in ("router", "fc1_latent_proj", "fc2_latent_proj", "fc2_norm"):
            if name in self._modules:
                ordered[name] = self._modules[name]
        ordered.update((name, module) for name, module in self._modules.items() if name not in ordered)
        self._modules = ordered

    def forward(self, hidden_states, input_ids=None):
        # BF16 input gradients depend on the shared/router/latent branch order.
        shared = self.shared_experts_compute(hidden_states)
        probabilities, routing_map = self.router(hidden_states, input_ids=input_ids)
        latent = self.fc1_latent_proj(hidden_states) if self.fc1_latent_proj is not None else hidden_states
        metadata = self.token_dispatcher.preprocess(routing_map)
        tokens, probabilities = self.token_dispatcher.dispatch_preprocess(latent, probabilities, metadata)
        tokens, probabilities = self.dispatch(tokens, probabilities, metadata)
        tokens, counts, probabilities = self.pre_routed_experts_compute(tokens, probabilities, metadata)
        output, bias = self.routed_experts_compute(tokens, counts, probabilities)
        output = self.post_routed_experts_compute(output, metadata)
        output = self.combine(output, metadata)
        output = self.post_combine(output, metadata, None)
        if self.fc2_norm is not None:
            output = self.fc2_norm(output)
        if self.fc2_latent_proj is not None:
            output = self.fc2_latent_proj(output)
        if shared is not None:
            output = output + shared
        return output, bias
