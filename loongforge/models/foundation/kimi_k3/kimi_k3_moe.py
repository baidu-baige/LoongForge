# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Megatron-Bridge under the Apache-2.0 License:
# https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/kimi/kimi_k3_layers.py

"""Kimi K3 latent-MoE extension."""

import torch
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer.moe.moe_layer import MoELayer

from .kimi_k3_ops import KimiRMSNorm, sum_grads_across_tp


class KimiK3MoELayer(MoELayer):
    """MCore MoE layer with K3's latent projections and post-combine RMSNorm."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        latent_size = self.config.moe_latent_size
        if not latent_size:
            raise ValueError("Kimi K3 MoE layers require a non-zero moe_latent_size")

        self.fc1_latent_proj = self._latent_linear(
            self.config.hidden_size, latent_size, self.config.init_method
        )
        self.fc2_latent_proj = self._latent_linear(
            latent_size, self.config.hidden_size, self.config.output_layer_init_method
        )
        self.routed_expert_norm = KimiRMSNorm(
            latent_size,
            self.config.layernorm_epsilon,
            device=torch.cuda.current_device(),
            dtype=self.config.params_dtype,
        )
        sum_grads_across_tp(self.routed_expert_norm)

    def _latent_linear(self, input_size: int, output_size: int, init_method) -> TELinear:
        return TELinear(
            input_size,
            output_size,
            parallel_mode="duplicated",
            config=self.config,
            init_method=init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            is_expert=False,
        )

    def router_and_preprocess(self, hidden_states: torch.Tensor, input_ids=None):
        """Route on the residual stream, then project into the latent space.

        Routing has to see the full-width hidden state (the router weight is
        hidden_size wide), so the projection goes after the router and before
        dispatch, which also keeps the dispatched payload at latent width.
        """
        residual = hidden_states
        probs, routing_map = self.router(hidden_states, input_ids=input_ids)
        metadata = self.token_dispatcher.preprocess(routing_map)

        hidden_states, _ = self.fc1_latent_proj(hidden_states)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, probs, metadata
        )
        return hidden_states, probs, metadata, residual

    def post_combine(
        self,
        output: torch.Tensor,
        metadata,
        shared_expert_output: torch.Tensor | None,
    ) -> torch.Tensor:
        """Normalize the combined routed output before projecting back to hidden size.

        The normalization is applied to the combined result, not per expert, so it
        cannot be folded into the expert modules.
        """
        output = self.token_dispatcher.combine_postprocess(output, metadata)
        output = self.routed_expert_norm(output)
        output, _ = self.fc2_latent_proj(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output


__all__ = ["KimiK3MoELayer"]
