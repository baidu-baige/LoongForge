# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Adapters"""

import torch
from dataclasses import dataclass
from typing import Union
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import  build_module
from .qwen2_vl_config import MLPAdapterConfig
from loongforge.models.common import BaseMegatronModule
from loongforge.models.utils import import_module


class Adapter(BaseMegatronModule):
    """Adaptor"""

    config_class = MLPAdapterConfig

    def __init__(
        self,
        config: TransformerConfig,
        input_size: int,
        output_size: int,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(config=config)
        self.hidden_size = input_size * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        if self.config.model_spec is None:
            model_spec = [
                "loongforge.models.encoder.qwen2_vl_vision_models.qwen2_vl_layer_spec",
                "get_adapeter_layer_with_te_spec",
            ]
        else:
            model_spec = self.config.model_spec
        submodules = import_module(model_spec, self.config)
        self.layernorm = build_module(
            submodules.layernorm,
            config=self.config,
            hidden_size=self.hidden_size if self.use_postshuffle_norm else input_size,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.hidden_size,
            self.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            parallel_mode=None,
            skip_weight_param_allocation=False,
        )

        self.activation_func = self.config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.hidden_size,
            output_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            parallel_mode=None,
            skip_weight_param_allocation=False,
        )
        if hasattr(config, 'freeze') and config.freeze:
            self.freeze()
            
    def forward(
        self, x: torch.Tensor, window_index: torch.LongTensor = None
    ) -> torch.Tensor:
        """Forward pass."""
        x = self.layernorm(
            x.view(-1, self.hidden_size)
            if self.use_postshuffle_norm
            else x
        ).view(-1, self.hidden_size)
        x, _ = self.linear_fc1(x)
        x = self.activation_func(x)
        x, _ = self.linear_fc2(x)
        if window_index is not None:
            reverse_indices = torch.argsort(window_index)
            x = x[reverse_indices, :].contiguous()
        return x
