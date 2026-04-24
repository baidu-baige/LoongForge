# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

""" Adapters """

import torch
import math
import transformer_engine as te
from megatron.core.transformer.spec_utils import build_module
from .internvl_config import InternMLPAdapterConfig
from loongforge.models.common import BaseMegatronModule
from loongforge.models.utils import import_module
from megatron.core.transformer.transformer_config import TransformerConfig


def _init_weights(m):
    if isinstance(m, (te.pytorch.Linear, torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(m.bias, -bound, bound)


class InternAdapter(BaseMegatronModule):
    """ Adapter  """

    config_class = InternMLPAdapterConfig

    def __init__(
        self, 
        config: TransformerConfig, 
        input_size: int, 
        **kwargs
        ) -> None:
        super().__init__(config=config)
        if self.config.model_spec is None:
            model_spec = [
                "loongforge.models.encoder.internvl_vision_models.internvl_layer_spec",
                "get_adapeter_layer_with_te_spec",
            ]
        else:
            model_spec = self.config.model_spec
        submodules = import_module(model_spec, self.config)

        self.layernorm = build_module(
            submodules.layernorm,
            config=config,
            hidden_size=input_size * int(1 / config.downsample_ratio) ** 2,
            eps=config.layernorm_epsilon,
        )
        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            input_size * int(1 / config.downsample_ratio) ** 2,
            config.ffn_hidden_size,
            bias=config.add_bias_linear,
        )

        self.activation_func = self.config.activation_func
        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            config.ffn_hidden_size,
            config.hidden_size,
            bias=config.add_bias_linear,
        )
        self.apply(_init_weights)
        if hasattr(config, 'freeze') and config.freeze:
            self.freeze()

    def forward(self, hidden_states, window_index=None, **kwargs):
        """ forward """
        hidden_states = self.layernorm(hidden_states)

        # [s, b, 4 * h/p]
        intermediate_parallel = self.linear_fc1(hidden_states)

        # activation function
        intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output = self.linear_fc2(intermediate_parallel)
        output = output.view(-1, output.size(-1))

        return output
