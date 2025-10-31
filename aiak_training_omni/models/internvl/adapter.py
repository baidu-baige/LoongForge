""" Adapters """

import torch
import math
from dataclasses import dataclass
import transformer_engine as te
from typing import Union
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.mlp import ActivationFuncModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from .internvl_config import AdapterConfig


def _init_weights(m):
    if isinstance(m, (te.pytorch.Linear, torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(m.bias, -bound, bound)


@dataclass
class AdapterSubmodules:
    """Adapter sub-modules."""
    layernorm: Union[ModuleSpec, type] = None
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None
    bias_activation_func_impl: Union[ModuleSpec, type] = None


class Adapter(MegatronModule):
    """ Adaptor  """

    def __init__(self, config: AdapterConfig, submodules: AdapterSubmodules, input_size: int) -> None:
        super().__init__(config=config)

        self.layernorm = build_module(
            submodules.layernorm,
            config=config,
            hidden_size=input_size,
            eps=config.layernorm_epsilon,
        )
        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            input_size,
            config.ffn_hidden_size,
            bias=config.add_bias_linear,
        )

        self.activation_func = ActivationFuncModule(self.config, submodules)
        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            config.ffn_hidden_size,
            config.hidden_size,
            bias=config.add_bias_linear,
        )
        self.apply(_init_weights)

    def forward(self, hidden_states):
        """ forward """
        hidden_states = self.layernorm(hidden_states)

        # [s, b, 4 * h/p]
        intermediate_parallel = self.linear_fc1(hidden_states)

        # activation function
        intermediate_parallel = self.activation_func(intermediate_parallel, torch.tensor(0))

        # [s, b, h]
        output = self.linear_fc2(intermediate_parallel)

        return output
