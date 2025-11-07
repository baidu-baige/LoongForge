"""Adapters for the transformer model."""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.dist_checkpointing.mapping import ShardedStateDict


@dataclass
class AdapterSubmodules:
    """Adapter sub-modules."""

    linear_proj: Union[ModuleSpec, type] = None
    layernorm: Union[ModuleSpec, type] = None
    mlp: Union[ModuleSpec, type] = None


class Adapter(MegatronModule):
    """Adapter module."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: AdapterSubmodules,
        input_size: int = None,
    ):
        super().__init__(config=config)

        self.input_size = input_size if input_size is not None else config.hidden_size

        self.conv = torch.nn.Conv2d(
            in_channels=input_size,
            out_channels=input_size,
            kernel_size=2,
            stride=2,
        )

        self.linear_proj = build_module(
            submodules.linear_proj,
            self.input_size,
            config.hidden_size,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            skip_bias_add=False,
            parallel_mode=None,
            skip_weight_param_allocation=False,
            tp_comm_buffer_name="proj",
        )

        self.layernorm = build_module(
            submodules.layernorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        )

        self.act = torch.nn.GELU()
        self.mlp = build_module(submodules.mlp, config=config)
        self.boi = torch.nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.eoi = torch.nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, x):
        """Forward method."""
        s, b, h = x.shape
        grid_size = int(s**0.5)
        x = x.view(grid_size, grid_size, b, h).permute(2, 3, 0, 1)
        x = self.conv(x)
        x = x.flatten(2).permute(2, 0, 1).contiguous()

        x, _ = self.linear_proj(x)
        x = self.act(self.layernorm(x))
        x, _ = self.mlp(x)

        boi = self.boi.expand(-1, b, -1)
        eoi = self.eoi.expand(-1, b, -1)
        x = torch.cat((boi, x, eoi))

        return x

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """sharded state dict"""
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(
                f"{prefix}{name}.", sharded_offsets, metadata
            )
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict
