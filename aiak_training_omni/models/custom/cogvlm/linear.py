"""Vision Expert Linear Layers"""

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


def get_expert_mask(token_type_ids, **kwargs):
    """torch.LongTensor(S, B) -> [torch.BoolTensor(S, B), torch.BoolTensor(S, B)]"""
    LANGUAGE_TOKEN_TYPE = 0
    VISION_TOKEN_TYPE = 1
    vision_token_mask = torch.zeros_like(token_type_ids, dtype=torch.bool)
    vision_token_mask[:-1, :] = (token_type_ids[:-1, :] == VISION_TOKEN_TYPE) & (
        token_type_ids[1:, :] == VISION_TOKEN_TYPE
    )
    language_token_mask = ~vision_token_mask
    return vision_token_mask, language_token_mask


@dataclass
class VisionExpertLinearSubmodules:
    """Vision Expert Linear Submodules"""

    vision_linear: Union[ModuleSpec, type] = None
    language_linear: Union[ModuleSpec, type] = None
    apply_mask_fn: Union[ModuleSpec, type] = None


class VisionExpertLinear(MegatronModule, ABC):
    """Vision Expert Linear Layers"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: TransformerConfig,
        submodules: VisionExpertLinearSubmodules,
        bias: bool = True,
        add_vision_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(config=config)

        self.language_linear = build_module(
            submodules.language_linear,
            input_size,
            output_size,
            config=config,
            bias=bias,
            **kwargs,
        )
        self.vision_linear = build_module(
            submodules.vision_linear,
            input_size,
            output_size,
            config=config,
            bias=bias or add_vision_bias,
            **kwargs,
        )
        self.language_linear.weight.is_language_expert_parameter = True
        if self.language_linear.bias is not None:
            self.language_linear.bias.is_language_expert_parameter = True
        self.vision_linear.weight.is_vision_expert_parameter = True
        if self.vision_linear.bias is not None:
            self.vision_linear.bias.is_vision_expert_parameter = True

        self.apply_mask_fn = build_module(submodules.apply_mask_fn)
        assert self.vision_linear.weight.shape == self.language_linear.weight.shape
        self.output_size = self.vision_linear.weight.shape[0]

    def forward(self, hidden_states, **kwargs):
        """forward pass for the model"""
        vision_token_mask, language_token_mask = self.apply_mask_fn(**kwargs)
        shape = list(hidden_states.shape)
        shape[-1] = self.output_size
        mixed = torch.empty(
            shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        mixed[vision_token_mask], _ = self.vision_linear(
            hidden_states[vision_token_mask]
        )
        mixed[language_token_mask], _ = self.language_linear(
            hidden_states[language_token_mask]
        )

        return mixed, None
