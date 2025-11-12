"""Omni Foundation Mixin Class: Abstract Base Class for Encoders/Foundation Models/Decoders
Alternative translations"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, List

import torch
from transformers import (
    BatchFeature,
    PreTrainedModel,
    PretrainedConfig,
    ProcessorMixin,
    AutoConfig,
)
import logging

logger = logging.getLogger(__name__)
import dataclasses
from dataclasses import fields, asdict
from megatron.core.transformer import TransformerConfig
from megatron.training.activations import squared_relu
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.module import MegatronModule
import torch.nn.functional as F
from copy import deepcopy


class BaseMegatronLanuageModule(LanguageModule):
    """Unified Base Class for Language Models"""

    def __init__(self, config: AutoConfig, **kwargs):
        super().__init__(config=config, **kwargs)
    
    def freeze(self):
        """Freeze model parameters during training to prevent them from being updated.
        """
        for param in self.parameters():
            param.requires_grad = False

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            dtype (`torch.dtype`, *optional*):
                Override the default `dtype` and load the model under this dtype.
        """
        # when we init a model from within another model (e.g. VLMs) and dispatch on FA2
        # a warning is raised that dtype should be fp16. Since we never pass dtype from within
        # modeling code, we can try to infer it here same way as done in `from_pretrained`
        # For BC on the old `torch_dtype`
        model = cls(config, **kwargs)
        return model


class BaseMegatronModule(MegatronModule):
    """A Unified Encoder Model Interface Supporting All Modalities."""
    def __init__(self, config: AutoConfig, **kwargs):
        super().__init__(config=config, **kwargs)

    def freeze(self):
        """Freeze model parameters during training to prevent them from being updated.
        """
        for param in self.parameters():
            param.requires_grad = False

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            dtype (`torch.dtype`, *optional*):
                Override the default `dtype` and load the model under this dtype.
        """
        # when we init a model from within another model (e.g. VLMs) and dispatch on FA2
        # a warning is raised that dtype should be fp16. Since we never pass dtype from within
        # modeling code, we can try to infer it here same way as done in `from_pretrained`
        # For BC on the old `torch_dtype`
        model = cls(config, **kwargs)
        return model


class BaseMegatronVisionModule(VisionModule):
    """Unified Abstract Base Class for Vision Models"""

    def __init__(self, config: AutoConfig, **kwargs):
        super().__init__(config=config, **kwargs)
    
    def freeze(self):
        """Freeze model parameters during training to prevent them from being updated.
        """
        for param in self.parameters():
            param.requires_grad = False

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            dtype (`torch.dtype`, *optional*):
                Override the default `dtype` and load the model under this dtype.
        """
        # when we init a model from within another model (e.g. VLMs) and dispatch on FA2
        # a warning is raised that dtype should be fp16. Since we never pass dtype from within
        # modeling code, we can try to infer it here same way as done in `from_pretrained`
        # For BC on the old `torch_dtype`
        model = cls(config, **kwargs)
        return model


# TODO: 需要抽象？


def base_position_id_func(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
) -> Dict[str, torch.Tensor]:
    """Base function to generate position ids."""
    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)
    if len(attention_mask.shape) == 1:
        attention_mask = attention_mask.unsqueeze(0)

    return dict(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


class PositionIDFuncCompose:
    """Composes multiple functions together"""

    def __init__(self, customized_funcs: List[Callable]):
        self.transforms = [base_position_id_func] + customized_funcs

    def __call__(self, **x):
        for t in self.transforms:
            x = t(**x)
        return x


class BaseFoundationModelMixin(PreTrainedModel, ABC):
    """统一的foundation model(LLM)模型混入类。"""

    def get_generation_position_id(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """获取生成位置id"""
        return None

    def get_position_id_func(self) -> List[Callable]:
        """获取位置id函数"""
        return None

    @property
    def position_id_func(self) -> Optional[Callable]:
        """获取位置id函数组合"""
        if self.get_position_id_func() is None:
            return None
        return PositionIDFuncCompose(self.get_position_id_func())


class BaseDecoderModelMixin(PreTrainedModel, ABC):
    """统一decoder模型混入类。"""

    @abstractmethod
    def forward_loss(
        self,
        decoder_inputs: Dict[str, torch.Tensor],
        foundation_outputs: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """计算decoder损失函数 ，用于训练阶段。"""
        raise NotImplementedError
