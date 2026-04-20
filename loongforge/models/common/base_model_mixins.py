# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Omni Foundation Mixin Class: Abstract Base Class for Encoders/Foundation Models/Decoders
Alternative translations"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from transformers import (
    PreTrainedModel,
    AutoConfig,
)
import logging
logger = logging.getLogger(__name__)
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.module import MegatronModule
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.process_groups_config import ProcessGroupCollection


class BaseMegatronLanguageModule(LanguageModule):
    """Unified Base Class for Language Models"""

    def __init__(self, config: AutoConfig, pg_collection: Optional[ProcessGroupCollection] = None, **kwargs):
        super().__init__(config=config, pg_collection=pg_collection, **kwargs)

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


class BaseMegatronVisionModule(MegatronModule):
    # TODO: Since VisionModule does not have any special logic, we inherit from MegatronModule instead.
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

    def get_dummy_input(self, device):
        """Get dummy inputs for vision models"""
        # Build patch_dim from config instead of hardcoding 3*2*14*14.
        # This keeps dummy forward compatible with different vision backbones
        # (e.g., non-default patch_size / temporal_patch_size / in_channels).
        patch_size = getattr(self.config, "patch_size", 14)
        temporal_patch_size = getattr(self.config, "temporal_patch_size", 2)
        in_channels = getattr(self.config, "in_channels", 3)
        patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
        # Keep a minimal valid [t, h, w] = [1, 2, 2] so token count is 4.
        return (
            torch.randn((4, patch_dim), dtype=torch.bfloat16, device=device),
            torch.tensor([[1, 2, 2]], dtype=torch.int32, device=device),
        )


class BaseDecoderModelMixin(PreTrainedModel, ABC):
    """Unified decoder model mixin class."""

    @abstractmethod
    def forward_loss(
        self,
        decoder_inputs: Dict[str, torch.Tensor],
        foundation_outputs: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Calculate decoder loss function, for training stage."""
        raise NotImplementedError
