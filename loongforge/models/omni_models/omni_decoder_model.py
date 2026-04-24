# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
OmniDecoderModel: Base decoder model class.
Provides unified interface for image/video/audio generation tasks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..common.base_model_mixins import BaseDecoderModelMixin


class OmniDecoderModel(BaseDecoderModelMixin):
    """Omni multimodal decoder model."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.modality: List[str] = []


    def forward_loss(
        self,
        decoder_inputs: Dict[str, torch.Tensor],
        foundation_outputs: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Calculate decoder losses."""
        pass

    def freeze(self, modality: Optional[str] = None) -> None:
        """Freeze decoder model parameters.

        Args:
            modality (str, optional): Specific modality to freeze ('image', 'video', 'audio'). 
                If None, freeze all modality decoders.
        """
        pass
