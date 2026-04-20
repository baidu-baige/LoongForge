# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""initialize the model"""

from ..common.base_model_mixins import (
    BaseMegatronVisionModule,
    BaseMegatronLanguageModule,
    BaseMegatronModule,
)
from .omni_encoder_model import OmniEncoderModel
from .omni_decoder_model import OmniDecoderModel
from .omni_combination_model import OmniCombinationModel
from .omni_model_provider import omni_model_provider

__all__ = [
    # Basic Mixins
    "BaseMegatronVisionModule",
    "BaseMegatronLanguageModule",
    "BaseMegatronModule",
    # Model classes
    "OmniEncoderModel",
    "OmniDecoderModel",
    "OmniCombinationModel",
    # Implementations
    "omni_model_provider",
]
