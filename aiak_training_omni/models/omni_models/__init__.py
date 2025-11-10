"""initialize the model"""

from ..common.base_model_mixins import (
    BaseMegatronVisionModuler,
    BaseMegatronLanuageModuler,
    BaseMegatronModuler,
)
from .omni_encoder_model import OmniEncoderModel
from .omni_decoder_model import OmniDecoderModel
from .omni_combination_model import OmniCombinationModel
from .omni_model_provider import omni_model_provider

__all__ = [
    # Basic Mixins
    "BaseMegatronVisionModuler",
    "BaseMegatronLanuageModuler",
    "BaseMegatronModuler",
    # Model classes
    "OmniEncoderModel",
    "OmniDecoderModel",
    "OmniCombinationModel",
    # Implementations
    "omni_model_provider",
]
