"""AIAK-Omni 多模态训练框架。"""

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
    # 基础Mixin
    "BaseMegatronVisionModuler",
    "BaseMegatronLanuageModuler",
    "BaseMegatronModuler"
    # 模型类
    "OmniEncoderModel",
    "OmniDecoderModel",
    "OmniCombinationModel",
    # 具体实现
    "omni_model_provider",
]
