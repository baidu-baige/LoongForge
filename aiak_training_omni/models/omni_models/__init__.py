"""AIAK-Omni 多模态训练框架。"""

from ..common.base_model_mixins import (
    BaseMegatronVisionModuler,
    BaseMegatronLanuageModuler,
    BaseMegatronModuler
)

from .configuration import (
    OmniModelConfig,
    OmniEncoderConfig,
    OmniDecoderConfig,
)

from .omni_encoder_model import OmniEncoderModel
from .omni_foundation_model import (
    OmniFoundationModel
)
from .omni_decoder_model import (
    OmniDecoderModel
)
from .omni_combination_model import OmniCombinationModel

from .omni_model_provider import (
    omni_model_provider
)

__all__ = [
    # 基础Mixin
    "BaseMegatronVisionModuler",
    "BaseMegatronLanuageModuler",
    "BaseMegatronModuler"

    
    # 配置类
    "OmniModelConfig",
    "OmniEncoderConfig",
    "OmniDecoderConfig",
    
    # 模型类
    "OmniEncoderModel",
    "OmniFoundationModel",
    "OmniDecoderModel",
    "OmniCombinationModel",
    
    # 具体实现
    
    
    "omni_model_provider",
]
