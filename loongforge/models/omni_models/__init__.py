# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""initialize the model"""

from importlib import import_module

from ..common.base_model_mixins import (
    BaseMegatronVisionModule,
    BaseMegatronLanguageModule,
    BaseMegatronModule,
)

_LAZY_EXPORTS = {
    "OmniEncoderModel": (".omni_encoder_model", "OmniEncoderModel"),
    "OmniDecoderModel": (".omni_decoder_model", "OmniDecoderModel"),
    "OmniCombinationModel": (".omni_combination_model", "OmniCombinationModel"),
    "omni_model_provider": (".omni_model_provider", "omni_model_provider"),
}


def __getattr__(name):
    """Resolve the multimodal models on first use.

    Importing them here instead would run while ``loongforge.models`` is still
    initializing, and ``omni_encoder_model`` imports back into that package.
    """
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value

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
