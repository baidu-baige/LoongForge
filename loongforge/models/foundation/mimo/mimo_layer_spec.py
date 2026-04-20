# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""MiMo layer spec."""

from typing import Optional, Tuple

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from loongforge.models.foundation.mimo.mimo_multi_token_prediction import (
    MimoMultiTokenPredictionLayer,
)
from loongforge.models.foundation.qwen2.qwen_layer_spec import (
    get_qwen2_layer_with_te_spec,
)


def get_mimo_decoder_block_and_mtp_spec(
    config: TransformerConfig, vp_stage: int = None
) -> Tuple[ModuleSpec, Optional[ModuleSpec]]:
    """Get MiMo decoder and MTP specs.

    MiMo uses a Qwen2-like decoder stack and optional MTP blocks.
    """
    decoder_layer_spec = get_qwen2_layer_with_te_spec(config)
    use_te = config.transformer_impl == "transformer_engine"

    mtp_block_spec = None
    if config.mtp_num_layers is not None:
        mtp_block_spec = get_gpt_mtp_block_spec(
            config,
            decoder_layer_spec,
            use_transformer_engine=use_te,
            vp_stage=vp_stage,
        )
        if mtp_block_spec is None:
            return decoder_layer_spec, mtp_block_spec

        rewritten_layer_specs = []
        for index, layer_spec in enumerate(mtp_block_spec.layer_specs):
            if not isinstance(layer_spec, ModuleSpec):
                raise TypeError(
                    "Expected each MTP layer spec to be ModuleSpec, "
                    + f"but got {type(layer_spec).__name__} at index {index}."
                )
            rewritten_layer_specs.append(
                ModuleSpec(
                    module=MimoMultiTokenPredictionLayer,
                    params=layer_spec.params,
                    submodules=layer_spec.submodules,
                )
            )
        mtp_block_spec.layer_specs = rewritten_layer_specs

    return decoder_layer_spec, mtp_block_spec
