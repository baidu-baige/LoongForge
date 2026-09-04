# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge model wrapper for the Kimi K3 language backbone."""

from typing import Optional

import torch

from megatron.core.process_groups_config import ProcessGroupCollection

from loongforge.models.foundation.base import BaseGPTModel
from loongforge.models.utils import import_module

from .kimi_k3_config import KimiK3Config


def _ignore_te_extra_state(module, incompatible_keys):
    """Allow BF16 checkpoints produced without TE extra-state tensors.

    ``_extra_state`` only ever carries Transformer Engine quantization metadata,
    which the offline converter does not emit. Filtering on the suffix rather
    than a per-module allow-list means a newly added TE module cannot silently
    turn into a load failure, and no real weight can be hidden.
    """
    incompatible_keys.missing_keys[:] = [
        key for key in incompatible_keys.missing_keys if not key.endswith("._extra_state")
    ]


class KimiK3Model(BaseGPTModel):
    """K3 model wrapper using the standard LoongForge GPT training entrypoint."""

    config_class = KimiK3Config

    def __init__(
        self,
        config: KimiK3Config,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        scatter_embedding_sequence_parallel: bool = True,
        language_embedding: Optional[torch.nn.Module] = None,
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        **kwargs,
    ) -> None:
        model_spec = config.model_spec or [
            "loongforge.models.foundation.kimi_k3.kimi_k3_layer_spec",
            "build_kimi_k3_spec",
        ]
        transformer_layer_spec = import_module(model_spec, config, vp_stage=vp_stage)
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=config.padded_vocab_size,
            max_sequence_length=config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=config.fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=not config.untie_embeddings_and_output_weights,
            position_embedding_type="none",
            language_embedding=language_embedding,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
        self.register_load_state_dict_post_hook(_ignore_te_extra_state)
