# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""default pretrain for generative models like GPTS"""

import os

from megatron.core.enums import ModelType
from megatron.core.utils import StragglerDetector

from typing import List
from megatron.core.transformer.multi_token_prediction import get_mtp_ranks

from loongforge.utils import (
    constants
)
from loongforge.models.omni_models.omni_model_provider import (
    omni_model_provider,
)


from loongforge.train.megatron_trainer import MegatronTrainer
from loongforge.train.trainer_builder import register_model_trainer

from loongforge.utils import constants, get_args, get_model_config

stimer = StragglerDetector()

def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        mtp_ranks = get_mtp_ranks(pp_ranks, config=get_model_config().foundation)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks

@register_model_trainer(model_family=constants.VisionLanguageModelFamilies.names(),
                        training_phase=constants.TrainingPhase.SFT)
def default_pretrain_trainer(train_args):
    """build trainer"""
    from loongforge.train.pretrain import pretrain_vlm
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=pretrain_vlm.train_valid_test_dataset_provider,
        model_provider=omni_model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=pretrain_vlm.forward_step,
        get_embedding_ranks=get_embedding_ranks,
    )

    return trainer
