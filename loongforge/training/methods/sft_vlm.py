# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""default pretrain for generative models like GPTS"""


from megatron.core.enums import ModelType

from loongforge.training.methods import pretrain_vlm
from loongforge.training.megatron_runner import pretrain


def train(args):
    """Run VLM SFT with the pretrain_vlm data provider and forward step."""
    pretrain_vlm.train_valid_test_dataset_provider.is_distributed = True
    pretrain(
        args,
        pretrain_vlm.train_valid_test_dataset_provider,
        pretrain_vlm.model_provider,
        ModelType.encoder_or_decoder,
        pretrain_vlm.forward_step,
        get_embedding_ranks=pretrain_vlm.get_embedding_ranks,
    )
