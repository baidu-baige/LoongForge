# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""default pretrain for generative models like GPTS"""


from functools import partial

from megatron.core.enums import ModelType

from loongforge.training.vlm import pretrain_vlm
from loongforge.engines.megatron.training_utils import pretrain


def train(args):
    """Run VLM SFT with the pretrain_vlm data provider and forward step."""
    pretrain_vlm.install_vlm_batch_layout(args)
    hetero_step = pretrain_vlm.build_hetero_step(args) if args.enable_full_hetero_dp else None
    dataset_provider = partial(pretrain_vlm.train_valid_test_dataset_provider, hetero_step=hetero_step)
    # Upstream reads this flag off the provider object; a partial does not inherit it.
    dataset_provider.is_distributed = True
    pretrain(
        args,
        dataset_provider,
        pretrain_vlm.model_provider,
        ModelType.encoder_or_decoder,
        partial(pretrain_vlm.forward_step, hetero_step=hetero_step),
        get_embedding_ranks=pretrain_vlm.get_embedding_ranks,
        hetero_step=hetero_step,
    )
