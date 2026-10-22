# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

""" Utils """

from typing import Dict

from loongforge.engines.megatron import get_args
from loongforge.models.base.context_parallel import get_inputs_on_this_cp_rank


def get_batch_on_this_cp_rank(batch: Dict):
    """slice batch along sequence dimension for context parallelism"""
    # TODO: Currently only split labels and loss_mask, may split other parameters in the future
    args = get_args()

    if args.context_parallel_size > 1:
        labels = batch.get('labels', None)
        loss_mask = batch.get('loss_mask', None)
        packed_seq_params = batch.get('packed_seq_params', None)

        assert labels is not None and loss_mask is not None, "labels and loss_mask must in batch"

        labels = get_inputs_on_this_cp_rank(labels.transpose(0, 1), packed_seq_params).transpose(0, 1)
        loss_mask = get_inputs_on_this_cp_rank(loss_mask.transpose(0, 1), packed_seq_params).transpose(0, 1)

        batch['labels'] = labels
        batch['loss_mask'] = loss_mask

    return batch
