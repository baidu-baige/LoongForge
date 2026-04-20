# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""default sft trainer for generative models like GPTS"""

import os
import torch

from functools import partial

from typing import List
from megatron.core.transformer.multi_token_prediction import get_mtp_ranks

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.utils import StragglerDetector
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.rerun_state_machine import get_rerun_state_machine

from megatron.training import get_timers
from megatron.training.utils import average_losses_across_data_parallel_group

from loongforge.utils import (
    constants,
    get_args,
    get_tokenizer,
    get_chat_template,
    print_rank_0,
)

from loongforge.models import get_model_provider, get_model_family
from loongforge.data import (
    SFTDataset,
    SFTDatasetConfig,
    BlendedHuggingFaceDatasetBuilder,
    DataCollatorForSupervisedDataset,
)

from loongforge.utils import constants, get_args, get_model_config

from loongforge.train.megatron_trainer import MegatronTrainer
from loongforge.train.trainer_builder import register_model_trainer

from .utils import (
    get_batch_on_this_tp_rank,
    get_batch_on_this_cp_rank,
    get_dataset_blend_from_list,
    build_sft_cyclic_iterators,
    build_sft_data_collator,
)
from loongforge.models.foundation.llm_model_provider import llm_model_provider


stimer = StragglerDetector()


def get_batch(data_iterator):
    """Generate a batch"""
    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    output = (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch["position_ids"],
        batch["attention_mask"],
        batch["packed_seq_params"],
    )

    return output


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        num_input_tokens (int): The number of tokens in the batch

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()

    loss = torch.sum(losses * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )

    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    loss_reduced_dict = {'lm loss': reporting_loss if not args.legacy_reporting_loss_reduction
                         else reporting_loss[0] / num_tokens}

    # calculate the number of tokens for this micro-batch
    if args.variable_seq_lengths:
        # for variable seq length, we need to calculate the number of tokens on fly
        # model output tensor shape is [B, S, H]
        num_input_tokens = output_tensor.shape[0] * output_tensor.shape[1]
        input_tokens = torch.tensor(num_input_tokens, dtype=torch.int, device=output_tensor.device)
        # sum across all dp ranks
        torch.distributed.all_reduce(input_tokens, group=mpu.get_data_parallel_group())
        loss_reduced_dict["total_inputs"] = input_tokens * args.context_parallel_size

    return loss, num_tokens, loss_reduced_dict


def forward_step(data_iterator, model, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: Megatron Model

    Returns:
        output_tensor: Output tensor
        loss_func: Loss function
        num_tokens: Number of tokens
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()

    global stimer
    with stimer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            position_ids,
            attention_mask,
            packed_seq_params,
        ) = get_batch(data_iterator)

    timers("batch-generator").stop()

    with stimer:
        if return_schedule_plan:
            assert args.overlap_moe_expert_parallel_comm, \
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            schedule_plan = model.build_schedule_plan(
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attention_mask, 
                labels=labels,
                packed_seq_params=packed_seq_params,
                loss_mask=loss_mask,
            )
            return schedule_plan, partial(loss_func, loss_mask)
        else:
            output_tensor = model(
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                packed_seq_params=packed_seq_params,
                loss_mask=loss_mask,
            )

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """
    Build the train test and validation datasets.

    Args:
        train_val_test_num_samples: List[int]

    Returns:
        train_iter: Iterator
        valid_iter: Iterator
        test_iter: Iterator
    """
    args = get_args()

    config = SFTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,  # max sequence length
        enable_discard_sample=args.enable_discard_sample,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path),
        ],
        split=args.split,
        path_to_cache=args.data_cache_path,
        tokenizer=get_tokenizer(),
        dataset=get_dataset_blend_from_list(args.sft_dataset),
        dataset_per_split=[
            get_dataset_blend_from_list(args.sft_train_dataset),
            get_dataset_blend_from_list(args.sft_valid_dataset),
            get_dataset_blend_from_list(args.sft_test_dataset),
        ],
        dataset_config_file=args.sft_dataset_config,
        streaming=args.sft_data_streaming,
        streaming_buffer_size=args.streaming_buffer_size,
        mix_strategy=args.sft_data_mix_strategy,
        chat_template=get_chat_template(),
        num_preprocess_workers=args.sft_num_preprocess_workers,
        train_on_prompt=args.train_on_prompt,
        ignore_index=constants.IGNORE_INDEX,
        eod_mask_loss=args.eod_mask_loss,
        history_mask_loss=args.history_mask_loss,
        is_tokenized=args.is_tokenized_data,
        packing=args.packing_sft_data,
        sort_batch=args.sft_sort_batch,
        packing_buffer_size=args.packing_buffer_size,
        context_parallel_size=args.context_parallel_size,
    )

    print_rank_0(
        f"> building sft train, validation, and test datasets for {args.model_name} ..."
    )

    train_ds, valid_ds, test_ds = BlendedHuggingFaceDatasetBuilder(
        cls=SFTDataset,
        sizes=train_val_test_num_samples,  # NOTE: not use now!
        is_built_on_rank=lambda: mpu.get_tensor_model_parallel_rank() == 0,
        config=config,
    ).build()

    # will use external dataloader type for sft
    data_collator = build_sft_data_collator(DataCollatorForSupervisedDataset)
    train_iter, valid_iter, test_iter = build_sft_cyclic_iterators(
        train_ds, valid_ds, test_ds, data_collator
    )
    print_rank_0(f"> finished creating {args.model_name} sft datasets ...")

    return train_iter, valid_iter, test_iter

def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        mtp_ranks = get_mtp_ranks(pp_ranks, config=get_model_config())
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks

@register_model_trainer(
    model_family=constants.LanguageModelFamilies.names(),
    training_phase=constants.TrainingPhase.SFT,
)
def default_sft_trainer(train_args):
    """build trainer"""
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=train_valid_test_datasets_provider,
        model_provider=llm_model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=forward_step,
        get_embedding_ranks=get_embedding_ranks,
    )

    return trainer
