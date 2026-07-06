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
from megatron.core.num_microbatches_calculator import get_num_microbatches

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

    # Extract chunk_group_size into args (for scheduler) and config (for model-side code).
    args = get_args()
    if args.enable_chunkpipe and "chunk_group_size" in batch:
        group_size = batch["chunk_group_size"][0].item()
        args.chunkpipe_current_chunk_group_size = group_size
        config = get_model_config()
        if config is not None and getattr(config, 'sft_chunkpipe_mode', False):
            # Always write chunkpipe_current_group_size from the batch data, regardless of
            # chunk_idx_in_group. This breaks the circular dependency where:
            #   1. group_size_cache missing group N → fallback gives wrong chunk_idx
            #   2. wrong chunk_idx ≠ 0 → get_batch() would not write group_size
            #   3. scheduler reads stale _discovered_gs → group_size_cache[N] = wrong value
            # By always writing, the scheduler post-step always gets the correct group_size
            # to update group_size_cache, even when chunk_idx was computed via fallback.
            config.chunkpipe_current_group_size = group_size

            # Loong-Megatron scheduler: the scheduler must set
            # config.chunkpipe_chunk_idx_in_group = 0 **before** invoking forward_step, so
            # that get_batch (called inside forward_step) observes the correct index here.
            #
            # Timing chain:
            #   scheduler sets chunk_idx = 0
            #     → forward_step()
            #       → get_batch() reads chunk_group_size from batch
            #         → writes config.chunkpipe_current_group_size   ← here
            #           → model code reads config.chunkpipe_current_group_size
            #
            # The ordering is currently correct but depends on this implicit call sequence.
            # If the scheduler logic is refactored, ensure this contract is preserved.
            if config.chunkpipe_chunk_idx_in_group == 0:
                # Composite scheduling info: applied at every real-group start
                # within the composite. The composite descriptor is the same
                # for every chunk inside one composite (sampler repeats it
                # per micro-batch), so re-writing on each real-group's chunk 0
                # is idempotent and preserves the value across the composite
                # for downstream readers (schedule.py + MLA chunk_keys).
                if "composite_component_sizes" in batch:
                    components = batch["composite_component_sizes"]
                    if components:
                        config.chunkpipe_component_sizes = list(components)
                        config.chunkpipe_composite_group_size = sum(components)
                    else:
                        # No composite info → fall back to trivial (single
                        # real group). Keeps behavior sane if the queues are
                        # somehow not populated (e.g. ad-hoc test paths).
                        config.chunkpipe_component_sizes = [group_size]
                        config.chunkpipe_composite_group_size = group_size

        # N_g and G for the per-sample loss path. N_g is per-chunk (same value
        # for all chunks in one source sequence), G_total is per-step (same
        # for all chunks in one step) and is the cross-rank sum of source
        # group counts. Both are read by loss_func from args. Because
        # micro_batch_size >= 1 chunk all share the same step, picking index 0
        # is correct.
        if "group_total_tokens" in batch:
            args.chunkpipe_group_total_tokens = batch["group_total_tokens"][0].item()
        if "step_num_groups" in batch:
            step_num_groups = batch["step_num_groups"][0].item()
            args.chunkpipe_step_num_groups = step_num_groups
            if config is not None and getattr(config, 'sft_chunkpipe_mode', False):
                config.chunkpipe_step_num_groups = step_num_groups

    mtp_batch = None
    if args.enable_chunkpipe and getattr(args, "sft_chunkpipe_mode", False):
        if "mtp_tokens" in batch:
            expected_length = args.chunksize + (args.mtp_num_layers or 0)
            assert batch["tokens"].size(1) == args.chunksize, (
                f"SFT chunkpipe main tokens must have base length "
                f"{args.chunksize}, got {batch['tokens'].size(1)}."
            )
            assert batch["mtp_tokens"].size(1) == expected_length, (
                f"SFT chunkpipe MTP tokens must have physical length "
                f"{expected_length}, got {batch['mtp_tokens'].size(1)}."
            )
            mtp_batch = {
                "tokens": batch["mtp_tokens"],
                "position_ids": batch["mtp_position_ids"],
                "labels": batch.get("mtp_labels"),
                "loss_mask": batch.get("mtp_loss_mask"),
                "group_total_tokens": batch.get("group_total_tokens"),
                "step_num_groups": batch.get("step_num_groups"),
            }

    output = (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch["position_ids"],
        batch["attention_mask"],
        batch["packed_seq_params"],
        mtp_batch,
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

    # SFT chunkpipe per-sample loss path (default when sft_chunkpipe_mode=True
    # and --calculate-per-token-loss is NOT set). Goal: final gradient equals
    # (1/G_total) * sum_g (1/N_g) * sum_k d S_{g,k}/d theta, i.e. per-source-
    # sequence equal weighting independent of chunk count per sequence and
    # robust to per-rank G_local fluctuation.
    #
    # Derivation (legacy=True 2-tuple path, no CP for brevity; cp=1):
    #   loss_func returns scaled = D * S_{g,k} * M_raw / (N_g * G_total * cp)
    #   schedules legacy path multiplies by cp / M_raw
    #   → backward sees D * S_{g,k} / (N_g * G_total)
    #   DDP grad all-reduce averages by 1/D
    #   → grad sees S_{g,k} / (N_g * G_total) accumulated over all chunks on
    #     all ranks: (1/G_total) sum_g (1/N_g) sum_k S_{g,k}  ✓
    #
    # The D pre-compensation symmetrically applies to the reporting tensor so
    # that the post-aggregation report (sum-across-mbs → all-reduce-sum →
    # divide-by-DPxCP-world-size in training_utils) recovers the same
    # (1/G_total) sum_g (1/N_g) sum_k S_{g,k} value. Using G_total instead of
    # G_local also fixes a latent precision bug in the equal-DP case where
    # LPT did not strictly guarantee G_local = G_total / D across ranks.
    config = get_model_config()
    sft_cp_mode = config is not None and getattr(config, "sft_chunkpipe_mode", False)
    if sft_cp_mode and not args.calculate_per_token_loss:
        N_g = args.chunkpipe_group_total_tokens
        G_total = args.chunkpipe_step_num_groups
        M_raw = get_num_microbatches()
        cp = mpu.get_context_parallel_world_size()
        D = mpu.get_data_parallel_world_size()

        denom = N_g * G_total * cp
        scaled = loss * (D * M_raw / denom)
        # Pre-compensate D*cp for the all_reduce + divide by (D*cp) in training_utils.py.
        # Without cp factor, the logged loss would be 1/cp times smaller than correct value.
        reporting = (D * cp * loss / (N_g * G_total)).detach()
        loss_reduced_dict = {'lm loss': reporting}
        return scaled, loss_reduced_dict

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
            mtp_batch,
        ) = get_batch(data_iterator)

    timers("batch-generator").stop()

    # SFT chunkpipe + MTP: MTP bridge tokens are appended during preprocessing.
    # The main model consumes the base chunksize slice; the MTP branch consumes
    # the full chunksize + mtp_num_layers tensors via extra_block_kwargs.
    extra_block_kwargs = None
    if mtp_batch is not None and mpu.is_pipeline_last_stage():
        extra_block_kwargs = {"mtp_batch": mtp_batch}

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
                extra_block_kwargs=extra_block_kwargs,
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
        enable_chunkpipe=args.enable_chunkpipe,
        chunksize=args.chunksize,
        mtp_num_layers=args.mtp_num_layers or 0,
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
