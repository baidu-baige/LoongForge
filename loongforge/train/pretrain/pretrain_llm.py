# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""default pretrain for generative models like GPTS"""

from functools import partial
from typing import List, Optional

import torch
from loongforge.models.foundation.llm_model_provider import llm_model_provider
from loongforge.train.megatron_trainer import MegatronTrainer, pretrain_with_mcore
from loongforge.train.trainer_builder import register_model_trainer
from loongforge.utils import constants, get_args, get_tokenizer, print_rank_0
from loongforge.utils.global_vars import get_model_config

from megatron.core import mpu, parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
    MockGPTDataset,
)
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.enums import ModelType
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.multi_token_prediction import get_mtp_ranks, mtp_on_this_rank
from megatron.core.utils import (
    StragglerDetector,
    get_attr_wrapped_model,
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training import get_timers
from megatron.training.utils import (
    is_first_or_last_pipeline_stage,
)

stimer = StragglerDetector()

_BATCH_KEYS = (
    "tokens",
    "labels",
    "loss_mask",
    "position_ids",
    "attention_mask",
    "cu_seqlens",
    "cu_seqlens_padded",
    "max_seqlen",
    "local_cp_size",
    "hybrid_cp_group",
)


def _mtp_on_this_rank(config, vp_stage=None):
    """Return whether MTP is built on this rank using the 0.20 explicit config API."""
    return mtp_on_this_rank(
        layout=config.pipeline_model_parallel_layout,
        mtp_num_layers=config.mtp_num_layers,
        ignore_virtual=False,
        vp_stage=vp_stage,
    )


def get_batch(data_iterator, vp_stage: Optional[int] = None):
    """Generate a batch."""
    args = get_args()
    config = get_model_config()
    has_cu_seqlens = getattr(args, "dataloader_inter_document_masking", False)
    is_hybrid_cp = getattr(args, "hybrid_context_parallel", False)
    mtp_on_rank = _mtp_on_this_rank(config, vp_stage)

    if (
        not is_first_or_last_pipeline_stage(vp_stage)
        and not mtp_on_rank
        and not has_cu_seqlens
    ):
        return None, None, None, None, None

    batch = {}
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    if tp_rank == 0:
        batch = next(data_iterator)
        for key in _BATCH_KEYS:
            value = batch.get(key)
            if value is not None:
                batch[key] = value.cuda(non_blocking=True)

    batch = get_batch_on_this_tp_rank(
        batch,
        has_cu_seqlens=has_cu_seqlens,
        is_hybrid_cp=is_hybrid_cp,
        create_attention_mask_in_dataloader=args.create_attention_mask_in_dataloader,
        broadcast_src_rank=parallel_state.get_tensor_model_parallel_src_rank(),
        broadcast_group=parallel_state.get_tensor_model_parallel_group(),
        cp_size=args.context_parallel_size,
        tp_rank=tp_rank,
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
        mtp_on_this_rank=mtp_on_rank,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        is_pipeline_first_stage=parallel_state.is_pipeline_first_stage(),
        is_pipeline_last_stage=parallel_state.is_pipeline_last_stage(),
    )

    if args.context_parallel_size > 1:
        batch = get_batch_on_this_cp_rank(
            batch,
            is_hybrid_cp=is_hybrid_cp,
            cp_group=parallel_state.get_context_parallel_group(),
            use_per_sequence_balancing=(
                getattr(args, "dataloader_inter_document_masking", False)
                and not args.training_phase == "sft"
            ),
        )

    return (
        batch.get("tokens"),
        batch.get("labels"),
        batch.get("loss_mask"),
        batch.get("attention_mask"),
        batch.get("position_ids"),
    )


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
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
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
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
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


def forward_step(data_iterator, model, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: Megatron Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()

    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator,
            vp_stage,
        )

    timers("batch-generator").stop()

    if args.enable_chunkpipe and args.mtp_num_layers and args.mtp_num_layers > 0 and mpu.is_pipeline_last_stage():
        raise NotImplementedError(
            "ChunkPipe MTP next-batch support is introduced in LF-C."
        )

    with stimer:
        if return_schedule_plan:
            assert args.overlap_moe_expert_parallel_comm, \
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            schedule_plan = model.build_schedule_plan(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )
            return schedule_plan, partial(loss_func, loss_mask)
        else:
            output_tensor = model(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                extra_block_kwargs=None,
            )
            if hasattr(output_tensor, "loss") and output_tensor.loss is not None:
                output_tensor = output_tensor.loss

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    For GPT-like models, if there are no special requirements, we should directly reuse the Megatron GPTDataset.
    """
    args = get_args()
    tokenizer = get_tokenizer()

    def _is_dataset_built_on_rank(vp_stage=None):
        return (
            is_first_or_last_pipeline_stage(vp_stage)
            or _mtp_on_this_rank(get_model_config(), vp_stage)
        ) and parallel_state.get_tensor_model_parallel_rank() == 0


    config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path),
        ],
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )

    print_rank_0(
        f"> building train, validation, and test datasets for {args.model_name} ..."
    )
    
    is_dataset_built = partial(_is_dataset_built_on_rank, vp_stage=vp_stage)

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset if not args.mock_data else MockGPTDataset,
        train_val_test_num_samples,
        partial(_is_dataset_built_on_rank, vp_stage=vp_stage),
        config,
    ).build()

    print_rank_0(f"> finished creating {args.model_name} datasets ...")

    return train_ds, valid_ds, test_ds

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
    training_phase=constants.TrainingPhase.PRETRAIN,
)
def default_pretrain_trainer(train_args):
    """build trainer"""
    runtime_kwargs = {"pretrain_func": pretrain_with_mcore}
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=train_valid_test_datasets_provider,
        model_provider=llm_model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=forward_step,
        get_embedding_ranks=get_embedding_ranks,
        **runtime_kwargs,
    )

    return trainer
