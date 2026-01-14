"""default sft trainer for generative models like GPTS"""

import os
import torch

from functools import partial

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.utils import StragglerDetector
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.rerun_state_machine import get_rerun_state_machine

from megatron.training import get_timers
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    average_losses_across_data_parallel_group,
)

from aiak_training_omni.utils import (
    constants,
    get_args,
    get_tokenizer,
    get_chat_template,
    print_rank_0,
)

from aiak_training_omni.models import get_model_provider, get_model_family
from aiak_training_omni.data import (
    SFTDataset,
    SFTDatasetConfig,
    BlendedHuggingFaceDatasetBuilder,
    DataCollatorForSupervisedDataset,
)

from aiak_training_omni.train.megatron_trainer import MegatronTrainer
from aiak_training_omni.train.trainer_builder import register_model_trainer

from .utils import (
    get_batch_on_this_tp_rank,
    get_dataset_blend_from_list,
    build_sft_cyclic_iterators,
    build_sft_data_collator,
)
from aiak_training_omni.models.foundation.llm_model_provider import llm_model_provider


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

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    if args.context_parallel_size > 1:
        loss = torch.cat(
            [torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)]
        )
        torch.distributed.all_reduce(
            loss,
            group=mpu.get_context_parallel_group(),
            op=torch.distributed.ReduceOp.SUM,
        )
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

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

    # reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss_reduced_dict = {"lm loss": averaged_loss[0]}

    # calculate the number of tokens for this micro-batch
    if args.variable_seq_lengths:
        # for variable seq length, we need to calculate the number of tokens on fly
        # model output tensor shape is [B, S, H]
        num_input_tokens = output_tensor.shape[0] * output_tensor.shape[1]
        input_tokens = torch.tensor(
            num_input_tokens, dtype=torch.int, device=output_tensor.device
        )
        # sum across all dp ranks
        torch.distributed.all_reduce(input_tokens, group=mpu.get_data_parallel_group())
        loss_reduced_dict["total_inputs"] = (
            input_tokens.item() * args.context_parallel_size
        )

    return loss, loss_reduced_dict


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: Megatron Model

    Returns:
        output_tensor: Output tensor
        loss_func: Loss function
        num_tokens: Number of tokens
    """
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
        output_tensor = model(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            packed_seq_params=packed_seq_params,
        )

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
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
        packing_batch_size=args.packing_batch_size,
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
    )

    return trainer
