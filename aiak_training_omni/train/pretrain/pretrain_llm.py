"""default pretrain for generative models like GPTS"""

import os
import torch

from functools import partial

from megatron.training import get_timers

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.utils import StragglerDetector

from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import (
    GPTDatasetConfig,
    MockGPTDataset,
    GPTDataset,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine

from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)

from aiak_training_omni.utils import constants, get_args, get_tokenizer, print_rank_0

from aiak_training_omni.models import get_model_provider, get_model_family

from aiak_training_omni.train.megatron_trainer import MegatronTrainer
from aiak_training_omni.train.trainer_builder import register_model_trainer
from aiak_training_omni.models.foundation.llm_model_provider import llm_model_provider


stimer = StragglerDetector()


def get_batch(data_iterator):
    """Generate a batch"""
    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


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
        a dict containing reporting metrics on the loss and number of tokens across the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    total_tokens = loss_mask.sum()
    loss = torch.cat(
        [torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)]
    )

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()

    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )

    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {"lm loss": (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: Megatron Model
    """
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()

    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator
        )

    timers("batch-generator").stop()

    with stimer:
        output_tensor = model(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    For GPT-like models, if there are no special requirements, we should directly reuse the Megatron GPTDataset.
    """
    args = get_args()
    tokenizer = get_tokenizer()

    def _is_dataset_built_on_rank():
        return (
            mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
        ) and mpu.get_tensor_model_parallel_rank() == 0

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

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset if not args.mock_data else MockGPTDataset,
        train_val_test_num_samples,
        _is_dataset_built_on_rank,
        config,
    ).build()

    print_rank_0(f"> finished creating {args.model_name} datasets ...")

    return train_ds, valid_ds, test_ds


@register_model_trainer(
    model_family=constants.LanguageModelFamilies.names(),
    training_phase=constants.TrainingPhase.PRETRAIN,
)
def default_pretrain_trainer(train_args):
    """build trainer"""
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=train_valid_test_datasets_provider,
        model_provider=llm_model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=forward_step,
    )

    return trainer
