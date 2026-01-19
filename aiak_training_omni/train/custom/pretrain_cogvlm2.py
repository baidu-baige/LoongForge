"""default pretrain for generative models like GPTS"""

import os
import torch

from functools import partial

from megatron.training import get_timers

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.utils import StragglerDetector

from megatron.core.transformer.enums import AttnMaskType

from aiak_training_omni.models.cogvlm.utils import (
    get_batch_on_this_tp_rank,
)

from aiak_training_omni.utils import constants, get_args, get_tokenizer, print_rank_0

from aiak_training_omni.models import get_model_provider, get_model_family

from aiak_training_omni.train.megatron_trainer import MegatronTrainer
from aiak_training_omni.train.trainer_builder import register_model_trainer
from aiak_training_omni.data.multimodal_dataset import CaptionDataset


stimer = StragglerDetector()


def model_provider(pre_process=True, post_process=True, vp_stage: int = None):
    """Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.

    Returns:
        MCoreModel: The returned model
    """
    args = get_args()
    model_family = get_model_family(args.model_name)
    model_provider = get_model_provider(model_family)
    assert model_provider is not None, f"model provider for {args.model_name} not found"
    return model_provider(pre_process, post_process, vp_stage)


def get_batch(data_iterator):
    """Generate a batch"""
    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    assert mpu.get_context_parallel_world_size() == 1, "not implemented"
    # batch = get_batch_on_this_cp_rank(batch)

    batch = (
        batch["images"],
        batch["input_ids"],
        batch["position_ids"],
        batch["attention_mask"],
        batch["token_type_ids"],
        batch["labels"],
        batch["loss_mask"],
        AttnMaskType.padding_causal,
    )

    return batch


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
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
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
        (
            images,
            input_ids,
            position_ids,
            attention_mask,
            token_type_ids,
            labels,
            loss_mask,
        ) = get_batch(data_iterator)

    timers("batch-generator").stop()

    with stimer:
        output_tensor = model(
            images,
            input_ids,
            position_ids,
            attention_mask,
            token_type_ids,
            labels,
        )

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.
    For GPT-like models, if there are no special requirements, we should directly reuse the Megatron GPTDataset.
    """
    args = get_args()
    tokenizer = get_tokenizer()

    dataset = CaptionDataset(
        root_dir=args.data_path[0],
        tokenizer=tokenizer,
        torch_type=args.params_dtype,
        template_version="base",
        patch_size=(args.patch_dim),
        image_size=(args.img_h, args.img_w),
        max_length=args.seq_length,
    )

    print_rank_0(f"> finished creating {args.model_name} datasets ...")

    return dataset, None, None


@register_model_trainer(
    model_family=constants.VisionLanguageModelFamilies.COGVLM2,
    training_phase=constants.TrainingPhase.PRETRAIN,
)
def default_pretrain_trainer(train_args):
    """build trainer"""
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=train_valid_test_datasets_provider,
        model_provider=model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=forward_step,
    )

    return trainer
