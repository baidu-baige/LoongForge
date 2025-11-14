"""Pretrain traniner for omni models"""

import os
import torch
from functools import partial

from megatron.training import get_timers

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import StragglerDetector
from megatron.core import parallel_state

from megatron.core.transformer.enums import AttnMaskType

from transformers import DataCollatorForSeq2Seq

from aiak_training_omni.utils import constants, get_args

# from aiak_training_omni.models.qwen_vl.utils import get_inputs_on_this_cp_rank_by_tex # TODO:@yizhan
from aiak_training_omni.train.megatron_trainer import MegatronTrainer
from aiak_training_omni.train.trainer_builder import register_model_trainer
from aiak_training_omni.train.sft.utils import (
    build_sft_data_collator,
    build_sft_cyclic_iterators,
)
from aiak_training_omni.data.multimodal.dataloader_provider import (
    get_train_dataset,
    get_train_loader,
    VLMPretrainCollator,
    IMAGE_TOKEN_ID,
    VIDEO_TOKEN_ID,
)
from aiak_training_omni.data.multimodal.vlm_task_encoder import VLMTaskEncoder
from datasets import load_from_disk

from megatron.core import parallel_state

from aiak_training_omni.models.omni_models.omni_model_provider import (
    omni_model_provider,
)
from aiak_training_omni.models.omni_models.utils import get_batch_on_this_cp_rank

stimer = StragglerDetector()



def get_batch_on_this_tp_rank(data_iterator):
    """Get the current micro-batch on this rank."""
    args = get_args()

    if data_iterator is not None and mpu.get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
    else:
        data = None

    tokens = tensor_parallel.broadcast_data(["tokens"], data, torch.int64)["tokens"]
    labels = tensor_parallel.broadcast_data(["labels"], data, torch.int64)["labels"]
    cu_lengths = tensor_parallel.broadcast_data(["cu_lengths"], data, torch.int32)["cu_lengths"]
    max_lengths = tensor_parallel.broadcast_data(["max_lengths"], data, torch.int32)["max_lengths"]
    position_ids = tensor_parallel.broadcast_data(["position_ids"], data, torch.int64)["position_ids"]
    loss_mask = tensor_parallel.broadcast_data(["loss_mask"], data, torch.int64)["loss_mask"]
    attn_mask_type_id = tensor_parallel.broadcast_data(["attn_mask_type_id"], data, torch.int64)["attn_mask_type_id"]
    attn_mask = tensor_parallel.broadcast_data(["attn_mask"], data, torch.float32)["attn_mask"]

    has_video = bool((tokens == VIDEO_TOKEN_ID).any())
    has_image = bool((tokens == IMAGE_TOKEN_ID).any())

    images = None
    image_grid_thw = None
    pixel_values_videos = None
    video_grid_thw = None
    if has_image:
        images = tensor_parallel.broadcast_data(["imgs"], data, torch.float32)["imgs"]
        image_grid_thw = tensor_parallel.broadcast_data(["image_grid_thw"], data, torch.int32)["image_grid_thw"]
    if has_video:
        pixel_values_videos = tensor_parallel.broadcast_data(["pixel_values_videos"], \
                                    data, torch.float32)["pixel_values_videos"]
        video_grid_thw = tensor_parallel.broadcast_data(["video_grid_thw"], data, torch.int32)["video_grid_thw"]
    attn_mask_type = AttnMaskType(int(attn_mask_type_id.item()))

    tokens = tokens.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    cu_lengths = cu_lengths.cuda(non_blocking=True)
    max_lengths = max_lengths.cuda(non_blocking=True)
    position_ids = position_ids.cuda(non_blocking=True)
    loss_mask = loss_mask.cuda(non_blocking=True)
    attn_mask = attn_mask.cuda(non_blocking=True)

    packed_seq_params = None
    if cu_lengths.shape != torch.Size([1, 1]):
        assert cu_lengths.shape[0] == 1, "micro-batch-size must be 1 for packing"
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_lengths[0],
            cu_seqlens_kv=cu_lengths[0],
            max_seqlen_q=max_lengths[0].item(),
            max_seqlen_kv=max_lengths[0].item(),
        )

    batch = {
        "images": images.cuda(non_blocking=True) if images is not None else None,
        "image_grid_thw": image_grid_thw.cuda(non_blocking=True) if image_grid_thw is not None else None,
        "pixel_values_videos": pixel_values_videos.cuda(non_blocking=True) if pixel_values_videos is not None else None,
        "video_grid_thw": video_grid_thw.cuda(non_blocking=True) if video_grid_thw is not None else None,
        "tokens": tokens,
        "attn_mask": attn_mask,
        "labels": labels,
        "cu_lengths": cu_lengths,
        "max_lengths": max_lengths,
        "position_ids": position_ids,
        "loss_mask": loss_mask,
        "attn_mask_type": attn_mask_type,
        "packed_seq_params": packed_seq_params,
    }

    return batch


def get_batch(data_iterator):
    """Generate a batch"""

    batch = get_batch_on_this_tp_rank(data_iterator)

    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor,
    loss_weight: torch.Tensor = None,
):
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

    if (loss_weight is not None and loss_weight.sum() == 0) or (loss_mask.sum() == 0):
        output_tensor = output_tensor * 0.0
        valid_mask = False
    else:
        valid_mask = True
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    if loss_weight is not None:
        shift_weights = loss_weight.view(-1)
        shift_weights_sum = shift_weights.sum()
        if (
            args.loss_reduction_all_gather and args.context_parallel_size > 1
        ):  # TODO: check args.loss_reduction_all_gather
            torch.distributed.all_reduce(
                shift_weights_sum,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_data_parallel_group(with_context_parallel=True),
            )
            shift_weights_sum = shift_weights_sum / mpu.get_data_parallel_world_size(
                with_context_parallel=True
            )
        loss = torch.cat(
            [
                torch.sum(losses.view(-1) * shift_weights)
                / (shift_weights_sum if valid_mask else 1.0).view(1),
                total_tokens.view(1),
            ]
        )
    else:
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

    loss_reduced_dict = {"lm loss": (reporting_loss[0], reporting_loss[1])}

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

    return (loss[0] * args.context_parallel_size, local_num_tokens, loss_reduced_dict)


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
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            tokens,
            attn_mask,
            labels,
            cu_lengths,
            max_lengths,
            position_ids,
            loss_mask,
            attn_mask_type,
            packed_seq_params,
        ) = get_batch(data_iterator)

    timers("batch-generator").stop()

    with stimer:
        output_tensor = model(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attn_mask,
            attn_mask_type=attn_mask_type,
            labels=labels,
            packed_seq_params=packed_seq_params,
            image_inputs=dict(
                images=images,
                image_grid_thw=image_grid_thw,
            ),
            video_inputs=dict(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            ),
        )

    return output_tensor, partial(loss_func, loss_mask)


GLOBAL_TRAIN_DATASET_SIZE = None


def train_valid_test_dataset_provider(train_val_test_num_samples):
    """Provides the datasets used by the trainer"""
    global GLOBAL_TRAIN_DATASET_SIZE
    args = get_args()

    if args.is_tokenized_data:
        rank = parallel_state.get_data_parallel_rank()
        save_path = os.path.join(args.data_path[0], "preprocess", str(rank))
        print(f"[rank{rank}] loading preprocessed dataset from {save_path}")
        train_ds = load_from_disk(save_path)
        collator = build_sft_data_collator(DataCollatorForSeq2Seq)
        train_data_iterator, valid_data_iterator, test_data_iterator = (
            build_sft_cyclic_iterators(train_ds, None, None, collator)
        )
        return train_data_iterator, None, None

    else:
        task_encoder = VLMTaskEncoder(args)
        train_dataset = get_train_dataset(task_encoder)

        try:
            GLOBAL_TRAIN_DATASET_SIZE = len(train_dataset)
        except TypeError:
            GLOBAL_TRAIN_DATASET_SIZE = getattr(train_dataset, "num_rows", None)

        collator = build_sft_data_collator(VLMPretrainCollator)
        train_dataloader = get_train_loader(train_dataset, collator)
        return train_dataloader, None, None


@register_model_trainer(
    model_family=constants.VisionLanguageModelFamilies.names(),
    training_phase=constants.TrainingPhase.PRETRAIN,
)
def default_vlm_pretrain_trainer(train_args):
    """build trainer"""
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=train_valid_test_dataset_provider,
        model_provider=omni_model_provider,
        model_type=ModelType.encoder_or_decoder,  # TODO: 异构tp/pp暂不支持
        forward_step_func=forward_step,
    )

    return trainer
