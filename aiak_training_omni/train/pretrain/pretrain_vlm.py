"""Pretrain traniner for omni models"""

import os
import torch
from typing import Tuple, Optional
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
)
from aiak_training_omni.data.multimodal.vlm_task_encoder import VLMTaskEncoder
from datasets import load_from_disk

from megatron.core.models.multimodal import context_parallel
from megatron.core import parallel_state
import torch.nn.functional as F

from aiak_training_omni.models.omni_models.omni_model_provider import (
    omni_model_provider,
)

IGNORE_INDEX = -100
PAD_TOKEN_ID = 151643
# TODO: get token id from tokenizer
image_token_id = 151655
video_token_id = 151656
vision_start_token_id = 151652
stimer = StragglerDetector()


def get_rope_index(
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slightly modified from Qwen2_5VLForConditionalGeneration.get_rope_index"""
    spatial_merge_size = 2
    mrope_position_deltas = []
    assert input_ids is not None

    if image_grid_thw is not None or video_grid_thw is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 0]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * 2

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 0] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.logical_not().long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 1, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def seq_padding_for_cp(data, tp_size=1, cp_size=1, has_sp=False):
    """Sequence padding for CP and/or SP

    Args:
        data (dict): Data from dataloader.
        tp_size (int): Tensor parallel size.
        cp_size (int): Context parallel size.
        has_sp (bool): Model uses sequence parallelism.

    Returns:
        data (dict): Padded data.
    """
    tokens = data["tokens"]
    labels = data["labels"]
    attn_mask = data["attn_mask"]
    cu_lengths = data["cu_lengths"]
    max_lengths = data["max_lengths"]

    valid_tokens = []
    valid_labels = []
    valid_attn_mask = []

    cu_seqlens_padded = [0]
    seq_lengths = cu_lengths[0, 1:] - cu_lengths[0, :-1]
    start = 0
    for length in seq_lengths:
        token = tokens[0, start : start + length]
        label = labels[0, start : start + length]
        mask = attn_mask[0, start : start + length]

        mp_padding_needed = context_parallel.get_padding(
            length, cp_size, tp_size, has_sp
        )

        input_ids = F.pad(token, (0, mp_padding_needed), "constant", PAD_TOKEN_ID)
        label = F.pad(label, (0, mp_padding_needed), "constant", IGNORE_INDEX)
        mask = F.pad(mask, (0, mp_padding_needed), "constant", True)

        valid_tokens.append(input_ids)
        valid_labels.append(label)
        valid_attn_mask.append(mask)

        cu_seqlens_padded.append(
            int(cu_seqlens_padded[-1] + length + mp_padding_needed)
        )

        start += length

    data["tokens"] = torch.cat(valid_tokens, dim=0).unsqueeze(0).to(tokens.dtype)
    data["labels"] = torch.cat(valid_labels, dim=0).unsqueeze(0).to(labels.dtype)
    data["attn_mask"] = (
        torch.cat(valid_attn_mask, dim=0).unsqueeze(0).to(attn_mask.dtype)
    )

    data["cu_lengths"] = torch.tensor(
        cu_seqlens_padded, dtype=cu_lengths.dtype
    ).unsqueeze(0)
    cu_seqlens_padded = torch.tensor(cu_seqlens_padded, dtype=torch.int32)
    seq_lens_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    data["max_lengths"] = torch.tensor(
        [seq_lens_padded.max().item()], dtype=max_lengths.dtype
    )

    return data


def get_batch_on_this_tp_rank(data_iterator):
    """Get the current micro-batch on this rank."""
    args = get_args()

    if data_iterator is not None and mpu.get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
        # When enabling CP or SP and using packing, it is necessary to pad the sequence.
        if (args.context_parallel_size > 1 or args.sequence_parallel) and args.packing_sft_data:
            data = seq_padding_for_cp(
                data=data,
                tp_size=args.tensor_model_parallel_size,
                cp_size=args.context_parallel_size,
                has_sp=args.sequence_parallel
            )
    else:
        data = None

    tokens = tensor_parallel.broadcast_data(["tokens"], data, torch.int64)["tokens"]
    labels = tensor_parallel.broadcast_data(["labels"], data, torch.int64)["labels"]
    attn_mask = tensor_parallel.broadcast_data(["attn_mask"], data, torch.bool)["attn_mask"]
    cu_lengths = tensor_parallel.broadcast_data(["cu_lengths"], data, torch.int32)["cu_lengths"]
    max_lengths = tensor_parallel.broadcast_data(["max_lengths"], data, torch.int32)["max_lengths"]

    has_video = video_token_id in tokens
    has_image = image_token_id in tokens

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
    batch = {
        "images": images.cuda(non_blocking=True) if images is not None else None,
        "image_grid_thw": image_grid_thw.cuda(non_blocking=True) if image_grid_thw is not None else None,
        "pixel_values_videos": pixel_values_videos.cuda(non_blocking=True) if pixel_values_videos is not None else None,
        "video_grid_thw": video_grid_thw.cuda(non_blocking=True) if video_grid_thw is not None else None,
        "tokens": tokens.cuda(non_blocking=True),
        "attn_mask": attn_mask.cuda(non_blocking=True),
        "labels": labels.cuda(non_blocking=True),
        "cu_lengths": cu_lengths.cuda(non_blocking=True),
        "max_lengths": max_lengths.cuda(non_blocking=True),
    }

    return batch


def get_ltor_masks_and_position_ids(
        batch
    ):
    """Build masks and position id for left to right model."""
    # Position ids. [3 X bs X seqlen]

    input_ids = batch['tokens']
    position_ids, _ = get_rope_index(
        input_ids=input_ids,
        image_grid_thw=batch.get('image_grid_thw'),
        video_grid_thw=batch.get('video_grid_thw'),
        attention_mask=batch['attn_mask']
    )
    labels = batch.get('labels')
    cu_lengths = batch['cu_lengths']
    max_lengths = batch['max_lengths']
    # Loss mask.
    loss_mask = torch.ones(labels.size(), dtype=torch.float, device=input_ids.device)
    loss_mask[labels == PAD_TOKEN_ID] = 0.0  # mask paddings
    if IGNORE_INDEX is not None:
        loss_mask[labels == IGNORE_INDEX] = 0.0  # mask prompts

    # Attention mask.
    attn_mask = batch['attn_mask']

    packed_seq_params = None

    loss_mask = (labels != -100).long()
    attn_mask_type = (
        AttnMaskType.padding_causal if attn_mask.any() else AttnMaskType.causal
    )

    labels = torch.roll(labels, shifts=-1, dims=1)
    if cu_lengths.shape == torch.Size([1, 1]):
        for i in range(attn_mask.shape[0]):
            loss_mask[i, (attn_mask[i] == False).sum() - 1] = 0
    else:
        for i in range(cu_lengths.shape[0]):
            for j in range(1, cu_lengths[i].shape[0]):
                loss_mask[i, cu_lengths[i][j] - 1] = 0

        assert cu_lengths.shape[0] == 1, "micro-batch-size must be 1 for packing"
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_lengths[0],
            cu_seqlens_kv=cu_lengths[0],
            max_seqlen_q=max_lengths[0].item(),
            max_seqlen_kv=max_lengths[0].item(),
        )
    batch['attn_mask'] = None
    batch['labels'] = labels
    batch['position_ids'] = position_ids
    batch['loss_mask'] = loss_mask
    batch['attn_mask_type'] = attn_mask_type
    batch['packed_seq_params'] = packed_seq_params

    return batch


def get_batch(data_iterator):
    """Generate a batch"""

    batch = get_batch_on_this_tp_rank(data_iterator)


    batch = get_ltor_masks_and_position_ids(batch)

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

        collator = build_sft_data_collator(DataCollatorForSeq2Seq)
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
