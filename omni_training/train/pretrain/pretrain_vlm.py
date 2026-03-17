# Copyright 2026 The OmniTraining Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Pretrain traniner for omni models"""

import os
import torch
from functools import partial
import copy

from megatron.training import get_timers

from typing import List
from megatron.core.transformer.multi_token_prediction import get_mtp_ranks

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import StragglerDetector
from megatron.core import parallel_state

from megatron.core.transformer.enums import AttnMaskType

from transformers import DataCollatorForSeq2Seq

from omni_training.utils import constants, get_args, get_model_config

# from omni_training.models.qwen_vl.utils import get_inputs_on_this_cp_rank_by_tex # TODO:@yizhan
from omni_training.train.megatron_trainer import MegatronTrainer
from omni_training.train.trainer_builder import register_model_trainer
from omni_training.train.sft.utils import (
    build_sft_data_collator,
    build_sft_cyclic_iterators,
)
from omni_training.data.multimodal.dataloader_provider import (
    get_train_dataset,
    get_train_loader,
    VLMPretrainCollator,
)
from omni_training.data.multimodal import build_task_encoder
from datasets import load_from_disk

from megatron.core import parallel_state

from omni_training.models.omni_models.omni_model_provider import (
    omni_model_provider,
)
from omni_training.models.omni_models.utils import get_batch_on_this_cp_rank
from omni_training.train.get_loss_func import default_loss_func
from omni_training.train.initialize import change_parallel_state, get_encoder_dp_size

from omni_training.utils.global_vars import get_model_config

stimer = StragglerDetector()


def get_batch_on_this_tp_rank(data_iterator):
    """Get the current micro-batch on this rank."""
    model_config = get_model_config()
    IMAGE_TOKEN_ID = getattr(model_config, "vision_token_id", 151655)
    VIDEO_TOKEN_ID = getattr(model_config, "video_token_id", 151656)
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
    attn_mask = tensor_parallel.broadcast_data(["attn_mask"], data, torch.bool)["attn_mask"]

    has_video = bool((tokens == VIDEO_TOKEN_ID).any())
    has_image = bool((tokens == IMAGE_TOKEN_ID).any())

    images = None
    image_grid_thw = None
    pixel_values_videos = None
    video_grid_thw = None
    if has_image:
        images = tensor_parallel.broadcast_data(["imgs"], data, torch.float32)["imgs"]
        image_grid_thw = tensor_parallel.broadcast_data(
            ["image_grid_thw"], data, torch.int32
        )["image_grid_thw"]
    if has_video:
        pixel_values_videos = tensor_parallel.broadcast_data(
            ["pixel_values_videos"], data, torch.float32
        )["pixel_values_videos"]
        video_grid_thw = tensor_parallel.broadcast_data(
            ["video_grid_thw"], data, torch.int32
        )["video_grid_thw"]

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
        "image_grid_thw": (
            image_grid_thw.cuda(non_blocking=True)
            if image_grid_thw is not None
            else None
        ),
        "pixel_values_videos": (
            pixel_values_videos.cuda(non_blocking=True)
            if pixel_values_videos is not None
            else None
        ),
        "video_grid_thw": (
            video_grid_thw.cuda(non_blocking=True)
            if video_grid_thw is not None
            else None
        ),
        "tokens": tokens,
        "attn_mask": attn_mask,
        "labels": labels,
        "cu_lengths": cu_lengths,
        "max_lengths": max_lengths,
        "position_ids": position_ids,
        "loss_mask": loss_mask,
        "packed_seq_params": packed_seq_params,
    }

    return batch


def get_batch(data_iterator):
    """Generate a batch"""

    batch = get_batch_on_this_tp_rank(data_iterator)

    batch = get_batch_on_this_cp_rank(batch)

    return batch


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10

batch_list = []
forward_step_calling_count = 0

def free_batch_list(batch_list):
    """
    Free the memory of the batch list.
    """
    for batch in batch_list:
        if isinstance(batch, dict):
            for k in list(batch.keys()):
                v = batch[k]
                if torch.is_tensor(v):
                    del v
                batch.pop(k)
        del batch

    batch_list.clear()
    torch.cuda.empty_cache()

def forward_step(data_iterator, model, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: Megatron Model
    """
    args = get_args()
    timers = get_timers()
    model_config = get_model_config()
    # Get the batch.
    timers("batch-generator", log_level=2).start()
    change_parallel_state('text_decoder')

    global stimer
    global forward_step_calling_count
    global batch_list

    _ImageEncoderDataParallelSize = get_encoder_dp_size('image_encoder')
    forward_group_id = forward_step_calling_count // _ImageEncoderDataParallelSize
    inner_group_id = forward_step_calling_count % _ImageEncoderDataParallelSize
    if inner_group_id == 0:
        free_batch_list(batch_list)
        with stimer(bdata=True):
            for _ in range(_ImageEncoderDataParallelSize):
                local_batch = copy.deepcopy(get_batch(data_iterator))
                batch_list.append(local_batch)

    timers("batch-generator").stop()

    with stimer:
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
            packed_seq_params,
        ) = batch_list[inner_group_id].values()

        loss_func = getattr(model_config, "loss_func", default_loss_func)

        if return_schedule_plan:
            assert args.overlap_moe_expert_parallel_comm, \
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            schedule_plan = model.build_schedule_plan(
                dict(
                images=images,
                image_grid_thw=image_grid_thw,
                ) if images is not None else None,
                dict(
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                ) if pixel_values_videos is not None else None,
                None,
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attn_mask,
                labels=labels,
                packed_seq_params=packed_seq_params,
                loss_mask=loss_mask,
            )
            return schedule_plan, partial(loss_func, loss_mask)
        else:
            output_tensor = model(
                dict(
                    images=images,
                    image_grid_thw=image_grid_thw,
                ) if images is not None else None,
                dict(
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                ) if pixel_values_videos is not None else None,
                None,
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attn_mask,
                labels=labels,
                packed_seq_params=packed_seq_params,
                enable_encoder_hetero_dp=args.enable_encoder_hetero_dp,
                batch_list=batch_list,
                forward_group_id=forward_group_id,
                inner_group_id=inner_group_id,
            )

        forward_step_calling_count += 1

    return output_tensor, partial(loss_func, loss_mask)  # TODO: add loss_weights data


GLOBAL_TRAIN_DATASET_SIZE = None


def train_valid_test_dataset_provider(train_val_test_num_samples, vp_stage=None):
    """Provides the datasets used by the trainer"""
    import omni_training.data.dp_balance.adaptor
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
        task_encoder = build_task_encoder(args)
        train_dataset = get_train_dataset(task_encoder)

        try:
            GLOBAL_TRAIN_DATASET_SIZE = len(train_dataset)
        except TypeError:
            GLOBAL_TRAIN_DATASET_SIZE = getattr(train_dataset, "num_rows", None)

        collator = build_sft_data_collator(VLMPretrainCollator)
        train_dataloader = get_train_loader(train_dataset, collator)
        return train_dataloader, None, None


def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        mtp_ranks = get_mtp_ranks(pp_ranks, config=get_model_config().foundation)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks

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
        model_type=ModelType.encoder_or_decoder,  # TODO: Heterogeneous TP/PP not supported yet
        forward_step_func=forward_step,
        get_embedding_ranks=get_embedding_ranks,
    )

    return trainer