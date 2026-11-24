# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 native multimodal training entry."""

from __future__ import annotations

from functools import partial
from typing import List

import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.utils import get_attr_wrapped_model
from megatron.training import get_timers
from megatron.training.utils import is_first_or_last_pipeline_stage

from loongforge.data.multimodal import build_task_encoder
from loongforge.data.multimodal.dataloader_provider import (
    get_train_dataset,
    get_train_loader,
    get_val_dataset,
)
from loongforge.data.multimodal.deepseek_v41_task_encoder import DeepseekV41PretrainCollator
from loongforge.models.foundation.llm_model_provider import llm_model_provider
from loongforge.train.megatron_trainer import MegatronTrainer
from loongforge.train.pretrain import pretrain_llm
from loongforge.train.sft import sft_llm
from loongforge.train.sft.utils import build_sft_data_collator
from loongforge.train.trainer_builder import register_model_trainer
from loongforge.utils import constants, get_args


def _batch_value(data, key):
    value = data.get(key) if data is not None else None
    if torch.is_tensor(value):
        return value.numel() > 0
    return value is not None


def _broadcast_optional(data, key, dtype):
    if mpu.get_tensor_model_parallel_rank() == 0:
        flag = torch.tensor([int(_batch_value(data, key))], dtype=torch.int32, device="cuda")
    else:
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    torch.distributed.broadcast(
        flag,
        src=mpu.get_tensor_model_parallel_src_rank(),
        group=mpu.get_tensor_model_parallel_group(),
    )
    if not bool(flag.item()):
        return None
    return tensor_parallel.broadcast_data([key], data, dtype)[key].cuda(non_blocking=True)


def _native_get_batch(data_iterator, vp_stage: int | None = None):
    if not is_first_or_last_pipeline_stage(vp_stage):
        return {
            key: None
            for key in (
                "tokens",
                "labels",
                "position_ids",
                "loss_mask",
                "attn_mask",
                "pixel_values",
                "image_grid_hws",
                "vision_token_types",
            )
        }
    if data_iterator is not None and mpu.get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
    else:
        data = None
    base = {}
    for key, dtype in (
        ("tokens", torch.int64),
        ("labels", torch.int64),
        ("position_ids", torch.int64),
        ("loss_mask", torch.int64),
        ("attn_mask", torch.bool),
    ):
        base[key] = tensor_parallel.broadcast_data([key], data, dtype)[key]
    batch = {key: value.cuda(non_blocking=True) for key, value in base.items()}
    batch["pixel_values"] = _broadcast_optional(data, "pixel_values", torch.bfloat16)
    batch["image_grid_hws"] = _broadcast_optional(data, "image_grid_hws", torch.int64)
    batch["vision_token_types"] = _broadcast_optional(data, "vision_token_types", torch.int32)
    return batch


def _native_forward_step(data_iterator, model, return_schedule_plan=False):
    timers = get_timers()
    timers("batch-generator", log_level=2).start()
    vp_stage = get_attr_wrapped_model(model, "vp_stage")
    batch = _native_get_batch(data_iterator, vp_stage)
    timers("batch-generator").stop()
    if return_schedule_plan:
        raise NotImplementedError("DeepSeek-V4.1 native multimodal schedule-plan execution is not enabled")
    extra = {
        key: batch[key] for key in ("pixel_values", "image_grid_hws", "vision_token_types") if batch[key] is not None
    }
    output = model(
        batch["tokens"],
        batch["position_ids"],
        batch["attn_mask"],
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
        extra_block_kwargs=extra,
    )
    if hasattr(output, "loss") and output.loss is not None:
        output = output.loss
    return output, partial(pretrain_llm.loss_func, batch["loss_mask"])


def _native_dataset_provider(train_val_test_num_samples, vp_stage=None):
    del train_val_test_num_samples, vp_stage
    args = get_args()
    task_encoder = build_task_encoder(args)
    train_dataset = get_train_dataset(task_encoder)
    valid_dataset = get_val_dataset(task_encoder) if args.eval_iters > 0 else None
    collator = build_sft_data_collator(DeepseekV41PretrainCollator)
    return (
        get_train_loader(train_dataset, collator),
        get_train_loader(valid_dataset, collator, restore_state=False) if valid_dataset is not None else None,
        None,
    )


def _model_provider(
    pre_process=True,
    post_process=True,
    parallel_output=True,
    vp_stage=None,
):
    return llm_model_provider(
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=parallel_output,
        vp_stage=vp_stage,
    )


def _embedding_ranks(pp_ranks: List[int]):
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
    return sorted(set(embedding_ranks))


def _build_trainer(train_args):
    native = bool(train_args.task_encoder)
    if native:
        if train_args.task_encoder.lower() != "deepseekv41taskencoder":
            raise ValueError("DeepSeek-V4.1 image batches require --task-encoder deepseekv41taskencoder")
        if train_args.packing_pretrain_data or train_args.packing_sft_data:
            raise ValueError("DeepSeek-V4.1 native sequence packing is not supported")
        train_args.packing_buffer_size = None
        train_args.dataloader_type = "external"
    text_trainer = sft_llm if train_args.training_phase == constants.TrainingPhase.SFT else pretrain_llm
    return MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=(
            _native_dataset_provider if native else text_trainer.train_valid_test_datasets_provider
        ),
        model_provider=_model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=(_native_forward_step if native else text_trainer.forward_step),
        get_embedding_ranks=_embedding_ranks,
    )


@register_model_trainer(
    model_family=constants.VisionLanguageModelFamilies.DEEPSEEK_V41,
    training_phase=constants.TrainingPhase.PRETRAIN,
    override=True,
)
def deepseek_v41_pretrain_trainer(train_args):
    return _build_trainer(train_args)


@register_model_trainer(
    model_family=constants.VisionLanguageModelFamilies.DEEPSEEK_V41,
    training_phase=constants.TrainingPhase.SFT,
    override=True,
)
def deepseek_v41_sft_trainer(train_args):
    return _build_trainer(train_args)
