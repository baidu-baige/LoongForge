# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Sft script for Ernie"""

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
from megatron.core.enums import ModelType
from megatron.training import get_timers

from loongforge.utils import get_args, print_rank_0
from loongforge.utils.constants import TrainingPhase, VisionLanguageModelFamilies
from loongforge.data.multimodal.ernie.data_utils import ErnieTensorDataset
from loongforge.train.megatron_trainer import MegatronTrainer
from loongforge.train.trainer_builder import register_model_trainer
from megatron.core import parallel_state
from megatron.core import mpu, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.utils import StragglerDetector
from loongforge.utils import get_model_config
from loongforge.models.omni_models.omni_model_provider import (
    omni_model_provider
)
from loongforge.train.get_loss_func import default_loss_func

logger = logging.getLogger(__name__)
stimer = StragglerDetector()

_VERIFY_DONE = False


def pad_to_len(data_i, loss_mask, packed_seq_params=None):
    """Pad to length for SP (sequence parallel).

    When packing is active, the last boundary of cu_seqlens is extended to
    include the padding tokens so that FlashAttention varlen sees a consistent
    total length.  The padding tokens belong to the last document segment and
    carry loss_mask=0, so they do not affect the loss.

    Position IDs for padding are filled with 0 (three dims) to avoid spurious
    large values leaking into the RoPE computation.
    """
    args = get_args()
    if args.tensor_model_parallel_size == 1:
        return data_i, loss_mask

    pad_to_multiple_of = 1
    pad_to_multiple_of *= (
        args.tensor_model_parallel_size if args.sequence_parallel else 1
    )
    remainder = data_i["input_ids"].shape[1] % pad_to_multiple_of
    if remainder == 0:
        return data_i, loss_mask
    pad_num = pad_to_multiple_of - remainder

    data_i["input_ids"] = F.pad(data_i["input_ids"], (0, pad_num), "constant", 0)
    data_i["token_type_ids"] = F.pad(data_i["token_type_ids"], (0, pad_num), "constant", 0)
    # Pad position_ids with zeros — padding tokens are masked out, their
    # position values are irrelevant.  Filling with 0 is consistent with the
    # ERNIE reference packing_dataloader._pad_to_max_length.
    data_i["position_ids"] = F.pad(data_i["position_ids"], (0, 0, 0, pad_num), "constant", 0)
    data_i["labels"] = F.pad(data_i["labels"], (0, pad_num), "constant", -100)
    loss_mask = F.pad(loss_mask, (0, pad_num), "constant", 0)

    # Keep cu_seqlens consistent with the padded tensor length.
    # Extend the last segment boundary to absorb the padding tokens, matching
    # the ERNIE reference behaviour where inbatch_pack_offset[-1] is set to
    # pad_to_max_seqlen.
    if packed_seq_params is not None:
        packed_seq_params.cu_seqlens_q[-1] += pad_num
        packed_seq_params.cu_seqlens_kv[-1] += pad_num

    return data_i, loss_mask


def get_batch(data_iterator):
    """Generate a batch"""
    # get batches based on the TP rank you are on
    if data_iterator is not None:
        data = next(data_iterator)
        # Packing produces (1, packed_len) tensors that must stay 2D for the
        # model.  Non-packing uses cu_lengths=[[0]] (numel==1); packing uses
        # cu_lengths=[[0, L1, ...]] (numel>1).  Use numel() to distinguish
        # rather than squeeze().shape, which collapses [[0]] to a scalar and
        # gives torch.Size([]) != torch.Size([1]) — a false positive.
        has_packing = "cu_lengths" in data and data["cu_lengths"].numel() > 1
        for key in data:
            if has_packing:
                data[key] = data[key].cuda()
            else:
                data[key] = data[key].squeeze(dim=0).cuda()
    else:
        data = None

    data_i = tensor_parallel.broadcast_data([
        "input_ids",
        "token_type_ids",
        "position_ids",
        "grid_thw",
        "image_type_ids",
        "labels",
    ], data, torch.int64)
    data_f = tensor_parallel.broadcast_data(["images"], data, torch.uint8)
    data_p = tensor_parallel.broadcast_data(["cu_lengths", "max_lengths"], data, torch.int32)

    cu_lengths = data_p["cu_lengths"]
    max_lengths = data_p["max_lengths"]

    # Build PackedSeqParams when packing is active (cu_lengths has >1 element,
    # i.e. at least one real segment boundary beyond the leading 0).
    packed_seq_params = None
    if cu_lengths.numel() > 1:
        assert cu_lengths.shape[0] == 1, "micro-batch-size must be 1 for packing"
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_lengths[0],
            cu_seqlens_kv=cu_lengths[0],
            max_seqlen_q=max_lengths[0].item(),
            max_seqlen_kv=max_lengths[0].item(),
        )
    # slice batch along sequence dimension for context parallelism
    assert mpu.get_context_parallel_world_size() == 1, "cp > 1 is not implemented"

    if packed_seq_params is not None:
        attention_mask = None
        attn_mask_type = None
    else:
        attention_mask = data_i['input_ids'].logical_not()
        attn_mask_type = AttnMaskType.causal

    loss_mask = (data_i['labels'] != -100).float()
    data_i, loss_mask = pad_to_len(data_i, loss_mask, packed_seq_params)
    batch = (
        data_f['images'],
        data_i['input_ids'],
        data_i['token_type_ids'],
        data_i['position_ids'],
        attention_mask,
        data_i['grid_thw'],
        data_i['image_type_ids'],
        data_i['labels'],
        loss_mask,
        attn_mask_type,
        packed_seq_params,
    )
    return batch


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: Megatron Model
    """
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    model_config = get_model_config()
    loss_func = getattr(model_config, "loss_func", default_loss_func)

    global stimer
    with stimer(bdata=True):
        images, input_ids, token_type_ids, position_ids, attention_mask, grid_thw, image_type_ids, \
            labels, loss_mask, attn_mask_type, packed_seq_params = get_batch(data_iterator)
    timers("batch-generator").stop()

    extra_input = {}
    model_config = get_model_config()
    image_mask = input_ids == model_config.foundation.im_patch_id
    # breakpoint()
    with stimer:
        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            image_inputs={"images": images, "image_grid_thw": grid_thw,
                "image_type_ids": image_type_ids, "image_mask": image_mask},
            labels=labels,
            packed_seq_params=packed_seq_params,
        )
    return loss, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider_energon(train_val_test_num_samples):
    """Build ERNIE datasets using the Energon pipeline.

    Activated when ``--task-encoder ErnieTaskEncoder`` is set.
    """
    from loongforge.data.multimodal import build_task_encoder
    from loongforge.data.multimodal.dataloader_provider import (
        get_train_dataset,
        get_train_loader,
    )

    args = get_args()
    task_encoder = build_task_encoder(args)
    train_ds = get_train_dataset(task_encoder)
    # No collator — encode_batch already produces the final dict format
    # that get_batch() expects (input_ids, token_type_ids, position_ids, etc.).
    train_dataloader = get_train_loader(train_ds, collator=None)
    print_rank_0(f"> finished creating {args.model_name} Energon datasets ...")
    return train_dataloader, None, None


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets."""
    args = get_args()

    # Use Energon pipeline if a task encoder is explicitly specified.
    if getattr(args, "task_encoder", None):
        return train_valid_test_datasets_provider_energon(train_val_test_num_samples)

    dataset = ErnieTensorDataset(
        args, args.data_path[0], args.train_iters * args.global_batch_size
    )

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_world_size = parallel_state.get_data_parallel_world_size()

    sampler = torch.utils.data.DistributedSampler(
        dataset, shuffle=False, num_replicas=dp_world_size, rank=dp_rank
    )
    # TODO: Batched inference is not supported yet.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
    )
    print_rank_0(f"> finished creating {args.model_name} datasets ...")

    return iter(dataloader), None, None


@register_model_trainer(
    model_family=[VisionLanguageModelFamilies.ERNIE4_5_VL],
    training_phase=TrainingPhase.SFT, override=True)
def default_pretrain_trainer(train_args):
    """build trainer"""
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=train_valid_test_datasets_provider,
        model_provider=omni_model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=forward_step,
    )
    return trainer
