# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

""" sft for internvl model """

import torch.distributed as dist

import math
import numpy as np
import os
import torch
import re
import copy
from functools import partial
from datasets import load_from_disk
from transformers import DataCollatorForSeq2Seq
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training import get_timers
from megatron.core import mpu, tensor_parallel
from loongforge.train.get_loss_func import default_loss_func
from megatron.core import parallel_state

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.utils import StragglerDetector
from loongforge.utils import (get_args, get_tokenizer, get_model_config)
from megatron.core.enums import ModelType
from loongforge.utils import constants
from loongforge.models import get_model_provider, get_model_family
from loongforge.train.megatron_trainer import MegatronTrainer
from loongforge.train.trainer_builder import register_model_trainer
from loongforge.data.multimodal.internvl.internvl_task_encoder import InternVLTaskEncoder
from loongforge.train.sft.utils import (
    build_sft_data_collator,
    build_sft_cyclic_iterators,
)
from loongforge.data.multimodal.dataloader_provider import (
    get_train_dataset,
    get_train_loader,
    VLMPretrainCollator,
)
from loongforge.models.omni_models.omni_model_provider import (
    omni_model_provider,
)

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
    #model_family = get_model_family(args.model_family)
    model_provider = get_model_provider(args.model_family)
    assert model_provider is not None, f'model provider for {args.model_name} not found'
    return model_provider(pre_process, post_process, vp_stage)


def get_packed_seq_params(attention_mask):
    """Get packed seq params """
    packed_seq_params = PackedSeqParams()
    packed_seq_params.qkv_format = "thd"
    packed_seq_params.cu_seqlens_q = attention_mask
    packed_seq_params.cu_seqlens_kv = attention_mask
    packed_seq_params.max_seqlen_q = (attention_mask[1:] - attention_mask[:-1]).max().item()
    packed_seq_params.max_seqlen_kv = packed_seq_params.max_seqlen_q

    return packed_seq_params


def get_batch(data_iterator):
    """Generate a batch"""
    # get batches based on the TP rank you are on
    args = get_args()

    if data_iterator is not None:
        sample = next(data_iterator)
        key_mapping = {'tokens': 'input_ids', 'attn_mask': 'attention_mask', 'imgs': 'pixel_values'}
        data = {key_mapping.get(k, k): v for k, v in sample.items()}
    else:
        data = None

    if data and data["loss_weight"] is not None:
        data["loss_weight"] = torch.tensor(data["loss_weight"], dtype=torch.float32)

    data_i = {}
    data_f = {}
    data_l = {}
    packed_seq_params = None
    attention_mask = None

    if args.packing_sft_data:
        data_a = tensor_parallel.broadcast_data(["attention_mask"], data, torch.int32)
        attention_mask = data_a["attention_mask"].squeeze_(0)
        packed_seq_params = get_packed_seq_params(attention_mask)
    else:
        data_a = tensor_parallel.broadcast_data(["attention_mask"], data, torch.bool)
        packed_seq_params = None
        attention_mask = ~(data_a["attention_mask"].unsqueeze(1).unsqueeze(1))

    if args.pipeline_model_parallel_size == 1:
        data_i = tensor_parallel.broadcast_data(["input_ids", "position_ids", "labels", "image_flags"], data,
                                                torch.int64)
        data_f = tensor_parallel.broadcast_data(["pixel_values"], data, torch.float32)
        if args.packing_sft_data:
            data_l = tensor_parallel.broadcast_data(["loss_weight"], data, torch.float32)
    elif mpu.is_pipeline_first_stage():
        data_i = tensor_parallel.broadcast_data(["input_ids", "position_ids", "image_flags"], data, torch.int64)
        data_f = tensor_parallel.broadcast_data(["pixel_values"], data, torch.float32)
    elif mpu.is_pipeline_last_stage():
        data_i = tensor_parallel.broadcast_data(["input_ids", "labels", "image_flags"], data, torch.int64)
        if args.packing_sft_data:
            data_l = tensor_parallel.broadcast_data(["loss_weight"], data, torch.float32)

    input_ids = data_i["input_ids"] if "input_ids" in data_i else None
    position_ids = data_i["position_ids"] if "position_ids" in data_i else None
    labels = data_i["labels"] if "labels" in data_i else None
    image_flags = data_i["image_flags"] if "image_flags" in data_i else None
    pixel_values = data_f["pixel_values"] if "pixel_values" in data_f else None
    loss_weight = data_l["loss_weight"] if "loss_weight" in data_l else None

    if labels is not None:
        labels = torch.roll(labels, shifts=-1, dims=1)
        labels[:, -1] = -100

    if loss_weight is not None:
        loss_weight = torch.roll(loss_weight, shifts=-1, dims=1)
        loss_weight[:, -1] = 0.0

    loss_mask = (labels != -100).int() if labels is not None else None

    # slice batch along sequence dimension for context parallelism

    batch = (pixel_values,
             position_ids,
             input_ids,
             image_flags,
             attention_mask,
             labels,
             loss_mask,
             packed_seq_params,
             loss_weight)
    return batch


def filter_ignore_data(input_ids, image_flags, img_context_token_id, num_image_token):
    """For qianfanvl """
    selected = (input_ids == img_context_token_id).sum().item()
    expected = (image_flags == 1).sum().item() * num_image_token
    return selected != expected


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: Megatron Model
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    args = get_args()
    global stimer
    with stimer(bdata=True):
        (pixel_values, position_ids, input_ids, image_flags, attention_mask, labels, loss_mask,
         packed_seq_params, loss_weights) = get_batch(data_iterator)

    timers('batch-generator').stop()
    with stimer:
        output_tensor = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_inputs={"images": pixel_values, "image_flags": image_flags},
            labels=labels,
            packed_seq_params=packed_seq_params,
        )
    # for qianfanvl
    if mpu.is_pipeline_last_stage():
        tokenizer = get_tokenizer().tokenizer
        image_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        num_image_token = int((args.force_image_size // args.patch_size) ** 2 * (args.down_sample_ratio ** 2))
        ignore_flag = filter_ignore_data(input_ids, image_flags, image_token_id,
                                            num_image_token)
        if ignore_flag:
            print(f"filter_ignore_data get True, skip current microbatch...")
            output_tensor = output_tensor * 0.0

    # print(f"packed_seq_params:{packed_seq_params}")
    model_config = get_model_config()
    loss_func = getattr(model_config, "loss_func", default_loss_func)
    return output_tensor, partial(loss_func, loss_mask, loss_weights)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Train_valid_test_datasets_provider """
    import loongforge.data.dp_balance.patches
    args = get_args()
    if mpu.get_tensor_model_parallel_rank() != 0:
        return None, None, None

    # Disable wandb if it causes issues
    if not hasattr(args, 'wandb_project') or args.wandb_project is None:
        print("Warning: Disabling wandb logging as wandb_project is not set")
        args.wandb_project = ""
        args.wandb_exp_name = ""
        args.wandb_save = False
        args.use_wandb = False

    # create dataloader
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
        tokenizer = get_tokenizer().tokenizer
        task_encoder = InternVLTaskEncoder(args, tokenizer)
        train_dataset = get_train_dataset(task_encoder)
        train_dataloader = get_train_loader(train_dataset)
        return train_dataloader, None, None


@register_model_trainer(model_family=constants.VisionLanguageModelFamilies.INTERN_VL,
                        training_phase=constants.TrainingPhase.SFT, override=True)
def default_sft_trainer(train_args):
    """Build trainer"""
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=train_valid_test_datasets_provider,
        model_provider=omni_model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=forward_step,
    )

    return trainer
