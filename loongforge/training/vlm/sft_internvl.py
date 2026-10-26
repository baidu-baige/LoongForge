# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

""" sft for internvl model """


import torch
from functools import partial
from loongforge.training.vlm import pretrain_vlm
from transformers import DataCollatorForSeq2Seq
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training import get_timers
from megatron.core import mpu, tensor_parallel
from loongforge.engines.megatron.distributed.loss_reduction import default_loss_func
from megatron.core import parallel_state

from megatron.core.utils import StragglerDetector
from loongforge.engines.megatron import (get_args, get_tokenizer, get_model_config)
from megatron.core.enums import ModelType
from loongforge.engines.megatron.runner import pretrain
from loongforge.data.vlm.internvl_task_encoder import InternVLTaskEncoder
from loongforge.engines.megatron.distributed.dp_balance.rebalance.layout import install_batch_layout
from loongforge.training.vlm.internvl_dp_balance import INTERNVL_BATCH_LAYOUT
from loongforge.engines.megatron.data_padding import (
    llm_pad_to_multiple_of,
)
from loongforge.data.llm.dataloader import (
    build_sft_data_collator,
    build_sft_cyclic_iterators,
    load_pretokenized_dataset,
)
from loongforge.data.vlm.dataloader import (
    get_train_dataset,
    get_train_loader,
)
from loongforge.engines.megatron.checkpointing import load_dataloader_state
from loongforge.training.sft_data_config import sft_loader_config
from loongforge.training.vlm.data_configs import vlm_data_config
from loongforge.models.multimodal.multimodal_model_provider import (
    omni_model_provider,
)
from loongforge.models.vision.internvl.internvl_config import IMG_CONTEXT_TOKEN

stimer = StragglerDetector()


def get_packed_seq_params(attention_mask):
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
        image_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        patch_size = get_model_config().image_encoder.patch_size
        num_image_token = int((args.force_image_size // patch_size) ** 2 * (args.down_sample_ratio ** 2))
        ignore_flag = filter_ignore_data(input_ids, image_flags, image_token_id,
                                            num_image_token)
        if ignore_flag:
            print("filter_ignore_data get True, skip current microbatch...")
            output_tensor = output_tensor * 0.0

    model_config = get_model_config()
    loss_func = getattr(model_config, "loss_func", default_loss_func)
    return output_tensor, partial(loss_func, loss_mask, loss_weights)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    import loongforge.engines.megatron.distributed.dp_balance.patches  # noqa: F401
    args = get_args()
    loader_config = sft_loader_config(args)
    data_config = vlm_data_config(args)
    if mpu.get_tensor_model_parallel_rank() != 0:
        return None, None, None

    # create dataloader
    if args.is_tokenized_data:
        dpr = parallel_state.get_data_parallel_rank()
        dpw = parallel_state.get_data_parallel_world_size()
        train_ds = load_pretokenized_dataset(args.data_path, dpr)
        collator = build_sft_data_collator(
            DataCollatorForSeq2Seq,
            loader_config,
            get_tokenizer(),
            pad_to_multiple_of=llm_pad_to_multiple_of(args),
        )
        train_data_iterator, valid_data_iterator, test_data_iterator = (
            build_sft_cyclic_iterators(
                train_ds,
                None,
                None,
                collator,
                loader_config,
                dpr,
                dpw,
                args.consumed_train_samples,
                args.consumed_valid_samples,
                args.skip_train,
                state=load_dataloader_state(args),
            )
        )
        return train_data_iterator, None, None
    else:
        tokenizer = get_tokenizer().tokenizer
        task_encoder = InternVLTaskEncoder(
            data_config, tokenizer, patch_size=get_model_config().image_encoder.patch_size
        )
        train_dataset = get_train_dataset(
            task_encoder,
            data_config,
            dp_rank=parallel_state.get_data_parallel_rank(),
            dp_size=parallel_state.get_data_parallel_world_size(),
            dp_group=parallel_state.get_data_parallel_group(),
        )
        train_dataloader = get_train_loader(train_dataset, state=load_dataloader_state(args))
        return train_dataloader, None, None


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    """Build the multimodal model with this run's args and model config."""
    return omni_model_provider(
        pre_process,
        post_process,
        vp_stage=vp_stage,
        args=get_args(),
        model_config=get_model_config(),
        runtime=pretrain_vlm.build_multimodal_runtime(get_args()),
    )


def train(args):
    """Run InternVL SFT."""
    install_batch_layout(args.model_family, INTERNVL_BATCH_LAYOUT)
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        args,
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
    )
