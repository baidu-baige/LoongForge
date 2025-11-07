# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""General utilities."""
import os
import sys
from datetime import datetime

import torch

try:
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_l2norm,
    )
except ImportError:
    try:
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        multi_tensor_applier = None

    try:
        from amp_C import multi_tensor_l2norm
    except ImportError:
        import warnings

        warnings.warn(
            f"Transformer Engine and Apex are not installed. "
            "Falling back to local implementations of "
            "multi_tensor_applier and multi_tensor_l2norm"
        )

        from megatron.core.utils import (
            local_multi_tensor_l2_norm as multi_tensor_l2norm,
            local_multi_tensor_applier as multi_tensor_applier,
        )

from megatron.training import (
    get_args,
    get_adlr_autoresume,
)
from megatron.core import DistributedDataParallel as DDP
from megatron.core import mpu
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.legacy.model import Float16Module
from megatron.legacy.model.module import param_is_not_shared


ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def get_batch_on_this_tp_rank(data_iterator):
    """Get the current micro-batch on this rank."""
    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                mpu.get_tensor_model_parallel_src_rank(),
                group=mpu.get_tensor_model_parallel_group(),
            )

    if mpu.get_tensor_model_parallel_rank() == 0:

        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None

        batch = {
            "images": data["images"].cuda(non_blocking=True),
            "input_ids": data["input_ids"].cuda(non_blocking=True),
            "position_ids": data["position_ids"].cuda(non_blocking=True),
            "attention_mask": data["attention_mask"].cuda(non_blocking=True),
            "token_type_ids": data["token_type_ids"].cuda(non_blocking=True),
            "labels": data["labels"].cuda(non_blocking=True),
            "loss_mask": data["loss_mask"].cuda(non_blocking=True),
        }

        _broadcast(batch["images"])
        _broadcast(batch["input_ids"])
        _broadcast(batch["labels"])
        _broadcast(batch["loss_mask"])
        _broadcast(batch["token_type_ids"])
        _broadcast(batch["attention_mask"])
        _broadcast(batch["position_ids"])

    else:

        images = torch.empty(
            args.micro_batch_size,
            3,
            args.img_h,
            args.img_w,
            dtype=args.params_dtype,
            device=torch.cuda.current_device(),
        )
        input_ids = torch.empty(
            (args.micro_batch_size, args.seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        labels = torch.empty(
            (args.micro_batch_size, args.seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        loss_mask = torch.empty(
            (args.micro_batch_size, args.seq_length),
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        token_type_ids = torch.empty(
            args.micro_batch_size,
            args.seq_length,
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        attention_mask = torch.empty(
            args.micro_batch_size,
            1,
            1,
            args.seq_length,
            dtype=torch.bool,
            device=torch.cuda.current_device(),
        )
        position_ids = torch.empty(
            args.micro_batch_size,
            args.seq_length,
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )

        _broadcast(images)
        _broadcast(input_ids)
        _broadcast(labels)
        _broadcast(loss_mask)
        _broadcast(token_type_ids)
        _broadcast(attention_mask)
        _broadcast(position_ids)

        batch = {
            "images": images,
            "input_ids": input_ids,
            "labels": labels,
            "token_type_ids": token_type_ids,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    return batch
