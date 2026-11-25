# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Runtime helpers shared by the MCore training adapters."""

import torch
from megatron.core import mpu
from megatron.core.rerun_state_machine import ChunkDataIterator


def get_next_batch_on_this_tp_rank(data_iterator):
    """Fetch and broadcast the next ChunkPipe chunk for MTP on the last stage."""
    from loongforge.utils import get_args

    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                mpu.get_tensor_model_parallel_src_rank(),
                group=mpu.get_tensor_model_parallel_group(),
            )

    if mpu.get_tensor_model_parallel_rank() == 0:
        is_none = not isinstance(data_iterator, ChunkDataIterator)
        if not is_none:
            data = data_iterator.get_next_chunk()
            is_none = data is None

        item_exist = torch.tensor(1 if is_none else 0, dtype=torch.int32, device="cuda")
        _broadcast(item_exist)
        if is_none:
            return None

        if args.enable_chunkpipe and getattr(args, "variable_seq_lengths", False):
            seq_len_t = torch.tensor(
                [data["tokens"].size(1)], dtype=torch.int64, device="cuda"
            )
            _broadcast(seq_len_t)

        fake_mask = torch.tensor(1, dtype=torch.int32, device="cuda")
        batch = {
            "tokens": data["tokens"].cuda(non_blocking=True),
            "labels": data["labels"].cuda(non_blocking=True),
            "loss_mask": data["loss_mask"].cuda(non_blocking=True),
            "attention_mask": fake_mask,
            "position_ids": data["position_ids"].cuda(non_blocking=True),
        }

        if args.pipeline_model_parallel_size == 1:
            for key in ("tokens", "labels", "loss_mask", "attention_mask", "position_ids"):
                _broadcast(batch[key])
        elif mpu.is_pipeline_first_stage():
            for key in ("tokens", "attention_mask", "position_ids"):
                _broadcast(batch[key])
        elif mpu.is_pipeline_last_stage():
            if (getattr(args, "mtp_num_layers", 0) or 0) > 0 and (
                mpu.get_tensor_model_parallel_world_size() > 1
            ):
                _broadcast(batch["tokens"])
                _broadcast(batch["position_ids"])
            for key in ("labels", "loss_mask", "attention_mask"):
                _broadcast(batch[key])

        return batch

    item_exist = torch.tensor(0, dtype=torch.int32, device="cuda")
    _broadcast(item_exist)
    if item_exist.item() == 1:
        return None

    tmp_seq_length = args.seq_length
    if args.enable_chunkpipe:
        tmp_seq_length = args.chunksize
        if getattr(args, "variable_seq_lengths", False):
            seq_len_t = torch.empty((1,), dtype=torch.int64, device="cuda")
            _broadcast(seq_len_t)
            tmp_seq_length = int(seq_len_t.item())

    tokens = torch.empty(
        (args.micro_batch_size, tmp_seq_length),
        dtype=torch.int64,
        device=torch.cuda.current_device(),
    )
    labels = torch.empty(
        (args.micro_batch_size, tmp_seq_length),
        dtype=torch.int64,
        device=torch.cuda.current_device(),
    )
    loss_mask = torch.empty(
        (args.micro_batch_size, tmp_seq_length),
        dtype=torch.float32,
        device=torch.cuda.current_device(),
    )
    attention_mask = torch.tensor(0, dtype=torch.int32, device="cuda")
    position_ids = torch.empty(
        (args.micro_batch_size, tmp_seq_length),
        dtype=torch.int64,
        device=torch.cuda.current_device(),
    )

    if args.pipeline_model_parallel_size == 1:
        for tensor in (tokens, labels, loss_mask, attention_mask, position_ids):
            _broadcast(tensor)
    elif mpu.is_pipeline_first_stage():
        labels = None
        loss_mask = None
        for tensor in (tokens, attention_mask, position_ids):
            _broadcast(tensor)
    elif mpu.is_pipeline_last_stage():
        if (getattr(args, "mtp_num_layers", 0) or 0) > 0 and (
            mpu.get_tensor_model_parallel_world_size() > 1
        ):
            _broadcast(tokens)
            _broadcast(position_ids)
        else:
            tokens = None
            position_ids = None
        for tensor in (labels, loss_mask, attention_mask):
            _broadcast(tensor)

    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
