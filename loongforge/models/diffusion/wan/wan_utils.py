# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""wan utils"""

import torch
from megatron.core import mpu
from loongforge.utils import get_args
from einops import rearrange
from megatron.core.parallel_state import get_context_parallel_group
from .communications import (
    split_forward_gather_backward,
    gather_forward_split_backward,
)


def wan_rope_apply(x, freqs, config, cu_seqlens=None, rotary_interleaved=False):
    """wan rope apply"""
    total_heads = config.num_attention_heads
    heads = x.shape[2]

    if config.context_parallel_size > 1:
        x = gather_forward_split_backward(
            x, get_context_parallel_group(), dim=0, grad_scale="up"
        )
    x = rearrange(x, "s b n d -> b s n d")

    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    if config.context_parallel_size > 1:
        x_out = split_forward_gather_backward(
            x_out, get_context_parallel_group(), dim=1, grad_scale="down"
        )
    x_out = rearrange(x_out, "b s (n d) -> s b n d", n=heads)

    return x_out.to(x.dtype).contiguous()


def send_batch(batch, broadcast):
    """send batch"""
    args = get_args()
    video_shape = torch.tensor(batch["latents"].shape, dtype=torch.int64).cuda(
        non_blocking=True
    )
    contxt_shape = torch.tensor(
        batch["prompt_emb"]["context"].shape, dtype=torch.int64
    ).cuda(non_blocking=True)
    broadcast(video_shape)
    broadcast(batch["latents"])
    broadcast(batch["training_target"])
    broadcast(batch["timestep"])
    broadcast(batch["scale"])

    broadcast(contxt_shape)
    broadcast(batch["prompt_emb"]["context"])

    image_emb = batch["image_emb"]
    image_info = torch.tensor([0, 0], dtype=torch.int64).cuda(non_blocking=True)

    if "clip_feature" in image_emb:
        image_info[0] = 1
    if "y" in image_emb:
        image_info[1] = 1

    broadcast(image_info)
    if image_info[0] == 1:
        clip_feature_shape = torch.tensor(
            image_emb["clip_feature"].shape, dtype=torch.int64
        ).cuda(non_blocking=True)
        broadcast(clip_feature_shape)
        broadcast(image_emb["clip_feature"])
    if image_info[1] == 1:
        y_shape = torch.tensor(image_emb["y"].shape, dtype=torch.int64).cuda(
            non_blocking=True
        )
        broadcast(y_shape)
        broadcast(image_emb["y"])

    args.micro_batch_size = video_shape.tolist()[0]

    return batch


def receive_batch(broadcast):
    """receive batch"""
    args = get_args()
    device = torch.cuda.current_device()
    # receive video
    video_shape = torch.empty(5, dtype=torch.int64, device=device)
    broadcast(video_shape)
    args.micro_batch_size = video_shape.tolist()[0]
    video = torch.empty(video_shape.tolist(), dtype=torch.bfloat16, device=device)
    training_target = torch.empty(
        video_shape.tolist(), dtype=torch.bfloat16, device=device
    )
    broadcast(video)
    broadcast(training_target)

    # receive timestep
    timestep = torch.empty([1], dtype=torch.bfloat16, device=device)
    broadcast(timestep)

    # scale
    scale = torch.empty([1], dtype=torch.float32, device=device)
    broadcast(scale)

    # receive context
    prompt_shape = torch.empty(4, dtype=torch.int64, device=device)
    broadcast(prompt_shape)
    prompt = torch.empty(prompt_shape.tolist(), dtype=torch.bfloat16, device=device)
    broadcast(prompt)
    # receive image
    image_info = torch.empty(2, dtype=torch.int64, device=device)
    broadcast(image_info)
    image_emb = {}
    if image_info[0] == 1:
        clip_shape = torch.empty(4, dtype=torch.int64, device=device)
        broadcast(clip_shape)
        clip_feature = torch.empty(
            clip_shape.tolist(), dtype=torch.bfloat16, device=device
        )
        broadcast(clip_feature)
        image_emb["clip_feature"] = clip_feature

    if image_info[1] == 1:
        y_shape = torch.empty(6, dtype=torch.int64, device=device)
        broadcast(y_shape)
        y = torch.empty(y_shape.tolist(), dtype=torch.bfloat16, device=device)
        broadcast(y)
        image_emb["y"] = y

    prompt_emb = {"context": prompt}
    batch = {
        "latents": video,
        "training_target": training_target,
        "timestep": timestep,
        "prompt_emb": prompt_emb,
        "image_emb": image_emb,
        "scale": scale,
    }

    return batch


def broadcast_on_cp_group(batch):
    """broadcast_on_cp_group,"""

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                mpu.get_context_parallel_src_rank(),
                group=mpu.get_context_parallel_group(),
            )

    if mpu.get_context_parallel_rank() == 0:
        return send_batch(batch, _broadcast)
    else:
        return receive_batch(_broadcast)


def broadcast_on_tp_group(batch):
    """get_batch_on_this_tp_rank,"""

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                mpu.get_tensor_model_parallel_src_rank(),
                group=mpu.get_tensor_model_parallel_group(),
            )

    if mpu.get_tensor_model_parallel_rank() == 0:
        return send_batch(batch, _broadcast)
    else:
        return receive_batch(_broadcast)
