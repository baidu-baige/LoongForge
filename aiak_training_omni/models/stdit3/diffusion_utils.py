""" Adapted from DiT
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
--------------------------------------------------------
References:
DiT:   https://github.com/facebookresearch/DiT/tree/main
GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
--------------------------------------------------------
"""

import numpy as np
import torch
from megatron.core import mpu
from megatron.training import get_args


def add_noise(batch, diffusion):
    """ add noise before TP communication """
    latent = batch["video"]
    noise = torch.randn_like(latent)
    timestep = diffusion.timestep_transform(latent, {
        "num_frames": batch["num_frames"],
        "height": batch["height"],
        "width": batch["width"],
    })
    batch["video_noised"] = diffusion.add_noise(latent, timestep, noise=noise)
    batch["timestep"] = timestep
    batch["labels"] = noise
    return batch


def send_batch(batch, broadcast):
    """ send batch """

    args = get_args()
    video_shape = torch.tensor(batch['video'].shape, dtype=torch.int64).cuda(non_blocking=True)
    broadcast(video_shape)
    args.micro_batch_size = video_shape.tolist()[0]

    broadcast(batch['video'])
    broadcast(batch['video_noised'])
    broadcast(batch['video_mask'])
    broadcast(batch['labels'])
    broadcast(batch['text_enc'])
    broadcast(batch['text_mask'])
    broadcast(batch['timestep'])
    broadcast(batch['fps'])
    return batch


def receive_batch(broadcast):
    """ receive batch  """

    args = get_args()
    device = torch.cuda.current_device()
    video_shape = torch.empty(5, dtype=torch.int64, device=device)
    broadcast(video_shape)
    args.micro_batch_size = video_shape.tolist()[0]

    video = torch.empty(video_shape.tolist(), dtype=torch.float32, device=device)
    video_noised = torch.empty_like(video, dtype=torch.float32, device=device)
    video_mask = torch.empty_like(video, dtype=torch.bool, device=device)
    text_enc = torch.empty((args.micro_batch_size, 1, args.max_text_length, args.caption_channels),
        dtype=torch.float32, device=device)
    text_mask = torch.empty((args.micro_batch_size, args.max_text_length), dtype=torch.bool, device=device)
    timestep = torch.empty((args.micro_batch_size,), dtype=torch.float32, device=device)
    labels = torch.empty_like(video, dtype=torch.float32, device=device)
    fps = torch.empty((args.micro_batch_size,), dtype=torch.int64, device=device)

    broadcast(video)
    broadcast(video_noised)
    broadcast(video_mask)
    broadcast(labels)
    broadcast(text_enc)
    broadcast(text_mask)
    broadcast(timestep)
    broadcast(fps)

    batch = {
        'video': video,
        'video_noised': video_noised,
        'video_mask': video_mask,
        'labels': labels,
        'text_enc': text_enc,
        'text_mask': text_mask,
        'timestep': timestep,
        'fps': fps
    }

    return batch


def broadcast_on_cp_group(batch):
    """ broadcast_on_cp_group, """

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
    """ get_batch_on_this_tp_rank, """
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