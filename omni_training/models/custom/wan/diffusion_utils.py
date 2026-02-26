"""Adapted from DiT
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

from omni_training.utils import get_args


def send_batch(batch, broadcast):
    """send batch"""

    args = get_args()
    video_shape = torch.tensor(batch["video"].shape, dtype=torch.int64).cuda(
        non_blocking=True
    )
    broadcast(video_shape)
    args.micro_batch_size = video_shape.tolist()[0]

    broadcast(batch["video"])
    broadcast(batch["video_noised"])
    broadcast(batch["video_mask"])
    broadcast(batch["labels"])
    broadcast(batch["text_enc"])
    broadcast(batch["text_mask"])
    broadcast(batch["timestep"])
    return batch


def receive_batch(broadcast):
    """receive batch"""

    args = get_args()
    device = torch.cuda.current_device()
    video_shape = torch.empty(5, dtype=torch.int64, device=device)
    broadcast(video_shape)
    args.micro_batch_size = video_shape.tolist()[0]

    video = torch.empty(video_shape.tolist(), dtype=torch.float32, device=device)
    video_noised = torch.empty_like(video, dtype=torch.float32, device=device)
    video_mask = torch.empty_like(video, dtype=torch.bool, device=device)
    text_enc = torch.empty(
        (args.micro_batch_size, 1, args.max_text_length, args.caption_channels),
        dtype=torch.float32,
        device=device,
    )
    text_mask = torch.empty(
        (args.micro_batch_size, args.max_text_length), dtype=torch.bool, device=device
    )
    timestep = torch.empty((args.micro_batch_size,), dtype=torch.int64, device=device)
    labels = torch.empty_like(video, dtype=torch.float32, device=device)

    broadcast(video)
    broadcast(video_noised)
    broadcast(video_mask)
    broadcast(labels)
    broadcast(text_enc)
    broadcast(text_mask)
    broadcast(timestep)

    batch = {
        "video": video,
        "video_noised": video_noised,
        "video_mask": video_mask,
        "labels": labels,
        "text_enc": text_enc,
        "text_mask": text_mask,
        "timestep": timestep,
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


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (
        1.0 + torch.tanh(np.sqrt(2.0 / torch.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )


def continuous_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a continuous Gaussian distribution.
    :param x: the targets
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    normalized_x = centered_x * inv_stdv
    log_probs = torch.distributions.Normal(
        torch.zeros_like(x), torch.ones_like(x)
    ).log_prob(normalized_x)
    return log_probs


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
        ),
    )
    assert log_probs.shape == x.shape
    return log_probs
