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
import inspect
from aiak_training_omni.utils import get_args
from einops import rearrange
from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_world_size,
)
from aiak_training_omni.models.stdit.communications import (
    split_forward_gather_backward,
    gather_forward_split_backward,
)


def wan_rope_apply(x, freqs, config, cu_seqlens=None, rotary_interleaved=False):
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


def dump(x, name="tensor.txt", line_number=inspect.currentframe().f_lineno, pp=2, cp=1):
    """dump tensor to file"""
    tensor_numpy = x.detach().cpu().float().reshape(-1).numpy()
    np.savetxt(
        f"{name}_line_{line_number}.txt_{pp}_{cp}_{torch.cuda.current_device()}",
        tensor_numpy,
        fmt="%.4f",
    )
