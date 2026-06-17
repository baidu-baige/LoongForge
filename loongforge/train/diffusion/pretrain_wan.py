# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""default pretrain for video diffusion model"""

import os
import torch
from functools import partial

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.utils import StragglerDetector


from megatron.training import get_timers
from megatron.training.utils import average_losses_across_data_parallel_group

from loongforge.utils import get_args, print_rank_0
from loongforge.utils.constants import TrainingPhase, CustomModelFamilies

from loongforge.data.video.latent_dataset import TensorDataset


from loongforge.data.video.packed_dataset import (
    PackedDataset,
    _patchify_for_conv3d,
    _patchify_latent,
)

from loongforge.models import get_model_provider, get_model_family
from loongforge.models.diffusion.wan.wan_flow_match import FlowMatchScheduler
from megatron.core.packed_seq_params import PackedSeqParams

from loongforge.train.megatron_trainer import MegatronTrainer
from loongforge.train.trainer_builder import register_model_trainer
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Subset
from megatron.core import parallel_state
import numpy as np
import math

from loongforge.models.diffusion.wan.gaussian_diffusion import (
    ModelMeanType,
    ModelVarType,
    LossType,
    GaussianDiffusion,
    get_named_beta_schedule,
)

from loongforge.models.diffusion.wan.wan_utils import (
    broadcast_on_tp_group,
    broadcast_on_cp_group,
)
from loongforge.models.diffusion.wan.wan_provider import wan_i2v_model_provider

SUPPORTED_MODELS = [
    CustomModelFamilies.WAN2_1_I2V,
    CustomModelFamilies.WAN2_2_I2V,
]
WAN_PATCH_SIZE = (1, 2, 2)


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
    assert args.tensor_model_parallel_size == 1, (
        "WAN model only supports TP=1. "
        "WanCrossAttention.norm_k uses global num_attention_heads for reshape, "
        "which produces incorrect results when TP>1."
    )
    args.max_position_embeddings = args.seq_length

    if args.model_name in ("wan2-1-i2v", "wan2-2-i2v") or args.model_family in SUPPORTED_MODELS:
        return wan_i2v_model_provider(pre_process, post_process, vp_stage)

    raise ValueError(f"Unsupported WAN model: {args.model_name}")


scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
scheduler.set_timesteps(1000, training=True)


def gen_time_steps(batch):
    """
    Generate time sequence.

    Args:
        latents (torch.Tensor): Input latent variables, shape (batch_size, dim).

    Returns:
        tuple: A tuple containing three elements: time sequence index, latents with noise added, and training target.

        - timestep (torch.Tensor): Time sequence index, shape (1,).
        - noisy_latents (torch.Tensor): Latents with noise added, same shape as input.
        - training_target (torch.Tensor): Training target, same shape as input.

    """
    # torch.manual_seed(10086)
    args = get_args()
    if args.model_name in ("wan2-1-i2v", "wan2-2-i2v") or args.model_family in SUPPORTED_MODELS:
        latents = batch.pop("input_latents")
        if latents.size(0) == 1:
            latents = latents.squeeze(0)
    seed = batch["seed"]
    max_timestep = args.max_timestep_boundary
    min_timestep = args.min_timestep_boundary
    assert max_timestep <= 1 and max_timestep >= 0, \
        "max_timestep should range from 0 to 1"
    assert min_timestep <= 1 and min_timestep >= 0, \
        "min_timestep should range from 0 to 1"
    assert min_timestep <= max_timestep, \
        f"min_timestep: {min_timestep} should <= max_timestep: {max_timestep}"
    max_timestep_boundary = int(max_timestep * scheduler.num_train_timesteps)
    min_timestep_boundary = int(min_timestep * scheduler.num_train_timesteps)

    device = torch.device("cuda")
    latents = latents.to(device=device)

    numpy_random_state = np.random.RandomState(seed=seed)
    noise_np = numpy_random_state.randn(*latents.shape)
    noise = torch.tensor(noise_np, dtype=latents.dtype, device=device)
    rand_int = numpy_random_state.randint(min_timestep_boundary, max_timestep_boundary)
    timestep_id = torch.tensor([rand_int])

    timestep = scheduler.timesteps[timestep_id].to(dtype=latents.dtype, device=device)
    noisy_latents = scheduler.add_noise(latents, noise, timestep)
    training_target = scheduler.training_target(latents, noise, timestep)
    scale = scheduler.training_weight(timestep)
    return timestep, noisy_latents, training_target, scale



def _build_packed_seq_params(batch):
    """Build separate PackedSeqParams for self-attention and cross-attention."""
    device = batch["seq_len_q"].device
    zero = torch.zeros(1, dtype=torch.int32, device=device)

    cu_seqlens_q_padded = torch.cat([zero, batch["seq_len_q_padded"].cumsum(0).to(torch.int32)])
    cu_seqlens_kv_padded = torch.cat([zero, batch["seq_len_kv_padded"].cumsum(0).to(torch.int32)])

    if batch["seq_len_q_padded"].sum() == batch["loss_mask"].shape[0]:
        cu_seqlens_q = cu_seqlens_q_padded
    else:
        cu_seqlens_q = torch.cat([zero, batch["seq_len_q"].cumsum(0).to(torch.int32)])

    if batch["seq_len_kv_padded"].sum() == batch["context"].shape[0]:
        cu_seqlens_kv = cu_seqlens_kv_padded
    else:
        cu_seqlens_kv = torch.cat([zero, batch["seq_len_kv"].cumsum(0).to(torch.int32)])

    max_seqlen_q = batch["seq_len_q_padded"].max().item()
    max_seqlen_kv = batch["seq_len_kv_padded"].max().item()

    self_attn_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_q_padded=cu_seqlens_q_padded,
        cu_seqlens_kv=cu_seqlens_q,
        cu_seqlens_kv_padded=cu_seqlens_q_padded,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_q,
    )
    cross_attn_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_q_padded=cu_seqlens_q_padded,
        cu_seqlens_kv=cu_seqlens_kv,
        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
    )

    self_attn_params._seq_len_q = batch["seq_len_q"].clone()
    self_attn_params._seq_len_q_padded = batch["seq_len_q_padded"].clone()
    cross_attn_params._seq_len_kv = batch["seq_len_kv"].clone()
    cross_attn_params._seq_len_kv_padded = batch["seq_len_kv_padded"].clone()

    return {
        "self_attention": self_attn_params,
        "cross_attention": cross_attn_params,
    }


def get_batch(data_iterator):
    """Generate a batch."""
    args = get_args()
    use_packing = getattr(args, 'packing_sft_data', False)

    should_broadcast_batch = False
    if use_packing:
        should_load_data = data_iterator is not None
    else:
        should_broadcast_batch = data_iterator is not None and mpu.get_context_parallel_world_size() > 1
        cp_src_rank = mpu.get_context_parallel_src_rank()
        should_load_data = data_iterator is not None and torch.distributed.get_rank() == cp_src_rank
        if data_iterator is not None and not should_load_data:
            data_iterator = None

    if should_load_data:
        batch = next(data_iterator)
        packed_seq_params = None
        grid_sizes = None

        if not use_packing:
            batch["timestep"], batch["latents"], batch["training_target"], \
                batch["scale"] = gen_time_steps(batch)
            if args.model_name in ("wan2-1-i2v", "wan2-2-i2v") or args.model_family in SUPPORTED_MODELS:
                batch.setdefault("prompt_emb", {})["context"] = batch.pop("context")
                image_emb = batch.setdefault("image_emb", {})
                if "y" in batch:
                    image_emb["y"] = batch.pop("y")
                if "clip_feature" in batch:
                    image_emb["clip_feature"] = batch.pop("clip_feature")
        else:
            packed_seq_params = _build_packed_seq_params(batch)
            grid_sizes = batch.get("grid_sizes")
            if "context" in batch:
                batch["prompt_emb"] = {"context": batch.pop("context")}
            if "image_emb" not in batch:
                batch["image_emb"] = {}
    else:
        batch = None
        packed_seq_params = None
        grid_sizes = None

    if batch:
        def move_to_device(x):
            if x is not None and isinstance(x, torch.Tensor):
                return x.cuda()
            return x

        for key, val in batch.items():
            if not isinstance(val, dict):
                batch[key] = move_to_device(val)
            else:
                for k, v in val.items():
                    batch[key][k] = move_to_device(v)

        if grid_sizes is not None:
            grid_sizes = grid_sizes.cuda()
        if packed_seq_params is not None:
            for attention_name, params in packed_seq_params.items():
                device_params = PackedSeqParams(
                    qkv_format=params.qkv_format,
                    cu_seqlens_q=params.cu_seqlens_q.cuda(),
                    cu_seqlens_q_padded=params.cu_seqlens_q_padded.cuda(),
                    cu_seqlens_kv=params.cu_seqlens_kv.cuda(),
                    cu_seqlens_kv_padded=params.cu_seqlens_kv_padded.cuda(),
                    max_seqlen_q=params.max_seqlen_q,
                    max_seqlen_kv=params.max_seqlen_kv,
                )
                if hasattr(params, '_seq_len_q'):
                    device_params._seq_len_q = params._seq_len_q.cuda()
                if hasattr(params, '_seq_len_q_padded'):
                    device_params._seq_len_q_padded = params._seq_len_q_padded.cuda()
                if hasattr(params, '_seq_len_kv'):
                    device_params._seq_len_kv = params._seq_len_kv.cuda()
                if hasattr(params, '_seq_len_kv_padded'):
                    device_params._seq_len_kv_padded = params._seq_len_kv_padded.cuda()
                packed_seq_params[attention_name] = device_params

        if use_packing and "input_latents_raw" in batch:
            input_latents_raw_list = batch.pop("input_latents_raw")
            y_raw_list = batch.pop("y_raw")
            noise_raw_list = batch.pop("noise_raw")
            timestep_id = batch.pop("timestep_id")

            timestep = scheduler.timesteps[timestep_id.cpu()].to(
                dtype=torch.bfloat16, device='cuda')

            latents_patched_list = []
            target_patched_list = []
            seq_len_q_padded = batch["seq_len_q_padded"]
            for sample_index, input_latents_raw in enumerate(input_latents_raw_list):
                input_latents = input_latents_raw.cuda()
                noise = noise_raw_list[sample_index].cuda()
                sample_timestep = timestep[sample_index:sample_index + 1]

                noisy_latents_without_y = scheduler.add_noise(input_latents, noise, sample_timestep)
                training_target_raw = scheduler.training_target(input_latents, noise, sample_timestep)

                y_raw = y_raw_list[sample_index]
                if y_raw is not None:
                    y_cuda = y_raw.to(device=input_latents.device, dtype=input_latents.dtype)
                    noisy_latents_with_y = torch.cat([noisy_latents_without_y, y_cuda], dim=0)
                else:
                    noisy_latents_with_y = noisy_latents_without_y

                latent_patched = _patchify_for_conv3d(noisy_latents_with_y, WAN_PATCH_SIZE)
                target_patched = _patchify_latent(training_target_raw, WAN_PATCH_SIZE)
                padded_len = seq_len_q_padded[sample_index].item()
                pad_q = int(padded_len - latent_patched.shape[0])
                if pad_q > 0:
                    latent_patched = torch.nn.functional.pad(latent_patched, (0, 0, 0, pad_q))
                    target_patched = torch.nn.functional.pad(target_patched, (0, 0, 0, pad_q))
                latents_patched_list.append(latent_patched)
                target_patched_list.append(target_patched)

            noisy_latents = torch.cat(latents_patched_list, dim=0)
            training_target = torch.cat(target_patched_list, dim=0)

            batch["latents"] = noisy_latents.unsqueeze(1)
            batch["training_target"] = training_target.unsqueeze(1)
            batch["timestep"] = timestep

    if not use_packing and should_broadcast_batch:
        batch = broadcast_on_cp_group(batch)

    if batch is None:
        return None, None, None, None, None, None, None, None, None, None, None

    video = batch["latents"]
    training_target = batch["training_target"]
    timestep = batch["timestep"]
    text = batch.get("prompt_emb", {}).get("context", batch.get("context"))
    scale = batch["scale"]
    image_emb = batch.get("image_emb", {})
    if "clip_feature" in image_emb:
        clip_feature = image_emb["clip_feature"]
        if clip_feature.dim() == 4 and clip_feature.size(0) == 1:
            clip_feature = clip_feature[0]
        image_emb["clip_feature"] = clip_feature.cuda().to(dtype=video.dtype)
    if "y" in image_emb:
        y = image_emb["y"]
        if y.dim() == 6 and y.size(0) == 1:
            y = y[0]
        image_emb["y"] = y.cuda().to(dtype=video.dtype)
    if text is not None:
        text = text.to(dtype=video.dtype)

    loss_mask = batch.get("loss_mask", None)
    seq_len_q_padded = batch.get("seq_len_q_padded", None)
    seq_len_q = batch.get("seq_len_q", None)

    return (
        video, timestep, text, image_emb, training_target, scale,
        packed_seq_params, grid_sizes, loss_mask, seq_len_q_padded, seq_len_q,
    )


def gaussian_diffusion():
    """Build a diffusion."""
    betas = get_named_beta_schedule("linear", 1000)
    model_mean_type = ModelMeanType.EPSILON
    model_var_type = ModelVarType.LEARNED_RANGE
    loss_type = LossType.MSE
    return GaussianDiffusion(
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type,
        device="cpu",
    )


def loss_func(training_target, timestep, scale, loss_mask, seq_len_q_padded, seq_len_q, noise_pred):
    """Compute the loss."""
    if loss_mask is not None:
        # Per-token MSE with loss_mask, scaled per-sample
        diff = (noise_pred.float() - training_target.float()) ** 2
        num_samples = scale.shape[0]

        if num_samples == 1:
            loss = (diff * loss_mask.unsqueeze(-1)).sum() / (loss_mask.sum() * diff.shape[-1] + 1e-8)
            loss = loss * scale[0]
        else:
            offsets = torch.cat([
                torch.zeros(1, dtype=seq_len_q_padded.dtype, device=seq_len_q_padded.device),
                seq_len_q_padded.cumsum(0),
            ])
            sample_losses = []
            for i in range(num_samples):
                start = int(offsets[i].item())
                end = int(offsets[i + 1].item())
                sample_diff = diff[start:end]
                sample_mask = loss_mask[start:end]
                sample_mse = (sample_diff * sample_mask.unsqueeze(-1)).sum() / (
                    sample_mask.sum() * sample_diff.shape[-1] + 1e-8
                )
                sample_losses.append(sample_mse * scale[i])
            loss = sum(sample_losses) / num_samples

        dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        averaged_losses = torch.cat([loss.clone().detach().view(1)])
        torch.distributed.all_reduce(averaged_losses, group=dp_cp_group)
        averaged_losses = averaged_losses / dp_cp_group.size()
        return loss, {"lm loss": averaged_losses[0]}
    else:
        # Standard MSE loss
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * scale
        dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        averaged_losses = torch.cat([loss.clone().detach().view(1)])
        torch.distributed.all_reduce(averaged_losses, group=dp_cp_group)
        averaged_losses = averaged_losses / dp_cp_group.size()
        return loss, {"lm loss": averaged_losses[0]}


def forward_step(diffusion, data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: Megatron Model
    """
    timers = get_timers()
    args = get_args()
    use_packing = getattr(args, 'packing_sft_data', False)

    # Get the batch.
    timers("batch-generator", log_level=2).start()

    global stimer
    with stimer(bdata=True):
        noisy_latents, timestep, text_enc, image_emb, training_target, scale, \
            packed_seq_params, grid_sizes, loss_mask, seq_len_q_padded, seq_len_q = get_batch(data_iterator)
    timers("batch-generator").stop()

    extra_input = {}

    with stimer:
        if use_packing:
            t_enc = text_enc if text_enc is not None else None
            image_emb = image_emb if image_emb is not None else {}
            noise_pred = model(
                noisy_latents,
                timestep,
                t_enc,
                **extra_input,
                **image_emb,
                packed_seq_params=packed_seq_params,
                grid_sizes=grid_sizes,
                use_gradient_checkpointing=True,
                use_gradient_checkpointing_offload=False,
            )
        else:
            t_enc = text_enc[0] if text_enc is not None else None
            image_emb = image_emb if image_emb is not None else {}
            noise_pred = model(
                noisy_latents,
                timestep,
                t_enc,
                **extra_input,
                **image_emb,
                use_gradient_checkpointing=True,
                use_gradient_checkpointing_offload=False,
            )
    return noise_pred, partial(loss_func, training_target, timestep, scale, loss_mask, seq_len_q_padded, seq_len_q)


def train_valid_test_datasets_provider(diffusion, train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets."""
    args = get_args()
    use_packing = getattr(args, 'packing_sft_data', False)

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_world_size = parallel_state.get_data_parallel_world_size()

    if use_packing:
        packing_buffer_size = getattr(args, 'packing_buffer_size', 512)
        seq_length = args.seq_length

        # steps_per_epoch controls how many packed bins PackedDataset produces
        # per epoch.  It is set large enough so the trainer never exhausts the
        # iterator before train_iters steps are done.
        steps_per_epoch = args.train_iters * args.global_batch_size

        dataset = PackedDataset(
            data_path=args.data_path[0],
            steps_per_epoch=steps_per_epoch,
            args=args,
            scheduler=scheduler,
            packing_buffer_size=packing_buffer_size,
            seq_length=seq_length,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
        # batch_size=None: DataLoader yields each item as-is (no batching /
        # collation).  PackedDataset already produces fully-merged packed
        # batches so no further assembly is needed.
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=True,
        )
    else:
        keep_keys = None
        if getattr(args, "model_name", None) in ("wan2-1-i2v", "wan2-2-i2v"):
            keep_keys = {
                "context", "input_latents", "y", "clip_feature",
                "height", "width", "num_frames",
                "max_timestep_boundary", "min_timestep_boundary",
            }
        dataset = TensorDataset(
            args.data_path[0],
            args.train_iters * args.global_batch_size,
            seed=args.seed,
            keep_keys=keep_keys,
        )
        sampler = torch.utils.data.DistributedSampler(
            dataset, shuffle=False, num_replicas=dp_world_size, rank=dp_rank
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=args.num_workers,
            sampler=sampler,
            pin_memory=True,
        )

    print_rank_0(f"> finished creating {args.model_name} datasets ...")

    return iter(dataloader), None, None


# Set random number seed
@register_model_trainer(
    model_family=SUPPORTED_MODELS, training_phase=TrainingPhase.PRETRAIN
)
def default_pretrain_trainer(train_args):
    """build trainer"""
    diffusion = gaussian_diffusion()
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=partial(
            train_valid_test_datasets_provider, diffusion
        ),
        model_provider=model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=partial(forward_step, diffusion),
    )

    return trainer
