# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Pretrain entry for Qwen-Image-Edit DiT."""

from functools import partial

import numpy as np
import torch
from megatron.core import mpu, parallel_state
from megatron.core.enums import ModelType
from megatron.core.utils import StragglerDetector
from megatron.training import get_timers
from torch.utils.data import DataLoader

from loongforge.data.video.latent_dataset import TensorDataset
from loongforge.models import get_model_provider
from loongforge.train.megatron_trainer import MegatronTrainer
from loongforge.train.trainer_builder import register_model_trainer
from loongforge.utils import get_args, get_model_config, print_rank_0
from loongforge.utils.constants import CustomModelFamilies, TrainingPhase

from loongforge.models.diffusion.qwen_image.qwen_image_flow_match import QwenImageFlowMatchScheduler

SUPPORTED_MODELS = [CustomModelFamilies.QWEN_IMAGE]
stimer = StragglerDetector()


def model_provider(pre_process=True, post_process=True, vp_stage: int = None):
    """Return the Qwen-Image model instance registered with LoongForge."""
    args = get_args()
    if args.context_parallel_size != 1:
        raise AssertionError("Qwen-Image uses FSDP + TP; set context parallel size to 1.")
    args.max_position_embeddings = args.seq_length
    provider = get_model_provider(CustomModelFamilies.QWEN_IMAGE)
    if provider is None:
        raise ValueError("Qwen-Image model provider is not registered")
    return provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


def _scheduler_for_shape(height, width):
    """Build a FlowMatch scheduler configured for the given latent shape."""
    scheduler = QwenImageFlowMatchScheduler()
    scheduler.set_timesteps(1000, training=True)
    return scheduler


def _normalize_latents(latents):
    """Squeeze / unsqueeze latents so they are ``[B, C, H, W]``."""
    if latents.dim() == 5 and latents.size(0) == 1:
        latents = latents.squeeze(0)
    if latents.dim() == 3:
        latents = latents.unsqueeze(0)
    return latents


def gen_time_steps(batch):
    """Sample a timestep, produce noisy latents and the FlowMatch training target."""
    args = get_args()
    latents = _normalize_latents(batch.pop("input_latents")).cuda()
    height = int(batch.get("height", latents.shape[-2] * 8))
    width = int(batch.get("width", latents.shape[-1] * 8))
    scheduler = _scheduler_for_shape(height, width)

    if "noise" in batch:
        noise = _normalize_latents(batch["noise"]).to(device=latents.device, dtype=latents.dtype)
    else:
        seed = int(batch["seed"])
        numpy_random_state = np.random.RandomState(seed=seed)
        noise_np = numpy_random_state.randn(*latents.shape)
        noise = torch.tensor(noise_np, dtype=latents.dtype, device=latents.device)

    max_timestep = args.max_timestep_boundary
    min_timestep = args.min_timestep_boundary
    if not (0 <= min_timestep <= max_timestep <= 1):
        raise ValueError("timestep boundaries must satisfy 0 <= min <= max <= 1")
    max_timestep_boundary = int(max_timestep * scheduler.num_train_timesteps)
    min_timestep_boundary = int(min_timestep * scheduler.num_train_timesteps)

    if "timestep_id" in batch:
        timestep_id = torch.as_tensor(batch["timestep_id"], dtype=torch.long).view(-1)[0]
    elif "timestep" in batch:
        timestep_value = torch.as_tensor(batch["timestep"]).float().view(-1)[0]
        timestep_id = torch.argmin((scheduler.timesteps - timestep_value.cpu()).abs())
    else:
        numpy_random_state = np.random.RandomState(seed=int(batch["seed"]))
        if min_timestep_boundary >= max_timestep_boundary:
            # Boundary check allows min == max (e.g. ablation with a fixed
            # timestep); ``randint`` requires low < high, so short-circuit here.
            timestep_id = torch.tensor(min_timestep_boundary, dtype=torch.long)
        else:
            timestep_id = torch.tensor(
                numpy_random_state.randint(min_timestep_boundary, max_timestep_boundary),
                dtype=torch.long,
            )

    timestep = scheduler.timesteps[timestep_id].to(dtype=latents.dtype, device=latents.device).view(1)
    if "latents" in batch:
        noisy_latents = _normalize_latents(batch["latents"]).to(device=latents.device, dtype=latents.dtype)
    else:
        noisy_latents = scheduler.add_noise(latents, noise, timestep)
    training_target = batch.get("training_target")
    if training_target is None:
        training_target = scheduler.training_target(latents, noise, timestep)
    else:
        training_target = _normalize_latents(training_target).to(device=latents.device, dtype=latents.dtype)
    if "scale" in batch:
        scale = torch.as_tensor(batch["scale"], device=latents.device, dtype=torch.float32).view(1)
    else:
        scale = scheduler.training_weight(timestep).to(device=latents.device, dtype=torch.float32)
    return timestep, noisy_latents, training_target, scale, height, width


def _move_tensor(x, dtype=None):
    """Move ``x`` to CUDA (optionally casting to ``dtype``); non-tensors pass through."""
    if not isinstance(x, torch.Tensor):
        return x
    x = x.cuda()
    return x.to(dtype=dtype) if dtype is not None else x


def _broadcast_tensor(tensor, src_rank, group):
    """Broadcast one CUDA tensor with small Python metadata."""
    is_src = torch.distributed.get_rank() == src_rank
    meta = [(tuple(tensor.shape), str(tensor.dtype).replace("torch.", ""))] if is_src else [None]
    torch.distributed.broadcast_object_list(meta, src=src_rank, group=group)
    shape, dtype_name = meta[0]
    if not is_src:
        tensor = torch.empty(shape, dtype=getattr(torch, dtype_name), device=torch.cuda.current_device())
    torch.distributed.broadcast(tensor, src=src_rank, group=group)
    return tensor


def _broadcast_optional_tensor(tensor, src_rank, group):
    """Broadcast ``tensor`` if not None on the source rank; otherwise return None everywhere."""
    is_src = torch.distributed.get_rank() == src_rank
    has_tensor = tensor is not None if is_src else None
    obj = [has_tensor]
    torch.distributed.broadcast_object_list(obj, src=src_rank, group=group)
    if not obj[0]:
        return None
    return _broadcast_tensor(tensor, src_rank, group)


def _broadcast_edit_latents(edit_latents, src_rank, group):
    """Broadcast an optional list of edit-latent tensors from ``src_rank``."""
    is_src = torch.distributed.get_rank() == src_rank
    if is_src:
        if edit_latents is None:
            meta = [0]
        elif isinstance(edit_latents, list):
            meta = [len(edit_latents)]
        else:
            meta = [1]
            edit_latents = [edit_latents]
    else:
        meta = [None]
        edit_latents = None
    torch.distributed.broadcast_object_list(meta, src=src_rank, group=group)
    count = meta[0]
    if count == 0:
        return None
    result = []
    for idx in range(count):
        local = edit_latents[idx] if is_src else None
        result.append(_broadcast_tensor(local, src_rank, group))
    return result[0] if count == 1 else result


def broadcast_qwen_image_batch_on_tp_group(batch):
    """Broadcast Qwen-Image batch tensors from TP source rank."""
    group = mpu.get_tensor_model_parallel_group()
    src_rank = mpu.get_tensor_model_parallel_src_rank()
    is_src = torch.distributed.get_rank() == src_rank
    meta = [{"height": int(batch["height"]), "width": int(batch["width"])}] if is_src else [None]
    torch.distributed.broadcast_object_list(meta, src=src_rank, group=group)
    if not is_src:
        batch = {"height": meta[0]["height"], "width": meta[0]["width"]}
    for key in ("latents", "timestep", "prompt_emb", "training_target", "scale"):
        batch[key] = _broadcast_tensor(batch.get(key) if is_src else None, src_rank, group)
    batch["prompt_emb_mask"] = _broadcast_optional_tensor(
        batch.get("prompt_emb_mask") if is_src else None, src_rank, group
    )
    batch["edit_latents"] = _broadcast_edit_latents(
        batch.get("edit_latents") if is_src else None, src_rank, group
    )
    return batch


def get_batch(data_iterator):
    """Pull the next batch, broadcast across TP, and return the tensors used by ``forward_step``."""
    args = get_args()
    should_broadcast_batch = data_iterator is not None and mpu.get_tensor_model_parallel_world_size() > 1
    should_load_data = data_iterator is not None and mpu.get_tensor_model_parallel_rank() == 0
    if data_iterator is not None and not should_load_data:
        data_iterator = None

    if should_load_data:
        batch = next(data_iterator)
        (
            batch["timestep"],
            batch["latents"],
            batch["training_target"],
            batch["scale"],
            batch["height"],
            batch["width"],
        ) = gen_time_steps(batch)
        if "prompt_emb" not in batch and "context" in batch:
            batch["prompt_emb"] = batch.pop("context")
    else:
        batch = None

    if batch:
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                target_dtype = batch["latents"].dtype if key not in {"height", "width", "seed"} else None
                batch[key] = _move_tensor(value, dtype=target_dtype)
            elif isinstance(value, list):
                batch[key] = [
                    _move_tensor(v, dtype=batch["latents"].dtype) if isinstance(v, torch.Tensor) else v
                    for v in value
                ]

    if should_broadcast_batch:
        batch = broadcast_qwen_image_batch_on_tp_group(batch)

    if batch is None:
        return None, None, None, None, None, None, None, None, None

    prompt_emb = batch["prompt_emb"]
    if prompt_emb.dim() == 4 and prompt_emb.size(0) == 1:
        prompt_emb = prompt_emb[0]
    if prompt_emb.dim() == 2:
        prompt_emb = prompt_emb.unsqueeze(0)
    prompt_emb_mask = batch.get("prompt_emb_mask")
    if prompt_emb_mask is not None and prompt_emb_mask.dim() == 3 and prompt_emb_mask.size(0) == 1:
        prompt_emb_mask = prompt_emb_mask[0]
    if prompt_emb_mask is not None and prompt_emb_mask.dim() == 1:
        prompt_emb_mask = prompt_emb_mask.unsqueeze(0)

    return (
        batch["latents"],
        batch["timestep"],
        prompt_emb.to(dtype=batch["latents"].dtype),
        prompt_emb_mask,
        batch.get("edit_latents"),
        batch["training_target"],
        batch["scale"],
        int(batch["height"]),
        int(batch["width"]),
    )


def loss_func(training_target, scale, noise_pred):
    """MSE loss between the DiT noise prediction and the FlowMatch target."""
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * scale
    dp_group = parallel_state.get_data_parallel_group()
    averaged_losses = torch.cat([loss.clone().detach().view(1)])
    torch.distributed.all_reduce(averaged_losses, group=dp_group)
    averaged_losses = averaged_losses / dp_group.size()
    return loss, {"lm loss": averaged_losses[0]}


def forward_step(diffusion, data_iterator, model):
    """Fetch a batch and run the DiT forward, returning ``(noise_pred, loss_partial)``."""
    timers = get_timers()
    timers("batch-generator", log_level=2).start()
    with stimer(bdata=True):
        (
            latents,
            timestep,
            prompt_emb,
            prompt_emb_mask,
            edit_latents,
            training_target,
            scale,
            height,
            width,
        ) = get_batch(data_iterator)
    timers("batch-generator").stop()
    with stimer:
        noise_pred = model(
            latents,
            timestep,
            prompt_emb,
            prompt_emb_mask=prompt_emb_mask,
            height=height,
            width=width,
            edit_latents=edit_latents,
            zero_cond_t=get_model_config().qwen_image_zero_cond_t,
        )
    return noise_pred, partial(loss_func, training_target, scale)


def train_valid_test_datasets_provider(diffusion, train_val_test_num_samples, vp_stage=None):
    """Build the train dataloader (TensorDataset over pre-cached Qwen-Image latents)."""
    args = get_args()
    keep_keys = {
        "input_latents",
        "latents",
        "edit_latents",
        "prompt_emb",
        "prompt_emb_mask",
        "context",
        "height",
        "width",
        "noise",
        "timestep",
        "timestep_id",
        "training_target",
        "scale",
        "seed",
    }
    dataset = TensorDataset(
        args.data_path[0],
        args.train_iters * args.global_batch_size,
        seed=args.seed,
        keep_keys=keep_keys,
    )
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        shuffle=False,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, sampler=sampler, pin_memory=True)
    print_rank_0(f"> finished creating {args.model_name} datasets ...")
    return iter(dataloader), None, None


@register_model_trainer(model_family=SUPPORTED_MODELS, training_phase=TrainingPhase.PRETRAIN)
def default_pretrain_trainer(train_args):
    """Register the default Qwen-Image pretrain trainer with MegatronTrainer."""
    diffusion = None
    return MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=partial(train_valid_test_datasets_provider, diffusion),
        model_provider=model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=partial(forward_step, diffusion),
    )
