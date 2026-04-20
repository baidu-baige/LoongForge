# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""default pretrain for video diffusion model"""

import torch
import os
from functools import partial

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.utils import StragglerDetector

from megatron.training import get_timers
from megatron.training.utils import average_losses_across_data_parallel_group

from loongforge.utils import get_args, print_rank_0
from loongforge.utils.constants import TrainingPhase, CustomModelFamilies

from loongforge.data.video.latent_dataset import TensorDataset

from loongforge.models import get_model_provider, get_model_family
from loongforge.models.diffusion.wan.wan_flow_match import FlowMatchScheduler

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
from loongforge.models.diffusion.wan.wan_provider import wan2_2_i2v_model_provider

SUPPORTED_MODELS = [
    CustomModelFamilies.WAN2_2_I2V
]


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
    cp = args.context_parallel_ulysses_degree
    text_length = math.ceil(args.max_text_length / cp) * cp
    if args.model_name == "wan2-2-i2v":
        args.seq_length = args.max_video_length + text_length + (6 + 1) * cp
    args.max_position_embeddings = args.seq_length
    print_rank_0(f"> calculated seq_length:  {args.seq_length}")

    if args.model_name == "wan2-2-i2v":
        model_provider = wan2_2_i2v_model_provider

    return model_provider(pre_process, post_process, vp_stage)


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
    if args.model_name == "wan2-2-i2v":
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


def get_batch(data_iterator):
    """Generate a batch"""
    # TODO: this is pretty hacky, find a better way
    args = get_args()
    if data_iterator is not None and mpu.get_context_parallel_rank() == 0:
        batch = next(data_iterator)
        batch["timestep"], batch["latents"], batch["training_target"], \
            batch["scale"] = gen_time_steps(batch)
        if args.model_name == "wan2-2-i2v":
            batch.setdefault("prompt_emb", {})["context"] = batch.pop("context")
            batch.setdefault("image_emb", {})["y"] = batch.pop("y")
    else:
        batch = None

    # get batches on TP0 and CP0 only
    def move_to_device(x):
        if x is not None and isinstance(x, torch.Tensor):
            return x.cuda()

    if batch:
        # batch["timestep"] = batch["timestep"][0]
        for key, val in batch.items():
            if not isinstance(val, dict):
                batch[key] = move_to_device(val)
            else:
                for k, v in val.items():
                    batch[key][k] = move_to_device(v)

    batch = broadcast_on_cp_group(batch)
    video = batch["latents"]  # B in T H W
    training_target = batch["training_target"]
    timestep = batch["timestep"]
    text = batch["prompt_emb"]["context"]  # B in T H W
    scale = batch["scale"]

    image_emb = batch["image_emb"]
    if "clip_feature" in image_emb:
        image_emb["clip_feature"] = image_emb["clip_feature"][0].cuda()
    if "y" in image_emb:
        image_emb["y"] = image_emb["y"][0].cuda()
    return video, timestep, text, image_emb, training_target, scale

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


def loss_func(training_target, timestep, scale, noise_pred):
    """Compute the loss."""
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    # loss = loss * scheduler.training_weight(timestep)
    loss = loss * scale
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {"lm loss": averaged_loss[0]}


def forward_step(diffusion, data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: Megatron Model
    """
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()

    global stimer
    with stimer(bdata=True):
        noisy_latents, timestep, text_enc, image_emb, training_target, scale = (
            get_batch(data_iterator)
        )
    timers("batch-generator").stop()

    extra_input = {}

    with stimer:
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
    return noise_pred, partial(loss_func, training_target, timestep, scale)


def train_valid_test_datasets_provider(diffusion, train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets."""
    args = get_args()

    dataset = TensorDataset(
        args.data_path[0], args.train_iters * args.global_batch_size
    )

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_world_size = parallel_state.get_data_parallel_world_size()

    sampler = torch.utils.data.DistributedSampler(
        dataset, shuffle=False, num_replicas=dp_world_size, rank=dp_rank
    )
    # TODO: Batched inference is not supported yet.
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
