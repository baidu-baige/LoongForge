"""default pretrain for video diffusion model"""

import torch

from functools import partial

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.utils import StragglerDetector

from megatron.training import get_timers
from megatron.training.utils import average_losses_across_data_parallel_group

from aiak_training_omni.utils import get_args, print_rank_0
from aiak_training_omni.utils.constants import TrainingPhase, VideoLanguageModelFamilies

from aiak_training_omni.data.video.latent_dataset import LatentDatasetFromCSV

from aiak_training_omni.models import get_model_provider, get_model_family

from aiak_training_omni.train.megatron_trainer import MegatronTrainer
from aiak_training_omni.train.trainer_builder import register_model_trainer

from aiak_training_omni.models.stdit.gaussian_diffusion import (
    ModelMeanType,
    ModelVarType,
    LossType,
    GaussianDiffusion,
    get_named_beta_schedule,
)

from aiak_training_omni.models.stdit.diffusion_utils import (
    broadcast_on_tp_group,
    broadcast_on_cp_group,
)


SUPPORTED_MODELS = [
    VideoLanguageModelFamilies.STDIT,
]


stimer = StragglerDetector()


def model_provider(pre_process=True, post_process=True):
    """Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.

    Returns:
        MCoreModel: The returned model
    """
    args = get_args()
    model_family = get_model_family(args.model_name)
    model_provider = get_model_provider(model_family)
    assert model_provider is not None, f"model provider for {args.model_name} not found"

    return model_provider(pre_process, post_process)


def get_batch(data_iterator):
    """Generate a batch"""
    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches on TP0 and CP0 only
    if (
        mpu.get_context_parallel_rank() == 0
        and mpu.get_tensor_model_parallel_rank() == 0
    ):
        assert data_iterator is not None
        batch = next(data_iterator)
        for key, val in batch.items():
            if val is not None and isinstance(val, torch.Tensor):
                batch[key] = val.cuda(non_blocking=True)
    else:
        batch = None

    if mpu.get_context_parallel_rank() == 0:
        batch = broadcast_on_tp_group(batch)

    batch = broadcast_on_cp_group(batch)

    video = batch["video"]  # B in T H W
    video_noised = batch["video_noised"]  # B in T H W
    video_mask = batch["video_mask"]
    text_enc = batch["text_enc"]  # B 1 S H
    text_mask = batch["text_mask"]
    labels = batch["labels"]
    timestep = batch["timestep"]  # B

    return video, video_noised, video_mask, text_enc, text_mask, timestep, labels


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


def loss_func(
    diffusion, loss_mask, target, video, video_noised, timestep, model_output
):
    """Computes the vision model loss (Cross entropy across vocabulary)

    Args:
        labels (Tensor): The labels of dimension [batch size, seq length]
        logits (Tensor): The final logits returned by the output layer of the transformer model

    Returns:
        Tensor: Loss tensor of dimensions [batch size, sequence_length]
    """

    args = get_args()
    assert model_output.shape[1] == args.latent_in_channels * 2, model_output.shape
    model_output, model_var_values = model_output.split(args.latent_in_channels, dim=1)
    frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
    vb = diffusion._vb_terms_bpd(
        model=lambda *args, r=frozen_out: r,
        x_start=video,
        x_t=video_noised,
        t=timestep,
        clip_denoised=False,
    )["output"]

    _dim = list(range(1, len(target.shape)))

    mse = torch.sum(
        ((target - model_output) * loss_mask) ** 2, dim=_dim
    ) / loss_mask.sum(dim=_dim)
    loss = (vb + mse).mean()
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
        video, video_noised, video_mask, text_enc, text_mask, timestep, labels = (
            get_batch(data_iterator)
        )

    timers("batch-generator").stop()

    with stimer:
        output_tensor = model(
            video_noised, video_mask, text_enc, text_mask, timestep, labels=labels
        )
    return output_tensor, partial(
        loss_func, diffusion, video_mask, labels, video, video_noised, timestep
    )


def train_valid_test_datasets_provider(diffusion, train_val_test_num_samples):
    """Build the train test and validation datasets."""
    args = get_args()

    train_ds = LatentDatasetFromCSV(
        args.data_path[0],
        diffusion,
        num_frames=args.num_latent_frames,
        max_height=args.max_latent_height,
        max_width=args.max_latent_width,
        max_text_length=args.max_text_length,
        frame_interval=args.latent_frame_interval,
    )

    print_rank_0(f"> finished creating {args.model_name} datasets ...")

    return train_ds, None, None


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
