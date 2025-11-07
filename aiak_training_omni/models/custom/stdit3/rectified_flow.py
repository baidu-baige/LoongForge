"""Open-Sora/opensora/schedulers/rf/rectified_flow.py"""

import torch
from einops import rearrange


def mean_flat(tensor: torch.Tensor, mask=None):
    """
    Take the mean over all non-batch dimensions.
    """
    if mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        assert tensor.dim() == 5
        assert tensor.shape[2] == mask.shape[1]
        tensor = rearrange(tensor, "b c t h w -> b t (c h w)")
        denom = mask.sum(dim=1) * tensor.shape[-1]
        loss = (tensor * mask.unsqueeze(2)).sum(dim=1).sum(dim=1) / denom
        return loss


class RFlowScheduler:
    """
    Rectified Flow Scheduler
    https://arxiv.org/abs/2107.06493
    """

    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        loc=0.0,
        scale=1.0,
        transform_scale=1.0,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps

        self.distribution = torch.distributions.LogisticNormal(
            torch.tensor([loc]), torch.tensor([scale])
        )
        self.sample_t = lambda x: self.distribution.sample()[0]

        # timestep transform
        self.transform_scale = transform_scale

    def timestep_transform(
        self,
        x_start,
        model_kwargs,
        base_resolution=512 * 512,
        base_num_frames=1,
    ):
        """transform the timestep"""
        t = self.sample_t(x_start)
        resolution = model_kwargs["height"] * model_kwargs["width"]
        ratio_space = (resolution / base_resolution).sqrt()
        # NOTE: currently, we do not take fps into account
        # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
        if model_kwargs["num_frames"][0] == 1:
            num_frames = torch.ones_like(model_kwargs["num_frames"])
        else:
            num_frames = model_kwargs["num_frames"] // 17 * 5
        ratio_time = (num_frames / base_num_frames).sqrt()

        ratio = ratio_space * ratio_time * self.transform_scale
        new_t = ratio * t / (1 + (ratio - 1) * t)

        new_t = new_t * self.num_timesteps
        return new_t

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
        noise: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(
            1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4]
        )

        return timepoints * original_samples + (1 - timepoints) * noise
