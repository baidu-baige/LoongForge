# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LingBot-VA under the Apache-2.0 License.
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

"""Native scheduler and sampling helpers for LingBot-VA."""

import torch


class LingBotVAFlowMatchScheduler:
    """Flow-matching schedule used by LingBot-VA training."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        extra_one_step: bool = False,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self.set_timesteps(100)

    def set_timesteps(self, num_inference_steps: int, training: bool = False) -> None:
        """Build sigma and timestep tensors for inference or training."""
        count = num_inference_steps + 1 if self.extra_one_step else num_inference_steps
        self.sigmas = torch.linspace(self.sigma_max, self.sigma_min, count)
        if self.extra_one_step:
            self.sigmas = self.sigmas[:-1]
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            weights = torch.exp(
                -2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2
            )
            weights = weights - weights.min()
            self.linear_timesteps_weights = weights * (
                num_inference_steps / weights.sum()
            )

    def add_noise(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        t_dim: int = 2,
    ):
        """Blend a sample with noise according to the selected timestep."""
        ids = torch.argmin(
            (self.timesteps[:, None] - timestep.detach().cpu()[None]).abs(), dim=0
        )
        shape = [1] * noise.ndim
        shape[t_dim] = ids.numel()
        sigma = self.sigmas[ids].to(sample).view(shape)
        return (1 - sigma) * sample + sigma * noise

    @staticmethod
    def training_target(
        sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ):
        """Return the velocity target used for flow matching training."""
        del timestep
        return noise - sample

    def training_weight(self, timestep: torch.Tensor):
        """Look up per-timestep loss weights for training."""
        ids = torch.argmin(
            (self.timesteps[:, None].to(timestep.device) - timestep[None]).abs(), dim=0
        )
        return self.linear_timesteps_weights.to(timestep.device)[ids]


def sample_timestep_id(
    count: int,
    num_train_timesteps: int,
    min_timestep_boundary: float = 0.0,
    max_timestep_boundary: float = 1.0,
) -> torch.Tensor:
    """Sample random timestep ids within the configured fractional bounds."""
    values = torch.rand(count)
    values = (
        values * (max_timestep_boundary - min_timestep_boundary) + min_timestep_boundary
    )
    return (values * num_train_timesteps).clamp(0, num_train_timesteps - 1).long()


def get_mesh_id(
    frames: int, height: int, width: int, token_type: int, action: bool = False
):
    """Create frame, height, width, and token-type grid ids."""
    frame_grid, height_grid, width_grid = torch.meshgrid(
        torch.arange(frames), torch.arange(height), torch.arange(width), indexing="ij"
    )
    if action:
        frame_grid = frame_grid + (torch.arange(1, height + 1) / (height + 1)).view(
            1, -1, 1
        )
        height_grid = torch.full_like(frame_grid, -1)
        width_grid = torch.full_like(frame_grid, -1)
    grid = torch.stack((frame_grid, height_grid, width_grid)).flatten(1)
    return torch.cat((grid, torch.full_like(grid[:1], token_type)), dim=0)
