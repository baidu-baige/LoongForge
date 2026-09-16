# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LingBot-VA (``wan_va/utils/scheduler.py`` and ``wan_va/utils/utils.py``)
# under the Apache-2.0 License.
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

"""Native scheduler and sampling helpers for LingBot-VA."""

import torch


class LingBotVAFlowMatchScheduler:
    """Flow-matching schedule shared by LingBot-VA training and inference.

    ``set_timesteps`` / ``add_noise`` / ``training_target`` / ``training_weight``
    serve the training path; ``step`` serves the inference denoising loop. Both
    read the same ``sigmas`` / ``timesteps`` built by ``set_timesteps``.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        extra_one_step: bool = False,
    ) -> None:
        """Store the schedule shape and build a default 100-step timestep table."""
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self._device_cache = {}
        self.set_timesteps(100)

    def set_timesteps(self, num_inference_steps: int, training: bool = False) -> None:
        """Build sigma and timestep tensors for inference or training."""
        self._device_cache.clear()
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

    def _cached_tensor(self, name: str, source: torch.Tensor, device, dtype=None):
        key = (name, str(device), dtype)
        cached = self._device_cache.get(key)
        if cached is None:
            cached = source.to(device=device, dtype=dtype)
            self._device_cache[key] = cached
        return cached

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

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Take one inference denoising step (Euler step on the flow-matching ODE).

        Inference-only; the training path uses ``add_noise`` / ``training_target``.
        """
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep)
        timesteps = self.timesteps.to(timestep.device)
        index = int(torch.argmin((timesteps - timestep.reshape(-1)[0]).abs()))
        sigma = self.sigmas[index].to(sample)
        if index + 1 < len(self.sigmas):
            sigma_next = self.sigmas[index + 1].to(sample)
        else:
            sigma_next = torch.zeros_like(sigma)
        return sample + model_output * (sigma_next - sigma)

    def sigma_from_ids(
        self,
        sample: torch.Tensor,
        timestep_ids: torch.Tensor,
        t_dim: int = 2,
    ):
        """Return broadcast-ready sigmas for already sampled schedule ids."""
        shape = [1] * sample.ndim
        shape[t_dim] = timestep_ids.numel()
        sigmas = self._cached_tensor("sigmas", self.sigmas, sample.device, sample.dtype)
        return sigmas[timestep_ids].view(shape)

    def add_noise_from_ids(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep_ids: torch.Tensor,
        t_dim: int = 2,
    ):
        """Blend a sample when the caller already has the sampled schedule ids."""
        sigma = self.sigma_from_ids(sample, timestep_ids, t_dim=t_dim)
        return (1 - sigma) * sample + sigma * noise

    def timesteps_from_ids(self, timestep_ids: torch.Tensor):
        """Materialize selected schedule values from a device-local cache."""
        timesteps = self._cached_tensor(
            "timesteps", self.timesteps, timestep_ids.device, self.timesteps.dtype
        )
        return timesteps[timestep_ids]

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

    def training_weight_from_ids(self, timestep_ids: torch.Tensor):
        """Look up weights without reconstructing ids from already-indexed timesteps."""
        weights = self._cached_tensor(
            "training_weights",
            self.linear_timesteps_weights,
            timestep_ids.device,
            self.linear_timesteps_weights.dtype,
        )
        return weights[timestep_ids]


def sample_timestep_id(
    count: int,
    num_train_timesteps: int,
    min_timestep_boundary: float = 0.0,
    max_timestep_boundary: float = 1.0,
    generator: "torch.Generator | None" = None,
) -> torch.Tensor:
    """Sample random timestep ids within the configured fractional bounds."""
    values = torch.rand(count, generator=generator)
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
