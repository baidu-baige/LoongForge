# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Rectified flow implementation for diffusion training."""

from typing import Callable

import torch
import torch.distributed
from diffusers import FlowMatchEulerDiscreteScheduler


class TrainTimeWeight:
    """Computes per-sample loss weight as a function of timestep."""

    def __init__(self, noise_scheduler, method: str = "uniform"):
        """Initialize the instance."""
        self.noise_scheduler = noise_scheduler
        self.method = method

    def __call__(self, timesteps: torch.Tensor, **kwargs) -> torch.Tensor:
        """Call the instance."""
        if self.method == "uniform":
            return torch.ones_like(timesteps)
        elif self.method == "sigma":
            sigmas = self.noise_scheduler.sigmas.to(device=timesteps.device, dtype=timesteps.dtype)
            schedule_timesteps = self.noise_scheduler.timesteps.to(device=timesteps.device, dtype=timesteps.dtype)
            step_indices = [(schedule_timesteps == t).nonzero().squeeze().item() for t in timesteps]
            sigma = sigmas[step_indices]
            return 1.0 / (sigma * (1.0 - sigma)).clamp(min=1e-6)
        return torch.ones_like(timesteps)


class TrainTimeSampler:
    """Samples training timesteps from a specified distribution."""
    _WAVER_MODE_S = 1.29

    def __init__(
        self,
        distribution: str = "uniform",
    ):
        """Initialize the instance."""
        self.distribution = distribution

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Sample time tensor for training

        Returns:
            torch.Tensor: Time tensor, shape (batch_size,)
        """
        if self.distribution == "uniform":
            t = torch.rand((batch_size,), generator=generator).to(device=device, dtype=dtype)  # [B]
        elif self.distribution == "logitnormal":
            t = torch.sigmoid(torch.randn((batch_size,), generator=generator)).to(device=device, dtype=dtype)  # [B]
        elif self.distribution == "waver":
            u = torch.rand((batch_size,), dtype=torch.float32, generator=generator)  # [B]
            t = 1.0 - u - self._WAVER_MODE_S * (torch.cos(torch.pi / 2.0 * u) ** 2 - 1 + u)  # [B]
            t = t.to(device=device, dtype=dtype)  # [B]
        else:
            raise NotImplementedError(f"Time distribution '{self.dist}' is not implemented.")

        return t  # [B]


class RectifiedFlow:
    """Rectified flow for training diffusion models."""
    def __init__(
        self,
        velocity_field: Callable,
        train_time_distribution: TrainTimeSampler | str = "uniform",
        train_time_weight_method: str = "uniform",
        use_dynamic_shift: bool = False,
        shift: int = 3,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        r"""Initialize the RectifiedFlow class.

        Args:
            velocity_field (`Callable`):
                A function that predicts the velocity given the current state and time.
            train_time_distribution (`TrainTimeSampler` or `str`, *optional*, defaults to `"uniform"`):
                Distribution for sampling training times.
                Can be an instance of `TrainTimeSampler` or a string specifying the distribution type.
            train_time_weight (`TrainTimeWeight` or `str`, *optional*, defaults to `"uniform"`):
                Weight applied to training times.
                Can be an instance of `TrainTimeWeight` or a string specifying the weight type.
        """
        self.velocity_field = velocity_field
        self.train_time_sampler: TrainTimeSampler = (
            train_time_distribution
            if isinstance(train_time_distribution, TrainTimeSampler)
            else TrainTimeSampler(train_time_distribution)
        )

        if use_dynamic_shift:
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=use_dynamic_shift)
        else:
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
        self.train_time_weight = TrainTimeWeight(self.noise_scheduler, train_time_weight_method)

        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = torch.dtype(dtype) if isinstance(dtype, str) else dtype

    def sample_train_time(self, batch_size: int, iteration: int | None = None) -> torch.Tensor:
        r"""This method calls the `TrainTimeSampler` to sample training times.

        Args:
            batch_size: Number of time values to sample.
            iteration: When provided, sampling uses a local generator seeded from
                ``(iteration, rank)`` so results are identical across independent runs
                regardless of prior global RNG state.

        Returns:
            t (`torch.Tensor`):
                A tensor of sampled training times with shape `(batch_size,)`,
                matching the class specified `device` and `dtype`.
        """
        # R2 patch: always diversify across DP ranks so each rank samples a
        # different t in [0, 1]. Without this, ranks share global RNG state
        # and yield identical timesteps, collapsing flow-matching coverage.
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        generator = torch.Generator()
        if iteration is not None and torch.are_deterministic_algorithms_enabled():
            generator.manual_seed(iteration * 65536 + rank)
        else:
            # Draw a base seed from global RNG (identical across ranks),
            # then offset by rank * 7919 (prime) to break rank symmetry while
            # keeping reproducibility once the global seed is fixed.
            base = int(torch.randint(0, 2**31 - 1, (1,)).item())
            generator.manual_seed(base + rank * 7919)
        time = self.train_time_sampler(batch_size, device=self.device, dtype=self.dtype, generator=generator)
        return time

    def get_discrete_timestamp(self, u, tensor_kwargs):
        r"""This method map time from 0,1 to discrete steps"""

        indices = (u.squeeze() * self.noise_scheduler.config.num_train_timesteps).long()  # [B]
        timesteps = self.noise_scheduler.timesteps.to(**tensor_kwargs)[indices]  # [B]
        return timesteps.unsqueeze(0) if timesteps.ndim == 0 else timesteps  # [B]

    def get_sigmas(self, timesteps, tensor_kwargs):  # timesteps: [B], returns [B]
        """Get sigma values for given timesteps."""
        sigmas = self.noise_scheduler.sigmas.to(**tensor_kwargs)  # [N_timesteps+1]
        schedule_timesteps = self.noise_scheduler.timesteps.to(**tensor_kwargs)  # [N_timesteps]
        step_indices = [(schedule_timesteps == t).nonzero().squeeze().tolist() for t in timesteps]
        assert len(step_indices) == timesteps.shape[0], "Number of indices do not match the given timesteps."
        sigma = sigmas[step_indices].flatten()  # [B]

        return sigma  # [B]

    def get_timesteps_sigmas_from_shift(
        self,
        t_raw,            # [B], from sample_train_time, in [0,1]
        shift,
        max_timestep=1000,
        tensor_kwargs=None,
    ):
        """Cosmos-equivalent analytic shift formula.

        timesteps = shift * (1 - t_raw) / (1 + (shift - 1) * (1 - t_raw)) * max_timestep
        sigmas    = timesteps / max_timestep
        """
        if tensor_kwargs is None:
            tensor_kwargs = {"device": t_raw.device, "dtype": torch.float32}
        t_raw = t_raw.to(**tensor_kwargs)
        t = 1.0 - t_raw  # invert (cosmos shifts the inverted time)
        timesteps = shift * t / (1.0 + (shift - 1.0) * t) * max_timestep  # [B]
        sigmas = timesteps / max_timestep  # [B]
        return timesteps, sigmas
