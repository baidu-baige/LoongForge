# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.

"""DreamZero flow-matching scheduler.

This scheduler is distinct from ``loongforge/models/diffusion/wan/wan_flow_match.py``.
The sigma sampling schedule (linear in [sigma_min, sigma_max], post-shifted via
``shift * sigma / (1 + (shift - 1) * sigma)``) and the ``shift`` default
(3.0) are part of the checkpoint-compatible training recipe.

Public surface (used by action_head_tf.py / inference):
- ``FlowMatchScheduler(num_inference_steps, num_train_timesteps, shift,
  sigma_max, sigma_min, inverse_timesteps, extra_one_step, reverse_sigmas)``
- ``set_timesteps(num_inference_steps, denoising_strength, training, shift)``
- ``add_noise(original_samples, noise, timestep)`` — batched via
  ``timesteps.unsqueeze(1)`` vs ``timestep.unsqueeze(0)``; supports per-sample
  timesteps (training path).
- ``training_target(sample, noise, timestep)`` → ``noise - sample`` (flow
  matching velocity target).
- ``training_weight(timestep)`` — looks up ``linear_timesteps_weights``
  computed during ``set_timesteps(training=True)``.
- ``step(model_output, timestep, sample, to_final)`` — Euler step; subtracts
  ``model_output * (sigma_next - sigma_curr)``.
- ``return_to_timestep(timestep, sample, sample_stablized)`` — inverse op.
"""

import torch


class FlowMatchScheduler():
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0)
        sigma = self.sigmas[timestep_id].to(device=original_samples.device, dtype=original_samples.dtype)
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        """Look up the per-timestep training loss weight."""
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(1) - timestep.unsqueeze(0).to(self.timesteps.device)).abs(),
            dim=0,
        )
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
