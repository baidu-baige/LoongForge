# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2024 DiffSynth-Studio
# Modifications Copyright (c) 2026 LoongForge Team.
#
# Derived from: DiffSynth-Studio
# Source: https://github.com/modelscope/DiffSynth-Studio
# See repository NOTICE for license details

"""FlowMatch scheduler for Qwen-Image."""

import math

import torch


class QwenImageFlowMatchScheduler:
    """FlowMatchScheduler("Qwen-Image") without pipeline dependencies."""

    def __init__(self):
        self.num_train_timesteps = 1000
        self.training = False

    @staticmethod
    def _calculate_shift_qwen_image(
        image_seq_len,
        base_seq_len=256,
        max_seq_len=8192,
        base_shift=0.5,
        max_shift=0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b

    @staticmethod
    def set_timesteps_qwen_image(
        num_inference_steps=100,
        denoising_strength=1.0,
        exponential_shift_mu=None,
        dynamic_shift_len=None,
    ):
        """Compute the sigma / timestep schedule used by Qwen-Image FlowMatch."""
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000
        shift_terminal = 0.02
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        if exponential_shift_mu is not None:
            mu = exponential_shift_mu
        elif dynamic_shift_len is not None:
            mu = QwenImageFlowMatchScheduler._calculate_shift_qwen_image(dynamic_shift_len)
        else:
            mu = 0.8
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        one_minus_z = 1 - sigmas
        scale_factor = one_minus_z[-1] / (1 - shift_terminal)
        sigmas = 1 - (one_minus_z / scale_factor)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    def set_training_weight(self):
        """Populate ``linear_timesteps_weights`` for the current timestep grid."""
        steps = 1000
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        weights = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            weights = weights * (len(self.timesteps) / steps)
            weights = weights + weights[1]
        self.linear_timesteps_weights = weights

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, **kwargs):
        """Configure sigmas / timesteps and optionally the training weights."""
        self.sigmas, self.timesteps = self.set_timesteps_qwen_image(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            **kwargs,
        )
        self.training = training
        if training:
            self.set_training_weight()

    def _index(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.detach().cpu()
        return torch.argmin((self.timesteps - timestep).abs())

    def add_noise(self, original_samples, noise, timestep):
        """Return the FlowMatch mixture ``(1 - sigma) * x + sigma * noise``."""
        sigma = self.sigmas[self._index(timestep)].to(device=original_samples.device, dtype=original_samples.dtype)
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample, noise, timestep):
        """Return the FlowMatch training target (velocity ``noise - sample``)."""
        return noise - sample

    def training_weight(self, timestep):
        """Return the per-timestep weighting factor used during training."""
        timestep_id = self._index(timestep.to(self.timesteps.device))
        return self.linear_timesteps_weights[timestep_id]
