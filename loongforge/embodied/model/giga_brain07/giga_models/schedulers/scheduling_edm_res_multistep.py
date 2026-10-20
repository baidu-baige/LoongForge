# Copyright 2024 TSAIL Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from typing import List, Optional, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput

try:
    import torch_npu  # noqa F401

    npu_is_available = True
except ImportError:
    npu_is_available = False


class EDMRESMultistepScheduler(SchedulerMixin, ConfigMixin):
    """Implements EDMRESMultistepScheduler in EDM formulation as presented in
    Qinsheng Zhang et al. 2023 [1]. `EDMRESMultistepScheduler` is a fast
    dedicated high-order solver for diffusion ODEs.

    [1] Zhang, et al. "Improved Order Analysis and Design of Exponential Integrator for Diffusion Models Sampling."
    https://arxiv.org/pdf/2308.02157

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        sigma_min (`float`, *optional*, defaults to 0.002):
            Minimum noise magnitude in the sigma schedule. This was set to 0.002 in the EDM paper [1]; a reasonable
            range is [0, 10].
        sigma_max (`float`, *optional*, defaults to 80.0):
            Maximum noise magnitude in the sigma schedule. This was set to 80.0 in the EDM paper [1]; a reasonable
            range is [0.2, 80.0].
        sigma_data (`float`, *optional*, defaults to 0.5):
            The standard deviation of the data distribution. This is set to 0.5 in the EDM paper [1].
        sigma_schedule (`str`, *optional*, defaults to `karras`):
            Sigma schedule to compute the `sigmas`. By default, we the schedule introduced in the EDM paper
            (https://arxiv.org/abs/2206.00364). Other acceptable value is "exponential". The exponential schedule was
            incorporated in this model: https://huggingface.co/stabilityai/cosxl.
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        solver_order (`int`, defaults to 2):
            The solver order which can be `1` or `2` It is recommended to use `solver_order=2` for guided
            sampling, and `solver_order=3` for unconditional sampling.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        lower_order_final (`bool`, defaults to `True`):
            Whether to use lower-order solvers in the final steps.
        final_sigmas_type (`str`, defaults to `"zero"`):
            The final `sigma` value for the noise schedule during the sampling process. If `"sigma_min"`, the final
            sigma is the same as the last sigma in the training schedule. If `zero`, the final sigma is set to 0.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        sigma_schedule: str = 'karras',
        num_train_timesteps: int = 1000,
        prediction_type: str = 'epsilon',
        rho: float = 7.0,
        solver_order: int = 2,
        lower_order_final: bool = True,
        final_sigmas_type: Optional[str] = 'zero',  # "zero", "sigma_min"
    ):
        self._dtype = torch.float64 if not npu_is_available else torch.float32

        ramp = torch.linspace(0, 1, num_train_timesteps, dtype=self.dtype)
        if sigma_schedule == 'karras':
            sigmas = self._compute_karras_sigmas(ramp)
        elif sigma_schedule == 'exponential':
            sigmas = self._compute_exponential_sigmas(ramp)

        self.timesteps = self.precondition_noise(sigmas)

        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device, dtype=self.dtype)])

        # setable values
        self.num_inference_steps = None
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to('cpu')  # to avoid too much CPU/GPU communication

    @property
    def init_noise_sigma(self):
        # standard deviation of the initial noise distribution
        return self.config.sigma_max

    @property
    def dtype(self):
        return self._dtype

    def to_dtype(self, dtype):
        self._dtype = dtype
        self.sigmas = self.sigmas.to(dtype)

    @property
    def step_index(self):
        """The index counter for current timestep.

        It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """The index for the first timestep.

        It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """Sets the begin index for the scheduler. This function should be run
        from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def add_condition_inputs(self, sample, sigma, cond_sample, cond_mask, cond_noise, cond_sigma):
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)
        while len(cond_sigma.shape) < len(sample.shape):
            cond_sigma = cond_sigma.unsqueeze(-1)
        if not self.config.prediction_type == 'rf':
            cond_sample = cond_sample + cond_noise * cond_sigma
            cond_c_in = 1 / ((cond_sigma**2 + self.config.sigma_data**2) ** 0.5)
            c_in = 1 / ((sigma**2 + self.config.sigma_data**2) ** 0.5)
            cond_sample = cond_sample * cond_c_in / c_in
        else:
            c_in = 1 - sigma / (1 + sigma)
            cond_sample = cond_sample / c_in
        new_sample = cond_mask * cond_sample + (1 - cond_mask) * sample
        return new_sample

    def precondition_inputs(self, sample, sigma):
        if self.config.prediction_type == 'rf':
            c_in = 1 - sigma / (1 + sigma)
        else:
            c_in = 1 / ((sigma**2 + self.config.sigma_data**2) ** 0.5)
        scaled_sample = sample * c_in
        return scaled_sample

    def precondition_noise(self, sigma):
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([sigma])
        if self.config.prediction_type == 'rf':
            c_noise = sigma / (1 + sigma)
        else:
            c_noise = 0.25 * torch.log(sigma)

        return c_noise

    def precondition_outputs(self, sample, model_output, sigma):
        sigma_data = self.config.sigma_data
        if self.config.prediction_type == 'rf':
            c_skip = 1 - sigma / (1 + sigma)
        else:
            c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)

        if self.config.prediction_type == 'epsilon':
            c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        elif self.config.prediction_type == 'rf':
            c_out = -(sigma / (1 + sigma))
        elif self.config.prediction_type == 'v_prediction':
            c_out = -sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        else:
            raise ValueError(f'Prediction type {self.config.prediction_type} is not supported.')

        denoised = c_skip * sample + c_out * model_output

        return denoised

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        """Ensures interchangeability with schedulers that need to scale the
        denoising model input depending on the current timestep. Scales the
        denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler
        algorithm.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = self.precondition_inputs(sample, sigma)

        return sample

    def set_timesteps(self, num_inference_steps: int = None, device: Union[str, torch.device] = None):
        """Sets the discrete timesteps used for the diffusion chain (to be run
        before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        self.num_inference_steps = num_inference_steps

        ramp = torch.linspace(0, 1, num_inference_steps, dtype=self.dtype)
        if self.config.sigma_schedule == 'karras':
            sigmas = self._compute_karras_sigmas(ramp)
        elif self.config.sigma_schedule == 'exponential':
            sigmas = self._compute_exponential_sigmas(ramp)

        sigmas = sigmas.to(device=device)
        self.timesteps = self.precondition_noise(sigmas)

        if self.config.final_sigmas_type == 'sigma_min':
            sigma_last = self.config.sigma_min
        elif self.config.final_sigmas_type == 'zero':
            sigma_last = 0
        else:
            raise ValueError(f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}")

        self.sigmas = torch.cat([sigmas, torch.tensor([sigma_last], dtype=self.dtype, device=device)])

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to('cpu')  # to avoid too much CPU/GPU communication

    def _compute_karras_sigmas(self, ramp, sigma_min=None, sigma_max=None) -> torch.Tensor:
        """Constructs the noise schedule of Karras et al.

        (2022).
        """
        sigma_min = sigma_min or self.config.sigma_min
        sigma_max = sigma_max or self.config.sigma_max

        rho = self.config.rho
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def _compute_exponential_sigmas(self, ramp, sigma_min=None, sigma_max=None) -> torch.Tensor:
        """Implementation closely follows k-diffusion.

        https://github.com/crowsonkb/k-diffusion/blob/6ab5146d4a5ef63901326489f31f1d8e7dd36b48/k_diffusion/sampling.py#L26
        """
        sigma_min = sigma_min or self.config.sigma_min
        sigma_max = sigma_max or self.config.sigma_max
        sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), len(ramp)).exp().flip(0)
        return sigmas

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor = None,
    ) -> torch.Tensor:
        """Convert the model output to the corresponding type the EDM RES
        algorithm needs.

        <Tip>

        The algorithm and model type are decoupled.

        </Tip>

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The converted model output.
        """
        sigma = self.sigmas[self.step_index]
        x0_pred = self.precondition_outputs(sample, model_output, sigma)
        return x0_pred

    def first_order_update(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """One step for the first-order(equivalent to DDIM).

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
        sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]

        coef_x0 = (sigma_s - sigma_t) / sigma_s
        coef_xs = sigma_t / sigma_s
        xt = coef_x0 * model_output + coef_xs * sample

        return xt

    def multistep_second_order_update(
        self,
        model_output_list: List[torch.Tensor],
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """One step for the second-order multistep.

        Args:
            model_output_list (`List[torch.Tensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )

        lambda_t = -torch.log(sigma_t)
        lambda_s0 = -torch.log(sigma_s0)
        lambda_s1 = -torch.log(sigma_s1)

        x0_s0, x0_s1 = model_output_list[-1], model_output_list[-2]

        dt = lambda_t - lambda_s0
        c2 = lambda_s1 - lambda_s0
        assert not torch.any(torch.isclose(dt, torch.zeros_like(dt), atol=1e-6)), 'Step size is too small'
        assert not torch.any(torch.isclose(c2, torch.zeros_like(dt), atol=1e-6)), 'Step size is too small'
        c2 = c2 / dt

        phi1_val, phi2_val = phi1(-dt), phi2(-dt)

        # Handle edge case where t = s = m
        b1 = torch.nan_to_num(phi1_val - 1.0 / c2 * phi2_val, nan=0.0)
        b2 = torch.nan_to_num(1.0 / c2 * phi2_val, nan=0.0)

        xt = torch.exp(-dt) * sample + dt * (b1 * x0_s0 + b2 * x0_s1)

        return xt

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        index_candidates = (schedule_timesteps == timestep).nonzero()

        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()

        return step_index

    def _init_step_index(self, timestep):
        """Initialize the step_index counter for the scheduler."""

        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        cond_sample: torch.Tensor = None,
        cond_mask: torch.Tensor = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """Predict the sample from the previous timestep by reversing the SDE.
        This function propagates the sample with the multistep Solver.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler")

        if self.step_index is None:
            self._init_step_index(timestep)

        # Improve numerical stability for small number of steps
        lower_order_final = (self.step_index == len(self.timesteps) - 1) and self.config.lower_order_final

        model_output = self.convert_model_output(model_output, sample=sample).to(self.dtype)
        if cond_sample is not None:
            model_output = cond_mask * cond_sample + (1 - cond_mask) * model_output
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.first_order_update(model_output, sample=sample)
        elif self.config.solver_order == 2 or self.lower_order_nums < 2:
            prev_sample = self.multistep_second_order_update(self.model_outputs, sample=sample)
        else:
            assert False

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == 'mps' and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def get_noise(self, sample, model_output, sigma):
        pred_original_sample = self.precondition_outputs(sample, model_output, sigma)
        noise = (sample - pred_original_sample) / sigma
        return noise

    def __len__(self):
        return self.config.num_train_timesteps


def phi1(t: torch.Tensor) -> torch.Tensor:
    """
    Compute the first order phi function: (exp(t) - 1) / t.

    Args:
        t: Input tensor.

    Returns:
        Tensor: Result of phi1 function.
    """
    input_dtype = t.dtype
    t = t.to(dtype=torch.float64 if not npu_is_available else torch.float32)
    return (torch.expm1(t) / t).to(dtype=input_dtype)


def phi2(t: torch.Tensor) -> torch.Tensor:
    """
    Compute the second order phi function: (phi1(t) - 1) / t.

    Args:
        t: Input tensor.

    Returns:
        Tensor: Result of phi2 function.
    """
    input_dtype = t.dtype
    t = t.to(dtype=torch.float64 if not npu_is_available else torch.float32)
    return ((phi1(t) - 1.0) / t).to(dtype=input_dtype)
