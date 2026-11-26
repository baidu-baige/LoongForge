# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 timestep distributions and sampler (Lane C).

The signatures follow ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.2; the
parameter names of the four distributions follow the four sets of Beta key names
recorded in ``config_layer_facts.md``.

The distribution is **algorithm** (R5): replacing logit_normal with beta is
replacing the training objective. So the **kind** of distribution is not exposed
in YAML, only its **parameters** are (the people tuning them sweep repeatedly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor

from .broadcast import DEFAULT_PER_SAMPLE, BroadcastSpec

__all__ = [
    "TimestepDistribution", "UniformShift", "LogitNormal", "BetaDist",
    "Constant", "RngPolicy", "TimestepSampler",
]


class TimestepDistribution(Protocol):
    """Protocol for a timestep distribution. One method: draw a batch.

    The return value must lie in **[0, 1]**: converting to sigma, scaling by
    1000 and the like belong to ``NoiseSchedule``, not to the distribution.
    Today both live in one function, so "change the sampling distribution" also
    changes the noise scale.
    """

    def sample(self, shape: tuple[int, ...], *,
               device: torch.device, dtype: torch.dtype) -> Tensor:
        """Draw timesteps of shape ``shape``, valued in [0, 1]."""
        ...


@dataclass(frozen=True)
class UniformShift:
    """Uniform distribution with a shift reparameterisation. The WAN default.

    ``shift=1.0`` is plain uniform. ``shift > 1`` pushes mass towards the
    high-noise end.

    Attributes:
        shift: Reparameterisation strength; must be positive.
    """

    shift: float = 1.0   # [YAML] diffusion.<stream>.shift, resolved by the provider (resolve_diffusion)

    def __post_init__(self) -> None:
        if self.shift <= 0:
            raise ValueError(
                f"UniformShift.shift must be positive, got {self.shift}")

    def sample(self, shape: tuple[int, ...], *,
               device: torch.device, dtype: torch.dtype) -> Tensor:
        """Draw ``u ~ U(0, 1)`` and push it through
        ``phi(u) = s*u / (1 + (s-1)*u)``.

        Same formula as the existing ``_phi`` in
        ``scheduler_continuous.py:37-39`` (used by ``shifted_uniform`` at
        ``:56-58``). That code then multiplies by ``num_train_timesteps``
        (``:70``) to blow [0, 1] up to [0, 1000]; **here we deliberately do
        not** -- the 1000x scaling is ``NoiseSchedule``'s business, and mixing
        it into the distribution makes "change the sampling distribution" also
        change the noise scale.

        With ``shift=1`` the map ``phi`` is the identity, so plain uniform falls
        out without a special-case branch.

        Args:
            shape: Output shape.
            device: Device for the draw.
            dtype: Floating dtype for the draw.

        Returns:
            Timesteps in [0, 1] of shape ``shape``.
        """
        u = torch.rand(shape, device=device, dtype=dtype)
        return self.shift * u / (1.0 + (self.shift - 1.0) * u)


@dataclass(frozen=True)
class LogitNormal:
    """Logit-normal distribution. The SD3 / pi05 default.

    Attributes:
        mean: Mean of the underlying normal, before the sigmoid.
        std: Standard deviation of the underlying normal; must be positive.
    """

    mean: float = 0.0    # [YAML] diffusion.<stream>.mean, read by the provider
    std: float = 1.0     # [YAML] diffusion.<stream>.std, read by the provider

    def __post_init__(self) -> None:
        if self.std <= 0:
            raise ValueError(
                f"LogitNormal.std must be positive, got {self.std}")

    def sample(self, shape: tuple[int, ...], *,
               device: torch.device, dtype: torch.dtype) -> Tensor:
        """``sigmoid(mean + std * z)`` with ``z ~ N(0, 1)``.

        The sigmoid guarantees the open interval (0, 1), so no clamp is needed.
        Never reaching the endpoints is precisely what distinguishes
        logit-normal from shifted-uniform: pure noise and pure clean both have
        probability zero.

        Args:
            shape: Output shape.
            device: Device for the draw.
            dtype: Floating dtype for the draw.

        Returns:
            Timesteps in (0, 1) of shape ``shape``.
        """
        z = torch.randn(shape, device=device, dtype=dtype)
        return torch.sigmoid(self.mean + self.std * z)


@dataclass(frozen=True)
class BetaDist:
    """Beta distribution. Four existing configs spelled the same thing with
    four different sets of key names; this collapses them to two.

    The cost of unifying the names is a **config migration table**; the benefit
    is that "which Beta is this, exactly" no longer requires reading four YAML
    files to answer.

    Attributes:
        alpha: First Beta shape parameter; must be positive.
        beta: Second Beta shape parameter; must be positive.
        eps: Lower bound on the denominator; same meaning as ``eps`` in the
            existing scheduler.
    """

    alpha: float = 1.0   # [YAML] diffusion.<stream>.alpha, read by the provider
    beta: float = 1.0    # [YAML] diffusion.<stream>.beta, read by the provider
    eps: float = 1e-6           # denominator floor; same as scheduler's ``eps``

    def __post_init__(self) -> None:
        for name in ("alpha", "beta"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(
                    f"BetaDist.{name} must be positive, got {value}"
                )
        if self.eps <= 0:
            raise ValueError(f"BetaDist.eps must be positive, got {self.eps}")

    def sample(self, shape: tuple[int, ...], *,
               device: torch.device, dtype: torch.dtype) -> Tensor:
        """Ratio of two standard gammas. Line-for-line the same as the existing
        ``scheduler_continuous.py:59-64`` (including ``clamp(min=eps)`` and the
        trailing ``clamp(0, 1)``).

        ``torch.distributions.Beta`` is avoided: with very small ``alpha`` /
        ``beta`` it draws exact 0 or 1, and ``x_t`` carries no gradient signal
        at t=0. The existing clamp combination is the one measured to be stable,
        so it is copied verbatim.

        Args:
            shape: Output shape.
            device: Device for the draw.
            dtype: Floating dtype of the returned tensor (the draw itself is
                done in float32).

        Returns:
            Timesteps in [0, 1] of shape ``shape``.
        """
        a = torch.full(shape, self.alpha, device=device, dtype=torch.float32)
        b = torch.full(shape, self.beta, device=device, dtype=torch.float32)
        ga = torch._standard_gamma(a)
        value = ga / (ga + torch._standard_gamma(b)).clamp(min=self.eps)
        return value.clamp(min=0.0, max=1.0).to(dtype=dtype)


@dataclass(frozen=True)
class Constant:
    """Constant timestep. **Not a debugging tool** -- it is the official way to
    say "this stream is not noised".

    ``value=0.0`` is a clean conditioning stream (VPP2's clean-conditioning
    video stream). It is used instead of writing ``if not add_noise`` inside the
    stream so that "the noise level of this stream" stays visible in the spec,
    which is what lets ``visibility.rules.noise_gate`` read it.

    Attributes:
        value: The constant timestep; must lie in [0, 1].
    """

    value: float = 0.0          # not a YAML knob: a constant timestep is the provider's choice (R5)

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(
                f"Constant.value must lie in [0, 1], got {self.value}"
            )

    def sample(self, shape: tuple[int, ...], *,
               device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.full(shape, self.value, device=device, dtype=dtype)


@dataclass(frozen=True)
class RngPolicy:
    """How one stream draws its randomness in a training step (Lane C v2 §1.1).

    Two decisions here are numerical facts of the implementation being
    reproduced, so they cannot be left to whatever order the code happens to
    call ``torch.rand`` in: with one global generator, swapping the noise and
    the timestep draw changes both values while the generator state after the
    step stays the same, which no RNG-state check can catch.

    Streams are drawn one after another in the key order of
    ``DiffusionSpec.timesteps``; within a stream ``order`` decides which comes
    first. The training timestep is ``t = sigma * num_train_timesteps`` (see
    ``RectifiedFlow``), drawn in float32 whatever the sample dtype.

    Attributes:
        order: ``"noise_first"`` draws epsilon and then the timestep (the Wan
            family: fastwam ``mot/fastwam.py:519-524``, VPP2
            ``libero_action_model.py:578-610``); ``"t_first"`` the reverse.
        sigma_dtype: ``"like_sample"`` casts t to the sample dtype and derives
            the noising sigma back from that t, which is what the Wan family
            does (``sample_training_t`` returns the cast t and ``add_noise``
            divides it by ``num_train_timesteps`` again); ``"float32"`` keeps t
            and sigma in float32, and sigma is cast to the sample dtype only
            for the noising arithmetic.
    """

    order: Literal["noise_first", "t_first"] = "noise_first"
    sigma_dtype: Literal["like_sample", "float32"] = "like_sample"

    def __post_init__(self) -> None:
        if self.order not in ("noise_first", "t_first"):
            raise ValueError(f"RngPolicy.order must be noise_first or t_first, got {self.order!r}")
        if self.sigma_dtype not in ("like_sample", "float32"):
            raise ValueError(
                f"RngPolicy.sigma_dtype must be like_sample or float32, got {self.sigma_dtype!r}")

    @property
    def draws(self) -> tuple[str, str]:
        """The two draws of one stream, in order: ``"noise"`` and ``"timestep"``."""
        return ("noise", "timestep") if self.order == "noise_first" else ("timestep", "noise")

    def quantize(self, sigma: Tensor, *, num_train_timesteps: int,
                 sample_dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Turn a float32 sigma into the step's ``(sigma, t)`` pair.

        Args:
            sigma: Float32 draw of the distribution, valued in [0, 1].
            num_train_timesteps: Scale from sigma to the training timestep.
            sample_dtype: Dtype of the stream's raw tensor.

        Returns:
            ``(sigma, t)``: sigma is the noising coefficient, t the training
            timestep the time embeddings and the sigma-dimension loss weights
            read. Under ``like_sample`` both are in ``sample_dtype``.
        """
        t = sigma * float(num_train_timesteps)
        if self.sigma_dtype == "float32":
            return sigma, t
        t = t.to(dtype=sample_dtype)
        return (t / float(num_train_timesteps)).to(dtype=sample_dtype), t


@dataclass(frozen=True)
class TimestepSampler:
    """Distribution plus broadcast spec. Step (1) of ``WorldModel.forward``
    calls it.

    ``broadcast`` is attached to the sampler rather than to the distribution:
    the same distribution can be drawn per sample or per frame, and "how many to
    draw" is exactly what ``broadcast.axes`` decides.

    Attributes:
        distribution: The timestep distribution to draw from.
        broadcast: Which label axes the drawn timesteps vary along.
        rng: Draw order of this stream's noise and timestep, and the dtype of
            the training timestep.
    """

    distribution: TimestepDistribution
    broadcast: BroadcastSpec = DEFAULT_PER_SAMPLE
    rng: RngPolicy = RngPolicy()

    def sample(self, batch_size: int, *, device: torch.device,
               dtype: torch.dtype = torch.float32,
               axis_sizes: tuple[int, ...] = ()) -> Tensor:
        """Draw a batch of timesteps, shaped ``(batch_size, *axis_sizes)``.

        ``axis_sizes`` must correspond one-to-one with ``broadcast.axes``: a
        length mismatch is a configuration error, not something broadcasting
        rules should paper over.

        Args:
            batch_size: Number of samples; must be positive.
            device: Device for the draw.
            dtype: Floating dtype for the draw.
            axis_sizes: Length of each axis named in ``broadcast.axes``, in the
                same order.

        Returns:
            A ``(batch_size, *axis_sizes)`` tensor of timesteps in [0, 1].

        Raises:
            ValueError: If ``batch_size`` is not positive, or ``axis_sizes``
                does not match ``broadcast.axes`` in length.
        """
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {batch_size}")
        if len(axis_sizes) != len(self.broadcast.axes):
            raise ValueError(
                f"TimestepSampler.sample: axis_sizes has {len(axis_sizes)} "
                f"items, broadcast.axes={self.broadcast.axes} requires "
                f"{len(self.broadcast.axes)} items"
            )
        return self.distribution.sample((batch_size, *axis_sizes),
                                        device=device, dtype=dtype)
