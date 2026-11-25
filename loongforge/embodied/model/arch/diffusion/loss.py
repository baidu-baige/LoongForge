# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 loss specs ``LossSpec`` / ``WeightRamp`` (Lane C).

The signatures follow ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.4; the
ruling on ``WeightRamp`` and the new interface **I4′** is the 11/13 conclusion
in the master plan §9.

The **number** of loss terms differs per model (2 / 3 / 5 terms), so this is a
list rather than a fixed set of parameters. That fixed arity is exactly the
constraint in OpenWAM-TS's two-tuple forward that ruled out InternW0-Δ and
AW-0, and this layer deliberately does not copy it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import torch
from torch import Tensor
from torch.nn import functional as F

from .broadcast import DEFAULT_PER_SAMPLE, BroadcastSpec
from .timesteps import Constant, TimestepSampler, UniformShift

__all__ = ["WeightRamp", "LossSpec", "LossObjective", "MeanSquaredError",
           "TokenMeanSquaredError", "BatchPadding", "CosineDistance",
           "TimestepWeight", "UniformWeight", "Bsmntw"]


class LossObjective(Protocol):
    """What is computed between prediction and target. One method, producing
    a **per-sample** ``[B]``.

    Why not a ``"mse"`` / ``"cosine"`` string: in the existing code that shows
    up as three separate ``if`` chains (weighting, logging, backward), so adding
    an objective means editing three places. As an object it has one
    implementation site.

    Why it returns ``[B]`` rather than a scalar: besides ``LossSpec.weight_at``
    there is per-sample weighting (``LossSpec.weight_of``), and applying that
    after the batch mean is not equivalent.
    """

    def __call__(self, prediction: Tensor, target: Tensor, *,
                 valid: Tensor | None = None) -> Tensor:
        """``[B, ...]`` x ``[B, ...]`` -> ``[B]``.

        ``valid`` is a bool ``[B, T]`` over the objective's token axis (True
        counts), from ``DiffusionRuntime.loss_valid``; an objective without a
        masked reduction raises when it is given one.
        """
        ...


def _reject_valid(valid: Tensor | None, *, objective: str) -> None:
    if valid is not None:
        raise ValueError(
            f"{objective} has no masked reduction, but this loss's source stream "
            "has a loss_valid mask; use TokenMeanSquaredError")


def _check_same_shape(prediction: Tensor, target: Tensor, *,
                      objective: str) -> None:
    """Prediction and target must have the **same shape**; no broadcasting.

    A head's output shape equals the shape of that stream's raw tensor (see the
    invariant in ``heads/flow.py``), so same-shape is guaranteed by design. A
    mismatch means some shape plumbing is wrong on one side, and broadcasting
    here would silently produce a number instead.

    Args:
        prediction: Model prediction.
        target: Regression target.
        objective: Objective name, used in the error message.

    Raises:
        ValueError: If the shapes differ, or the rank is below 2.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"{objective}: prediction {tuple(prediction.shape)} and target "
            f"{tuple(target.shape)} have different shapes. A head's output "
            "shape must equal that stream's raw tensor shape; a mismatch is "
            "not covered up by broadcasting"
        )
    if prediction.ndim < 2:
        raise ValueError(
            f"{objective}: expected at least 2 dims [B, ...], "
            f"got {tuple(prediction.shape)}"
        )


@dataclass(frozen=True)
class MeanSquaredError:
    """The regression objective of flow matching. **12 of the 13 models use
    it**, so it is the default.

    Means over every axis except batch, giving ``[B]``. It does not sum first
    and divide later: the element counts of different streams differ by orders
    of magnitude (video tokens outnumber action tokens by two decimal places),
    so summing would let tensor size, rather than ``LossSpec.weight``, decide
    the relative weight of two losses.
    """

    def __call__(self, prediction: Tensor, target: Tensor, *,
                 valid: Tensor | None = None) -> Tensor:
        _reject_valid(valid, objective="MeanSquaredError")
        _check_same_shape(prediction, target, objective="MeanSquaredError")
        error = (prediction.float() - target.float()).square()
        return error.flatten(1).mean(dim=1)


@dataclass(frozen=True)
class TokenMeanSquaredError:
    """Per-token MSE, then the mean over each sample's valid tokens. ``[B]``.

    The tokens lie on ``axis``: float32 squared error, mean over every other
    non-batch axis, then the mean over ``axis`` counting only ``valid``
    positions (denominator clamped at 1, so a sample with no valid token
    contributes 0, not NaN). Without ``valid`` every token counts.

    The order of the two means is part of the number: averaging all ``T * D``
    elements at once (``MeanSquaredError``) rounds differently. This is the
    reduction of the Wan action losses (VPP2 ``_training_loss_action_only``,
    fastwam ``mot/fastwam.py:613-621``) and, with ``axis=2``, of fastwam's
    per-frame video loss (``:476``, ``.mean(dim=(1, 3, 4))``) -- Lane C v2
    ``PerSampleMean``.

    Attributes:
        axis: The token axis: 1 for actions ``[B, T, D]``, 2 for the latent
            frames of a video ``[B, C, F, H, W]``.
    """

    axis: int = 1

    def __post_init__(self) -> None:
        if self.axis < 1:
            raise ValueError(
                f"TokenMeanSquaredError.axis must be at least 1 (axis 0 is the "
                f"batch), got {self.axis}")

    def __call__(self, prediction: Tensor, target: Tensor, *,
                 valid: Tensor | None = None) -> Tensor:
        _check_same_shape(prediction, target, objective="TokenMeanSquaredError")
        if prediction.ndim < 3 or self.axis >= prediction.ndim or not all(prediction.shape):
            raise ValueError(
                f"TokenMeanSquaredError(axis={self.axis}): expected a nonempty "
                "[B, ...] prediction with the token axis and at least one "
                f"feature axis, got {tuple(prediction.shape)}")
        features = tuple(dim for dim in range(1, prediction.ndim) if dim != self.axis)
        token_loss = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(
            dim=features)
        if valid is None:
            return token_loss.mean(dim=1)
        if tuple(valid.shape) != tuple(token_loss.shape):
            raise ValueError(
                f"TokenMeanSquaredError: valid {tuple(valid.shape)} must be "
                f"[B, T] = {tuple(token_loss.shape)}")
        counted = valid.to(device=token_loss.device, dtype=torch.bool).to(dtype=token_loss.dtype)
        return (token_loss * counted).sum(dim=1) / counted.sum(dim=1).clamp(min=1.0)


_PADDING_UNITS = ("step", "video_frame")


@dataclass(frozen=True)
class BatchPadding:
    """Per-sample padding of one stream, read from the batch: padded units do
    not count in that stream's flow losses.

    The data pipelines pad an action chunk past the episode end and flag it in
    ``action_is_pad``; fastwam flags padded video frames in ``image_is_pad``.
    Both old implementations average only over the units that are not padding
    (``model/fastwam/mot/fastwam.py:464-506``, ``:613-621``; master VPP2
    ``_training_loss_action_only``). ``WorldModel.add_noise`` turns this
    declaration into ``DiffusionRuntime.loss_valid``.

    A batch without ``pad_key`` has no padding, as in the old code
    (``sample.get("action_is_pad", None)``). The keys are batch bindings like
    ``StreamSpec.batch_key``, so they are written in the provider, not YAML.

    Attributes:
        pad_key: Batch key of a bool ``[B, N]`` mask, True where the unit is
            padding.
        keep_key: Batch key of a bool ``[B, N]`` mask of padded units that
            still count (VPP2's terminal steps); ``unit="step"`` only.
        unit: ``"step"``: one column per unit on axis 1 of the raw tensor
            (action steps). ``"video_frame"``: one column per pixel frame,
            folded onto the latent frames on axis 2 by the VAE temporal stride:
            column 0 is latent frame 0, each next ``stride`` columns one latent
            frame, padded only when all its columns are (``:487-499``).
    """

    pad_key: str
    keep_key: str | None = None
    unit: str = "step"

    def __post_init__(self) -> None:
        if not self.pad_key or self.keep_key == "":
            raise ValueError(
                "BatchPadding: pad_key must be a nonempty batch key, and so must "
                f"keep_key when given; got pad_key={self.pad_key!r}, "
                f"keep_key={self.keep_key!r}")
        if self.keep_key == self.pad_key:
            raise ValueError(
                f"BatchPadding: keep_key and pad_key are both {self.pad_key!r}")
        if self.unit not in _PADDING_UNITS:
            raise ValueError(
                f"BatchPadding.unit must be one of {_PADDING_UNITS}, got {self.unit!r}")
        if self.keep_key is not None and self.unit != "step":
            raise ValueError(
                f"BatchPadding.keep_key applies to unit='step' only; a padded "
                f"{self.unit} has no counterpart of a kept terminal step")

    @property
    def axis(self) -> int:
        """Axis of the raw tensor holding the units: 1 for steps, 2 for frames."""
        return 2 if self.unit == "video_frame" else 1

    @property
    def batch_keys(self) -> tuple[str, ...]:
        """The batch keys this declaration reads, in reading order."""
        return (self.pad_key,) if self.keep_key is None else (self.pad_key, self.keep_key)

    def padded(self, batch: Mapping[str, Any], raw: Tensor, *, stream: str,
               temporal_stride: int) -> Tensor | None:
        """Which units of ``raw`` are padding and do not count.

        Only shapes are checked, never values, so this does not sync the host.

        Args:
            batch: The training batch.
            raw: The stream's raw tensor; its batch size and the unit count on
                ``self.axis`` are what the mask must match.
            stream: Stream name, for error messages.
            temporal_stride: ``GeometrySpec.vae_temporal_stride``, used by
                ``unit="video_frame"``.

        Returns:
            bool ``[B, units]``, True where the unit is padding and not kept;
            None when the batch has no ``pad_key``.

        Raises:
            ValueError: The batch has ``keep_key`` without ``pad_key``, or a
                mask does not match the raw tensor.
        """
        pad = batch.get(self.pad_key)
        keep = None if self.keep_key is None else batch.get(self.keep_key)
        if pad is None:
            if keep is not None:
                raise ValueError(
                    f"stream {stream!r}: the batch has {self.keep_key!r} but not "
                    f"{self.pad_key!r}, so the padded units it keeps are undefined")
            return None
        if raw.ndim <= self.axis or (self.unit == "video_frame" and raw.ndim != 5):
            raise ValueError(
                f"stream {stream!r}: BatchPadding(unit={self.unit!r}) counts units "
                f"on axis {self.axis} of a raw "
                f"{'video [B, C, F, H, W]' if self.unit == 'video_frame' else 'tensor'}, "
                f"but the raw tensor is {tuple(raw.shape)}")
        units = int(raw.shape[self.axis])
        pad = self._columns(pad, self.pad_key, raw, stream=stream)
        if self.unit == "video_frame":
            pad = self._fold_frames(pad, units, temporal_stride, stream=stream)
        elif pad.shape[1] != units:
            raise ValueError(
                f"stream {stream!r}: batch[{self.pad_key!r}] has {pad.shape[1]} "
                f"columns, the raw tensor {units} units on axis {self.axis}")
        if keep is not None:
            keep = self._columns(keep, self.keep_key, raw, stream=stream)
            if keep.shape != pad.shape:
                raise ValueError(
                    f"stream {stream!r}: batch[{self.keep_key!r}] is "
                    f"{tuple(keep.shape)}, batch[{self.pad_key!r}] {tuple(pad.shape)}")
            pad = pad & ~keep
        return pad

    @staticmethod
    def _columns(mask: Any, key: str, raw: Tensor, *, stream: str) -> Tensor:
        if not isinstance(mask, Tensor) or mask.ndim != 2 or mask.shape[0] != raw.shape[0]:
            shape = tuple(mask.shape) if isinstance(mask, Tensor) else type(mask).__name__
            raise ValueError(
                f"stream {stream!r}: batch[{key!r}] must be a [B={raw.shape[0]}, N] "
                f"mask, got {shape}")
        return mask.to(device=raw.device, dtype=torch.bool)

    def _fold_frames(self, pad: Tensor, frames: int, stride: int, *, stream: str) -> Tensor:
        columns = int(pad.shape[1])
        if stride < 1 or columns != 1 + stride * (frames - 1):
            raise ValueError(
                f"stream {stream!r}: batch[{self.pad_key!r}] has {columns} pixel-frame "
                f"columns, which do not fold onto {frames} latent frames with VAE "
                f"temporal stride {stride} (that needs 1 + {stride} x ({frames} - 1))")
        tail = pad[:, 1:].reshape(pad.shape[0], frames - 1, stride).all(dim=2)
        return torch.cat([pad[:, :1], tail], dim=1)


@dataclass(frozen=True)
class CosineDistance:
    """``1 - cos``, taken along the **last dimension**. Used by the
    self-distillation alignment family.

    The existing ``semantic_future`` of InternW0-Δ is this case
    (``losses.py:58-72``, computed directly on 3072-dim raw hidden states).
    Not interchangeable with MSE: it is insensitive to vector norm, MSE is not.

    Attributes:
        eps: Numerical floor passed to ``cosine_similarity``; must be positive.
    """

    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise ValueError(
                f"CosineDistance.eps must be positive, got {self.eps}")

    def __call__(self, prediction: Tensor, target: Tensor, *,
                 valid: Tensor | None = None) -> Tensor:
        _reject_valid(valid, objective="CosineDistance")
        _check_same_shape(prediction, target, objective="CosineDistance")
        cos = torch.nn.functional.cosine_similarity(
            prediction.float(), target.float(), dim=-1, eps=self.eps)
        distance = 1.0 - cos
        # cos already consumed the last dim: ``[B, D]`` => ``[B]`` (no further
        # mean needed), ``[B, L, D]`` => ``[B, L]`` (mean over the token axis)
        if distance.ndim == 1:
            return distance
        return distance.flatten(1).mean(dim=1)


class TimestepWeight(Protocol):
    """Sigma-dimension loss weight: per-sample weight from the step's timestep.

    The input is the training timestep ``DiffusionRuntime.model_timesteps``
    (already cast per ``RngPolicy``), not sigma: the Wan family computes its
    weight from the cast t in float32, and multiplying sigma back by
    ``num_train_timesteps`` misses by one bf16 step (Lane C v2 §1.4).
    """

    def check(self, sampler: TimestepSampler, *, loss: str) -> None:
        """Construction-time check against the source stream's sampler."""
        ...

    def __call__(self, t: Tensor, *, sampler: TimestepSampler,
                 num_train_timesteps: int) -> Tensor | None:
        """``[B]`` training timesteps -> ``[B]`` weights, or None for no weighting."""
        ...


@dataclass(frozen=True)
class UniformWeight:
    """No sigma-dimension weighting. The default."""

    def check(self, sampler: TimestepSampler, *, loss: str) -> None:
        del sampler, loss

    def __call__(self, t: Tensor, *, sampler: TimestepSampler,
                 num_train_timesteps: int) -> None:
        # None, not ones: the reducer skips the multiply instead of running it.
        del t, sampler, num_train_timesteps


@lru_cache(maxsize=None)
def _bsmntw_stats(shift: float, steps: int) -> tuple[float, float]:
    """Minimum and mean-normalizer of the bell curve on the training grid.

    Same grid as master ``WanContinuousFlowMatchScheduler``: descending fp64 u
    excluding the endpoint 0, pushed through the shift and scaled to steps.
    """
    u = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
    t = shift * u / (1.0 + (shift - 1.0) * u) * float(steps)
    y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
    y_min = float(y.min().item())
    return y_min, float((y - y_min).mean().item())


@dataclass(frozen=True)
class Bsmntw:
    """Bell-shaped, mean-normalized timestep weight (DiffSynth ``bsmntw``).

    ``exp(-2 * ((t - T/2) / T)^2)``, shifted so its minimum on the training
    grid is 0 and divided by its mean there. The grid depends on the source
    stream's ``UniformShift.shift``, which is read from the sampler rather
    than written here a second time. Used by the VPP2 action loss and both
    fastwam losses (master ``scheduler.py:training_weight``).

    Attributes:
        eps: Floor added to the normalizer; master uses 1e-10.
    """

    eps: float = 1e-10

    def check(self, sampler: TimestepSampler, *, loss: str) -> None:
        if isinstance(sampler.distribution, Constant):
            # The grid minimum sits at u=1, so a Constant(1.0) stream would get
            # weight 0 forever: a silently dead loss term (v2 T24).
            raise ValueError(
                f"LossSpec({loss!r}) weights a Constant timestep stream with Bsmntw, "
                "whose weight at sigma=1 is exactly 0; use UniformWeight")
        if not isinstance(sampler.distribution, UniformShift):
            raise ValueError(
                f"LossSpec({loss!r}): Bsmntw normalizes on the UniformShift training "
                f"grid, but the source stream draws from {type(sampler.distribution).__name__}")

    def __call__(self, t: Tensor, *, sampler: TimestepSampler,
                 num_train_timesteps: int) -> Tensor:
        y_min, norm = _bsmntw_stats(float(sampler.distribution.shift), int(num_train_timesteps))
        t = t.to(dtype=torch.float32)
        steps = float(num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        weight = (y - y_min) / (norm + self.eps)
        return weight.reshape(()) if weight.numel() == 1 else weight


@dataclass(frozen=True)
class WeightRamp:
    """Linear ramp-up / decay of a loss weight over training steps.

    Why it must live in the spec rather than in the trainer: the weight curve
    changes the converged result, and "at which step does this loss switch on"
    is today scattered across ``if`` statements in training scripts, so running
    a different experiment means editing code. In the spec it is configuration,
    and I4′ lets the trainer only read it, never compute it.

    Attributes:
        start_step: Step at which the ramp begins.
        end_step: Step at which the end value is reached; equal to
            ``start_step`` gives a step function.
        start_value: Weight at and before ``start_step``.
        end_value: Weight at and after ``end_step``.
    """

    start_step: int = 0         # step where the ramp begins; 0 when YAML adds the ramp
    end_step: int = 0           # step reaching end value; == start => step. YAML heads.<head>.warmup_steps
    start_value: float = 1.0    # weight at and before start_step; 0.0 when YAML adds the ramp
    end_value: float = 1.0      # weight at and after end_step. YAML heads.<head>.weight

    def __post_init__(self) -> None:
        if self.start_step < 0 or self.end_step < 0:
            raise ValueError(
                f"WeightRamp steps must be non-negative, got "
                f"start_step={self.start_step}, end_step={self.end_step}"
            )
        if self.end_step < self.start_step:
            raise ValueError(
                f"WeightRamp.end_step({self.end_step}) must not be less than "
                f"start_step({self.start_step})"
            )

    @property
    def is_constant(self) -> bool:
        """Constant weight. Default construction lands here, and the whole ramp
        computation is then skipped."""
        return self.start_value == self.end_value

    def scale(self, step: int) -> float:
        """Weight at step ``step``.

        Below ``start_step`` it is ``start_value``, at or above ``end_step`` it
        is ``end_value``, and linearly interpolated in between.

        Args:
            step: Global training step; must be non-negative.

        Returns:
            The weight multiplier for that step.

        Raises:
            ValueError: If ``step`` is negative.
        """
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        if self.is_constant:
            return self.start_value
        if step <= self.start_step:
            return self.start_value
        if step >= self.end_step:
            return self.end_value
        span = self.end_step - self.start_step
        ratio = (step - self.start_step) / span
        return self.start_value + ratio * (self.end_value - self.start_value)


@dataclass(frozen=True)
class LossSpec:
    """One loss term, logged as ``loss_<name>`` in the ``metrics`` dict (I4).

    Exactly **one** of ``stream`` and ``head`` may be set: a loss is computed
    either on the output of one stream (the video stream's flow matching loss)
    or on one output head (the action head's L1). Setting both is ambiguous;
    setting neither is a dead configuration.

    Attributes:
        name: Term name; the ``metrics`` key is ``loss_<name>``.
        stream: Stream this loss is computed on, or None.
        head: Head this loss is computed on, or None.
        objective: What is computed between prediction and target.
        weight: Constant weight.
        ramp: Step-dependent weight; overrides ``weight`` when not None.
        broadcast: Which label axes the per-sample weighting varies along.
        enabled: False forces the weight to 0 without removing the term.
        weight_of: Sigma-dimension per-sample weight, read from the source
            stream's training timestep; ``UniformWeight`` means none.
    """

    name: str
    stream: str | None = None
    head: str | None = None
    objective: LossObjective = MeanSquaredError()
    # What is computed. **Not YAML-overridable**: changing the objective is
    # changing the training problem (R5). It has a default because 12 of the 13
    # models use MSE, so writing it once is enough
    weight: float = 1.0                     # [YAML] heads.<head>.weight: constant weight
    ramp: WeightRamp | None = None          # [YAML] heads.<head>.warmup_steps: not None => overrides weight
    broadcast: BroadcastSpec = DEFAULT_PER_SAMPLE
    # broadcast decides "along which axes the loss is weighted before it is
    # averaged": models with a per-frame timestep weight per frame, and the axis
    # names must come from the layout labels (I2)
    enabled: bool = True                    # [YAML] heads.<head>.enabled: False => weight 0, term kept
    weight_of: TimestepWeight = UniformWeight()
    # Algorithm, not YAML (R5): which sigma weighting is part of the objective

    def __post_init__(self) -> None:
        if (self.stream is None) == (self.head is None):
            raise ValueError(
                f"LossSpec({self.name!r}): exactly one of stream and head must "
                f"be set, got stream={self.stream!r}, head={self.head!r}"
            )
        if self.weight < 0:
            raise ValueError(
                f"LossSpec({self.name!r}).weight must be non-negative, "
                f"got {self.weight}"
            )

    @property
    def source(self) -> str:
        """Who this loss is computed on. Used in error messages and by V8
        reconciliation."""
        return self.stream if self.stream is not None else str(self.head)

    def weight_at(self, step: int) -> float:
        """Effective weight at step ``step``. A None ``ramp`` means constant
        ``weight``.

        ``enabled=False`` yields 0.0 rather than dropping the term from the
        list: dropping it would make the key set of ``metrics`` vary with the
        configuration, which breaks downstream dashboards.

        Args:
            step: Global training step.

        Returns:
            The weight to multiply this loss term by.
        """
        if not self.enabled:
            return 0.0
        if self.ramp is None:
            return self.weight
        return self.ramp.scale(step)
