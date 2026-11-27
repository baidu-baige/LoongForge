# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 supervision specs, objectives, gates, reducers and loss masks (Lane C).

Decoupled from ``diffusion/`` so non-diffusion objectives (self-distillation
``CosineDistance``, curriculum ``WeightRamp``, batch-sourced regression or
classification targets, per-sample gates, and custom reducers) live outside the
diffusion package. Diffusion-specific timestep weights (``Bsmntw``) implement
``TimestepWeight`` inside ``diffusion/weights.py``.

The **number** of loss terms differs per model (1 / 2 / 3 / 5 terms), so
``WorldModelSpec.losses`` is a tuple rather than a fixed set of parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch import Tensor
from torch.nn import functional as F

from .broadcast import DEFAULT_PER_SAMPLE, BroadcastSpec

if TYPE_CHECKING:
    from .diffusion.timesteps import TimestepSampler
    from .layout import TokenLayout

__all__ = [
    "BatchGate",
    "LossMaskSpec",
    "CosineDistance",
    "LossGate",
    "LossObjective",
    "LossSpec",
    "MeanSquaredError",
    "Reducer",
    "TimestepWeight",
    "TokenMeanSquaredError",
    "UniformWeight",
    "WeightRamp",
]


class LossObjective(Protocol):
    """What is computed between prediction and target, producing ``[B]``.

    Why not a ``"mse"`` / ``"cosine"`` string: in the existing code that shows
    up as three separate ``if`` chains (weighting, logging, backward), so adding
    an objective means editing three places. As an object it has one
    implementation site.

    Why it returns ``[B]`` rather than a scalar: besides ``LossSpec.weight_at``
    there is per-sample weighting (``LossSpec.weight_of``) and gating
    (``LossSpec.gate``), and applying either after the batch mean is not
    equivalent. Models whose timestep weight varies per frame or token pass a
    ``Reducer`` on ``LossSpec.reducer``, which receives the unreduced error via
    ``unreduced`` when implemented.
    """

    def __call__(self, prediction: Tensor, target: Tensor, *,
                 valid: Tensor | None = None) -> Tensor:
        """``[B, ...]`` x ``[B, ...]`` -> ``[B]``.

        ``valid`` is a bool ``[B, T]`` over the objective's token axis (True
        counts); an objective without a masked reduction raises when given one.
        """
        ...


class Reducer(Protocol):
    """Reduces an uncollapsed error tensor to a 0-D scalar term (Lane C v2 §1.5).

    Used when weighting or masking must happen *before* collapsing intra-sample
    axes (e.g. per-frame ``bsmntw`` in DreamZero / LingBot-VA, whole-batch
    masked token means in GR00T, or DP-collective reductions in InternW0-Delta).
    """

    def __call__(self, error: Tensor, *,
                 weight: Tensor | None = None,
                 valid: Tensor | None = None,
                 active: Tensor | None = None,
                 layout: TokenLayout | None = None) -> Tensor:
        """Reduce ``error`` ``[B, ...]`` to a 0-D scalar tensor."""
        ...


class LossGate(Protocol):
    """Per-sample runtime gate for a loss term (Lane C v2 §1.9, §1.17).

    Returns a bool ``[B]`` tensor (True counts) or ``None`` when every sample in
    the batch is active. Gated-out samples contribute 0 while keeping the head
    output attached to the autograd graph so DDP/FSDP never hangs on an all-off
    micro-batch.
    """

    @property
    def batch_keys(self) -> tuple[str, ...]:
        """Batch keys the shell should forward into ``WorldModel`` when present."""
        ...

    def __call__(self, batch: Mapping[str, Any], *,
                 timesteps: Mapping[str, Tensor]) -> Tensor | None:
        """Return bool ``[B]`` active mask, or ``None`` for all active."""
        ...


@dataclass(frozen=True)
class BatchGate:
    """Gate a loss term by a bool ``[B]`` mask stored in ``batch[key]``.

    Covers datasets where only a subset of samples carry supervision for a head
    (e.g. DreamZero's ``has_real_action``). When ``key`` is absent from the
    batch, all samples are active.

    Attributes:
        key: Non-empty batch key holding a bool ``[B]`` tensor.
    """

    key: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError(f"BatchGate.key must be a non-empty string, got {self.key!r}")

    @property
    def batch_keys(self) -> tuple[str, ...]:
        return (self.key,)

    def __call__(self, batch: Mapping[str, Any], *,
                 timesteps: Mapping[str, Tensor]) -> Tensor | None:
        del timesteps
        mask = batch.get(self.key)
        if mask is None:
            return None
        if not isinstance(mask, Tensor) or mask.ndim != 1:
            shape = tuple(mask.shape) if isinstance(mask, Tensor) else type(mask).__name__
            raise ValueError(f"BatchGate({self.key!r}): expected a 1-D [B] mask, got {shape}")
        return mask.to(dtype=torch.bool)


def _reject_valid(valid: Tensor | None, *, objective: str) -> None:
    if valid is not None:
        raise ValueError(
            f"{objective} has no masked reduction, but this loss has a "
            "loss_valid mask; use TokenMeanSquaredError or a Reducer")


def _check_same_shape(prediction: Tensor, target: Tensor, *,
                      objective: str) -> None:
    """Prediction and target must have the **same shape**; no broadcasting."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"{objective}: prediction {tuple(prediction.shape)} and target "
            f"{tuple(target.shape)} have different shapes. A head's output "
            "shape must equal its target shape; a mismatch is not covered up "
            "by broadcasting"
        )
    if prediction.ndim < 2:
        raise ValueError(
            f"{objective}: expected at least 2 dims [B, ...], "
            f"got {tuple(prediction.shape)}"
        )


@dataclass(frozen=True)
class MeanSquaredError:
    """Elementwise squared error averaged over every non-batch axis -> ``[B]``.

    **12 of the 13 models use MSE**, so it is the default. It does not sum first
    and divide later: video tokens outnumber action tokens by two orders of
    magnitude, so summing would let tensor size rather than ``LossSpec.weight``
    decide the relative weight of two losses.
    """

    def unreduced(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Elementwise float32 squared error ``[B, ...]`` for a ``Reducer``."""
        _check_same_shape(prediction, target, objective="MeanSquaredError")
        return (prediction.float() - target.float()).square()

    def __call__(self, prediction: Tensor, target: Tensor, *,
                 valid: Tensor | None = None) -> Tensor:
        _reject_valid(valid, objective="MeanSquaredError")
        return self.unreduced(prediction, target).flatten(1).mean(dim=1)


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
    per-frame video loss (``:476``, ``.mean(dim=(1, 3, 4))``).

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

    def unreduced(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Per-unit float32 MSE ``[B, T]`` collapsed over feature axes."""
        _check_same_shape(prediction, target, objective="TokenMeanSquaredError")
        if prediction.ndim < 3 or self.axis >= prediction.ndim or not all(prediction.shape):
            raise ValueError(
                f"TokenMeanSquaredError(axis={self.axis}): expected a nonempty "
                "[B, ...] prediction with the token axis and at least one "
                f"feature axis, got {tuple(prediction.shape)}")
        features = tuple(dim for dim in range(1, prediction.ndim) if dim != self.axis)
        return F.mse_loss(prediction.float(), target.float(), reduction="none").mean(
            dim=features)

    def __call__(self, prediction: Tensor, target: Tensor, *,
                 valid: Tensor | None = None) -> Tensor:
        token_loss = self.unreduced(prediction, target)
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
class LossMaskSpec:
    """Per-sample loss mask read from the batch to exclude padded units.

    Reads dataset padding flags (such as ``action_is_pad`` or ``image_is_pad``)
    and masks those padded steps/frames out of loss reduction without changing
    any static tensor shape (unlike L0/L1 alignment ``is_padding`` slots).
    Usable both per stream on ``WorldModelSpec.loss_masks`` and per loss term
    on ``LossSpec.loss_mask``. A batch without ``pad_key`` has no masked units,
    matching legacy ``sample.get("action_is_pad", None)``.

    Attributes:
        pad_key: Batch key of a bool ``[B, N]`` mask, True where the unit is
            padding.
        keep_key: Batch key of a bool ``[B, N]`` mask of padded units that
            still count (VPP2's terminal steps); ``unit="step"`` only.
        unit: ``"step"``: one column per unit on axis 1 of the tensor.
            ``"video_frame"``: one column per pixel frame, folded onto the
            latent frames on axis 2 by the VAE temporal stride.
    """

    pad_key: str
    keep_key: str | None = None
    unit: str = "step"

    def __post_init__(self) -> None:
        if not self.pad_key or self.keep_key == "":
            raise ValueError(
                "LossMaskSpec: pad_key must be a nonempty batch key, and so must "
                f"keep_key when given; got pad_key={self.pad_key!r}, "
                f"keep_key={self.keep_key!r}")
        if self.keep_key == self.pad_key:
            raise ValueError(
                f"LossMaskSpec: keep_key and pad_key are both {self.pad_key!r}")
        if self.unit not in _PADDING_UNITS:
            raise ValueError(
                f"LossMaskSpec.unit must be one of {_PADDING_UNITS}, got {self.unit!r}")
        if self.keep_key is not None and self.unit != "step":
            raise ValueError(
                f"LossMaskSpec.keep_key applies to unit='step' only; a padded "
                f"{self.unit} has no counterpart of a kept terminal step")

    @property
    def axis(self) -> int:
        """Axis of the tensor holding the units: 1 for steps, 2 for frames."""
        return 2 if self.unit == "video_frame" else 1

    @property
    def batch_keys(self) -> tuple[str, ...]:
        """The batch keys this declaration reads, in reading order."""
        return (self.pad_key,) if self.keep_key is None else (self.pad_key, self.keep_key)

    def padded(self, batch: Mapping[str, Any], raw: Tensor, *, stream: str,
               temporal_stride: int) -> Tensor | None:
        """Which units of ``raw`` are padding and do not count.

        Only shapes are checked, never values, so this does not sync the host.
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
                f"stream {stream!r}: LossMaskSpec(unit={self.unit!r}) counts units "
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
    self-distillation alignment family (InternW0-Delta ``semantic_future``).

    Attributes:
        eps: Numerical floor passed to ``cosine_similarity``; must be positive.
    """

    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise ValueError(
                f"CosineDistance.eps must be positive, got {self.eps}")

    def unreduced(self, prediction: Tensor, target: Tensor) -> Tensor:
        """``1 - cos`` collapsed along the feature axis ``-1``."""
        _check_same_shape(prediction, target, objective="CosineDistance")
        cos = torch.nn.functional.cosine_similarity(
            prediction.float(), target.float(), dim=-1, eps=self.eps)
        return 1.0 - cos

    def __call__(self, prediction: Tensor, target: Tensor, *,
                 valid: Tensor | None = None) -> Tensor:
        _reject_valid(valid, objective="CosineDistance")
        distance = self.unreduced(prediction, target)
        if distance.ndim == 1:
            return distance
        return distance.flatten(1).mean(dim=1)


class TimestepWeight(Protocol):
    """Timestep-dimension loss weight: per-sample or per-unit weight from ``t``.

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
        """Training timesteps -> weights, or ``None`` for no weighting."""
        ...


@dataclass(frozen=True)
class UniformWeight:
    """No timestep-dimension weighting. The default."""

    def check(self, sampler: TimestepSampler, *, loss: str) -> None:
        del sampler, loss

    def __call__(self, t: Tensor, *, sampler: TimestepSampler,
                 num_train_timesteps: int) -> None:
        del t, sampler, num_train_timesteps


@dataclass(frozen=True)
class WeightRamp:
    """Linear ramp-up / decay of a loss weight over training steps.

    Attributes:
        start_step: Step at which the ramp begins.
        end_step: Step at which the end value is reached; equal to
            ``start_step`` gives a step function.
        start_value: Weight at and before ``start_step``.
        end_value: Weight at and after ``end_step``.
    """

    start_step: int = 0
    end_step: int = 0
    start_value: float = 1.0
    end_value: float = 1.0

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
        """Constant weight. Default construction lands here."""
        return self.start_value == self.end_value

    def scale(self, step: int) -> float:
        """Weight multiplier at global training ``step``."""
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
    """One supervision loss term on ``WorldModelSpec.losses``, logged as ``loss_<name>``.

    Exactly **one** of ``stream`` and ``head`` may be set: a loss is computed
    either on the output of one stream or on one output head. Setting both is
    ambiguous; setting neither is a dead configuration.

    Attributes:
        name: Term name; the ``metrics`` key is ``loss_<name>``.
        stream: Stream this loss is computed on, or None.
        head: Head this loss is computed on, or None.
        objective: Pointwise / per-sample objective (MSE by default).
        weight: Constant scalar coefficient.
        ramp: Step-dependent weight; overrides ``weight`` when not None.
        broadcast: Which label axes the timestep weighting varies along.
        enabled: False forces the weight to 0 without removing the term.
        weight_of: Timestep-dimension weight (``UniformWeight`` means none).
        target_key: Optional batch key holding the supervision target directly
            (offline teacher features, autoregressive labels, auxiliary heads).
        loss_mask: Optional per-term ``LossMaskSpec`` read from the batch to
            exclude padded steps/frames from loss reduction.
        gate: Optional ``LossGate`` for per-sample runtime gating; logs
            ``loss_<name>_active`` when set.
        reducer: Optional ``Reducer`` taking unreduced error, timestep weights,
            ``valid``, ``active`` and ``layout`` before collapsing to a scalar.
    """

    name: str
    stream: str | None = None
    head: str | None = None
    objective: LossObjective = MeanSquaredError()
    weight: float = 1.0
    ramp: WeightRamp | None = None
    broadcast: BroadcastSpec = DEFAULT_PER_SAMPLE
    enabled: bool = True
    weight_of: TimestepWeight = UniformWeight()
    target_key: str | None = None
    loss_mask: LossMaskSpec | None = None
    gate: LossGate | None = None
    reducer: Reducer | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("LossSpec.name must be non-empty")
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
        if self.target_key is not None and (not isinstance(self.target_key, str) or not self.target_key):
            raise ValueError(
                f"LossSpec({self.name!r}).target_key must be a non-empty batch key, "
                f"got {self.target_key!r}"
            )
        if self.loss_mask is not None and not isinstance(self.loss_mask, LossMaskSpec):
            raise TypeError(
                f"LossSpec({self.name!r}).loss_mask must be a LossMaskSpec, "
                f"got {type(self.loss_mask).__name__}"
            )

    @property
    def source(self) -> str:
        """Who this loss is computed on."""
        return self.stream if self.stream is not None else str(self.head)

    def weight_at(self, step: int) -> float:
        """Effective scalar coefficient at global ``step``."""
        if not self.enabled:
            return 0.0
        if self.ramp is None:
            return self.weight
        return self.ramp.scale(step)
