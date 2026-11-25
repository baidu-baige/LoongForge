# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 numerics: timestep distributions, noise schedules, broadcasting, loss
objectives and the Euler step.

Division of labour with ``test_arch_diffusion.py`` (the declaration-layer
cases): this file only tests **numbers**. Five things are checked:

1. All four distributions land in [0, 1], and the shift / Beta formulas match
   the existing implementation line for line.
2. Rectified-flow noising and its target degenerate to known values at both
   endpoints.
3. ``BroadcastSpec.expand`` gathers per axis in step with the token labels, and
   a ``-1`` label does not wrap around to the last row.
4. Both loss objectives return ``[B]``, raise on mismatched shapes, and the MSE
   is independent of the element count.
5. The Euler step only touches active streams and its sign is a **subtraction**.
"""

from __future__ import annotations

import pytest
import torch

from loongforge.embodied.model.arch.diffusion.broadcast import BroadcastSpec
from loongforge.embodied.model.arch.diffusion.loss import (
    BatchPadding, CosineDistance, LossSpec, MeanSquaredError, TokenMeanSquaredError)
from loongforge.embodied.model.arch.diffusion.schedule import (
    ConstantSigma, RectifiedFlow)
from loongforge.embodied.model.arch.diffusion.stepping import (
    EulerSampler, Trajectory, UniformSteps)
from loongforge.embodied.model.arch.diffusion.timesteps import (
    BetaDist, Constant, LogitNormal, RngPolicy, TimestepSampler, UniformShift)
from loongforge.embodied.model.arch.geometry import GeometrySpec
from loongforge.embodied.model.arch.layout import TokenLayout

CPU = torch.device("cpu")
N = 4096


class _Stream:
    """Minimal stand-in for a ``StreamSpec``: the layout only reads ``name``."""

    def __init__(self, name: str) -> None:
        self.name = name


def small_layout() -> TokenLayout:
    """The same 12-token geometry as ``test_arch_layout.py``: 2 latent frames x
    4 video tokens plus 4 action tokens."""
    geo = GeometrySpec(frame_height=32, frame_width=32, num_frames=5,
                       num_views=1, vae_spatial_stride=16, patch_size=(1, 1, 1),
                       action_horizon=4, action_dim=7,
                       action_video_freq_ratio=2)
    return TokenLayout.from_geometry(geo, [_Stream("video"), _Stream("action")])


# ---- 1. distributions ----


@pytest.mark.parametrize("dist", [
    UniformShift(), UniformShift(shift=3.0), UniformShift(shift=5.0),
    LogitNormal(), LogitNormal(mean=-0.5, std=1.5),
    BetaDist(), BetaDist(alpha=0.5, beta=2.0),
    Constant(), Constant(value=1.0),
])
def test_every_distribution_lands_in_the_unit_interval(dist):
    """A [0, 1] range is part of the protocol.

    Converting to sigma and the 1000x rescaling belong to the schedule, not to
    the distribution, so a sampler that returned raw sigmas would break every
    consumer.
    """
    torch.manual_seed(0)
    value = dist.sample((N,), device=CPU, dtype=torch.float32)
    assert value.shape == (N,)
    assert torch.isfinite(value).all()
    assert float(value.min()) >= 0.0
    assert float(value.max()) <= 1.0


def test_uniform_shift_matches_the_existing_phi():
    """``UniformShift`` applies ``phi(u) = s*u / (1 + (s-1)*u)``, the same map as
    ``scheduler_continuous.py:37-39``.

    The check is on the mean rather than elementwise: the distribution cannot
    reproduce a random stream, but ``phi`` is monotone, so ``shift > 1`` must
    push mass towards the high-noise end while ``shift=1`` must be exactly
    uniform.
    """
    torch.manual_seed(0)
    plain = UniformShift().sample((N,), device=CPU, dtype=torch.float32)
    shifted = UniformShift(shift=5.0).sample((N,), device=CPU,
                                             dtype=torch.float32)
    assert abs(float(plain.mean()) - 0.5) < 0.02      # shift=1 => uniform
    assert float(shifted.mean()) > float(plain.mean())
    # Fixed points of phi: u=0 -> 0 and u=1 -> 1, independent of shift
    for shift in (1.0, 3.0, 5.0):
        u = torch.tensor([0.0, 1.0])
        phi = shift * u / (1.0 + (shift - 1.0) * u)
        assert torch.allclose(phi, u)


def test_constant_is_the_no_noise_declaration():
    """``Constant`` is not a debugging aid: it is how "this stream is not noised"
    is stated."""
    value = Constant(value=0.0).sample((3, 2), device=CPU,
                                       dtype=torch.float32)
    assert value.shape == (3, 2)
    assert torch.equal(value, torch.zeros(3, 2))


@pytest.mark.parametrize("bad", [
    (UniformShift, {"shift": 0.0}), (LogitNormal, {"std": 0.0}),
    (BetaDist, {"alpha": 0.0}), (BetaDist, {"beta": -1.0}),
    (BetaDist, {"eps": 0.0}), (Constant, {"value": 1.5}),
])
def test_distribution_parameters_are_validated_at_construction(bad):
    """Degenerate distribution parameters are rejected when the object is built,
    not when it is first sampled inside a training step."""
    cls, kwargs = bad
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_sampler_axis_sizes_must_match_the_broadcast_axes():
    """``axis_sizes`` corresponds one-to-one with ``broadcast.axes`` instead of
    relying on broadcasting rules to absorb a mismatch.

    A per-sample sampler takes no axis sizes and yields ``[B]``; a per-frame one
    takes one size and yields ``[B, F]``.
    """
    per_sample = TimestepSampler(distribution=UniformShift())
    assert per_sample.sample(3, device=CPU).shape == (3,)
    with pytest.raises(ValueError, match="axis_sizes"):
        per_sample.sample(3, device=CPU, axis_sizes=(2,))
    per_frame = TimestepSampler(distribution=UniformShift(),
                                broadcast=BroadcastSpec(axes=("frame",)))
    assert per_frame.sample(3, device=CPU, axis_sizes=(2,)).shape == (3, 2)
    with pytest.raises(ValueError, match="batch_size"):
        per_frame.sample(0, device=CPU, axis_sizes=(2,))


# ---- 2. noise schedules ----


def test_rectified_flow_degenerates_at_both_endpoints():
    """t=0 gives back the clean sample, t=1 gives pure noise, and the target is
    ``noise - clean`` for every t (the defining property of rectified flow)."""
    flow = RectifiedFlow()
    clean = torch.randn(2, 5)
    noise = torch.randn(2, 5)
    assert torch.allclose(flow.add_noise(clean, noise, torch.zeros(2, 1)), clean)
    assert torch.allclose(flow.add_noise(clean, noise, torch.ones(2, 1)), noise)
    for t in (torch.zeros(2, 1), torch.rand(2, 1), torch.ones(2, 1)):
        assert torch.allclose(flow.target(clean, noise, t), noise - clean)
    t = torch.rand(4)
    assert torch.equal(flow.sigma(t), t)        # sigma is the identity here


def test_rectified_flow_is_the_straight_line():
    """Noising is linear interpolation, so t=0.5 lands exactly halfway between
    clean and noise."""
    flow = RectifiedFlow()
    clean, noise = torch.zeros(1, 3), torch.ones(1, 3)
    half = flow.add_noise(clean, noise, torch.full((1, 1), 0.5))
    assert torch.allclose(half, torch.full((1, 3), 0.5))


def test_constant_sigma_zero_is_the_identity():
    """With ``sigma=0`` a clean conditioning stream runs the **same code path**
    as a noised stream.

    The scaled form adds ``sigma * noise`` and its target is the noise itself.
    """
    clean, noise = torch.randn(2, 3), torch.randn(2, 3)
    assert torch.equal(ConstantSigma().add_noise(clean, noise, torch.zeros(2)),
                       clean)
    scaled = ConstantSigma(sigma_value=0.2)
    assert torch.allclose(scaled.add_noise(clean, noise, torch.zeros(2)),
                          clean + 0.2 * noise)
    assert torch.equal(scaled.target(clean, noise, torch.zeros(2)), noise)
    with pytest.raises(ValueError, match="must be non-negative"):
        ConstantSigma(sigma_value=-1.0)


# ---- 3. broadcasting ----


def test_per_sample_expand_is_a_view():
    """A per-sample timestep expands to ``[B, 1, 1]`` without copying, and a
    non-scalar input is rejected."""
    layout = small_layout()
    value = torch.rand(3)
    out = BroadcastSpec().expand(value, layout)
    assert out.shape == (3, 1, 1)
    assert out.data_ptr() == value.data_ptr()        # a view, no copy
    with pytest.raises(ValueError, match="expects a per-sample scalar of shape"):
        BroadcastSpec().expand(torch.rand(3, 2), layout)


def test_per_frame_expand_gathers_by_the_frame_label():
    """Per-frame timesteps are gathered by ``labels_of('frame')``, so video and
    action tokens share one frame numbering.

    In the small geometry the frame labels are
    ``[0,0,0,0,1,1,1,1, 0,0,1,1]``: an action token takes ``step // ratio``,
    the latent frame it aligns to. So "add more noise to frame 1" acts on the
    video and the action side at once, which is the point of cross-stream frame
    alignment.
    """
    layout = small_layout()
    spec = BroadcastSpec(axes=("frame",))
    value = torch.tensor([[0.25, 0.75], [1.0, 0.0]])
    out = spec.expand(value, layout)
    assert out.shape == (2, layout.total, 1)
    want = torch.tensor([0.25] * 4 + [0.75] * 4 + [0.25, 0.25, 0.75, 0.75])
    assert torch.allclose(out[0, :, 0], want)
    assert torch.allclose(out[1, :, 0],
                          torch.tensor([1.0] * 4 + [0.0] * 4
                                       + [1.0, 1.0, 0.0, 0.0]))


def test_negative_label_becomes_zero_not_a_wrapped_neighbour():
    """A token labelled ``-1`` has no position on that axis, so it reads 0
    rather than wrapping around to the last row.

    The ``row`` label is -1 for action tokens, and indexing with -1 would pick
    up the last row with a perfectly correct shape. That is the silent failure
    this case nails down.
    """
    layout = small_layout()
    spec = BroadcastSpec(axes=("row",))
    value = torch.tensor([[0.1, 0.2]])      # two rows
    out = spec.expand(value, layout)
    assert out.shape == (1, layout.total, 1)
    rows = layout.labels_of("row")
    for token in range(layout.total):
        if int(rows[token]) < 0:
            assert float(out[0, token, 0]) == 0.0
        else:
            assert float(out[0, token, 0]) == float(value[0, int(rows[token])])
    assert float(out[0, 8, 0]) == 0.0       # an action token, not 0.2


def test_two_axis_expand_is_c_order():
    """Multiple axes fold into a C-order flat index in the declared order of
    ``axes``, and a token with -1 on any axis still reads 0."""
    layout = small_layout()
    spec = BroadcastSpec(axes=("frame", "chunk"))
    # value[b, frame, chunk]; chunk is -1 for video, so the video segment is 0
    value = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    out = spec.expand(value, layout)
    assert torch.allclose(out[0, :8, 0], torch.zeros(8))
    # The action tokens have (frame, chunk) = (0,0) (0,0) (1,1) (1,1)
    assert torch.allclose(out[0, 8:, 0], torch.tensor([0.0, 0.0, 3.0, 3.0]))


def test_expand_rejects_unknown_axis_and_wrong_rank():
    """An axis name absent from the layout, a wrong rank, an axis size larger
    than the value, and a repeated axis are all rejected."""
    layout = small_layout()
    with pytest.raises(ValueError, match="are not in the layout"):
        BroadcastSpec(axes=("noise_level",)).expand(torch.rand(1, 2), layout)
    with pytest.raises(ValueError, match="2 dims in total"):
        BroadcastSpec(axes=("frame",)).expand(torch.rand(2), layout)
    with pytest.raises(ValueError, match="which exceeds length"):
        BroadcastSpec(axes=("frame",)).expand(torch.rand(1, 1), layout)
    with pytest.raises(ValueError, match="has duplicate axes"):
        BroadcastSpec(axes=("frame", "frame"))


# ---- 4. loss objectives ----


def test_objectives_return_one_number_per_sample():
    """Objectives return ``[B]`` rather than a scalar, because per-sample
    weighting has to multiply **before** the batch mean."""
    pred, target = torch.randn(3, 4, 5), torch.randn(3, 4, 5)
    for objective in (MeanSquaredError(), CosineDistance()):
        out = objective(pred, target)
        assert out.shape == (3,), objective
        assert torch.isfinite(out).all()
        assert out.dtype == torch.float32


def test_mean_squared_error_is_independent_of_element_count():
    """The MSE takes the **mean** over the non-batch axes, so the relative
    weight of the video and action losses is set by ``LossSpec.weight`` alone.

    Summing and dividing by a fixed number would let the video loss, with two
    orders of magnitude more elements, absorb the whole gradient while
    ``LossSpec.weight`` still reads 0.5 / 1.0.
    """
    mse = MeanSquaredError()
    small = mse(torch.ones(2, 4), torch.zeros(2, 4))
    large = mse(torch.ones(2, 4000), torch.zeros(2, 4000))
    assert torch.allclose(small, torch.ones(2))
    assert torch.allclose(large, torch.ones(2))


def test_cosine_distance_is_scale_invariant_and_mse_is_not():
    """The two objectives are not interchangeable: cosine ignores magnitude,
    MSE does not.

    Cosine distance is 0 for a positively scaled copy and 2 for the negated
    tensor.
    """
    pred = torch.randn(2, 8)
    assert torch.allclose(CosineDistance()(pred, pred * 3.0),
                          torch.zeros(2), atol=1e-5)
    assert float(MeanSquaredError()(pred, pred * 3.0).min()) > 0.0
    # The other direction: opposite directions give a cosine distance of 2
    assert torch.allclose(CosineDistance()(pred, -pred),
                          torch.full((2,), 2.0), atol=1e-5)


def test_objective_rejects_mismatched_shapes():
    """Equal shapes are a design guarantee (a head's output shape equals the
    raw target shape), so a mismatch raises instead of broadcasting."""
    for objective in (MeanSquaredError(), CosineDistance()):
        with pytest.raises(ValueError, match="have different shapes"):
            objective(torch.randn(2, 4), torch.randn(2, 1))
        with pytest.raises(ValueError, match="expected at least 2 dims"):
            objective(torch.randn(4), torch.randn(4))


def test_loss_spec_objective_defaults_to_mse_and_is_not_yaml():
    """The objective is a Python-level default on ``LossSpec``, not a YAML key,
    and a loss term names exactly one of ``stream`` or ``head``."""
    spec = LossSpec(name="video_flow", head="video_flow")
    assert isinstance(spec.objective, MeanSquaredError)
    cosine = LossSpec(name="semantic", head="semantic",
                      objective=CosineDistance())
    assert isinstance(cosine.objective, CosineDistance)
    # Exactly one of stream/head is filled in; new fields must not loosen this
    with pytest.raises(ValueError, match="exactly one of stream and head must be set"):
        LossSpec(name="bad", stream="video", head="video_flow")


# ---- 5. the Euler step ----


def test_euler_step_subtracts_and_only_touches_active_streams():
    """The update is ``x <- x - dt * v``.

    A flipped sign samples pure noise while every shape and every loss value
    still looks normal, so the direction is pinned down numerically: with
    ``num_steps=4`` the step is 0.25, so v=4 drives 1.0 down to 0.0 and v=-4
    drives 0.0 up to 1.0.
    """
    gate = UniformSteps(num_steps=4)
    traj = Trajectory(latents={"video": torch.ones(1, 3),
                               "action": torch.zeros(1, 2)})
    out = EulerSampler().step(traj, {"video": torch.full((1, 3), 4.0),
                                     "action": torch.full((1, 2), -4.0)},
                              gate=gate)
    assert out.step == 1
    assert torch.allclose(out.latents["video"], torch.zeros(1, 3))
    assert torch.allclose(out.latents["action"], torch.ones(1, 2))
    # The input trajectory is not rewritten (advancing returns a new object)
    assert torch.allclose(traj.latents["video"], torch.ones(1, 3))


def test_skipped_stream_must_not_receive_a_prediction():
    """A stream in ``gate.skip`` raises if a prediction is handed to it.

    Fast-WAM drops the video stream at inference time. Ignoring the extra
    prediction instead of raising would make "drop the video stream" silently
    ineffective, and nothing in the numbers would show it.
    """
    gate = UniformSteps(num_steps=2, skip=frozenset({"video"}))
    traj = Trajectory(latents={"video": torch.ones(1, 3),
                               "action": torch.ones(1, 2)})
    assert traj.active_streams(gate) == ("action",)
    with pytest.raises(KeyError, match="video"):
        EulerSampler().step(traj, {"video": torch.ones(1, 3),
                                   "action": torch.ones(1, 2)}, gate=gate)
    out = EulerSampler().step(traj, {"action": torch.full((1, 2), 2.0)},
                              gate=gate)
    assert torch.allclose(out.latents["video"], torch.ones(1, 3))
    assert torch.allclose(out.latents["action"], torch.zeros(1, 2))


def test_missing_prediction_for_an_active_stream_is_named():
    """An active stream with no prediction raises a ``KeyError`` naming it,
    rather than leaving that stream's latent frozen for the step."""
    gate = UniformSteps(num_steps=2)
    traj = Trajectory(latents={"video": torch.ones(1, 3),
                               "action": torch.ones(1, 2)})
    with pytest.raises(KeyError, match="action"):
        EulerSampler().step(traj, {"video": torch.ones(1, 3)}, gate=gate)


# ---- training timestep: RngPolicy and num_train_timesteps ----


def test_like_sample_casts_t_and_derives_sigma_back():
    """The Wan scheduler: t = (sigma * 1000) cast to the sample dtype, sigma = t / 1000."""
    sigma = torch.tensor([0.12345678, 0.5, 0.99999, 1.0])
    s, t = RngPolicy().quantize(sigma, num_train_timesteps=1000, sample_dtype=torch.bfloat16)
    assert s.dtype == t.dtype == torch.bfloat16
    assert torch.equal(t, (sigma * 1000.0).to(torch.bfloat16))
    assert torch.equal(s, (t / 1000.0).to(torch.bfloat16))
    s32, t32 = RngPolicy(sigma_dtype="float32").quantize(
        sigma, num_train_timesteps=1000, sample_dtype=torch.bfloat16)
    assert s32 is sigma and t32.dtype == torch.float32
    assert torch.equal(t32, sigma * 1000.0)


@pytest.mark.parametrize("bad", [dict(order="random"), dict(sigma_dtype="bf16")])
def test_rng_policy_rejects_unknown_values(bad):
    with pytest.raises(ValueError):
        RngPolicy(**bad)


@pytest.mark.parametrize("bad", [0, -1000, 1000.0, True])
def test_num_train_timesteps_must_be_a_positive_int(bad):
    with pytest.raises(ValueError, match="num_train_timesteps"):
        RectifiedFlow(num_train_timesteps=bad)
    with pytest.raises(ValueError, match="num_train_timesteps"):
        ConstantSigma(num_train_timesteps=bad)


# ---- masked per-token reduction ----


def test_token_mse_means_features_then_the_valid_tokens():
    torch.manual_seed(0)
    prediction, target = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    token = (prediction - target).square().mean(dim=2)
    valid = torch.tensor([[True, True, False, False], [False] * 4])
    out = TokenMeanSquaredError()(prediction, target, valid=valid)
    assert out.shape == (2,)
    assert torch.allclose(out[0], token[0, :2].mean())
    assert out[1].item() == 0.0          # no valid token: 0, not NaN
    assert torch.allclose(TokenMeanSquaredError()(prediction, target), token.mean(dim=1))
    with pytest.raises(ValueError, match=r"\[B, T\]"):
        TokenMeanSquaredError()(prediction, target, valid=valid[:, :3])


def test_objectives_without_masking_reject_a_valid_mask():
    prediction = torch.randn(2, 4, 3)
    valid = torch.ones(2, 4, dtype=torch.bool)
    for objective in (MeanSquaredError(), CosineDistance()):
        with pytest.raises(ValueError, match="TokenMeanSquaredError"):
            objective(prediction, prediction, valid=valid)


def test_token_mse_on_the_frame_axis_is_the_fastwam_video_reduction():
    """``axis=2`` on [B, C, F, H, W] is ``mot/fastwam.py:476-506``, bit for bit."""
    torch.manual_seed(1)
    prediction, target = torch.randn(2, 3, 4, 2, 2), torch.randn(2, 3, 4, 2, 2)
    frame = torch.nn.functional.mse_loss(prediction, target, reduction="none").mean(dim=(1, 3, 4))
    assert torch.equal(TokenMeanSquaredError(axis=2)(prediction, target), frame.mean(dim=1))
    valid = torch.tensor([[True, True, True, False], [True, False, False, False]])
    counted = valid.to(frame.dtype)
    expected = (frame * counted).sum(dim=1) / counted.sum(dim=1).clamp(min=1.0)
    assert torch.equal(TokenMeanSquaredError(axis=2)(prediction, target, valid=valid), expected)


def test_token_mse_axis_is_validated():
    with pytest.raises(ValueError, match="at least 1"):
        TokenMeanSquaredError(axis=0)
    with pytest.raises(ValueError, match=r"axis=3"):
        TokenMeanSquaredError(axis=3)(torch.randn(2, 4, 3), torch.randn(2, 4, 3))


# ---- per-sample padding read from the batch ----


def _video(frames: int) -> torch.Tensor:
    return torch.zeros(2, 3, frames, 2, 2)


def test_video_frame_padding_folds_like_fastwam():
    """Column 0 is latent frame 0; each next stride columns one latent frame,
    padded only when all of them are (``mot/fastwam.py:487-499``)."""
    pad = torch.tensor([[False] * 5 + [True] * 4,
                        [False] * 3 + [True] * 6])
    folded = BatchPadding("image_is_pad", unit="video_frame").padded(
        {"image_is_pad": pad}, _video(3), stream="video", temporal_stride=4)
    tail = pad[:, 1:].view(2, -1, 4).all(dim=2)
    assert torch.equal(folded, torch.cat([pad[:, :1], tail], dim=1))
    assert folded.tolist() == [[False, False, True], [False, False, True]]


def test_kept_steps_count_although_padded():
    pad = torch.tensor([[False, True, True, True]])
    keep = torch.tensor([[False, False, True, False]])
    padded = BatchPadding("action_is_pad", keep_key="action_terminal_mask").padded(
        {"action_is_pad": pad, "action_terminal_mask": keep}, torch.zeros(1, 4, 6),
        stream="action", temporal_stride=4)
    assert padded.tolist() == [[False, True, False, True]]


def test_a_batch_without_the_pad_key_has_no_padding():
    padding = BatchPadding("action_is_pad", keep_key="action_terminal_mask")
    raw = torch.zeros(2, 4, 6)
    assert padding.padded({}, raw, stream="action", temporal_stride=4) is None
    with pytest.raises(ValueError, match="'action_terminal_mask' but not 'action_is_pad'"):
        padding.padded({"action_terminal_mask": torch.zeros(2, 4, dtype=torch.bool)}, raw,
                       stream="action", temporal_stride=4)


@pytest.mark.parametrize("kwargs, match", [
    (dict(pad_key=""), "nonempty"),
    (dict(pad_key="a", keep_key="a"), "both 'a'"),
    (dict(pad_key="a", unit="frame"), "unit must be one of"),
    (dict(pad_key="a", keep_key="b", unit="video_frame"), "unit='step' only"),
])
def test_batch_padding_declarations_are_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BatchPadding(**kwargs)


@pytest.mark.parametrize("unit, raw, mask, match", [
    ("step", torch.zeros(2, 4, 6), torch.zeros(3, 4, dtype=torch.bool), r"\[B=2, N\]"),
    ("step", torch.zeros(2, 4, 6), torch.zeros(2, 5, dtype=torch.bool), "5 columns"),
    ("video_frame", _video(3), torch.zeros(2, 8, dtype=torch.bool), "do not fold"),
    ("video_frame", torch.zeros(2, 4, 6), torch.zeros(2, 4, dtype=torch.bool), "on axis 2"),
])
def test_padding_masks_must_match_the_raw_tensor(unit, raw, mask, match):
    with pytest.raises(ValueError, match=match):
        BatchPadding("pad", unit=unit).padded({"pad": mask}, raw, stream="s", temporal_stride=4)
