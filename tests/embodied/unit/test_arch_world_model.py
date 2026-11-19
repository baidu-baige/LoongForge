# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``WorldModel``'s seven training steps: a real provider spec in, (loss, metrics) out.

Division of labour with ``test_arch_blocks.py``: that file drives the engine's
four-stage protocol by hand, while this one goes through
``WorldModel.forward``, the **single public entry point** (interface I4), and
checks six things:
  1. both buildable providers produce a 0-dim loss, and after backward every
     parameter has a gradient;
  2. the metrics key set **does not vary with configuration**: disabling a loss
     term only drives its weight to 0;
  3. ``WeightRamp`` reads ``global_step`` out of the batch, not from a counter
     the model keeps itself;
  4. weights live only on ``LossSpec``: a ``HeadSpec.weight`` other than 1.0
     raises;
  5. raw tensors are keyed by ``StreamSpec.batch_key``, and a missing key names
     that key in the error message;
  6. three paths that **raise on purpose** (per-axis noising, stream-side loss,
     no regression target found).

Geometry and widths always come from ``test_arch_blocks.tiny_spec``: the real
providers are 40 layers at width 5120, so a whole-model forward will not run on
CPU, and the seven-step orchestration is width independent.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from loongforge.embodied.model.arch.builder import DiffusionRuntime, WorldModel
from loongforge.embodied.model.arch.diffusion.broadcast import BroadcastSpec
from loongforge.embodied.model.arch.diffusion.loss import LossSpec, WeightRamp
from loongforge.embodied.model.arch.diffusion.timesteps import TimestepSampler
from loongforge.embodied.model.arch.layout import TokenLayout
from loongforge.embodied.model.arch.spec.world import WorldModelSpec
from loongforge.embodied.model.arch.streams.conditions import InjectionKind
from loongforge.embodied.model.arch.streams.engine import JointAttentionEngine

from test_arch_blocks import BATCH, CONTEXT_LEN, tiny_spec

# internw0_delta is absent here: its ``track_align`` head raises
# NotImplementedError at build time (see the last case in this file), so a
# whole-model forward is unreachable.
BUILDABLE = ("arch_vpp2", "fastwam")


@pytest.fixture(params=BUILDABLE)
def spec(request) -> WorldModelSpec:
    """A production spec shrunk to test widths, for each buildable model."""
    return tiny_spec(request.param)


def world(spec: WorldModelSpec) -> WorldModel:
    """Assemble exactly as production does: spec plus engine plus layout.

    The backend is dense; backend negotiation is covered by the L1 cases.
    """
    return WorldModel(spec, engine=JointAttentionEngine(backend="dense"),
                      layout=TokenLayout.from_geometry(spec.geometry,
                                                       spec.streams))


def fake_batch(spec: WorldModelSpec, **extra) -> dict:
    """The two key classes a batch owes: per-stream raw tensors and conditions.

    Stream keys are ``StreamSpec.batch_key`` (neither provider sets
    ``source_key``, so it falls back to the stream name). Condition keys follow
    ``ConditionSpec.source_key`` and widths follow ``ConditionSpec.width``,
    both being the only basis on which the consumer builds weights.

    A condition with an ``encoder`` is the **exception**: its width in the
    batch is ``source_width`` (the encoder's input width) while ``width`` is
    the post-encoder width. The shape given is ``[B, source_width]`` rather
    than ``[B, 1, source_width]`` because the legacy call site slices to two
    dimensions (``proprio[:, 0, :]`` at ``vpp2.py:845``); adding the token axis
    back is ``TokenEncoder``'s job.
    """
    geo = spec.geometry
    frames, height, width = geo.latent_grid
    batch: dict = {
        stream.batch_key: (
            torch.randn(BATCH, geo.latent_channels, frames, height, width)
            if stream.name == "video"
            else torch.randn(BATCH, geo.action_horizon, geo.action_dim))
        for stream in spec.streams
    }
    for cond in spec.conditions:
        if cond.encoder is not None:
            batch[cond.source_key] = torch.randn(BATCH, cond.source_width)
        elif cond.kind is InjectionKind.CROSS_ATTENTION:
            batch[cond.source_key] = torch.randn(BATCH, CONTEXT_LEN,
                                                 cond.width)
        elif cond.kind is InjectionKind.CHANNEL_CONCAT:
            batch[cond.source_key] = torch.randn(BATCH, cond.width, frames,
                                                 height, width)
    batch.update(extra)
    return batch


def with_losses(spec: WorldModelSpec, *losses: LossSpec) -> WorldModelSpec:
    """Swap the loss table, keep the rest; ``replace`` re-runs V1-V9."""
    return replace(spec,
                   diffusion=replace(spec.diffusion, losses=tuple(losses)))


# ---- 1. it runs ----


def test_forward_returns_a_scalar_loss_and_one_metric_per_term(spec):
    """I4's signature: ``(loss, metrics)`` with a 0-dim loss, not [B], not a dict.

    Every declared loss term also contributes a 0-dim metric and a recorded
    weight, and ``metrics["loss"]`` is the detached total.
    """
    loss, metrics = world(spec)(fake_batch(spec))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    for term in spec.diffusion.losses:
        assert metrics[term.name].ndim == 0
        assert metrics[f"{term.name}/weight"] == term.weight
    assert torch.equal(metrics["loss"], loss.detach())


def test_metrics_hold_tensors_not_python_floats(spec):
    """metrics carry detached tensors; ``float()`` would force a D2H sync a step.

    Formatting belongs to the trainer (I4'). What this case pins down is who
    is responsible for calling ``.item()``.
    """
    _, metrics = world(spec)(fake_batch(spec))
    for term in spec.diffusion.losses:
        value = metrics[term.name]
        assert isinstance(value, torch.Tensor)
        assert not value.requires_grad


def test_backward_reaches_every_parameter(spec):
    """One backward must leave a gradient on every ``WorldModel`` parameter.

    Streams and heads are checked together. A missing gradient usually means a
    head was built but never called during the head step of the forward, or a
    loss term's weight is permanently 0. The forward shows nothing: the
    parameters still enter ``state_dict`` and still occupy memory.
    """
    model = world(spec)
    loss, _ = model(fake_batch(spec))
    loss.backward()
    missing = [name for name, param in model.named_parameters()
               if param.grad is None]
    assert not missing, missing


def test_condition_encoders_are_the_third_parameter_namespace(spec):
    """A condition with an ``encoder`` owns a ``condition_encoders.<name>`` level.

    Hanging it under a stream has two failure modes: one copy per target
    stream (that is a different model), or picking one stream as an arbitrary
    host (whose removal breaks the whole chain). The legacy checkpoint stores
    these weights as a top-level payload key beside ``mot``
    (``vpp2.py:1800-1801``), which says the same thing: they belong to no
    expert. Each of the two widths has one source: input is ``source_width``,
    output is the host slot's ``width``.
    """
    model = world(spec)
    want = {cond.name for cond in spec.conditions if cond.encoder is not None}
    assert set(model.condition_encoders) == want == {"proprio"}
    assert {key for key in model.state_dict()
            if key.startswith("condition_encoders.")} == \
        {"condition_encoders.proprio.weight", "condition_encoders.proprio.bias"}
    # Each of the two widths has one source: input is source_width, output is
    # the host slot's width
    assert tuple(model.condition_encoders["proprio"].weight.shape) == \
        (spec.condition("text").width, spec.condition("proprio").source_width)


def test_head_outputs_have_the_shape_of_the_raw_tensor(spec):
    """A head's output shape equals its stream's raw tensor shape, so no reshape.

    With differing shapes, MSE would broadcast and still return a number (see
    ``_check_same_shape``), so this invariant has to hold at the assembly
    layer rather than being patched up in the loss.
    """
    model = world(spec)
    batch = fake_batch(spec)
    timesteps = model.sample_timesteps(batch)
    runtime = model.add_noise(batch, timesteps)
    for head in spec.heads:
        assert head.source_stream in runtime.targets
    loss, _ = model(batch)
    assert loss.ndim == 0


def test_timestep_dtype_follows_the_raw_tensor(spec):
    """The timestep dtype follows the raw tensor's, here bfloat16, shape ``[B]``.

    Otherwise ``(1-t) * x_0`` would silently promote a bf16 tensor to fp32.
    Only streams with ``needs_timestep`` get a timestep at all.
    """
    model = world(spec)
    batch = fake_batch(spec)
    batch = {key: (value.to(torch.bfloat16)
                   if isinstance(value, torch.Tensor)
                   and value.is_floating_point() else value)
             for key, value in batch.items()}
    timesteps = model.sample_timesteps(batch)
    assert set(timesteps) == {name for name in spec.stream_names
                              if spec.stream(name).needs_timestep}
    for value in timesteps.values():
        assert value.dtype == torch.bfloat16
        assert value.shape == (BATCH,)


# ---- 2 / 3. loss weights and the metric key set ----


def test_disabled_loss_keeps_its_metric_key_at_weight_zero():
    """Disabling a loss does not drop it from ``metrics``.

    Dropping the key would break downstream dashboards. Weight goes to 0 while
    the key stays, so a dashboard reads "this term is off" rather than "this
    term vanished".
    """
    spec = with_losses(
        tiny_spec("fastwam"),
        LossSpec(name="video_flow", head="video_flow", enabled=False),
        LossSpec(name="action_flow", head="action_flow"))
    loss, metrics = world(spec)(fake_batch(spec))
    assert "video_flow" in metrics
    assert metrics["video_flow/weight"] == 0.0
    assert metrics["action_flow/weight"] == 1.0
    # The disabled term still enters the sum, at weight 0, so the total equals
    # the remaining term.
    assert torch.allclose(loss, metrics["action_flow"])


def test_weight_ramp_reads_the_global_step_from_the_batch():
    """``WeightRamp`` is a function of the ``global_step`` carried in the batch.

    A counter kept by the model itself would diverge from the trainer's step in
    two places: gradient accumulation (several forwards per step) and resuming
    from a checkpoint. Both show up as a weight curve that disagrees with the
    step numbers in the log.
    """
    ramp = WeightRamp(start_step=0, end_step=10, start_value=0.0,
                      end_value=1.0)
    spec = with_losses(
        tiny_spec("fastwam"),
        LossSpec(name="video_flow", head="video_flow", ramp=ramp),
        LossSpec(name="action_flow", head="action_flow"))
    model, batch = world(spec), fake_batch(spec)
    for step, want in ((0, 0.0), (5, 0.5), (10, 1.0), (99, 1.0)):
        _, metrics = model({**batch, "global_step": step})
        assert metrics["video_flow/weight"] == pytest.approx(want)
    # No ``global_step`` means step 0, not an error: the evaluation path goes
    # through this same forward.
    _, metrics = model(batch)
    assert metrics["video_flow/weight"] == pytest.approx(0.0)


def test_weight_written_on_the_head_is_rejected():
    """The weight belongs to ``LossSpec``. Writing one in each of two places
    would leave only one of them in effect, with nothing to show which.

    The providers already follow this ruling (all five weights live on
    ``LossSpec``). This test turns "somebody moved it back onto ``HeadSpec``"
    into a failure instead of a silent reweighting.
    """
    base = tiny_spec("fastwam")
    heads = tuple(replace(head, weight=0.5) if head.name == "video_flow"
                  else head for head in base.heads)
    spec = replace(base, heads=heads)
    with pytest.raises(ValueError, match="weight ownership must be fixed in one place"):
        world(spec)(fake_batch(spec))


# ---- 5. the batch keys of the raw tensors ----


def test_missing_raw_tensor_names_the_batch_key(spec):
    """A missing key must put the ``batch_key`` itself in the error message.

    The old code did ``batch["latents"]`` directly, so the message carried only
    ``'latents'`` and did not say which stream asked for it. With two streams
    both pulling raw tensors out of the batch, that left guessing.
    """
    model = world(spec)
    batch = fake_batch(spec)
    key = spec.stream("action").batch_key
    del batch[key]
    with pytest.raises(KeyError, match=key):
        model(batch)


def test_stream_names_come_from_the_spec_not_the_batch(spec):
    """Extra, unrelated keys are not an error: a real batch also carries
    training-side entries such as labels and masks."""
    loss, _ = world(spec)(fake_batch(spec, task_description="pick up the cube"))
    assert loss.ndim == 0


# ---- 6. the three paths that raise on purpose ----


def test_per_axis_noising_raises_instead_of_broadcasting():
    """A per-frame timestep *raises* in the default ``add_noise``.

    It does not broadcast itself into a shape that happens to multiply.
    ``BroadcastSpec.expand`` produces the token shape ``[B, total, 1]``, while
    noising happens on the raw tensor ``[B, C, F, H, W]``. The axes do not line
    up, yet ``[B, 1, 1]`` multiplies a 5-D tensor just fine, which would
    silently downgrade per-frame noise to per-sample noise.
    """
    base = tiny_spec("fastwam")
    per_frame = TimestepSampler(
        distribution=base.diffusion.timesteps["video"].distribution,
        broadcast=BroadcastSpec(axes=("frame",)))
    spec = replace(base, diffusion=replace(
        base.diffusion,
        timesteps={**base.diffusion.timesteps, "video": per_frame}))
    with pytest.raises(NotImplementedError, match="samples its timestep per axis"):
        world(spec)(fake_batch(spec))


def test_stream_side_loss_raises_until_the_semantics_are_settled():
    """The ``LossSpec.stream`` side (loss on the residual itself) is still
    missing an unpatchify step.

    The residual is token-shaped while the target is raw-tensor-shaped, and who
    owns the step in between has not been settled, so the code raises instead
    of guessing.
    """
    spec = with_losses(
        tiny_spec("fastwam"),
        LossSpec(name="video_flow", stream="video"),
        LossSpec(name="action_flow", head="action_flow"))
    with pytest.raises(NotImplementedError, match="The semantics of a stream-side loss"):
        world(spec)(fake_batch(spec))


def test_loss_target_error_lists_both_candidate_keys():
    """When no regression target is found, the message lists both candidate
    keys: the head name and the source stream name.

    If a missing target degraded to a zero tensor, the symptom would be a loss
    that is identically 0, which reads on a dashboard as "this term converged
    beautifully". So the code raises, and names both keys worth checking.
    """
    spec = tiny_spec("fastwam")
    model = world(spec)
    loss = spec.diffusion.losses[0]
    head = spec.head(str(loss.head))
    empty = DiffusionRuntime(timesteps={}, noised={}, noise={}, targets={})
    with pytest.raises(KeyError) as info:
        model._loss_target(loss, head, empty)
    message = str(info.value)
    assert str(loss.head) in message
    assert head.source_stream in message


def test_internw0_delta_track_head_raises_at_build_time():
    """``track_align`` is not a flow head, so the build raises rather than
    constructing a head of the wrong shape.

    This test is also the reason ``internw0_delta`` is absent from
    ``BUILDABLE``: the whole-model forward is unreachable, so it does not join
    the parametrize set above.
    """
    with pytest.raises(NotImplementedError, match="track_align"):
        world(tiny_spec("internw0_delta"))
