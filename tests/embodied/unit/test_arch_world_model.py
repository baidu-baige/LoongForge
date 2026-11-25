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
     is rejected when the spec is built;
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

from loongforge.embodied.model.arch.builder import (
    STEP_HOOK_EVENTS, DiffusionDraw, DiffusionRuntime, WorldModel)
from loongforge.embodied.model.arch.diffusion.broadcast import BroadcastSpec
from loongforge.embodied.model.arch.diffusion.loss import BatchPadding, Bsmntw, LossSpec, WeightRamp
from loongforge.embodied.model.arch.diffusion.timesteps import RngPolicy, TimestepSampler
from loongforge.embodied.model.arch.spec.world import WorldModelSpec
from loongforge.embodied.model.arch.streams.conditions import InjectionKind
from loongforge.embodied.model.arch.streams.engine import JointAttentionEngine

from test_arch_blocks import BATCH, CONTEXT_LEN, tiny_spec

# internw0_delta is absent here: its ``track_align`` head raises
# NotImplementedError at build time (see the last case in this file), so a
# whole-model forward is unreachable.
BUILDABLE = ("vpp2", "fastwam")


@pytest.fixture(params=BUILDABLE)
def spec(request) -> WorldModelSpec:
    """A production spec shrunk to test widths, for each buildable model."""
    return tiny_spec(request.param)


def world(spec: WorldModelSpec) -> WorldModel:
    """Assemble exactly as production does: spec plus engine plus layout.

    The backend is dense; backend negotiation is covered by the L1 cases.
    """
    return WorldModel(spec, engine=JointAttentionEngine(backend="dense"),
                      layout=spec.build_layout())


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
        assert metrics[f"loss_{term.name}"].ndim == 0
        assert metrics[f"{term.name}/weight"] == term.weight
    assert torch.equal(metrics["loss"], loss.detach())
    # The trainer averages over ranks only the keys that contain "loss" (T23).
    weights = {f"{term.name}/weight" for term in spec.diffusion.losses}
    assert all("loss" in key for key in set(metrics) - weights)


def test_metrics_hold_tensors_not_python_floats(spec):
    """metrics carry detached tensors; ``float()`` would force a D2H sync a step.

    Formatting belongs to the trainer (I4'). What this case pins down is who
    is responsible for calling ``.item()``.
    """
    _, metrics = world(spec)(fake_batch(spec))
    for term in spec.diffusion.losses:
        value = metrics[f"loss_{term.name}"]
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
    runtime = model.add_noise(batch, model.draw(batch))
    for head in spec.heads:
        assert head.source_stream in runtime.targets
    loss, _ = model(batch)
    assert loss.ndim == 0


def test_timestep_dtype_follows_the_raw_tensor(spec):
    """Sigma and the training t follow the raw tensor's dtype (bf16), ``[B]``.

    Otherwise ``(1-t) * x_0`` would silently promote a bf16 tensor to fp32
    (``RngPolicy.sigma_dtype="like_sample"``, the default). Only streams with
    ``needs_timestep`` get a timestep at all.
    """
    model = world(spec)
    batch = fake_batch(spec)
    batch = {key: (value.to(torch.bfloat16)
                   if isinstance(value, torch.Tensor)
                   and value.is_floating_point() else value)
             for key, value in batch.items()}
    draw = model.draw(batch)
    expected = {name for name in spec.stream_names if spec.stream(name).needs_timestep}
    assert set(draw.sigma) == set(draw.model_timesteps) == set(draw.noise) == expected
    for value in (*draw.sigma.values(), *draw.model_timesteps.values()):
        assert value.dtype == torch.bfloat16
        assert value.shape == (BATCH,)


def test_float32_sigma_runs_a_bf16_forward_without_promoting_x_t(spec):
    """``sigma_dtype="float32"`` keeps sigma and t in float32 through a bf16 step.

    x_t stays bf16, and the time embedding takes the float32 t the way master's
    ``pre_dit`` does: cast to the weights' dtype.
    """
    timesteps = {name: replace(sampler, rng=RngPolicy(sigma_dtype="float32"))
                 for name, sampler in spec.diffusion.timesteps.items()}
    spec = replace(spec, diffusion=replace(spec.diffusion, timesteps=timesteps))
    model = world(spec).to(torch.bfloat16)
    batch = {key: (value.to(torch.bfloat16)
                   if isinstance(value, torch.Tensor) and value.is_floating_point() else value)
             for key, value in fake_batch(spec).items()}
    seen = {}
    model.register_step_hook(lambda m, event, value: seen.setdefault(event, value))
    loss, _ = model(batch)
    assert torch.isfinite(loss)
    draw, runtime = seen["draw"], seen["runtime"]
    for name in draw.sigma:
        assert draw.sigma[name].dtype == draw.model_timesteps[name].dtype == torch.float32
        assert runtime.noised[name].dtype == torch.bfloat16


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
    assert "loss_video_flow" in metrics
    assert metrics["video_flow/weight"] == 0.0
    assert metrics["action_flow/weight"] == 1.0
    # The disabled term still enters the sum, at weight 0, so the total equals
    # the remaining term.
    assert torch.allclose(loss, metrics["loss_action_flow"])


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
    model.eval()
    _, metrics = model(batch)
    assert metrics["video_flow/weight"] == pytest.approx(0.0)


def test_training_a_ramped_loss_without_a_step_is_rejected():
    """Without ``global_step`` the ramp would stay at step 0, silently (v2 §1.9)."""
    ramp = WeightRamp(start_step=0, end_step=10, start_value=0.0, end_value=1.0)
    spec = with_losses(
        tiny_spec("fastwam"),
        LossSpec(name="video_flow", head="video_flow", ramp=ramp),
        LossSpec(name="action_flow", head="action_flow"))
    model = world(spec)
    with pytest.raises(ValueError, match="no global_step"):
        model(fake_batch(spec))


def test_weight_written_on_the_head_is_rejected():
    """The weight belongs to ``LossSpec``. Writing one in each of two places
    would leave only one of them in effect, with nothing to show which.

    The providers already follow this ruling (all five weights live on
    ``LossSpec``). This test turns "somebody moved it back onto ``HeadSpec``"
    into a failure when the spec is built, before any step runs, instead of a
    silent reweighting.
    """
    base = tiny_spec("fastwam")
    heads = tuple(replace(head, weight=0.5) if head.name == "video_flow"
                  else head for head in base.heads)
    with pytest.raises(ValueError, match="HeadSpec.weight must stay 1.0"):
        replace(base, heads=heads)


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


# ---- step 1: one draw, in the declared order ----


def _replay(model: WorldModel, batch: dict, draws: tuple[str, str]) -> dict:
    """Draw by hand, stream by stream in the declared key order."""
    out = {}
    for name, sampler in model.spec.diffusion.timesteps.items():
        raw = batch[model.spec.stream(name).batch_key]
        for kind in draws:
            out[name, kind] = (torch.randn_like(raw) if kind == "noise" else
                               sampler.sample(raw.shape[0], device=raw.device, dtype=torch.float32))
    return out


@pytest.mark.parametrize("order", ["noise_first", "t_first"])
def test_draw_follows_the_declared_order(order):
    """Streams in ``DiffusionSpec.timesteps`` key order, ``RngPolicy.order`` within.

    Either order leaves the generator in the same state (same draws, other
    sequence), so the values are compared, and the opposite order must miss.
    """
    spec = tiny_spec("vpp2")
    rng = RngPolicy(order=order)
    spec = replace(spec, diffusion=replace(spec.diffusion, timesteps={
        name: replace(sampler, rng=rng) for name, sampler in spec.diffusion.timesteps.items()}))
    model = world(spec)
    batch = fake_batch(spec)
    torch.manual_seed(3)
    draw = model.draw(batch)
    torch.manual_seed(3)
    manual = _replay(model, batch, rng.draws)
    torch.manual_seed(3)
    swapped = _replay(model, batch, rng.draws[::-1])
    steps = spec.diffusion.schedule.num_train_timesteps
    for name in spec.diffusion.timesteps:
        sigma, t = rng.quantize(manual[name, "timestep"], num_train_timesteps=steps,
                                sample_dtype=batch[name].dtype)
        assert torch.equal(draw.noise[name], manual[name, "noise"])
        assert torch.equal(draw.sigma[name], sigma)
        assert torch.equal(draw.model_timesteps[name], t)
    assert not torch.equal(draw.noise["action"], swapped["action", "noise"])


def test_add_noise_consumes_the_draw_and_draws_nothing(spec):
    model = world(spec)
    batch = fake_batch(spec)
    draw = model.draw(batch)
    state = torch.get_rng_state()
    runtime = model.add_noise(batch, draw)
    assert torch.equal(torch.get_rng_state(), state)
    for name in draw.noise:
        assert runtime.noise[name] is draw.noise[name]
        assert runtime.timesteps[name] is draw.sigma[name]
        assert runtime.model_timesteps[name] is draw.model_timesteps[name]


def test_engine_receives_the_training_timestep_not_sigma():
    """The time embeddings read ``sigma * num_train_timesteps`` (Wan: 0..1000)."""
    spec = tiny_spec("vpp2")
    model = world(spec)
    seen = {}
    for name, block in model.blocks.items():
        def prepare(x, *, timestep=None, _name=name, _prepare=block.prepare, **kwargs):
            seen[_name] = timestep
            return _prepare(x, timestep=timestep, **kwargs)
        block.prepare = prepare
    batch = fake_batch(spec)
    torch.manual_seed(1)
    draw = model.draw(batch)
    torch.manual_seed(1)
    model(batch)
    for name in spec.diffusion.timesteps:
        assert torch.equal(seen[name], draw.model_timesteps[name])
    assert bool((seen["video"] == 1000).all())


def test_overriding_the_removed_sample_timesteps_is_rejected():
    """forward no longer calls it, so an override would silently be dead code."""
    with pytest.raises(TypeError, match="replaced by draw"):
        class Legacy(WorldModel):  # noqa: F841 -- definition itself must fail
            def sample_timesteps(self, batch):
                return {}


def test_bsmntw_on_a_constant_timestep_stream_is_rejected():
    """At sigma=1 the bsmntw weight is exactly 0: a dead loss term (v2 T24)."""
    with pytest.raises(ValueError, match="exactly 0"):
        with_losses(tiny_spec("vpp2"),
                    LossSpec(name="video_flow", head="video_flow", weight_of=Bsmntw()),
                    LossSpec(name="action_flow", head="action_flow"))


def test_timestep_weight_on_a_stream_without_timesteps_is_rejected():
    """compute_losses only weights a stream that draws a timestep; elsewhere the
    weight would be skipped without a word."""
    base = tiny_spec("fastwam")
    group = base.attention_groups[0]
    streams = tuple(replace(stream, needs_timestep=False, timestep_embed_width=None)
                    if stream.name == "action" else stream for stream in group.streams)
    timesteps = {name: sampler for name, sampler in base.diffusion.timesteps.items()
                 if name != "action"}
    losses = tuple(replace(loss, weight_of=Bsmntw()) if loss.head == "action_flow" else loss
                   for loss in base.diffusion.losses)
    with pytest.raises(ValueError, match="draws none"):
        replace(base, attention_groups=(replace(group, streams=streams),),
                diffusion=replace(base.diffusion, timesteps=timesteps, losses=losses))


# ---- step hook: read-only observation of steps 1, 3, 6 and 7 ----


def test_step_hook_reports_each_stage_once_in_order(spec):
    model = world(spec)
    seen = []
    handle = model.register_step_hook(lambda m, event, value: seen.append((m, event, value)))
    batch = fake_batch(spec)
    loss, metrics = model(batch)
    assert [event for _, event, _ in seen] == list(STEP_HOOK_EVENTS)
    assert all(m is model for m, _, _ in seen)
    draw, runtime, outputs, (hook_loss, hook_metrics) = (value for _, _, value in seen)
    assert isinstance(draw, DiffusionDraw) and isinstance(runtime, DiffusionRuntime)
    assert all(runtime.noise[name] is draw.noise[name] for name in draw.noise)
    assert set(outputs) == {head.name for head in spec.heads}
    assert hook_loss is loss and dict(hook_metrics) == metrics
    handle.remove()
    model(batch)
    assert len(seen) == len(STEP_HOOK_EVENTS)


def test_step_hook_is_read_only_and_leaves_the_step_unchanged(spec):
    """Same seed, same loss and generator state with or without an observer."""
    model = world(spec)
    batch = fake_batch(spec)
    torch.manual_seed(5)
    expected, _ = model(batch)
    state = torch.get_rng_state()

    def hook(m, event, value):
        if event == "outputs":
            with pytest.raises(TypeError):
                value["extra"] = None
        return "ignored"

    model.register_step_hook(hook)
    torch.manual_seed(5)
    loss, _ = model(batch)
    assert torch.equal(loss, expected)
    assert torch.equal(torch.get_rng_state(), state)


# ---- clean prefix frames and batch padding (Lane C v2 §1.3, §1.16, T36, T37) ----


def _zero_outputs(runtime: DiffusionRuntime) -> dict:
    return {"video_flow": torch.zeros_like(runtime.targets["video"]),
            "action_flow": torch.zeros_like(runtime.targets["action"])}


def _video_loss(model: WorldModel, runtime: DiffusionRuntime, outputs: dict) -> torch.Tensor:
    return model.compute_losses(None, runtime, outputs, step=0)[1]["loss_video_flow"]


def test_clean_prefix_frames_go_in_as_x0_and_stay_out_of_the_loss():
    """fastwam's first latent frame (``TokenTimestep(clean_prefix=1)``): x_t is
    x_0 there and the plain noising elsewhere, the target is untouched, and the
    video loss never sees frame 0 (``mot/fastwam.py:529``, ``:597-600``)."""
    spec = tiny_spec("fastwam")
    assert spec.stream("video").token_timestep.clean_prefix == 1
    model = world(spec)
    batch = fake_batch(spec)
    runtime = model.add_noise(batch, model.draw(batch))
    clean, eps = batch["video"], runtime.noise["video"]
    sigma = runtime.timesteps["video"].to(clean.dtype).view(-1, 1, 1, 1, 1)
    schedule = spec.diffusion.schedule
    assert torch.equal(runtime.noised["video"][:, :, :1], clean[:, :, :1])
    assert torch.equal(runtime.noised["video"][:, :, 1:],
                       schedule.add_noise(clean, eps, sigma)[:, :, 1:])
    assert torch.equal(runtime.targets["video"], schedule.target(clean, eps, sigma))
    outputs = _zero_outputs(runtime)
    base = _video_loss(model, runtime, outputs)
    outputs["video_flow"][:, :, :1] = 1e3
    assert torch.equal(_video_loss(model, runtime, outputs), base)
    outputs["video_flow"][:, :, 1:] = 1.0
    assert not torch.equal(_video_loss(model, runtime, outputs), base)


def test_clean_prefix_needs_the_raw_video_shape():
    spec = tiny_spec("fastwam")
    model = world(spec)
    batch = fake_batch(spec)
    batch["video"] = batch["video"][:, :, :1]
    with pytest.raises(ValueError, match="keeps its first 1 latent frames clean"):
        model.add_noise(batch, model.draw(batch))


def test_padded_action_steps_drop_out_except_the_kept_ones():
    """vpp2 counts ``~(action_is_pad & ~action_terminal_mask)`` (master
    ``_training_loss_action_only``); a batch without the keys masks nothing."""
    spec = tiny_spec("vpp2")
    model = world(spec)
    pad = torch.tensor([[False, False, True, True], [False, True, True, True]])
    keep = torch.tensor([[False, False, True, False], [False] * 4])
    batch = fake_batch(spec, action_is_pad=pad, action_terminal_mask=keep)
    runtime = model.add_noise(batch, model.draw(batch))
    assert torch.equal(runtime.loss_valid["action"], ~(pad & ~keep))
    plain = fake_batch(spec)
    assert model.add_noise(plain, model.draw(plain)).loss_valid == {}


def test_all_padded_action_steps_give_a_zero_action_loss():
    """A sample with no valid step contributes 0, not NaN (v2 T21)."""
    spec = tiny_spec("vpp2")
    pad = torch.ones(BATCH, spec.geometry.action_horizon, dtype=torch.bool)
    _, metrics = world(spec)(fake_batch(spec, action_is_pad=pad))
    assert metrics["loss_action_flow"].item() == 0.0


def test_video_frame_padding_drops_the_clean_prefix_column():
    """``image_is_pad`` folds onto latent frames, then the clean first frame's
    column goes, as fastwam's video loss does (``mot/fastwam.py:487-499``).

    13 pixel frames make 4 latent frames: the first alone, then 4 pixel frames
    each. Padding pixel frames 7-12 pads only latent frame 3, since latent
    frame 2 (pixel frames 5-8) is padding only when all of its frames are.
    """
    base = tiny_spec("fastwam")
    padding = {"video": BatchPadding("image_is_pad", unit="video_frame")}
    spec = replace(base, diffusion=replace(base.diffusion, batch_padding=padding))
    model = world(spec)
    assert (spec.geometry.num_frames, spec.geometry.latent_grid[0]) == (13, 4)
    pad = torch.zeros(BATCH, spec.geometry.num_frames, dtype=torch.bool)
    pad[1, 7:] = True
    batch = fake_batch(spec, image_is_pad=pad)
    runtime = model.add_noise(batch, model.draw(batch))
    assert runtime.loss_valid["video"].tolist() == [[True, True, True], [True, True, False]]


def test_padding_on_a_stream_without_a_timestep_is_rejected():
    base = tiny_spec("vpp2")
    with pytest.raises(ValueError, match="draw no timestep"):
        replace(base.diffusion, batch_padding={"text": BatchPadding("context_mask")})

