# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 stream modules: every provider's streams build and run the four-stage
protocol.

Division of labour with ``test_arch_providers.py``: that file only reconciles
declarations against each other, this one **really allocates the parameters and
really runs a forward/backward**. Three things are checked:

1. The four-stage protocol runs under the engine, and each stream emits as many
   tokens as its segment in the token layout is long.
2. ``state_dict`` keys match the existing submodule names and contain no complex
   RoPE frequency table (it is deliberately not a buffer: a complex buffer would
   add an unexpected key and would not survive bf16 conversion).
3. The entry-stage shape checks really reject malformed inputs, instead of
   letting the error drift downstream and blow up at the concatenation point.

The real providers are too large to instantiate here (vpp2 is 40 layers × 5120
width), so every case shrinks the stream scalars to 16 layers × 32 width and the
geometry to a (4, 2, 2) patch grid = 16 video tokens + 4 action tokens. The
geometry shrink is handed to the provider rather than applied afterwards, since
the provider derives token labels and head widths from it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from loongforge.embodied.model.arch.builder import join_stream_visibility
from loongforge.embodied.model.arch.spec.core import build, get_provider
from loongforge.embodied.model.arch.spec.world import WorldModelSpec
from loongforge.embodied.model.arch.streams.blocks import (
    ActionStream, WanVideoStream, build_context_projector)
from loongforge.embodied.model.arch.streams.conditions import (
    ConditionBus, InjectionKind)
from loongforge.embodied.model.arch.streams.engine import JointAttentionEngine
from loongforge.embodied.model.arch.streams.spec import TokenTimestep
from loongforge.embodied.model.arch.visibility.mask import AttentionPlan

import loongforge.embodied.model.arch.spec.providers  # noqa: F401  registers on import

MODEL_TYPES = ("vpp2", "fastwam", "internw0_delta")

# Two narrow streams. ``num_heads`` must match across the two streams (joint
# attention concatenates along the token axis) and so must ``head_dim`` (V3).
# ``residual_width`` is deliberately unequal, and 32 != 2 * 8 is deliberate too:
# "residual width != attention width" is the normal case for the action stream,
# so the tests have to exercise that path.
# The layer count is 16 rather than 2 because the two intermediate-layer heads of
# InternW0-Δ read layer 8 and layer 14; a smaller stack would be rejected at
# construction time instead of reaching the forward pass this file tests.
TINY_SCALARS = dict(
    video_residual_width=32, video_num_heads=2, video_head_dim=8,
    video_num_layers=16, video_ffn_width=32,
    action_residual_width=16, action_num_heads=2, action_head_dim=8,
    action_num_layers=16, action_ffn_width=16,
)
ACTION_DIM = 6
BATCH = 2
CONTEXT_LEN = 3


def tiny_spec(model_type: str, **overrides: object) -> WorldModelSpec:
    """The real provider topology with shrunken widths and geometry.

    The condition table and the visibility rules are kept as the provider
    declares them.

    The shrink goes **through the provider's** ``geometry=`` override rather
    than ``dataclasses.replace``: the provider derives visibility, token
    labels and the flow heads from the resolved geometry, so replacing the
    geometry afterwards would leave those baked at full size.

    Args:
        model_type: A registered provider name.
        **overrides: Further provider keyword arguments, for cases that need
            to vary a topology parameter rather than take its default. Passing
            ``geometry`` is a ``TypeError``, since the shrink owns it.

    Returns:
        The shrunken spec.
    """
    probe = get_provider(model_type)(action_width=ACTION_DIM, **TINY_SCALARS)
    geo = probe.geometry
    # latent grid (4, 4, 4) / patch (1, 2, 2) => patch grid (4, 2, 2) = 16
    # tokens. The three models use different VAE strides (8 for vpp2, 16 for
    # the other two), so the pixel sizes are derived from the stride instead of
    # hardcoded; hardcoding would break the divisibility check for two of the
    # three models.
    # Four latent frames, not two: InternW0-Δ declares ``condition_frames=3``,
    # so its video stream's ``TokenTimestep(clean_prefix=3)`` needs a fourth
    # frame to carry noise. Two frames make that declaration contradict the
    # geometry and construction fails, which is the right failure for a real
    # config but is only an artefact of shrinking here.
    return get_provider(model_type)(
        action_width=ACTION_DIM,
        geometry=dict(
            frame_height=4 * geo.vae_spatial_stride,
            frame_width=4 * geo.vae_spatial_stride,
            num_views=1,
            num_frames=1 + 3 * geo.vae_temporal_stride,
            action_horizon=4,
            action_video_freq_ratio=2,
            action_dim=ACTION_DIM,
        ),
        **TINY_SCALARS, **overrides)


@pytest.fixture(params=MODEL_TYPES)
def spec(request) -> WorldModelSpec:
    return tiny_spec(request.param)


def build_streams(spec: WorldModelSpec) -> dict[str, torch.nn.Module]:
    """Same single call as step 1 of ``WorldModel.__init__``: ``build(stream.impl,
    …)``."""
    return {stream.name: build(stream.impl, spec=stream,
                               geometry=spec.geometry, world=spec)
            for stream in spec.streams}


def fake_inputs(spec: WorldModelSpec) -> dict[str, torch.Tensor]:
    """Per-stream **raw input tensors**: video ``[B, C, F, H, W]``, action
    ``[B, T, D]``."""
    geo = spec.geometry
    frames, height, width = geo.latent_grid
    return {
        "video": torch.randn(BATCH, geo.latent_channels, frames, height, width),
        "action": torch.randn(BATCH, geo.action_horizon, geo.action_dim),
    }


def fake_bus(spec: WorldModelSpec, *, skip: str | None = None) -> ConditionBus:
    """Materialise one bus from the condition table, widths from
    ``ConditionSpec.width``.

    ``skip`` drives the "one condition missing" case: the consumer side must
    raise and name the registered conditions rather than silently substitute a
    zero tensor and keep going.

    The ``APPEND_TO_SLOT`` value is an **already encoded** token
    (``[B, 1, host slot width]``): what sits on the bus is the ``value``, and
    the encoder runs before it (in ``materialize_conditions``). One token rather
    than ``CONTEXT_LEN`` because the existing proprio path produces exactly one
    (the ``unsqueeze(1)`` at ``vpp2.py:531-533``), and "the slot got longer after
    the append" is only observable when the two segment lengths differ.
    """
    geo = spec.geometry
    frames, height, width = geo.latent_grid
    bus = ConditionBus()
    for cond in spec.conditions:
        if cond.name == skip:
            continue
        if cond.kind is InjectionKind.CROSS_ATTENTION:
            value = torch.randn(BATCH, CONTEXT_LEN, cond.width)
        elif cond.kind is InjectionKind.CHANNEL_CONCAT:
            value = torch.randn(BATCH, cond.width, frames, height, width)
        elif cond.kind is InjectionKind.APPEND_TO_SLOT:
            value = torch.randn(BATCH, 1, cond.width)
        else:
            continue
        bus.add(cond.materialize(value))
    return bus


def run_engine(spec: WorldModelSpec, modules, bus: ConditionBus):
    """Run the engine once. The backend is always dense; backend negotiation is
    an L1 case (covered in the providers file)."""
    group = spec.attention_groups[0]
    layout = spec.build_layout()
    plan = AttentionPlan.from_visibility(
        join_stream_visibility(spec.visibility), layout)
    engine = JointAttentionEngine(backend="dense")
    timesteps = {name: torch.rand(BATCH) for name in group.stream_names}
    return layout, engine.run(
        group, fake_inputs(spec), modules, plan, layout,
        conditions=bus, timesteps=timesteps, captures=spec.captures())


# ---- 1. the four-stage protocol runs ----


def test_streams_build_and_run_through_the_engine(spec):
    """Each stream emits exactly as many tokens as its layout segment is long.

    The hidden state also has to be finite, which catches an uninitialised
    modulation or a NaN leaking out of the entry stage.
    """
    modules = build_streams(spec)
    assert isinstance(modules["video"], WanVideoStream)
    assert isinstance(modules["action"], ActionStream)
    layout, out = run_engine(spec, modules, fake_bus(spec))
    for stream in spec.streams:
        state = out.require(stream.name)
        assert state.hidden.shape == (BATCH, layout.range_of(stream.name).length,
                                      stream.residual_width), stream.name
        assert torch.isfinite(state.hidden).all(), stream.name


def test_captured_layers_come_back_from_the_engine(spec):
    """Intermediate layers declared by the head table are really captured.

    InternW0-Δ is the only model with intermediate-layer heads, so without this
    case a dropped capture would go unnoticed.
    """
    modules = build_streams(spec)
    layout, out = run_engine(spec, modules, fake_bus(spec))
    for stream, layers in spec.captures().items():
        for layer_id in layers:
            got = out.require_capture(stream, layer_id)
            assert got.shape == (BATCH, layout.range_of(stream).length,
                                 spec.stream(stream).residual_width)


def test_every_stream_parameter_gets_a_gradient(spec):
    """After one backward pass every parameter of every stream has a gradient.

    The usual cause of a missing gradient is a submodule that was allocated but
    never called by the four-stage protocol (for example modulation taking the
    zero-tensor branch, or ``time_projection`` being bypassed), which a forward
    pass alone cannot reveal.
    """
    modules = build_streams(spec)
    _, out = run_engine(spec, modules, fake_bus(spec))
    loss = sum(out.require(name).hidden.square().mean()
               for name in spec.stream_names)
    loss.backward()
    for name, module in modules.items():
        for param_name, param in module.named_parameters():
            assert param.grad is not None, f"{name}.{param_name}"


# ---- 2. state_dict key names ----


def test_state_dict_has_no_complex_rope_table(spec):
    """The RoPE frequency table is deliberately not a buffer.

    A complex tensor does not survive bf16 conversion and would also add an
    unexpected key when loading an existing checkpoint.
    """
    for module in build_streams(spec).values():
        for key, value in module.state_dict().items():
            assert not value.is_complex(), key
            assert "freqs" not in key, key


def test_submodule_names_match_the_existing_checkpoint(spec):
    """Submodule names match the existing model, so a weight migration is only a
    prefix move rather than a key-by-key rename."""
    modules = build_streams(spec)
    video = set(modules["video"].state_dict())
    for key in ("patch_embedding.weight", "patch_embedding.bias",
                "time_embedding.0.weight", "time_projection.1.weight",
                "blocks.0.modulation", "blocks.0.self_attn.q.weight",
                "blocks.0.self_attn.norm_q.weight", "blocks.0.self_attn.o.weight",
                "blocks.0.ffn.0.weight", "blocks.0.ffn.2.weight"):
        assert key in video, key
    action = set(modules["action"].state_dict())
    assert "action_encoder.weight" in action
    assert not any(k.startswith("patch_embedding") for k in action)


# The cross-attention slots of each model's video stream. The vpp2 i2v path runs
# attention twice and adds the results (``wan_video_dit.py:187-204``), so CLIP
# and text are two separate slots, not one slot holding two segments.
VIDEO_SLOTS = {"vpp2": ("clip_image", "text"), "fastwam": ("text",),
               "internw0_delta": ("text",)}


def test_cross_attention_slots_are_one_subtree_each(spec):
    """One slot means one ``cross_attn.kv.<slot>`` subtree plus one
    ``context_embedding.<slot>``.

    The existing model writes the two slots as parallel fields ``cross_attn.k``
    and ``cross_attn.k_img`` (``wan_video_dit.py:170-176``), so changing the slot
    count means editing the class definition. With one subtree per slot name,
    adding a slot touches only the ``cross_kv_slots`` line.
    """
    modules = build_streams(spec)
    keys = set(modules["video"].state_dict())
    slots = VIDEO_SLOTS[spec.model_type]
    assert tuple(spec.stream("video").cross_kv_slots) == slots
    for slot in slots:
        assert any(k.startswith(f"context_embedding.{slot}.") for k in keys), slot
        for leaf in ("k.weight", "v.weight", "norm_k.weight"):
            assert f"blocks.0.cross_attn.kv.{slot}.{leaf}" in keys, (slot, leaf)
    assert "blocks.0.cross_attn.q.weight" in keys
    assert "blocks.0.cross_attn.o.weight" in keys
    # No parallel fields such as ``k_img`` are allowed beside the slot names
    assert not any(k.startswith("blocks.0.cross_attn.k") for k in keys
                   if not k.startswith("blocks.0.cross_attn.kv."))
    # The action stream has a single text slot (both action DiTs read text only)
    assert tuple(spec.stream("action").cross_kv_slots) == ("text",)


def test_context_projector_form_follows_the_params(spec):
    """The projector shape is an implementation detail carried by ``Spec.params``,
    and its key names match the existing model.

    The vpp2 CLIP path is a five-stage ``img_emb`` (``proj.{0,1,3,4}``) while the
    text path is a three-stage ``text_embedding``. The two are not
    interchangeable: they do not even have the same number of parameters.
    """
    keys = set(build_streams(spec)["video"].state_dict())
    text = {k for k in keys if k.startswith("context_embedding.text.")}
    assert text == {"context_embedding.text.0.weight",
                    "context_embedding.text.0.bias",
                    "context_embedding.text.2.weight",
                    "context_embedding.text.2.bias"}
    if spec.model_type == "vpp2":
        clip = {k for k in keys
                if k.startswith("context_embedding.clip_image.")}
        assert clip == {f"context_embedding.clip_image.proj.{i}.{suffix}"
                        for i in (0, 1, 3, 4)
                        for suffix in ("weight", "bias")}


# ---- 3. entry-stage and condition shape checks ----


def test_video_entry_rejects_wrong_rank_and_channels(spec):
    """The video entry stage requires rank 5 and exactly ``latent_channels``.

    Both errors are raised at the entry stage so that the caller sees the bad
    input, not a mismatch far downstream.
    """
    video = build_streams(spec)["video"]
    geo = spec.geometry
    bus = fake_bus(spec)
    timestep = torch.rand(BATCH)
    frames, height, width = geo.latent_grid
    with pytest.raises(ValueError, match="the entry takes 5-D"):
        video.prepare(torch.randn(BATCH, geo.latent_channels, height, width),
                      timestep=timestep, conditions=bus)
    # Pre-concatenating the channel-concat conditions before calling is rejected:
    # otherwise the same channels get concatenated twice, the entry conv input
    # channel count happens to match (36 = 16 + 20), and the error only shows up
    # in the loss
    with pytest.raises(ValueError, match="latent channels"):
        video.prepare(torch.randn(BATCH, geo.latent_channels + 1, frames,
                                  height, width),
                      timestep=timestep, conditions=bus)


def test_action_entry_rejects_wrong_horizon_and_width(spec):
    """The action entry stage requires rank 3 and the declared horizon and dim.

    Either number being wrong means the batch was built for a different model
    config, which is worth a named error at the boundary.
    """
    action = build_streams(spec)["action"]
    geo = spec.geometry
    bus = fake_bus(spec)
    timestep = torch.rand(BATCH)
    with pytest.raises(ValueError, match="the entry takes 3-D"):
        action.prepare(torch.randn(BATCH, geo.action_dim),
                       timestep=timestep, conditions=bus)
    with pytest.raises(ValueError, match="action_horizon"):
        action.prepare(torch.randn(BATCH, geo.action_horizon + 1, geo.action_dim),
                       timestep=timestep, conditions=bus)
    with pytest.raises(ValueError, match="action_dim"):
        action.prepare(torch.randn(BATCH, geo.action_horizon, geo.action_dim + 1),
                       timestep=timestep, conditions=bus)


def test_missing_condition_is_named_in_the_error(spec):
    """A condition missing from the bus raises ``KeyError`` listing the
    registered names, instead of continuing with a zero tensor."""
    modules = build_streams(spec)
    with pytest.raises(KeyError, match="text"):
        run_engine(spec, modules, fake_bus(spec, skip="text"))


def test_timestep_declaration_is_checked_both_ways(spec):
    """``needs_timestep`` and the timestep passed to ``prepare`` must agree in
    both directions."""
    modules = build_streams(spec)
    bus = fake_bus(spec)
    inputs = fake_inputs(spec)
    with pytest.raises(ValueError, match="needs_timestep=True"):
        modules["action"].prepare(inputs["action"], timestep=None,
                                  conditions=bus)
    # The other direction: a pure condition stream (such as the EWAM vl stream)
    # that receives a timestep must also fail
    pure = replace(spec.stream("action"), needs_timestep=False,
                   timestep_embed_width=None)
    module = build(pure.impl, spec=pure, geometry=spec.geometry, world=spec)
    with pytest.raises(ValueError, match="needs_timestep=False"):
        module.prepare(inputs["action"], timestep=torch.rand(BATCH),
                       conditions=bus)


def test_pure_condition_stream_runs_without_modulation(spec):
    """A ``needs_timestep=False`` stream takes the zero-modulation path rather
    than a second code path.

    When ``state.modulation is None`` the modulation degenerates to the
    per-layer learnable part only, and the four-stage protocol is unchanged.
    The EWAM vl stream is exactly this case.
    """
    pure = replace(spec.stream("action"), needs_timestep=False,
                   timestep_embed_width=None)
    module = build(pure.impl, spec=pure, geometry=spec.geometry, world=spec)
    state = module.prepare(fake_inputs(spec)["action"],
                           conditions=fake_bus(spec))
    assert state.modulation is None
    assert "t_embed" not in state.extras
    qkv = module.pre_attn(state, local_layer_id=0)
    attn = torch.randn(BATCH, state.num_tokens, pure.num_heads, pure.head_dim)
    after = module.post_attn(state, attn, local_layer_id=0)
    assert qkv.q.shape == (BATCH, state.num_tokens, pure.num_heads,
                           pure.head_dim)
    assert after.hidden.shape == state.hidden.shape
    assert module.finalize(after) is after


# ---- 4. per-token timestep modulation ----


def _legacy_token_timesteps(geo, timestep: torch.Tensor,
                            clean_prefix: int) -> torch.Tensor:
    """fastwam's own formula, transcribed from ``wan/dit.py:762-779``.

    ``ones(B, F, tokens_per_frame) * t.view(B, 1, 1)`` followed by assigning 0
    into the clean prefix, then flattened. InternW0-Delta reaches the same
    values the other way round (expand over the noised frames, concatenate the
    zeros in front, ``wan_video_dit.py:820-840``), so one transcription covers
    both.
    """
    frames, rows, cols = geo.patch_grid
    tokens = torch.ones(timestep.shape[0], frames, rows * cols) \
        * timestep.view(-1, 1, 1)
    tokens[:, :clean_prefix, :] = 0
    return tokens.reshape(timestep.shape[0], -1)


def test_token_timestep_reproduces_the_legacy_token_timesteps(spec):
    """The per-token expansion equals the formula in the source models.

    fastwam declares ``clean_prefix=1`` and InternW0-Delta
    ``clean_prefix=condition_frames``; vpp2 declares nothing and keeps the
    per-sample path, where the timestep passes through untouched.
    """
    video = spec.stream("video")
    module = build_streams(spec)["video"]
    timestep = torch.rand(BATCH)
    got = module._expand_timestep(timestep)
    if video.token_timestep is None:
        assert torch.equal(got, timestep)
        return
    assert torch.equal(got, _legacy_token_timesteps(
        spec.geometry, timestep, video.token_timestep.clean_prefix))
    assert got.shape == (BATCH, spec.geometry.padded_seq_len)


def test_per_frame_and_per_sample_timesteps_agree(spec):
    """``[B]`` and ``[B, noised frames]`` are two spellings of one tensor.

    The second form is what an L3 ``BroadcastSpec(axes=("frame",))`` emits and
    what InternW0-Delta's ``pre_dit`` takes; feeding the same value through
    both must land on the same per-token vector, otherwise the two entry points
    disagree about where the clean prefix sits.
    """
    video = spec.stream("video")
    scope = video.token_timestep
    if scope is None:
        pytest.skip(f"{spec.model_type} has per-sample video modulation")
    module = build_streams(spec)["video"]
    noised = spec.geometry.patch_grid[0] - scope.clean_prefix
    timestep = torch.rand(BATCH)
    assert torch.equal(
        module._expand_timestep(timestep),
        module._expand_timestep(timestep[:, None].expand(BATCH, noised)))


def test_token_timestep_gives_every_token_its_own_modulation(spec):
    """``prepare`` grows a token axis on ``t_embed`` and on the modulation.

    The slots that reach a layer must be ``[B, L, W]`` rather than the
    per-sample ``[B, 1, W]``, and the clean prefix's slots must differ from the
    noised frames' -- otherwise the whole per-token path would be a no-op that
    still passes every shape check.
    """
    video = spec.stream("video")
    scope = video.token_timestep
    module = build_streams(spec)["video"]
    state = module.prepare(fake_inputs(spec)["video"],
                           timestep=torch.rand(BATCH),
                           conditions=fake_bus(spec))
    tokens = state.num_tokens
    slot_count = video.modulation_slots.width
    if scope is None:
        assert state.extras["t_embed"].shape == (BATCH, video.residual_width)
        assert state.modulation.shape == (BATCH, slot_count,
                                          video.residual_width)
    else:
        assert state.extras["t_embed"].shape == (BATCH, tokens,
                                                 video.residual_width)
        assert state.modulation.shape == (BATCH, tokens, slot_count,
                                          video.residual_width)
    slots = module._modulation(module.blocks[0], state)
    expected = (BATCH, tokens if scope else 1, video.residual_width)
    for name, slot in slots.items():
        assert slot.shape == expected, name
    if scope is not None:
        per_frame = spec.geometry.tokens_per_frame
        clean = slots["shift_msa"][:, :per_frame]
        noised = slots["shift_msa"][:, -per_frame:]
        assert not torch.allclose(clean, noised)


def test_per_token_modulation_needs_an_axis_to_broadcast_along(spec):
    """An action stream cannot carry ``token_timestep``.

    ``ActionStream`` has no frame axis, so the base ``_expand_timestep``
    refuses instead of silently running the per-sample path and producing a
    modulation one rank short of what ``_modulation`` then unpacks.
    """
    action = replace(spec.stream("action"),
                     token_timestep=TokenTimestep(clean_prefix=1))
    module = build(action.impl, spec=action, geometry=spec.geometry,
                   world=spec)
    with pytest.raises(ValueError, match="only per-sample"):
        module.prepare(fake_inputs(spec)["action"],
                       timestep=torch.rand(BATCH), conditions=fake_bus(spec))


def test_token_timestep_rejects_a_wrong_timestep_shape(spec):
    """A timestep that is neither ``[B]`` nor ``[B, noised frames]`` raises.

    Without the check a ``[B, F]`` tensor (the per-frame vector *including* the
    clean prefix, which is the easiest form to pass by mistake) would be
    reshaped into the token axis and shift every frame's modulation by one.
    """
    video = replace(spec.stream("video"),
                    token_timestep=TokenTimestep(clean_prefix=1))
    module = build(video.impl, spec=video, geometry=spec.geometry, world=spec)
    frames = spec.geometry.patch_grid[0]
    with pytest.raises(ValueError, match="token_timestep"):
        module.prepare(fake_inputs(spec)["video"],
                       timestep=torch.rand(BATCH, frames),
                       conditions=fake_bus(spec))


def test_clean_prefix_must_leave_a_noised_frame(spec):
    """A clean prefix as long as the latent stack is a contradiction.

    It would mean no frame carries noise at all, so the sampled timestep would
    never reach any token and the flow loss would have no signal.
    """
    video = replace(spec.stream("video"),
                    token_timestep=TokenTimestep(
                        clean_prefix=spec.geometry.patch_grid[0]))
    module = build(video.impl, spec=video, geometry=spec.geometry, world=spec)
    with pytest.raises(ValueError, match="no noised frame"):
        module.prepare(fake_inputs(spec)["video"],
                       timestep=torch.rand(BATCH), conditions=fake_bus(spec))


def test_token_timestep_requires_a_timestep_at_all(spec):
    """``token_timestep`` on a ``needs_timestep=False`` stream is rejected.

    The pair describes "modulation that varies per token" and "no modulation at
    all", which cannot both hold. Caught in ``StreamSpec.__post_init__`` so a
    provider typo fails before any parameter is allocated.
    """
    with pytest.raises(ValueError, match="needs_timestep=False"):
        replace(spec.stream("video"), needs_timestep=False,
                timestep_embed_width=None,
                token_timestep=TokenTimestep(clean_prefix=1))
    with pytest.raises(ValueError, match="non-negative"):
        TokenTimestep(clean_prefix=-1)


def test_context_projector_rejects_unknown_form_and_slot(spec):
    """An unknown projector form and a projector keyed on a slot that does not
    exist are both rejected at build time.

    A typo in the ``context_proj`` slot name would otherwise leave that slot
    with no projector and surface as a width mismatch inside cross attention.
    """
    with pytest.raises(ValueError, match="unknown projector shape"):
        build_context_projector("mlp3", in_width=8, out_width=8)
    video = spec.stream("video")
    impl = replace(video.impl,
                   params={**video.impl.params, "context_proj": {"typo": "mlp2"}})
    with pytest.raises(ValueError, match="cross_kv_slots"):
        build(impl, spec=video, geometry=spec.geometry, world=spec)


def test_entry_channels_are_the_conv_input_channels(spec):
    """The entry conv input channels equal ``latent_channels +
    concat_channels``, which is computed rather than declared a second time."""
    video = build_streams(spec)["video"]
    want = spec.geometry.latent_channels + spec.concat_channels("video")
    assert video.patch_embedding.weight.shape[1] == want
    assert tuple(video.patch_embedding.weight.shape[2:]) == spec.geometry.patch_size
    action = build_streams(spec)["action"]
    assert action.action_encoder.weight.shape[1] == spec.geometry.action_dim


# ---- 4. APPEND_TO_SLOT really lands in the host slot ----


def test_appended_condition_lengthens_the_host_slot(spec):
    """The proprio token really enters the text slot KV: after projection the
    slot length is ``CONTEXT_LEN + 1``.

    The appended token reuses the host slot's per-layer ``k/v/norm_k``, so
    whether it landed is invisible in ``state_dict`` and only the slot length
    shows it. That is why this case checks a length rather than the presence of
    a key. A slot with no appended condition (vpp2's clip_image) keeps its
    length.
    """
    inputs = fake_inputs(spec)
    bus = fake_bus(spec)
    for name, module in build_streams(spec).items():
        state = module.prepare(inputs[name], timestep=torch.rand(BATCH),
                               conditions=bus)
        kv = state.extras["cross_kv"]
        assert kv["text"].shape[1] == CONTEXT_LEN + 1, name
        if "clip_image" in kv:
            assert kv["clip_image"].shape[1] == CONTEXT_LEN, name


def test_disabled_appended_condition_leaves_the_slot_alone(spec):
    """Disabling the appended condition returns the slot to ``CONTEXT_LEN``
    instead of appending a zero token.

    A zero token would show up as one extra constant KV column in attention:
    numerically alive, invisible on any dashboard.
    """
    off = replace(spec, conditions=tuple(
        replace(cond, enabled=False) if cond.name == "proprio" else cond
        for cond in spec.conditions))
    stream = off.stream("video")
    module = build(stream.impl, spec=stream, geometry=off.geometry, world=off)
    state = module.prepare(fake_inputs(off)["video"],
                           timestep=torch.rand(BATCH), conditions=fake_bus(off))
    assert state.extras["cross_kv"]["text"].shape[1] == CONTEXT_LEN
