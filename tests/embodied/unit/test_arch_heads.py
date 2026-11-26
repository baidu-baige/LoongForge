# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L4 output heads: all four forms build, emit the right shapes, and use the
existing key names.

Division of labour with ``test_arch_blocks.py``: that file tests the L2
four-stage stream protocol, this one runs the L4 heads on its output. It reuses
``tiny_spec`` / ``run_engine`` from there, so the hidden states the heads
consume are the ones **the engine really computed**, not a ``randn``.

Four things are checked:

1. Every head of all three providers builds, and the forward output shape equals
   the shape of that stream's raw input tensor.
2. ``state_dict`` key names (``modulation`` / ``norm`` / ``head``) match the
   existing model, and the ``identity`` head has **no parameters at all**.
3. ``needs_timestep_embedding`` and the timestep embedding passed in agree in both
   directions, and a wrong embedding width raises.
4. A wrong ``form``, an ``identity`` head with unequal widths, and
   ``track_align`` not being a flow head all surface at **construction time**
   rather than drifting into a strict checkpoint load or into the loss.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from loongforge.embodied.model.arch.heads.flow import (
    IdentityHead, LinearHead, NormLinearHead, PatchFlowHead, build_flow_head)
from loongforge.embodied.model.arch.spec.core import Spec, build
from loongforge.embodied.model.arch.spec.providers.internw0_delta.spec import (
    track_query_head)
from loongforge.embodied.model.arch.spec.world import WorldModelSpec

from test_arch_blocks import (BATCH, MODEL_TYPES, build_streams, fake_bus,
                              run_engine, tiny_spec)

# The head table of each provider. The form names are an implementation detail,
# but "which head is which form" is the criterion this file checks.
HEAD_FORMS = {
    "vpp2": {"video_flow": PatchFlowHead, "action_flow": LinearHead},
    "fastwam": {"video_flow": PatchFlowHead, "action_flow": LinearHead},
    "internw0_delta": {"video_flow": PatchFlowHead, "action_flow": LinearHead,
                       "future_delta": NormLinearHead,
                       "semantic_future": IdentityHead},
}
# track_align is not a flow head (it is a decoder with learnable queries), so it
# raises at construction time
UNBUILT_HEADS = {"internw0_delta": ("track_align",)}


@pytest.fixture(params=MODEL_TYPES)
def spec(request) -> WorldModelSpec:
    return tiny_spec(request.param)


def build_heads(spec: WorldModelSpec) -> dict[str, torch.nn.Module]:
    """Same single call as step 3 of ``WorldModel.__init__``, skipping the heads
    that cannot be built."""
    skip = UNBUILT_HEADS.get(spec.model_type, ())
    return {head.name: build(head.impl, spec=head, geometry=spec.geometry,
                             world=spec)
            for head in spec.heads if head.name not in skip}


def run_heads(spec: WorldModelSpec):
    """Run the engine, then the heads.

    The timestep a head consumes is the source stream's
    ``extras['t_embed']``, that is the embedding rather than the scalar.
    """
    modules = build_streams(spec)
    layout, out = run_engine(spec, modules, fake_bus(spec))
    heads = build_heads(spec)
    results = {}
    for head in spec.heads:
        if head.name not in heads:
            continue
        timestep_embedding = (
            out.require(head.source_stream).extras["t_embed"]
            if head.needs_timestep_embedding else None)
        results[head.name] = heads[head.name](
            head.source.resolve(out), timestep_embedding=timestep_embedding)
    return layout, heads, results


# ---- 1. heads construct, and their output shape is the stream's raw shape ----


def test_every_head_is_the_declared_form(spec):
    heads = build_heads(spec)
    want = HEAD_FORMS[spec.model_type]
    assert set(heads) == set(want)
    for name, cls in want.items():
        assert isinstance(heads[name], cls), name
        assert heads[name].output_width == spec.head(name).output_width


def test_head_output_shape_matches_the_stream_raw_tensor(spec):
    """The video head emits ``[B, C, F, H, W]``, the action head ``[B, T, D]``.

    This invariant makes the L3 regression target and the head output the same
    shape, so the loss step never reshapes. The video head's patch unfold
    happens *inside* the head, which is why the result is a latent grid rather
    than a token sequence.
    """
    geo = spec.geometry
    _, _, out = run_heads(spec)
    frames, height, width = geo.latent_grid
    assert out["video_flow"].shape == (BATCH, geo.latent_channels,
                                       frames, height, width)
    assert out["action_flow"].shape == (BATCH, geo.action_horizon,
                                        geo.action_dim)
    for tensor in out.values():
        assert torch.isfinite(tensor).all()


def test_intermediate_layer_heads_read_the_captured_hidden(spec):
    """Intermediate-layer heads read ``captures``, shaped ``[B, L, residual_width]``."""
    if spec.model_type != "internw0_delta":
        pytest.skip("only InternW0-Delta has intermediate-layer heads")
    layout, _, out = run_heads(spec)
    tokens = layout.range_of("video").length
    video_width = spec.stream("video").residual_width
    # The identity head returns the hidden state unchanged, so the width has
    # to still be the residual width.
    assert out["semantic_future"].shape == (BATCH, tokens, video_width)
    # The norm_linear head projects to 192.
    assert out["future_delta"].shape == (BATCH, tokens, 192)


def test_every_head_parameter_gets_a_gradient(spec):
    _, heads, out = run_heads(spec)
    loss = sum(tensor.square().mean() for tensor in out.values())
    loss.backward()
    for name, head in heads.items():
        for param_name, param in head.named_parameters():
            assert param.grad is not None, f"{name}.{param_name}"


# ---- 2. state_dict key names ----


def test_head_state_dict_keys_match_the_existing_checkpoint(spec):
    """The three names ``modulation`` / ``norm`` / ``head`` match the old code.

    The old ``future_delta`` was an ``nn.Sequential``, so its keys were ``0.*``
    and ``1.*``. Splitting it into ``norm`` and ``head`` makes porting weights a
    *determinate* rename instead of a question ("is entry 0 the norm or the
    linear?") that can only be answered by reading the source.
    """
    heads = build_heads(spec)
    assert set(heads["video_flow"].state_dict()) == {
        "modulation", "head.weight", "head.bias"}
    assert set(heads["action_flow"].state_dict()) == {
        "head.weight", "head.bias"}
    if spec.model_type == "internw0_delta":
        assert set(heads["future_delta"].state_dict()) == {
            "norm.weight", "norm.bias", "head.weight", "head.bias"}
        # The identity head has no parameters: the old ``semantic_future``
        # computed ``1 - cos`` straight on the raw hidden state
        # (``losses.py:58-72``) and had no projection layer.
        assert not list(heads["semantic_future"].parameters())
        assert not heads["semantic_future"].state_dict()


def test_patch_flow_linear_width_includes_the_patch_volume(spec):
    """The head's Linear output width is ``output_width * prod(patch_size)``.

    The ``* prod(patch_size)`` factor stays inside the head, so
    ``HeadSpec.output_width`` only records the channel count *after* the unfold.
    One field, one meaning.
    """
    import math
    geo = spec.geometry
    head = build_heads(spec)["video_flow"]
    want = geo.latent_channels * math.prod(geo.patch_size)
    assert head.head.weight.shape[0] == want
    assert head.head.weight.shape[1] == spec.stream("video").residual_width
    # Two modulation slots, as wide as the source stream's residual width (the
    # timestep embedding is added straight onto it).
    assert head.modulation.shape == (1, 2, spec.stream("video").residual_width)
    # The norm is elementwise_affine=False: the affine part is carried by the
    # two modulation slots instead.
    assert not list(head.norm.parameters())


def test_unpatchify_is_the_exact_inverse_of_the_entry_flatten(spec):
    """The unfold axis order is the strict inverse of the entry flatten.

    Getting the order wrong here is *silent*. The test builds a tensor where
    every latent voxel holds its own linear index, pushes it through the
    inverse of the entry convolution (a hand-written patchify) and then through
    the head's ``_unpatchify``; it has to come back unchanged. The shapes agree
    either way, so only an element-wise comparison shows the time axis and the
    spatial axes being mixed up.
    """
    geo = spec.geometry
    head = build_heads(spec)["video_flow"]
    frames, height, width = geo.latent_grid
    channels = geo.latent_channels
    volume = torch.arange(BATCH * channels * frames * height * width,
                          dtype=torch.float32)
    volume = volume.reshape(BATCH, channels, frames, height, width)
    patch_t, patch_h, patch_w = geo.patch_size
    gf, gh, gw = geo.patch_grid
    # Hand-written patchify, in the same flatten order as
    # Conv3d(kernel=stride=patch): the last dim runs (pt, ph, pw, C) and the
    # token dim runs (f, h, w) in C order.
    packed = volume.reshape(BATCH, channels, gf, patch_t, gh, patch_h,
                            gw, patch_w)
    packed = packed.permute(0, 2, 4, 6, 3, 5, 7, 1)
    tokens = packed.reshape(BATCH, gf * gh * gw, -1)
    assert tokens.shape[1] == geo.video_seq_len
    assert torch.equal(head._unpatchify(tokens), volume)


# ---- 3. the timestep is declared, and checked in both directions ----


def test_timestep_declaration_is_checked_both_ways(spec):
    heads = build_heads(spec)
    video, action = heads["video_flow"], heads["action_flow"]
    video_width = spec.stream("video").residual_width
    hidden = torch.randn(BATCH, 4, video_width)
    with pytest.raises(ValueError, match="needs_timestep_embedding=True"):
        video(hidden, timestep_embedding=None)
    with pytest.raises(ValueError, match="needs_timestep_embedding=False"):
        action(torch.randn(BATCH, 4, spec.stream("action").residual_width),
               timestep_embedding=torch.randn(BATCH, 16))


def test_head_rejects_the_raw_timestep_scalar(spec):
    """A head consumes the timestep *embedding*, not the scalar in ``[0, 1]``.

    In the old code the ``t`` of ``Head.forward(x, t)`` was already the
    embedding: it is added straight onto the modulation. The two have different
    widths, so passing the wrong one broadcasts to a wrong shape at the
    addition *without raising*.
    """
    video = build_heads(spec)["video_flow"]
    width = spec.stream("video").residual_width
    hidden = torch.randn(BATCH, 4, width)
    with pytest.raises(ValueError, match="consumes the timestep"):
        video(hidden, timestep_embedding=torch.rand(BATCH))
    with pytest.raises(ValueError, match="consumes the timestep"):
        video(hidden, timestep_embedding=torch.randn(BATCH, width + 1))


def test_head_rejects_hidden_of_the_wrong_width(spec):
    """The head's input width is *looked up* (the source stream's
    ``residual_width``), not declared as a field."""
    action = build_heads(spec)["action_flow"]
    width = spec.stream("action").residual_width
    with pytest.raises(ValueError, match="consumes the hidden state of source stream"):
        action(torch.randn(BATCH, 4, width + 1))
    with pytest.raises(ValueError, match="consumes the hidden state of source stream"):
        action(torch.randn(BATCH, width))


# ---- 4. a wrong declaration shows up at construction time ----


def test_unknown_form_is_rejected_with_the_options(spec):
    head = spec.head("action_flow")
    with pytest.raises(ValueError, match="has unknown params"):
        build_flow_head(spec=head, geometry=spec.geometry, world=spec,
                        form="mlp")
    # ``form`` deliberately has no default: the four forms all have different
    # parameter counts, so guessing wrong surfaces as missing or unexpected
    # keys during a strict load, far away from where the form was declared.
    with pytest.raises(TypeError):
        build_flow_head(spec=head, geometry=spec.geometry, world=spec)


def test_identity_form_requires_equal_widths(spec):
    """``identity`` has no projection, so ``output_width`` must equal the
    source stream's residual width."""
    head = spec.head("action_flow")
    bad = replace(head, output_width=spec.stream("action").residual_width + 1)
    with pytest.raises(ValueError, match="identity"):
        build_flow_head(spec=bad, geometry=spec.geometry, world=spec,
                        form="identity")
    good = replace(head, output_width=spec.stream("action").residual_width)
    assert isinstance(
        build_flow_head(spec=good, geometry=spec.geometry, world=spec,
                        form="identity"), IdentityHead)


def test_patch_flow_only_accepts_the_two_slot_table(spec):
    from loongforge.embodied.model.arch.streams.spec import DEFAULT_WAN_SLOTS
    head = replace(spec.head("video_flow"), modulation_slots=DEFAULT_WAN_SLOTS)
    with pytest.raises(ValueError, match="accepts only two-slot modulation"):
        build(replace(head.impl, params={"form": "patch_flow"}),
              spec=head, geometry=spec.geometry, world=spec)


def test_track_align_is_not_a_flow_head(spec):
    """It is a decoder student network with learnable queries, which none of
    the four flow forms can express."""
    if spec.model_type != "internw0_delta":
        pytest.skip("only InternW0-Delta has track_align")
    head = spec.head("track_align")
    assert head.impl.impl is track_query_head
    with pytest.raises(NotImplementedError, match="track_bridge"):
        build(head.impl, spec=head, geometry=spec.geometry, world=spec)


def test_head_impl_params_carry_only_the_form(spec):
    """No params other than ``form`` are allowed: everything else is already a
    field on one of the spec layers."""
    for head in spec.heads:
        if head.impl.impl is build_flow_head:
            assert set(head.impl.params) == {"form"}, head.name
        else:
            assert isinstance(head.impl, Spec)
