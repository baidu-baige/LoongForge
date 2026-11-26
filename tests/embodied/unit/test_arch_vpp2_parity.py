# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Numerical parity between this architecture and the VPP2 source it replaces.

The shape of the check is "two independent implementations, one set of
weights". Both sides run their **own** full forward; nothing is shared except
the numbers in the parameters, which travel from the legacy side to this one
through ``transfer.vpp2_to_arch``. Nothing here wraps or calls a VPP2 module
inside the new code, so a regression cannot hide behind a shim.

Three layers are compared separately, because a mismatch in each has a
different cause:

====================  ==================================================
entry                 ``StreamModule.prepare`` against the legacy
                      ``pre_dit``: patchify, the timestep embedding, the
                      projected per-slot context, and the RoPE table.
every joint layer      ``MoT.forward``'s per-layer callback against the
                      engine's ``captures``. The two fire at the same
                      point, so the first mismatching layer is reported
                      directly instead of only the exit being wrong.
heads                 the two L4 heads against legacy
                      ``Head``/``unpatchify`` and the bare action
                      ``Linear``.
visibility            the compiled ``[S, S]`` matrix against
                      ``VPP2._build_mot_attention_mask``, cell for cell.
                      This one is **not** implied by the three above: the
                      numerical sides share one mask, so they agree under a
                      wrong one.
====================  ==================================================

The comparison is **exact equality**, not a tolerance. The two sides issue the
same primitive operations in the same order, so in float32 they agree
bit-for-bit; a tolerance would let a genuine reordering through.

The VPP2 source is not a dependency of this repository. Its location comes
from ``$VPP2_SOURCE_ROOT``, falling back to the sibling checkout, and the whole
module skips when neither is importable, so CI never depends on an external
tree.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from loongforge.embodied.model.arch.builder import (WorldModel,
                                                    materialize_conditions)
from loongforge.embodied.model.arch.spec.world import WorldModelSpec
from loongforge.embodied.model.arch.streams.engine import EngineOutput
from loongforge.embodied.model.arch.factory import (PROPRIO_PAYLOAD_KEY,
                                                    vpp2_to_arch)

from arch_parity import (assert_identical, assert_same_mask, condition_frames,
                         structural_mask)
from test_arch_blocks import BATCH, tiny_spec
from test_arch_world_model import world


def _vpp2_root() -> Path:
    """Return the directory that should hold the ``vpp2`` package.

    ``$VPP2_SOURCE_ROOT`` wins outright when set, so that a checkout
    elsewhere can be pointed at without editing this file and a typo in it
    surfaces as a skip naming the path rather than as a silent fall-through to
    some other tree. Without it, the sibling ``projects/`` tree of the usual
    working copy is tried as a convenience.
    """
    env = os.environ.get("VPP2_SOURCE_ROOT")
    if env:
        return Path(env)
    return (Path(__file__).resolve().parents[3].parent / "projects"
            / "video-prediction-policy-2" / "src")


_ROOT = _vpp2_root()
if not (_ROOT / "vpp2").is_dir():
    pytest.skip(f"no vpp2 package under {_ROOT}; point $VPP2_SOURCE_ROOT at "
                "the directory holding it to enable the parity comparison",
                allow_module_level=True)
sys.path.insert(0, str(_ROOT))

# The two DiT files need only torch and einops: their ``flash_attention`` is
# an SDPA wrapper, so no attention kernel has to be installed to run this.
pytest.importorskip("einops", reason="the VPP2 DiT modules import einops")
_mot = pytest.importorskip("vpp2.models.wan21_14b.mot")
_video_dit = pytest.importorskip("vpp2.models.wan21_14b.wan_video_dit")
_action_dit = pytest.importorskip("vpp2.models.wan21_14b.action_dit")
# Only for ``_build_mot_attention_mask``, called unbound below. The module
# also reaches omegaconf and PIL, which is why it is skipped separately from
# the three above rather than imported with them.
_vpp2 = pytest.importorskip("vpp2.models.wan21_14b.vpp2")

# Length of the fake text context. Only "more than one token, and not equal
# to any other length in the fixture" matters.
TEXT_LEN = 5
# Not configurable on the legacy side: ``I2VCrossAttention.forward`` raises
# below 257 context tokens and splits the context at exactly 257
# (``wan_video_dit.py:188-194``), which is how it tells the image slot from
# the text slot. The new side keeps the two slots apart by name instead, so
# this number appears here only to satisfy the legacy half.
CLIP_TOKENS = 257


# ---- building the two sides ----


@dataclass
class Legacy:
    """The VPP2 half: the two experts, their joint driver, and the encoder.

    ``proprio`` is a plain ``Linear`` rather than part of ``mot`` because that
    is where it lives in the source as well -- it hangs off ``VPP2`` itself
    and belongs to neither expert (``vpp2.py:119``), which is also why
    ``vpp2_to_arch`` reads it from the top level of the payload.
    """

    mot: Any
    video: Any
    action: Any
    proprio: torch.nn.Module

    def payload(self) -> dict[str, Any]:
        """Build a legacy-shaped ckpt out of the live legacy parameters."""
        return {"mot": self.mot.state_dict(),
                PROPRIO_PAYLOAD_KEY: self.proprio.state_dict()}


def build_legacy(spec: WorldModelSpec) -> Legacy:
    """Construct the VPP2 modules from the same numbers as ``spec``.

    Every scalar is read off the spec rather than written twice. Two sets of
    literals would let the sides drift and the symptom would be a shape error
    at load, which says nothing about which side is wrong.
    """
    geometry = spec.geometry
    video_spec, action_spec = spec.streams
    channels = spec.concat_channels(video_spec.name)
    text_width = spec.condition("text").width
    video = _video_dit.WanVideoDiT(
        model_type="i2v", patch_size=tuple(geometry.patch_size),
        in_dim=geometry.latent_channels + channels,
        hidden_dim=video_spec.residual_width, ffn_dim=video_spec.ffn_width,
        freq_dim=video_spec.timestep_embed_width, text_dim=text_width,
        out_dim=geometry.latent_channels, num_heads=video_spec.num_heads,
        attn_head_dim=video_spec.head_dim,
        num_layers=video_spec.num_layers, condition_channels=channels)
    action = _action_dit.ActionDiT(
        hidden_dim=action_spec.residual_width,
        action_dim=geometry.action_dim, ffn_dim=action_spec.ffn_width,
        text_dim=text_width, freq_dim=action_spec.timestep_embed_width,
        eps=1e-6, num_heads=action_spec.num_heads,
        attn_head_dim=action_spec.head_dim,
        num_layers=action_spec.num_layers)
    # The three flags pin the joint (non-cached) path. With
    # ``use_fixed_video_layers`` set or ``use_video_tokens`` on, ``MoT``
    # switches to its prefill/cache branch (``mot.py:839``), which the new
    # engine has no counterpart for; checkpointing would only change where
    # the activations live, and off keeps the comparison one forward long.
    mot = _mot.MoT({video_spec.name: video, action_spec.name: action},
                   mot_checkpoint_mixed_attn=False,
                   use_fixed_video_layers=None, use_video_tokens=False)
    return Legacy(
        mot=mot.eval(), video=video, action=action,
        proprio=torch.nn.Linear(spec.condition("proprio").source_width,
                                spec.condition("proprio").width).eval())


@dataclass
class Sides:
    """Both implementations plus the one batch and timestep pair they share.

    The timesteps are ``[B]`` on purpose. The new side's sampler may broadcast
    a timestep over a layout axis (per frame, for instance), which the legacy
    ``pre_dit`` cannot express at all, so the parity fixture supplies them
    directly instead of going through ``sample_timesteps``.
    """

    spec: WorldModelSpec
    model: WorldModel
    legacy: Legacy
    batch: dict[str, torch.Tensor]
    video_timestep: torch.Tensor
    action_timestep: torch.Tensor
    load_report: Any


def build_sides() -> Sides:
    """Build both sides, move the legacy weights over, and draw one batch."""
    torch.manual_seed(0)
    spec = tiny_spec("vpp2")
    geometry = spec.geometry
    video_spec, action_spec = spec.streams
    model = world(spec).eval()
    legacy = build_legacy(spec)
    report = model.load_state_dict(vpp2_to_arch(legacy.payload()),
                                   strict=True)
    frames, rows, cols = geometry.latent_grid
    batch = {
        video_spec.batch_key: torch.randn(BATCH, geometry.latent_channels,
                                          frames, rows, cols),
        action_spec.batch_key: torch.randn(BATCH, geometry.action_horizon,
                                           geometry.action_dim),
        spec.condition("clip_image").source_key: torch.randn(
            BATCH, CLIP_TOKENS, spec.condition("clip_image").width),
        spec.condition("text").source_key: torch.randn(
            BATCH, TEXT_LEN, spec.condition("text").width),
        spec.condition("cond_latent").source_key: torch.randn(
            BATCH, spec.concat_channels(video_spec.name), frames, rows, cols),
        spec.condition("proprio").source_key: torch.randn(
            BATCH, spec.condition("proprio").source_width),
    }
    return Sides(spec=spec, model=model, legacy=legacy, batch=batch,
                 video_timestep=torch.rand(BATCH),
                 action_timestep=torch.rand(BATCH), load_report=report)


@dataclass
class Run:
    """One forward on each side, under one choice of text key padding mask."""

    new: EngineOutput
    new_heads: dict[str, torch.Tensor]
    new_video: Any
    new_action: Any
    legacy_entry: dict[str, dict[str, Any]]
    legacy_tokens: dict[str, torch.Tensor]
    legacy_layers: dict[int, torch.Tensor]
    legacy_heads: dict[str, torch.Tensor]


def _legacy_context(sides: Sides, text_valid: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Append the proprio token to the **raw** text context, legacy style.

    Both sides concatenate before the projector, not after: there is one
    ``text_embedding`` and the legacy source cats the encoded proprio onto the
    raw context ahead of it (``vpp2.py:520-540``). The mask grows by the one
    all-valid column in the same step, which is what the new side's
    ``_embed_context`` does off ``world.appended_to``.
    """
    spec = sides.spec
    raw = sides.batch[spec.condition("text").source_key]
    token = sides.legacy.proprio(
        sides.batch[spec.condition("proprio").source_key].unsqueeze(1))
    ones = torch.ones(BATCH, 1, dtype=torch.bool)
    return (torch.cat([raw, token], dim=1),
            torch.cat([text_valid, ones], dim=1))


def _run_new(sides: Sides, text_valid: torch.Tensor) -> tuple[Any, ...]:
    """Run this architecture: bus, both ``prepare``s, the engine, the heads.

    ``engine.run`` is called directly rather than through
    ``WorldModel.forward`` for two reasons: the timesteps have to be the
    ``[B]`` ones this fixture chose, and every joint layer has to be captured,
    while ``spec.captures()`` requests only the layers some head declared.
    """
    spec, model = sides.spec, sides.model
    video_spec, action_spec = spec.streams
    bus = materialize_conditions(spec.conditions, sides.batch,
                                 training=False,
                                 encoders=model.condition_encoders)
    bus.conditions["text"] = replace(bus.conditions["text"],
                                     key_valid=text_valid)
    group = spec.attention_groups[0]
    timesteps = {video_spec.name: sides.video_timestep,
                 action_spec.name: sides.action_timestep}
    inputs = {stream.name: sides.batch[stream.batch_key]
              for stream in spec.streams}
    layers = tuple(range(group.num_layers))
    output = model.engine.run(
        group, inputs, {name: model.blocks[name]
                        for name in group.stream_names},
        model.plans[0], model.layout, conditions=bus, timesteps=timesteps,
        captures={name: layers for name in group.stream_names})
    states = {name: model.blocks[name].prepare(
        inputs[name], timestep=timesteps[name], conditions=bus)
        for name in group.stream_names}
    return (output, model._run_heads(output), states[video_spec.name],
            states[action_spec.name])


def _run_legacy(sides: Sides, text_valid: torch.Tensor,
                joint_mask: torch.Tensor) -> tuple[Any, ...]:
    """Run VPP2 end to end on its own entry output.

    Only ``joint_mask`` crosses over from the new side, and it has to: the
    legacy code builds a video-to-video mask only
    (``build_video_to_video_mask``) and has no notion of the joint
    ``[S, S]`` matrix that ``MoT.forward`` nevertheless demands, so the caller
    is its sole source there. Everything else -- tokens, RoPE, modulation,
    projected context -- is the legacy side's own ``pre_dit`` output, which
    keeps the two implementations independent.

    The video input arrives with the channel conditions already concatenated
    so that ``pre_dit`` skips ``_build_condition``. Deriving the 4-channel
    mask and the 16-channel conditioning latent is data-pipeline work; the new
    architecture declares the finished 20 channels as one ``CHANNEL_CONCAT``
    condition and expects them in the batch.
    """
    spec, legacy = sides.spec, sides.legacy
    video_spec, action_spec = spec.streams
    context, context_mask = _legacy_context(sides, text_valid)
    entry = {
        video_spec.name: legacy.video.pre_dit(
            x=torch.cat([sides.batch[video_spec.batch_key],
                         sides.batch[
                             spec.condition("cond_latent").source_key]],
                        dim=1),
            timestep=sides.video_timestep, context=context,
            context_mask=context_mask,
            clip_fea=sides.batch[spec.condition("clip_image").source_key],
            condition_latents=None),
        action_spec.name: legacy.action.pre_dit(
            action_tokens=sides.batch[action_spec.batch_key],
            timestep=sides.action_timestep, context=context,
            context_mask=context_mask),
    }
    per_layer: dict[int, torch.Tensor] = {}
    tokens = legacy.mot(
        {name: state["tokens"] for name, state in entry.items()},
        joint_mask,
        {name: state["freqs"] for name, state in entry.items()},
        {name: {"context": state["context"], "mask": state["context_mask"]}
         for name, state in entry.items()},
        {name: state["t_mod"] for name, state in entry.items()},
        video_feature_callback=lambda index, hidden: per_layer.__setitem__(
            index, hidden))
    heads = {
        "video_flow": legacy.video.unpatchify(
            legacy.video.head(tokens[video_spec.name],
                              entry[video_spec.name]["t"]),
            spec.geometry.patch_grid),
        "action_flow": legacy.action.head(tokens[action_spec.name]),
    }
    return entry, tokens, per_layer, heads


def run_both(sides: Sides, text_valid: torch.Tensor) -> Run:
    """Run both sides under one mask, with grads off and both sides in eval."""
    with torch.no_grad():
        new, new_heads, new_video, new_action = _run_new(sides, text_valid)
        entry, tokens, per_layer, heads = _run_legacy(
            sides, text_valid, sides.model.plans[0].structural)
    return Run(new=new, new_heads=new_heads, new_video=new_video,
               new_action=new_action, legacy_entry=entry,
               legacy_tokens=tokens, legacy_layers=per_layer,
               legacy_heads=heads)


def padded_text_mask() -> torch.Tensor:
    """A text mask with a padded tail, the case the slot mask exists for."""
    mask = torch.ones(BATCH, TEXT_LEN, dtype=torch.bool)
    mask[:, -2:] = False
    return mask


@pytest.fixture(scope="module")
def sides() -> Sides:
    """Both sides, built once: 16 layers twice is the expensive part here."""
    return build_sides()


@pytest.fixture(scope="module")
def run(sides: Sides) -> Run:
    """The padded-text run that most cases compare."""
    return run_both(sides, padded_text_mask())


# ---- the weights have to arrive before anything can be compared ----


def test_the_legacy_weights_load_with_nothing_missing_or_unexpected(sides):
    """The premise of every comparison below: one set of numbers, two models.

    Without this, a parity failure cannot be told apart from a remap failure,
    and the two call for opposite fixes. ``test_arch_transfer.py`` pins the
    rule table against generated key names; this pins it against the key
    names the real VPP2 modules actually produce.
    """
    assert not sides.load_report.missing_keys
    assert not sides.load_report.unexpected_keys


# ---- the entry ----


def test_the_entry_produces_the_same_tokens_and_modulation(run, sides):
    """``prepare`` against ``pre_dit``, for both streams.

    Three separate things ride on this one comparison: the patchify/flatten
    order, the sinusoidal timestep embedding, and the ``[B, 6, W]``
    pre-combination modulation. They are asserted together because a single
    ``prepare`` call produces all three and splitting them would triple the
    forward passes without isolating anything new -- each has its own name in
    the message.
    """
    video, action = sides.spec.streams
    for stream, state in ((video, run.new_video), (action, run.new_action)):
        entry = run.legacy_entry[stream.name]
        assert_identical(state.hidden, entry["tokens"],
                         f"{stream.name} entry tokens")
        assert_identical(state.modulation, entry["t_mod"],
                         f"{stream.name} modulation")
        assert_identical(state.extras["t_embed"], entry["t"],
                         f"{stream.name} timestep embedding")


def test_the_rope_tables_are_identical(run, sides):
    """The 3-D video table and the 1-D action table, sliced to the token count.

    The new side tabulates once in ``__init__`` and keeps the whole table, so
    the legacy per-call slice is what the comparison has to match. A wrong
    axis split here is silent: the shape is right either way and only the
    numbers move.
    """
    video, action = sides.spec.streams
    for stream, state in ((video, run.new_video), (action, run.new_action)):
        length = state.hidden.shape[1]
        table = sides.model.blocks[stream.name]._freqs
        assert_identical(table[:length],
                         run.legacy_entry[stream.name]["freqs"],
                         f"{stream.name} rope table")


def test_the_projected_context_matches_slot_by_slot(run, sides):
    """One legacy tensor, two named slots here, and the split point is 257.

    The legacy context is a single ``cat([image, text])`` that
    ``I2VCrossAttention`` re-splits at a hardcoded 257 every layer. Here the
    slots are separate entries of ``extras["cross_kv"]``, so the comparison
    has to re-join them in ``cross_kv_slots`` order -- that order is what
    decides whether the legacy split lands on the slot boundary, and getting
    it backwards would feed the image keys to the text projection.
    """
    video, action = sides.spec.streams
    joined = torch.cat([run.new_video.extras["cross_kv"][slot]
                        for slot in video.cross_kv_slots], dim=1)
    assert run.new_video.extras["cross_kv"]["clip_image"].shape[1] \
        == CLIP_TOKENS
    assert_identical(joined, run.legacy_entry[video.name]["context"],
                     "video projected context")
    assert action.cross_kv_slots == ("text",)
    assert_identical(run.new_action.extras["cross_kv"]["text"],
                     run.legacy_entry[action.name]["context"],
                     "action projected context")


def test_the_text_slot_mask_carries_the_padded_tail_and_the_proprio_token(
        run, sides):
    """The new per-slot mask against the legacy single combined mask.

    The legacy mask covers image and text in one row of length
    ``257 + TEXT_LEN + 1`` and is then sliced at 257 inside the cross
    attention, leaving the image branch maskless. The new side never builds
    the image half: ``clip_image`` is absent from ``cross_kv_mask``, which is
    what makes its attention call the maskless one. So the two agree on the
    text half, and the new side's silence about the image half is the claim
    being checked on that side.
    """
    video, action = sides.spec.streams
    legacy_mask = run.legacy_entry[video.name]["context_mask"]
    text_half = legacy_mask[:, 0, CLIP_TOKENS:]
    assert set(run.new_video.extras["cross_kv_mask"]) == {"text"}
    assert_identical(run.new_video.extras["cross_kv_mask"]["text"].int(),
                     text_half.int(), "video text slot mask")
    # proprio declares no mask, so the token it appends is valid, and it has
    # to be the **last** column because that is where its value was appended
    assert not text_half[:, -2:-1].any()
    assert text_half[:, -1].all()
    assert_identical(
        run.new_action.extras["cross_kv_mask"]["text"].int(),
        run.legacy_entry[action.name]["context_mask"][:, 0, :].int(),
        "action text slot mask")

# ---- every joint layer, then the exit ----


def test_every_joint_layer_matches_and_the_first_bad_one_is_named(run, sides):
    """Layer by layer, because only the exit would say "somewhere upstream".

    The two capture points coincide: the legacy callback fires right after the
    post-block (``mot.py:952-954``) and the engine stores the hidden state at
    the same place (``engine.py:587-589``), so layer *i* means the same thing
    on both sides and no offset correction is needed.

    Only the video stream has a legacy callback, which is also the stream
    worth walking: it is the one with two cross-attention slots. The action
    stream is covered at the exit below, and a divergence there would have to
    come through the shared attention anyway.
    """
    video, _ = sides.spec.streams
    layers = range(sides.spec.attention_groups[0].num_layers)
    assert set(run.legacy_layers) == set(layers)
    bad = [index for index in layers
           if not torch.equal(run.legacy_layers[index],
                              run.new.captured[(video.name, index)])]
    if bad:
        first = bad[0]
        diff = (run.legacy_layers[first]
                - run.new.captured[(video.name, first)]).abs().max()
        raise AssertionError(
            f"first mismatching joint layer {first} of "
            f"{len(list(layers))}, max abs {diff.item():.3e}; "
            f"{len(bad)} layers differ")


def test_the_engine_exit_hidden_states_match(run, sides):
    """Both streams at the exit, after ``finalize``.

    Worth asserting even with every layer already compared: ``finalize`` runs
    after the last layer, and for this family it is deliberately a no-op (the
    final norm sits in the head). A stray operation there would show up only
    here.
    """
    for stream in sides.spec.streams:
        assert_identical(run.new.states[stream.name].hidden,
                         run.legacy_tokens[stream.name],
                         f"{stream.name} exit hidden state")


def test_the_two_output_heads_match(run, sides):
    """``patch_flow`` against ``Head`` + ``unpatchify``, ``linear`` against it.

    The video head is the interesting half: its modulation slots, the
    ``x prod(patch_size)`` exit width, and the unflatten all have to line up,
    and the unflatten is the one step whose failure keeps the shape right
    while stirring the time axis into the spatial axes.
    """
    for name in ("video_flow", "action_flow"):
        assert_identical(run.new_heads[name], run.legacy_heads[name],
                         f"head {name}")
    geometry = sides.spec.geometry
    assert tuple(run.new_heads["video_flow"].shape) == (
        BATCH, geometry.latent_channels, *geometry.latent_grid)

# ---- the two inputs that would make all of the above vacuous ----


def test_an_all_valid_text_mask_also_matches_and_gives_a_different_answer(
        sides, run):
    """Parity holds under a second mask, and the mask is not decoration.

    Two claims in one forward. "Also matches" rules out a shared bug that
    ignores ``key_valid`` on both sides, which would make every comparison
    above pass with the mask doing nothing. "Different answer" rules out the
    opposite reading of the same symptom: that this fixture's padded columns
    happen not to move the result, in which case the first claim would be
    free.
    """
    unmasked = run_both(sides, torch.ones(BATCH, TEXT_LEN, dtype=torch.bool))
    for stream in sides.spec.streams:
        assert_identical(unmasked.new.states[stream.name].hidden,
                         unmasked.legacy_tokens[stream.name],
                         f"{stream.name} exit hidden state, no padding")
        assert not torch.equal(unmasked.new.states[stream.name].hidden,
                               run.new.states[stream.name].hidden)


# ---- the visibility matrix, against the builder VPP2 uses ----

# The libero config, which is what ``_vpp2_visibility`` was written against:
# ``video_attention_mask_mode: bidirectional`` (``train_action.yaml:191``).
# It is written here rather than read off ``build_legacy``'s expert because it
# is **not** the class default -- ``WanVideoDiT`` defaults to
# ``first_frame_causal`` (``wan_video_dit.py:336``) -- and nothing else in
# this file needs the legacy mask, so that constructor does not pass it.
MASK_MODE = "bidirectional"


def legacy_structural(spec: WorldModelSpec) -> torch.Tensor:
    """Build the ``[S, S]`` matrix VPP2 itself would attend under.

    ``_build_mot_attention_mask`` is called unbound on a stand-in, because the
    two attributes it reads are the whole of its dependence on ``VPP2``: the
    video expert's own mask builder (``vpp2.py:889``) and the visible frame
    count (``:896``, supplied as the explicit argument instead). Constructing
    a real ``VPP2`` would load the Wan components off disk.

    Args:
        spec: The spec under comparison. It supplies both sequence lengths,
            the tokens per frame and the condition frame count, so the two
            sides cannot be measured at different sizes.

    Returns:
        The legacy mask, ``[total, total]`` bool on CPU.
    """
    expert = SimpleNamespace(video_attention_mask_mode=MASK_MODE)
    expert.build_video_to_video_mask = (
        _video_dit.WanVideoDiT.build_video_to_video_mask.__get__(expert))
    layout = spec.build_layout()
    return _vpp2.VPP2._build_mot_attention_mask(
        SimpleNamespace(video_expert=expert),
        video_seq_len=layout.range_of("video").length,
        action_seq_len=layout.range_of("action").length,
        video_tokens_per_frame=spec.geometry.tokens_per_frame,
        device=torch.device("cpu"),
        action_visible_video_frames=condition_frames(spec))


@pytest.mark.parametrize("visible_frames", (1, 2, 3))
def test_the_visibility_matrix_matches_the_legacy_builder(visible_frames):
    """Cell for cell against ``_build_mot_attention_mask``.

    Nothing above this line implies it. ``_run_legacy`` hands the legacy
    driver **this** side's ``plan.structural`` (:348), so every forward
    comparison in this file holds just as well under a mask both sides got
    equally wrong. That is not hypothetical: the first version of
    ``_vpp2_visibility`` was 24 of 784 cells too strict in the action rows and
    no numerical case moved.

    Three configurations rather than the shipped one, because the declaration
    may not contain a token count. The rule names the condition phase and the
    frame count enters through the phase label, so an expression that happens
    to match at three frames and not at one is exactly the regression worth
    catching.
    """
    spec = tiny_spec("vpp2", video_prefix_frames=visible_frames)
    assert condition_frames(spec) == visible_frames
    assert_same_mask(structural_mask(spec), legacy_structural(spec),
                     f"visibility at video_prefix_frames={visible_frames}")


def test_the_matrix_fed_to_mot_is_the_one_compared_above(sides):
    """``model.plans[0]`` is what the case above speaks for.

    That case compiles the declaration directly, which is what makes one spec
    per configuration affordable; this one closes the gap to the object the
    harness really hands ``MoT.forward``. It also records the one shape
    conversion the harness has to get right: ``MoT.forward`` rejects anything
    but a 2-D square mask, while ``plan.materialize("dense")`` returns
    ``[1, 1, S, S]``.

    The two all/any assertions stay because they fail differently: a matrix of
    all True would leave every comparison in this file saying nothing about
    visibility, while the equality below would still pass if the legacy
    builder were handed a wrong frame count.
    """
    structural = sides.model.plans[0].structural
    total = sides.model.layout.total
    assert tuple(structural.shape) == (total, total)
    assert structural.dtype is torch.bool
    assert not structural.all()
    assert structural.any()
    assert_same_mask(structural, structural_mask(sides.spec),
                     "plans[0] against the compiled declaration")

