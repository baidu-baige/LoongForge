# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Numerical parity between this architecture and the InternW0-Delta source.

Same shape of check as ``test_arch_vpp2_parity.py`` -- two independent
implementations, one set of weights, moved across by
``transfer.internw0_delta_to_arch`` -- and the same four comparison layers
(entry, every joint layer, the two flow heads, and the visibility matrix
against this source's own builder). Read that module first; this one records
only what differs.

Four things differ, and the first two are the reason this harness exists at
all rather than the VPP2 one being parametrized:

1. **The video timestep is per token and its zeros are built in two places.**
   The caller supplies ``condition_frames - 1`` leading zeros
   (``losses.py:624-634``) and ``pre_dit`` prepends one more
   (``wan_video_dit.py:826-836``), so a ``[B]`` draw on this side has to
   become ``[B, F-1]`` on that side for the two to mean the same thing.
   ``TokenTimestep(clean_prefix=condition_frames)`` is the single declaration
   that replaces both, and
   ``test_the_clean_prefix_covers_all_three_condition_frames`` is what pins
   the composition.
2. **The two streams hand the joint driver different-rank modulation in one
   call**: ``[B, S, 6, W]`` for video, ``[B, 6, W]`` for action.
3. **Layer capture is declarative**: ``capture_layers={"video": {...}}`` turns
   the return into a 2-tuple (``mixture_of_transformers.py:730-739``), where
   VPP2 takes a callback. Both streams can be captured here, so both are
   walked layer by layer.
4. **No CLIP branch**: one cross-attention slot per stream
   (``wan_video_dit.py:473-475`` asserts ``has_image_input`` and
   ``require_clip_embedding`` off), so there is no 257-token split point.

The comparison is exact equality, as for VPP2. Both sides build their RoPE
tables in float64 and rotate through a complex multiply, and the attention is
a plain SDPA wrapper (``wan_video_dit.py:14-23``), so nothing in the path
reassociates.

Everything runs on CPU in float32. The source is not a dependency of this
repository: its location comes from ``$INTERNW0_DELTA_SOURCE_ROOT``, falling
back to the sibling checkout, and the whole module skips when neither is
importable.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from loongforge.embodied.model.arch.builder import (WorldModel,
                                                    materialize_conditions)
from loongforge.embodied.model.arch.spec.world import WorldModelSpec
from loongforge.embodied.model.arch.streams.engine import EngineOutput
from loongforge.embodied.model.arch.factory import (PROPRIO_PAYLOAD_KEY,
                                                    internw0_delta_to_arch)

from arch_parity import (assert_close, assert_identical, assert_same_mask,
                         condition_frames, deviation, drop_heads,
                         structural_mask)
from test_arch_blocks import BATCH, tiny_spec
from test_arch_world_model import world


def _source_root() -> Path:
    """Return the directory that should hold the ``wam`` package.

    ``$INTERNW0_DELTA_SOURCE_ROOT`` wins outright when set, so a checkout
    elsewhere can be pointed at without editing this file and a typo in it
    surfaces as a skip naming the path rather than as a silent fall-through to
    some other tree.
    """
    env = os.environ.get("INTERNW0_DELTA_SOURCE_ROOT")
    if env:
        return Path(env)
    return (Path(__file__).resolve().parents[3].parent / "projects"
            / "InternW0-Delta" / "src")


_ROOT = _source_root()
if not (_ROOT / "wam").is_dir():
    pytest.skip(f"no wam package under {_ROOT}; point "
                "$INTERNW0_DELTA_SOURCE_ROOT at the directory holding it to "
                "enable the parity comparison", allow_module_level=True)
sys.path.insert(0, str(_ROOT))

# ``wam.model.__init__`` imports the model factory, which reaches a tokenizer,
# so transformers and omegaconf are paid for by any ``wam.model.*`` import
# even though nothing here builds a text encoder. flash_attn is **not** in the
# chain: the ``flash_attention`` at ``wan_video_dit.py:14`` is an SDPA wrapper.
pytest.importorskip("einops", reason="the DiT modules import einops")
pytest.importorskip("transformers",
                    reason="wam.model.__init__ reaches AutoTokenizer")
_video_dit = pytest.importorskip("wam.model.backbones.wan22.wan_video_dit")
_action_dit = pytest.importorskip("wam.model.modules.experts.action_dit")
_mot = pytest.importorskip("wam.model.modules.mot.mixture_of_transformers")
_proprio = pytest.importorskip("wam.model.modules.memory.proprio_encoder")
# The mask builder is a free function in its own module and imports nothing
# but torch, so it is cheap; it is listed separately because it is the only
# import here that the numerical comparison does not use.
_frame_attention = pytest.importorskip(
    "wam.model.modules.conditioning.frame_attention")

# Length of the fake text context. Only "more than one token, and not equal to
# any other length in the fixture" matters.
TEXT_LEN = 5


@dataclass(frozen=True)
class Legacy:
    """The source-side modules, built and in eval mode.

    Attributes:
        mot: The joint driver holding both experts.
        video: The video expert, kept for ``pre_dit`` and its head.
        action: The action expert, kept for ``pre_dit`` and its head.
        proprio: The proprioception encoder, a sibling of ``mot`` on the
            source model rather than a part of it
            (``world_action_model.py:95-98``).
    """

    mot: Any
    video: Any
    action: Any
    proprio: Any

    def payload(self) -> dict[str, Any]:
        """Return the payload ``internw0_delta_to_arch`` consumes."""
        return {"mot": self.mot.state_dict(),
                PROPRIO_PAYLOAD_KEY: self.proprio.state_dict()}


def build_legacy(spec: WorldModelSpec) -> Legacy:
    """Build the source model to the shape ``spec`` describes.

    Every number is read off the spec, so the two sides cannot drift apart
    silently: one edit to ``tiny_spec`` moves both.

    Two flags carry the load. ``seperated_timestep=True`` is what makes the
    video timestep per token rather than per sample, and
    ``fuse_vae_embedding_in_latents`` has to be set both here and at the
    ``pre_dit`` call -- the constructor flag sizes the patch embedding, the
    call-time one is asserted (``wan_video_dit.py:826``).

    ``attention_backend="sdpa"`` with ``mot_checkpoint_mixed_attn=False`` and
    ``compile_mode="off"`` is what keeps the joint driver runnable on CPU: the
    flex backend compiles a block mask, and checkpointing needs grad.

    Args:
        spec: The provider spec both sides are built from.

    Returns:
        The source modules, in eval mode.
    """
    geo = spec.geometry
    video_stream, action_stream = spec.streams
    text_width = spec.condition("text").width
    video = _video_dit.WanVideoDiT(
        hidden_dim=video_stream.residual_width,
        in_dim=geo.latent_channels,
        ffn_dim=video_stream.ffn_width,
        out_dim=geo.latent_channels,
        text_dim=text_width,
        freq_dim=video_stream.timestep_embed_width,
        eps=1e-6,
        patch_size=tuple(geo.patch_size),
        num_heads=video_stream.num_heads,
        attn_head_dim=video_stream.head_dim,
        num_layers=video_stream.num_layers,
        has_image_input=False,
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True).eval()
    action = _action_dit.ActionDiT(
        hidden_dim=action_stream.residual_width,
        action_dim=geo.action_dim,
        ffn_dim=action_stream.ffn_width,
        text_dim=text_width,
        freq_dim=action_stream.timestep_embed_width,
        eps=1e-6,
        num_heads=action_stream.num_heads,
        attn_head_dim=action_stream.head_dim,
        num_layers=action_stream.num_layers).eval()
    mot = _mot.MoT({"video": video, "action": action},
                   mot_checkpoint_mixed_attn=False,
                   attention_backend="sdpa",
                   compile_mode="off").eval()
    proprio_spec = spec.condition("proprio")
    proprio = _proprio.ProprioContextEncoder(proprio_spec.source_width,
                                             proprio_spec.width).eval()
    return Legacy(mot=mot, video=video, action=action, proprio=proprio)


def parity_spec() -> WorldModelSpec:
    """Return the provider spec with the three distillation heads removed.

    ``test_the_distillation_heads_are_out_of_scope`` records why each one has
    no numerical counterpart. ``semantic_future`` stays, because it is a plain
    projection on this side and the layer-by-layer comparison already covers
    the hidden state it reads.
    """
    return drop_heads(tiny_spec("internw0_delta"),
                      ("track_align", "future_delta"))


@dataclass(frozen=True)
class Sides:
    """Both implementations, holding one set of weights, and their input.

    Attributes:
        spec: The spec both sides were built from.
        model: This architecture's world model, in eval mode.
        legacy: The source modules, loaded from the same weights.
        load_report: What ``load_state_dict`` said, so the first case can
            assert on it instead of the load raising inside a fixture.
        batch: Stream latents and raw condition sources, keyed as the spec
            declares.
        timesteps: One scalar draw per stream, ``[B]``. Both sides expand it,
            each in its own way.
    """

    spec: WorldModelSpec
    model: WorldModel
    legacy: Legacy
    load_report: Any
    batch: dict[str, torch.Tensor]
    timesteps: dict[str, torch.Tensor]


def build_sides() -> Sides:
    """Build both sides, move the weights across, and draw one batch.

    The direction is legacy to new: the source modules are the ones randomly
    initialized, and ``internw0_delta_to_arch`` renames their state dict onto
    this architecture. A ``strict=True`` load is therefore also the first
    assertion of the harness -- a rename rule that misses shows up here as a
    missing or unexpected key rather than as a numerical difference later.

    Returns:
        Both sides, ready to run.
    """
    torch.manual_seed(0)
    spec = parity_spec()
    geo = spec.geometry
    video_stream, action_stream = spec.streams
    model = world(spec).eval()
    legacy = build_legacy(spec)
    report = model.load_state_dict(internw0_delta_to_arch(legacy.payload()),
                                   strict=True)

    frames, rows, cols = geo.latent_grid
    text = spec.condition("text")
    proprio = spec.condition("proprio")
    batch = {
        video_stream.batch_key: torch.randn(BATCH, geo.latent_channels,
                                            frames, rows, cols),
        action_stream.batch_key: torch.randn(BATCH, geo.action_horizon,
                                             geo.action_dim),
        text.source_key: torch.randn(BATCH, TEXT_LEN, text.width),
        proprio.source_key: torch.randn(BATCH, proprio.source_width),
    }
    timesteps = {"video": torch.rand(BATCH), "action": torch.rand(BATCH)}
    return Sides(spec=spec, model=model, legacy=legacy, load_report=report,
                 batch=batch, timesteps=timesteps)


def padded_text_mask(valid: int) -> torch.Tensor:
    """Return a ``[B, TEXT_LEN]`` key mask whose tail positions are padding.

    Args:
        valid: Number of leading positions that hold real tokens.

    Returns:
        A bool mask, True where the key is real.
    """
    mask = torch.zeros(BATCH, TEXT_LEN, dtype=torch.bool)
    mask[:, :valid] = True
    return mask


@dataclass(frozen=True)
class Run:
    """One forward pass of each side, with every intermediate kept.

    Attributes:
        engine: This architecture's joint-attention output, with every layer
            of both streams captured.
        states: This architecture's per-stream entry state, from
            ``prepare``, so the entry comparison does not have to re-derive it.
        heads: This architecture's head outputs, by head name.
        tokens: The source hidden states at the joint stack's exit, by stream.
        captured: The source per-layer outputs, ``captured[stream][layer]``.
        legacy_heads: The source head outputs, under this architecture's head
            names so the two dicts can be walked together.
        pre: The source ``pre_dit`` return, by stream. Holds the entry tokens,
            the RoPE table, both timestep embeddings, and the projected
            context.
    """

    engine: EngineOutput
    states: dict[str, Any]
    heads: dict[str, torch.Tensor]
    tokens: dict[str, torch.Tensor]
    captured: dict[str, dict[int, torch.Tensor]]
    legacy_heads: dict[str, torch.Tensor]
    pre: dict[str, dict[str, Any]]


def _legacy_context(sides: Sides,
                    text_valid: torch.Tensor) -> tuple[torch.Tensor,
                                                       torch.Tensor]:
    """Build the source-side cross-attention context and its key mask.

    The source appends the proprioception token to the **raw** text context,
    before either expert's text projection
    (``append_proprio_to_context``, ``proprio_encoder.py:20-48``). That is
    what ``InjectionKind.APPEND_TO_SLOT`` declares on this side, and it is why
    the appended token is projected by the same per-layer weights as the text
    it rides with.

    Args:
        sides: Both implementations and the batch.
        text_valid: ``[B, TEXT_LEN]`` key mask for the text positions.

    Returns:
        The context, ``[B, TEXT_LEN + 1, W]``, and its mask, ``[B,
        TEXT_LEN + 1]``. The appended token is always valid.
    """
    spec = sides.spec
    raw = sides.batch[spec.condition("text").source_key]
    source = sides.batch[spec.condition("proprio").source_key]
    token = sides.legacy.proprio(source.unsqueeze(1))
    context = torch.cat([raw, token], dim=1)
    mask = torch.cat([text_valid,
                      torch.ones(BATCH, 1, dtype=torch.bool)], dim=1)
    return context, mask


def _run_new(sides: Sides, text_valid: torch.Tensor) -> tuple[Any, ...]:
    """Run this architecture: the bus, both ``prepare``s, engine, heads.

    ``engine.run`` is called directly rather than through
    ``WorldModel.forward`` for the same two reasons as in the VPP2 harness:
    the timesteps have to be the ``[B]`` draws this fixture made, and every
    joint layer has to be captured, while ``spec.captures()`` asks only for
    the layers some head declared.
    """
    spec, model = sides.spec, sides.model
    group = spec.attention_groups[0]
    inputs = {stream.name: sides.batch[stream.batch_key]
              for stream in spec.streams}
    bus = materialize_conditions(spec.conditions, sides.batch, training=False,
                                 encoders=model.condition_encoders)
    bus.conditions["text"] = replace(bus.conditions["text"],
                                     key_valid=text_valid)
    layers = tuple(range(group.num_layers))
    engine = model.engine.run(
        group, inputs,
        {name: model.blocks[name] for name in group.stream_names},
        model.plans[0], model.layout, conditions=bus,
        timesteps=sides.timesteps,
        captures={name: layers for name in group.stream_names})
    states = {name: model.blocks[name].prepare(
        inputs[name], timestep=sides.timesteps[name], conditions=bus)
        for name in group.stream_names}
    return engine, model._run_heads(engine), states


def _frame_timestep(sides: Sides) -> torch.Tensor:
    """Expand the ``[B]`` video draw the way the source's caller does.

    ``losses.py:624-634`` builds ``condition_frames - 1`` leading zero columns
    and ``pre_dit`` prepends one more (``wan_video_dit.py:826-836``), so the
    two zero paddings compose to ``condition_frames`` clean frames. The ``- 1``
    here is the half of that composition the caller owns.
    """
    frames = sides.spec.geometry.latent_grid[0]
    clean = sides.spec.streams[0].token_timestep.clean_prefix
    draw = sides.timesteps["video"]
    return torch.cat([torch.zeros(BATCH, clean - 1),
                      draw[:, None].expand(BATCH, frames - clean)], dim=1)


def _run_legacy(sides: Sides, text_valid: torch.Tensor) -> tuple[Any, ...]:
    """Run the source: both ``pre_dit``s, the joint driver, both heads.

    The joint driver takes the visibility matrix this architecture planned
    (``model.plans[0].structural``) rather than building its own, so a
    difference in the comparison is a difference in the layers and not in
    which tokens may see which.

    ``capture_layers`` turns the return into a 2-tuple
    (``mixture_of_transformers.py:730-739``); both streams are asked for, so
    both can be walked layer by layer.
    """
    spec, legacy = sides.spec, sides.legacy
    video_stream, action_stream = spec.streams
    group = spec.attention_groups[0]
    context, mask = _legacy_context(sides, text_valid)
    video_pre = legacy.video.pre_dit(
        x=sides.batch[video_stream.batch_key],
        timestep=_frame_timestep(sides), context=context, context_mask=mask,
        fuse_vae_embedding_in_latents=True)
    action_pre = legacy.action.pre_dit(
        action_tokens=sides.batch[action_stream.batch_key],
        timestep=sides.timesteps["action"], context=context,
        context_mask=mask)
    pre = {"video": video_pre, "action": action_pre}
    layers = set(range(group.num_layers))
    tokens, captured = legacy.mot(
        {name: pre[name]["tokens"] for name in pre},
        sides.model.plans[0].structural,
        {name: pre[name]["freqs"] for name in pre},
        {name: {"context": pre[name]["context"],
                "mask": pre[name]["context_mask"]} for name in pre},
        {name: pre[name]["t_mod"] for name in pre},
        capture_layers={name: layers for name in pre})
    heads = {
        "video_flow": legacy.video.unpatchify(
            legacy.video.head(tokens["video"], video_pre["t"]),
            spec.geometry.patch_grid),
        "action_flow": legacy.action.head(tokens["action"]),
    }
    return tokens, captured, heads, pre


def run_both(sides: Sides, text_valid: torch.Tensor) -> Run:
    """Run both sides on one batch under one text key mask.

    Args:
        sides: Both implementations and the batch.
        text_valid: ``[B, TEXT_LEN]`` key mask for the text positions. The
            appended proprioception token is valid on both sides regardless.

    Returns:
        Every intermediate of both passes.
    """
    with torch.no_grad():
        engine, heads, states = _run_new(sides, text_valid)
        tokens, captured, legacy_heads, pre = _run_legacy(sides, text_valid)
    return Run(engine=engine, states=states, heads=heads, tokens=tokens,
               captured=captured, legacy_heads=legacy_heads, pre=pre)


@pytest.fixture(scope="module")
def sides() -> Sides:
    """Both sides, built once: 16 layers twice is the expensive part here."""
    return build_sides()


@pytest.fixture(scope="module")
def run(sides: Sides) -> Run:
    """The padded-text run that most cases compare."""
    return run_both(sides, padded_text_mask(TEXT_LEN - 2))


# ---- the weights have to arrive before anything can be compared ----


def test_the_legacy_weights_load_with_nothing_missing_or_unexpected(sides):
    """The premise of every comparison below: one set of numbers, two models.

    ``test_arch_transfer.py`` pins the rule table against generated key names;
    this pins it against the key names the real ``MoT`` and
    ``ProprioContextEncoder`` produce. The two payload keys are the source's
    own top-level ones -- the proprioception encoder is a sibling of ``mot``,
    not a part of it, which is why it travels separately.
    """
    assert not sides.load_report.missing_keys
    assert not sides.load_report.unexpected_keys


# ---- the entry ----


def test_the_entry_produces_the_same_tokens_and_modulation(run, sides):
    """``prepare`` against ``pre_dit``, for both streams.

    The two streams differ in the rank of what they hand the joint driver:
    ``[B, S, 6, W]`` for video, whose timestep is per token, and ``[B, 6, W]``
    for action, whose timestep is per sample. One loop covers both because
    ``assert_identical`` compares shapes first, so a stream that quietly
    produced the other rank fails by shape rather than by broadcast.
    """
    for stream in sides.spec.streams:
        state = run.states[stream.name]
        entry = run.pre[stream.name]
        assert_identical(state.hidden, entry["tokens"],
                         f"{stream.name} entry tokens")
        assert_identical(state.modulation, entry["t_mod"],
                         f"{stream.name} modulation")
        assert_identical(state.extras["t_embed"], entry["t"],
                         f"{stream.name} timestep embedding")
    assert run.states["video"].modulation.dim() == 4
    assert run.states["action"].modulation.dim() == 3


def test_the_clean_prefix_covers_all_three_condition_frames(run, sides):
    """One declaration here against two zero paddings there.

    The source splits the clean prefix between its caller, which supplies
    ``condition_frames - 1`` zero columns, and ``pre_dit``, which prepends one
    more. This side declares the total once as
    ``TokenTimestep(clean_prefix=condition_frames)``. Both halves are pinned:
    the column count the caller hands over, and the resulting token-wise
    embedding, which must be constant across the clean prefix, constant across
    the noisy tail, and different between the two. Without the last clause an
    off-by-one in either place would still pass the first.
    """
    stream = sides.spec.streams[0]
    clean = stream.token_timestep.clean_prefix
    frames, rows, cols = sides.spec.geometry.patch_grid
    per_frame = rows * cols
    caller = _frame_timestep(sides)
    assert tuple(caller.shape) == (BATCH, frames - 1)
    assert int((caller[0] == 0).sum()) == clean - 1

    prefix = clean * per_frame
    embed = run.states[stream.name].extras["t_embed"]
    assert tuple(embed.shape[:2]) == (BATCH, frames * per_frame)
    assert torch.equal(embed[:, :prefix],
                       embed[:, :1].expand(BATCH, prefix, -1))
    tail = embed.shape[1] - prefix
    assert torch.equal(embed[:, prefix:],
                       embed[:, prefix:prefix + 1].expand(BATCH, tail, -1))
    assert not torch.equal(embed[:, 0], embed[:, prefix])


def test_the_rope_tables_are_identical(run, sides):
    """The 3-D video table and the 1-D action table, sliced to the token count.

    The new side tabulates once in ``__init__`` and keeps the whole table, so
    the legacy per-call slice is what the comparison has to match. Both sides
    build the bases in float64 and hold ``freqs`` as complex128, so this is an
    equality and not a tolerance: a wrong axis split is otherwise silent,
    since the shape is right either way and only the numbers move.
    """
    for stream in sides.spec.streams:
        length = run.states[stream.name].hidden.shape[1]
        table = sides.model.blocks[stream.name]._freqs
        assert table.dtype is torch.complex128
        assert_identical(table[:length], run.pre[stream.name]["freqs"],
                         f"{stream.name} rope table")


def test_the_projected_context_matches_in_the_single_slot(run, sides):
    """One cross-attention slot per stream, and the proprioception token in it.

    There is no CLIP branch in this model (``wan_video_dit.py:473-475``), so
    unlike VPP2 there is no 257-token split point and the comparison is one
    tensor against one tensor. What is being checked is the appended
    proprioception token: it rides inside the text slot, so it is projected by
    the text slot's per-layer weights. Had it been given a slot of its own,
    the projection would be a different set of numbers and this would fail
    while the shapes still agreed.
    """
    for stream in sides.spec.streams:
        assert stream.cross_kv_slots == ("text",)
        context = run.states[stream.name].extras["cross_kv"]["text"]
        assert context.shape[1] == TEXT_LEN + 1
        assert_identical(context, run.pre[stream.name]["context"],
                         f"{stream.name} projected context")


def test_the_text_slot_mask_carries_the_padded_tail_and_the_proprio_token(
        run, sides):
    """The new ``[B, L]`` slot mask against the legacy ``[B, S, L]`` one.

    The source broadcasts its key mask over the query axis before handing it
    to attention; this side keeps it per key and lets the attention call
    broadcast. Taking row 0 is therefore the whole conversion, and the rows
    being identical is itself part of the claim -- a per-query mask would mean
    the two sides are masking different things.

    The last column is the appended proprioception token, which declares no
    mask and so must be valid on both sides. The two before it are this
    fixture's padding.
    """
    for stream in sides.spec.streams:
        legacy_mask = run.pre[stream.name]["context_mask"]
        assert legacy_mask.dim() == 3
        assert torch.equal(legacy_mask,
                           legacy_mask[:, :1].expand_as(legacy_mask))
        mask = run.states[stream.name].extras["cross_kv_mask"]["text"]
        assert set(run.states[stream.name].extras["cross_kv_mask"]) == {"text"}
        assert_identical(mask.int(), legacy_mask[:, 0, :].int(),
                         f"{stream.name} text slot mask")
        assert not mask[:, TEXT_LEN - 2:TEXT_LEN].any()
        assert mask[:, -1].all()

# ---- every joint layer, then the exit ----


def test_every_joint_layer_matches_in_both_streams(run, sides):
    """Layer by layer, because only the exit would say "somewhere upstream".

    Both capture points are the post-block output, on both sides
    (``mixture_of_transformers.py:730-739`` and ``engine.py:587-589``), so
    layer *i* means the same thing and no offset correction is needed.

    Both streams are walked, unlike the VPP2 harness: ``capture_layers`` takes
    a name per stream, so there is no reason to settle for one.
    """
    layers = range(sides.spec.attention_groups[0].num_layers)
    for stream in sides.spec.streams:
        captured = run.captured[stream.name]
        assert set(captured) == set(layers)
        bad = [index for index in layers
               if not torch.equal(
                   captured[index],
                   run.engine.captured[(stream.name, index)])]
        if bad:
            first = bad[0]
            worst = deviation(run.engine.captured[(stream.name, first)],
                              captured[first])
            raise AssertionError(
                f"{stream.name}: first mismatching joint layer {first} of "
                f"{len(list(layers))}, max abs {worst:.3e}; "
                f"{len(bad)} layers differ")


def test_the_engine_exit_hidden_states_match(run, sides):
    """Both streams at the exit, after ``finalize``.

    Worth asserting even with every layer already compared: ``finalize`` runs
    after the last layer, and for this family it is deliberately a no-op (the
    final norm sits in the head). A stray operation there would show up only
    here.
    """
    for stream in sides.spec.streams:
        assert_identical(run.engine.states[stream.name].hidden,
                         run.tokens[stream.name],
                         f"{stream.name} exit hidden state")


def test_the_two_flow_heads_match(run, sides):
    """``patch_flow`` against ``Head`` + ``unpatchify``, ``linear`` against it.

    The video head is the interesting half. Its modulation is per token here,
    which takes the source's 3-D ``t_mod`` branch (``:400-416``) rather than
    the 2-D one VPP2 uses, and the unflatten is the one step whose failure
    keeps the shape right while stirring the time axis into the spatial axes.
    """
    for name in ("video_flow", "action_flow"):
        assert_identical(run.heads[name], run.legacy_heads[name],
                         f"head {name}")
    geometry = sides.spec.geometry
    assert tuple(run.heads["video_flow"].shape) == (
        BATCH, geometry.latent_channels, *geometry.latent_grid)
    assert tuple(run.heads["action_flow"].shape) == (
        BATCH, geometry.action_horizon, geometry.action_dim)

# ---- the inputs that would make all of the above vacuous ----


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
        assert_identical(unmasked.engine.states[stream.name].hidden,
                         unmasked.tokens[stream.name],
                         f"{stream.name} exit hidden state, no padding")
        assert not torch.equal(unmasked.engine.states[stream.name].hidden,
                               run.engine.states[stream.name].hidden)


# ---- the visibility matrix, against the builder the source uses ----


def legacy_structural(spec: WorldModelSpec) -> torch.Tensor:
    """Build the ``[S, S]`` matrix the source itself would attend under.

    ``build_frame_memory_attention_mask`` is a free function, so unlike the
    VPP2 side there is no stand-in to assemble. Two of its arguments are left
    at their defaults and both are the same open item: with
    ``num_future_delta_tokens=0`` the delta rectangles are not written at all,
    which also makes ``num_anchor_condition_frames`` dead. Delta tokens have
    no declared position in ``GeometrySpec`` yet (ruling 2 of the design
    doc's §2.2.2), and ``Phase.DELTA`` is reserved against the day they do.

    Args:
        spec: The spec under comparison. It supplies both sequence lengths,
            the tokens per frame and the condition frame count, so the two
            sides cannot be measured at different sizes.

    Returns:
        The legacy mask, ``[total, total]`` bool on CPU.
    """
    layout = spec.build_layout()
    return _frame_attention.build_frame_memory_attention_mask(
        video_seq_len=layout.range_of("video").length,
        action_seq_len=layout.range_of("action").length,
        video_tokens_per_frame=spec.geometry.tokens_per_frame,
        num_condition_frames=condition_frames(spec),
        device=torch.device("cpu"))


@pytest.mark.parametrize("frames", (1, 2, 3))
def test_the_visibility_matrix_matches_the_legacy_builder(frames):
    """Cell for cell against ``build_frame_memory_attention_mask``.

    Nothing above this line implies it. ``_run_legacy`` hands the source's
    ``MoT`` **this** side's ``plan.structural`` (:417), so the layer-by-layer
    comparison holds just as well under a mask both sides got equally wrong.
    The first version of ``_internw0_delta_visibility`` was 48 of 784 cells
    too strict in the video rows and no numerical case moved.

    This model is the reason the query side needed a selector at all: the
    condition rows are causal among themselves while the noisy-future rows are
    bidirectional over all real video, which is two intra-stream rules inside
    one stream. Three configurations, because the declaration contains no
    token count -- the frame count enters only through the phase label.
    """
    spec = tiny_spec("internw0_delta", condition_frames=frames)
    assert condition_frames(spec) == frames
    assert_same_mask(structural_mask(spec), legacy_structural(spec),
                     f"visibility at condition_frames={frames}")


def test_the_matrix_fed_to_mot_is_the_one_compared_above(sides):
    """``model.plans[0]`` is what the case above speaks for.

    That case compiles the declaration directly, which is what makes one spec
    per configuration affordable; this one closes the gap to the object the
    harness really hands ``MoT.forward``. It also records the one shape
    conversion this harness has to get right: ``MoT.forward`` validates a 2-D
    square mask, while ``plan.materialize("dense")`` returns ``[1, 1, S, S]``.

    The two all/any assertions stay because they fail differently: a matrix of
    all True would leave every comparison above saying nothing about
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

# ---- the two things the transfer does that are not a rename ----


def test_the_proprio_type_embedding_is_folded_into_the_bias():
    """The source's third parameter has no counterpart, and need not have one.

    ``ProprioContextEncoder.forward`` is ``super().forward(input) +
    type_embedding`` with nothing non-linear in between
    (``proprio_encoder.py:16-17``), so the extra vector is a second bias and
    ``_attach_proprio`` adds it into the first. The shipped initializer is
    ``zeros``, which is why every comparison above is exact; this case is the
    only one that gives it a value, and it is also the only one that cannot be
    exact -- ``W x + (b + e)`` and ``(W x + b) + e`` reassociate one addition.

    Both halves matter. Agreement to float32 rounding says the fold is the
    right arithmetic; the inequality against the unfolded run says the vector
    reaches the output at all, without which the first half would hold for a
    parameter that was being dropped.
    """
    sides = build_sides()
    mask = padded_text_mask(TEXT_LEN - 2)
    before = run_both(sides, mask)
    with torch.no_grad():
        embedding = sides.legacy.proprio.type_embedding
        assert not embedding.any(), "the shipped initializer is zeros"
        embedding.copy_(torch.randn_like(embedding))
    report = sides.model.load_state_dict(
        internw0_delta_to_arch(sides.legacy.payload()), strict=True)
    assert not report.missing_keys and not report.unexpected_keys
    after = run_both(sides, mask)
    for stream in sides.spec.streams:
        assert_close(after.engine.states[stream.name].hidden,
                     after.tokens[stream.name],
                     f"{stream.name} exit, type embedding folded",
                     atol=1e-5)
        assert not torch.equal(after.tokens[stream.name],
                               before.tokens[stream.name])


def test_the_distillation_heads_have_no_numerical_counterpart():
    """Why the other three heads are out of scope, with the numbers measured.

    ``future_delta`` is the one with a real disagreement: this side declares
    the **per-frame** width, while the source flattens the frame axis into the
    projection, so the two differ by the frame count and a strict load would
    fail on shape rather than on a name. ``HeadSpec.output_width`` has no
    frame axis to hold the difference, and the delta tokens themselves have no
    declared position in ``GeometrySpec``, so the head cannot be compared
    until Lane A adds both.

    ``semantic_future`` has no source module at all -- the term is a cosine
    computed directly on two captured hidden states
    (``losses.py:58-71``), and both of those states are already compared layer
    by layer above, so there is nothing left for a head comparison to add.

    ``track_align`` cannot be built on this side: its ``track_query_head``
    raises ``NotImplementedError``, which is also why ``drop_heads`` has to
    remove it before the model is constructed.
    """
    spec = tiny_spec("internw0_delta")
    geometry = spec.geometry
    patch_dim = geometry.latent_channels * math.prod(geometry.patch_size)
    assert spec.head("future_delta").output_width == patch_dim
    assert spec.head("semantic_future").output_width \
        == spec.stream("video").residual_width

    frames = 2
    video = build_legacy(spec).video
    video.configure_future_delta_queries(
        num_tokens=geometry.patch_grid[1] * geometry.patch_grid[2],
        patch_dim=patch_dim, num_future_delta_frames=frames)
    legacy_width = video.future_delta_head[1].out_features
    assert legacy_width == patch_dim * frames

    pytest.skip(
        f"future_delta exits at {patch_dim} per frame here against "
        f"{legacy_width} flattened there and HeadSpec has no frame axis; "
        "semantic_future has no source module, only a cosine on two captured "
        "hidden states; track_align cannot be built")
