# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Numerical parity between this architecture and the fastwam source.

Same shape of check as ``test_arch_vpp2_parity.py`` -- two independent
implementations, one set of weights, moved across by
``transfer.fastwam_to_arch`` -- and the same four comparison layers (entry,
every joint layer, the two flow heads, and the visibility matrix against this
source's own builder). Read that module first; this one records only what
differs.

Four things differ:

1. **This one runs on the GPU and compares to a tolerance.** fastwam's RoPE is
   the FlashAttention Triton interleaved rotary
   (``wan/dit.py:27``, ``:90-115``, ``:132-148``) with **no eager fallback**,
   and the MoT path modulates through ``triton_norm_modulate``. Both compute
   the same expression as this architecture in a different order, so the
   agreement is to float32 rounding and not bit exact. ``_ATOL`` is the budget;
   every case that uses it is comparing across one of those two kernels.
   TF32 is turned off for the duration, or the tolerance would have to be
   about a thousand times wider and would stop meaning anything.
2. **Layer capture has to be monkeypatched.** There is no callback and no
   capture argument, and ``nn.Module`` hooks do not fire because ``MoT``
   calls the blocks' pieces directly rather than calling the blocks. The one
   place every expert passes through once per layer is
   ``_apply_post_with_optional_checkpoint`` (``mot/model.py:264``), which
   receives ``block=`` as a keyword, so ``id(block)`` is what identifies the
   stream and the layer.
3. **The source is in this repository**, under
   ``loongforge/embodied/model/fastwam/``, so there is no environment variable
   and no checkout to find. The gates that remain are the GPU and
   ``flash_attn``.
4. **The visibility comparison is a bound, not an equality.** The other two
   providers match their source's mask cell for cell; fastwam's declaration is
   deliberately stricter than its source's, so the comparison asserts "never
   more permissive" and then pins the remaining gap as an exact set. See
   ``test_the_visibility_matrix_is_the_legacy_one_minus_future_frames``.

The video timestep is per token here as well, but with a one-frame clean
prefix rather than three (``wan/dit.py:766-775``: ``[:, 0, :] = 0``), which is
what ``TokenTimestep(clean_prefix=1)`` declares.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from loongforge.embodied.model.arch.builder import (WorldModel,
                                                    materialize_conditions)
from loongforge.embodied.model.arch.spec.world import WorldModelSpec
from loongforge.embodied.model.arch.streams.engine import EngineOutput
from loongforge.embodied.model.arch.factory import (PROPRIO_PAYLOAD_KEY,
                                                    fastwam_to_arch)

from arch_parity import (assert_close, assert_identical, assert_same_mask,
                         deviation, structural_mask)
from test_arch_blocks import BATCH, tiny_spec
from test_arch_world_model import world

if not torch.cuda.is_available():
    pytest.skip("the fastwam rotary is a Triton kernel with no eager "
                "fallback, so this comparison needs a GPU",
                allow_module_level=True)
pytest.importorskip("flash_attn",
                    reason="wan/dit.py:27 imports the Triton rotary from it")

_video_dit = pytest.importorskip(
    "loongforge.embodied.model.fastwam.wan.dit")
_action_dit = pytest.importorskip(
    "loongforge.embodied.model.fastwam.action.dit")
_mot = pytest.importorskip("loongforge.embodied.model.fastwam.mot.model")
# Only for ``_build_mot_attention_mask``, called unbound below. Listed apart
# from the three above because nothing numerical here needs it, and it is
# gated like them rather than placed in an ungated module because the two
# builders it reaches live behind the same ``flash_attn`` import.
_fastwam = pytest.importorskip("loongforge.embodied.model.fastwam.mot.fastwam")

DEVICE = torch.device("cuda")

# Length of the fake text context. Only "more than one token, and not equal to
# any other length in the fixture" matters.
TEXT_LEN = 5

# Agreement budget. Measured, not guessed: the drift comes from the Triton
# rotary and from ``triton_norm_modulate``, both of which reassociate a
# float32 expression this architecture evaluates eagerly. Raise it only
# together with a note saying what moved.
_ATOL = 1e-5


@dataclass(frozen=True)
class Legacy:
    """The source-side modules, built, moved to the GPU, and in eval mode.

    Attributes:
        mot: The joint driver holding both experts.
        video: The video expert, kept for ``pre_dit`` and its head.
        action: The action expert, kept for ``pre_dit`` and its head.
        proprio: The proprioception encoder. A bare ``nn.Linear`` here
            (``mot/fastwam.py:78``), with no extra parameter of its own, so
            the transfer is a plain rename and not a fold.
    """

    mot: Any
    video: Any
    action: Any
    proprio: torch.nn.Linear

    def payload(self) -> dict[str, Any]:
        """Return the checkpoint payload ``fastwam_to_arch`` consumes."""
        return {"mot": self.mot.state_dict(),
                PROPRIO_PAYLOAD_KEY: self.proprio.state_dict()}


def build_legacy(spec: WorldModelSpec) -> Legacy:
    """Build the source model to the shape ``spec`` describes.

    Every number is read off the spec, so the two sides cannot drift apart
    silently: one edit to ``tiny_spec`` moves both.

    ``seperated_timestep=True`` plus the call-time
    ``fuse_vae_embedding_in_latents=True`` is not a choice here: every other
    combination raises (``wan/dit.py:569-571`` and ``:777``). Together they
    select the per-token timestep branch.

    ``mot_checkpoint_mixed_attn=False`` would matter only under ``training``,
    but it is passed anyway so the harness does not depend on the eval flag
    for correctness.

    Args:
        spec: The provider spec both sides are built from.

    Returns:
        The source modules, on the GPU and in eval mode.
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
        fuse_vae_embedding_in_latents=True)
    action = _action_dit.ActionDiT(
        hidden_dim=action_stream.residual_width,
        action_dim=geo.action_dim,
        ffn_dim=action_stream.ffn_width,
        text_dim=text_width,
        freq_dim=action_stream.timestep_embed_width,
        eps=1e-6,
        num_heads=action_stream.num_heads,
        attn_head_dim=action_stream.head_dim,
        num_layers=action_stream.num_layers)
    mot = _mot.MoT({"video": video, "action": action},
                   mot_checkpoint_mixed_attn=False).to(DEVICE).eval()
    proprio_spec = spec.condition("proprio")
    proprio = torch.nn.Linear(proprio_spec.source_width,
                              proprio_spec.width).to(DEVICE).eval()
    return Legacy(mot=mot, video=video, action=action, proprio=proprio)


@contextmanager
def _capturing(legacy: Legacy) -> Iterator[dict[str, dict[int,
                                                          torch.Tensor]]]:
    """Record every expert's post-block output, keyed by stream and layer.

    ``MoT`` offers no capture argument and no callback, and module hooks do
    not fire because it calls the blocks' pieces rather than the blocks. What
    it does do is route every expert through
    ``_apply_post_with_optional_checkpoint`` exactly once per layer, with
    ``block=`` among the keywords (``mot/model.py:653-663``), so wrapping that
    one method sees every layer of both streams and ``id(block)`` says which.

    The wrapper is set on the instance and deleted afterwards, which uncovers
    the class method again; nothing is patched globally.

    Args:
        legacy: The source modules whose joint driver to instrument.

    Yields:
        ``captured[stream][layer]``, filled as the forward runs.
    """
    index = {id(expert.blocks[layer]): (name, layer)
             for name, expert in legacy.mot.mixtures.items()
             for layer in range(len(expert.blocks))}
    captured: dict[str, dict[int, torch.Tensor]] = {
        name: {} for name in legacy.mot.mixtures}
    original = legacy.mot._apply_post_with_optional_checkpoint

    def wrapper(**kwargs: Any) -> torch.Tensor:
        """Call the real method, then file its output under (stream, layer)."""
        tokens = original(**kwargs)
        name, layer = index[id(kwargs["block"])]
        captured[name][layer] = tokens
        return tokens

    legacy.mot._apply_post_with_optional_checkpoint = wrapper
    try:
        yield captured
    finally:
        del legacy.mot._apply_post_with_optional_checkpoint


@dataclass(frozen=True)
class Sides:
    """Both implementations, holding one set of weights, and their input.

    Attributes:
        spec: The spec both sides were built from.
        model: This architecture's world model, on the GPU and in eval mode.
        legacy: The source modules, loaded from the same weights.
        load_report: What ``load_state_dict`` said, so the first case can
            assert on it instead of the load raising inside a fixture.
        batch: Stream latents and raw condition sources, keyed as the spec
            declares.
        timesteps: One scalar draw per stream, ``[B]``.
    """

    spec: WorldModelSpec
    model: WorldModel
    legacy: Legacy
    load_report: Any
    batch: dict[str, torch.Tensor]
    timesteps: dict[str, torch.Tensor]


@dataclass(frozen=True)
class Run:
    """One forward pass of each side, with every intermediate kept.

    Attributes:
        engine: This architecture's joint-attention output, with every layer
            of both streams captured.
        states: This architecture's per-stream entry state, from ``prepare``.
        heads: This architecture's head outputs, by head name.
        tokens: The source hidden states at the joint stack's exit, by stream.
        captured: The source per-layer outputs, ``captured[stream][layer]``.
        legacy_heads: The source head outputs, under this architecture's head
            names so the two dicts can be walked together.
        pre: The source ``pre_dit`` return, by stream.
    """

    engine: EngineOutput
    states: dict[str, Any]
    heads: dict[str, torch.Tensor]
    tokens: dict[str, torch.Tensor]
    captured: dict[str, dict[int, torch.Tensor]]
    legacy_heads: dict[str, torch.Tensor]
    pre: dict[str, dict[str, Any]]


def build_sides() -> Sides:
    """Build both sides, move the weights across, and draw one batch.

    The direction is source to new: the fastwam modules are the ones randomly
    initialized, and ``fastwam_to_arch`` renames their state dict onto this
    architecture. A ``strict=True`` load is therefore also the first assertion
    of the harness -- a rename rule that misses shows up here as a missing or
    unexpected key rather than as a numerical difference later.

    Returns:
        Both sides, on the GPU, ready to run.
    """
    torch.manual_seed(0)
    spec = tiny_spec("fastwam")
    geo = spec.geometry
    video_stream, action_stream = spec.streams
    model = world(spec).to(DEVICE).eval()
    legacy = build_legacy(spec)
    report = model.load_state_dict(fastwam_to_arch(legacy.payload()),
                                   strict=True)

    frames, rows, cols = geo.latent_grid
    text = spec.condition("text")
    proprio = spec.condition("proprio")
    batch = {
        video_stream.batch_key: torch.randn(BATCH, geo.latent_channels,
                                            frames, rows, cols,
                                            device=DEVICE),
        action_stream.batch_key: torch.randn(BATCH, geo.action_horizon,
                                             geo.action_dim, device=DEVICE),
        text.source_key: torch.randn(BATCH, TEXT_LEN, text.width,
                                     device=DEVICE),
        proprio.source_key: torch.randn(BATCH, proprio.source_width,
                                        device=DEVICE),
    }
    timesteps = {"video": torch.rand(BATCH, device=DEVICE),
                 "action": torch.rand(BATCH, device=DEVICE)}
    return Sides(spec=spec, model=model, legacy=legacy, load_report=report,
                 batch=batch, timesteps=timesteps)


def padded_text_mask(valid: int) -> torch.Tensor:
    """Return a ``[B, TEXT_LEN]`` key mask whose tail positions are padding.

    Args:
        valid: Number of leading positions that hold real tokens.

    Returns:
        A bool mask on the GPU, True where the key is real.
    """
    mask = torch.zeros(BATCH, TEXT_LEN, dtype=torch.bool, device=DEVICE)
    mask[:, :valid] = True
    return mask


def _legacy_context(sides: Sides,
                    text_valid: torch.Tensor) -> tuple[torch.Tensor,
                                                       torch.Tensor]:
    """Build the source-side cross-attention context and its key mask.

    The source appends the proprioception token to the **raw** text context,
    before either expert's text projection
    (``_append_proprio_to_context``, ``mot/fastwam.py:248-269``). That is what
    ``InjectionKind.APPEND_TO_SLOT`` declares on this side, and it is why the
    appended token is projected by the same per-layer weights as the text it
    rides with.

    Args:
        sides: Both implementations and the batch.
        text_valid: ``[B, TEXT_LEN]`` key mask for the text positions.

    Returns:
        The context, ``[B, TEXT_LEN + 1, W]``, and its mask, ``[B,
        TEXT_LEN + 1]``. The appended token is always valid
        (``mot/fastwam.py:266``).
    """
    spec = sides.spec
    raw = sides.batch[spec.condition("text").source_key]
    source = sides.batch[spec.condition("proprio").source_key]
    token = sides.legacy.proprio(source.unsqueeze(1))
    context = torch.cat([raw, token], dim=1)
    ones = torch.ones(BATCH, 1, dtype=torch.bool, device=DEVICE)
    return context, torch.cat([text_valid, ones], dim=1)


def _run_new(sides: Sides, text_valid: torch.Tensor) -> tuple[Any, ...]:
    """Run this architecture: the bus, both ``prepare``s, engine, heads.

    ``engine.run`` is called directly rather than through
    ``WorldModel.forward`` for the same two reasons as in the VPP2 harness:
    the timesteps have to be the ``[B]`` draws this fixture made, and every
    joint layer has to be captured, while ``spec.captures()`` asks only for
    the layers some head declared.
    """
    spec, model = sides.spec, sides.model
    group = spec.groups[0]
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


def _run_legacy(sides: Sides, text_valid: torch.Tensor) -> tuple[Any, ...]:
    """Run the source: both ``pre_dit``s, the joint driver, both heads.

    The joint driver takes the visibility matrix this architecture planned
    (``model.plans[0].structural``) rather than building its own, so a
    difference in the comparison is a difference in the layers and not in
    which tokens may see which.

    The video timestep is the plain ``[B]`` draw: unlike InternW0-Delta, this
    source expands it to a per-token tensor itself and zeroes the first frame
    (``wan/dit.py:766-772``), so the caller has nothing to pad.
    """
    spec, legacy = sides.spec, sides.legacy
    video_stream, action_stream = spec.streams
    context, mask = _legacy_context(sides, text_valid)
    video_pre = legacy.video.pre_dit(
        x=sides.batch[video_stream.batch_key],
        timestep=sides.timesteps["video"], context=context, context_mask=mask,
        fuse_vae_embedding_in_latents=True)
    action_pre = legacy.action.pre_dit(
        action_tokens=sides.batch[action_stream.batch_key],
        timestep=sides.timesteps["action"], context=context,
        context_mask=mask)
    pre = {"video": video_pre, "action": action_pre}
    with _capturing(legacy) as captured:
        tokens = legacy.mot(
            {name: pre[name]["tokens"] for name in pre},
            sides.model.plans[0].structural,
            {name: pre[name]["freqs"] for name in pre},
            {name: {"context": pre[name]["context"],
                    "mask": pre[name]["context_mask"]} for name in pre},
            {name: pre[name]["t_mod"] for name in pre})
    heads = {
        "video_flow": legacy.video.post_dit(tokens["video"], video_pre),
        "action_flow": legacy.action.post_dit(tokens["action"], action_pre),
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


@pytest.fixture(scope="module", autouse=True)
def _real_float32() -> Iterator[None]:
    """Turn TF32 off for this module, and put it back afterwards.

    TF32 keeps 10 mantissa bits, so every matmul in the stack would land
    around 1e-3 of the float32 answer. The comparison would still pass with a
    wide enough tolerance, but it would no longer be able to tell a reordered
    expression from a wrong one, which is the only thing it is for.
    """
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


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
    this pins it against the key names the real fastwam ``MoT`` produces. The
    two payload keys are the source's own top-level ones -- the proprioception
    encoder is a sibling of ``mot``, not a part of it, which is why it travels
    separately.
    """
    assert not sides.load_report.missing_keys
    assert not sides.load_report.unexpected_keys


def test_the_rotary_really_is_the_triton_kernel_with_no_fallback(sides):
    """Why this whole module is a tolerance and not an equality.

    If the source ever grew an eager rotary, or this harness ever stopped
    reaching the fused one, the right fix would be to tighten every
    comparison below to equality rather than to keep the tolerance for no
    reason. This case is what would notice.
    """
    rope = _video_dit._TritonRoPE
    assert issubclass(rope, torch.autograd.Function)
    assert _video_dit.apply_rotary.__module__.startswith("flash_attn")
    assert not hasattr(_video_dit, "rope_apply_eager")
    assert sides.spec.streams[0].head_dim % 2 == 0

# ---- the entry ----


def test_the_entry_produces_the_same_tokens_and_modulation(run, sides):
    """``prepare`` against ``pre_dit``, for both streams.

    Three separate things ride on this one comparison: the patchify/flatten
    order, the sinusoidal timestep embedding, and the pre-combination
    modulation. They are asserted together because a single ``prepare`` call
    produces all three and splitting them would double the forward passes
    without isolating anything new -- each has its own name in the message.

    The two streams differ in rank, as in InternW0-Delta: ``[B, S, 6, W]`` for
    video, whose timestep is per token, and ``[B, 6, W]`` for action, whose
    timestep is per sample.
    """
    for stream in sides.spec.streams:
        state = run.states[stream.name]
        entry = run.pre[stream.name]
        assert_close(state.hidden, entry["tokens"],
                     f"{stream.name} entry tokens", atol=_ATOL)
        assert_close(state.modulation, entry["t_mod"],
                     f"{stream.name} modulation", atol=_ATOL)
        assert_close(state.extras["t_embed"], entry["t"],
                     f"{stream.name} timestep embedding", atol=_ATOL)
    assert run.states["video"].modulation.dim() == 4
    assert run.states["action"].modulation.dim() == 3


def test_the_clean_prefix_covers_the_first_latent_frame_only(run, sides):
    """One frame, not three: ``[:, 0, :] = 0`` and nothing else.

    This is the whole difference between fastwam's timestep and
    InternW0-Delta's, and it is declared in one place on this side as
    ``TokenTimestep(clean_prefix=1)``. The embedding has to be constant across
    the clean prefix, constant across the noisy tail, and different between
    the two; without the last clause an off-by-one would still pass the first.
    """
    stream = sides.spec.streams[0]
    clean = stream.token_timestep.clean_prefix
    assert clean == 1
    frames, rows, cols = sides.spec.geometry.patch_grid
    prefix = clean * rows * cols
    embed = run.states[stream.name].extras["t_embed"]
    assert tuple(embed.shape[:2]) == (BATCH, frames * rows * cols)
    assert torch.equal(embed[:, :prefix],
                       embed[:, :1].expand(BATCH, prefix, -1))
    tail = embed.shape[1] - prefix
    assert torch.equal(embed[:, prefix:],
                       embed[:, prefix:prefix + 1].expand(BATCH, tail, -1))
    assert not torch.equal(embed[:, 0], embed[:, prefix])


def test_the_rope_tables_are_identical_through_the_cos_sin_split(run, sides):
    """One complex table here, a ``(cos, sin)`` pair there, and no kernel yet.

    The source tabulates in float64 and holds complex128 exactly as this side
    does, then splits into two float32 tables for the Triton kernel
    (``prepare_rope_freqs``, ``wan/dit.py:118-129``:
    ``freqs[:, 0].real.float()`` and ``.imag.float()``). Nothing fused has run
    at this point, so this is the one comparison in the module that is an
    equality -- and it is where a wrong axis split would show, since the
    shapes agree either way and only the numbers move.

    The ``.to`` is not incidental: ``_freqs`` is deliberately **not** a
    registered buffer (``wan_streams.py:244-247`` -- complex128 cannot survive
    ``Module.to(dtype=bf16)`` and the key is absent from existing
    checkpoints), so ``model.to(DEVICE)`` leaves it on the host and the
    rotation moves it at use time.
    """
    for stream in sides.spec.streams:
        length = run.states[stream.name].hidden.shape[1]
        table = sides.model.blocks[stream.name]._freqs[:length]
        assert table.dtype is torch.complex128
        cos, sin = run.pre[stream.name]["freqs"]
        assert_identical(table[:, 0].real.float().to(cos.device), cos,
                         f"{stream.name} rope cos")
        assert_identical(table[:, 0].imag.float().to(sin.device), sin,
                         f"{stream.name} rope sin")


def test_the_projected_context_matches_in_the_single_slot(run, sides):
    """One cross-attention slot per stream, and the proprioception token in it.

    fastwam has no CLIP branch (``has_image_input`` is asserted off at
    ``wan/dit.py:567``), so unlike VPP2 there is no 257-token split point and
    the comparison is one tensor against one tensor. What is being checked is
    the appended proprioception token: it rides inside the text slot, so it is
    projected by the text slot's weights. Had it been given a slot of its own
    the projection would be a different set of numbers, and this would fail
    while the shapes still agreed.
    """
    for stream in sides.spec.streams:
        assert stream.cross_kv_slots == ("text",)
        context = run.states[stream.name].extras["cross_kv"]["text"]
        assert context.shape[1] == TEXT_LEN + 1
        assert_close(context, run.pre[stream.name]["context"],
                     f"{stream.name} projected context", atol=_ATOL)


def test_the_text_slot_mask_carries_the_padded_tail_and_the_proprio_token(
        run, sides):
    """The new ``[B, L]`` slot mask against the legacy ``[B, S, L]`` one.

    The source broadcasts its key mask over the query axis before handing it
    to attention (``wan/dit.py:836`` and ``action/dit.py:331``); this side
    keeps it per key and lets the attention call broadcast. Taking row 0 is
    therefore the whole conversion, and the rows being identical is itself
    part of the claim -- a per-query mask would mean the two sides are masking
    different things. The action-conditioned branch at ``wan/dit.py:806-827``
    is the one that would break that, and it is off here.

    The last column is the appended proprioception token, which declares no
    mask and so must be valid on both sides. The two before it are this
    fixture's padding.
    """
    for stream in sides.spec.streams:
        legacy_mask = run.pre[stream.name]["context_mask"]
        assert legacy_mask.dim() == 3
        assert torch.equal(legacy_mask,
                           legacy_mask[:, :1].expand_as(legacy_mask))
        extras = run.states[stream.name].extras
        assert set(extras["cross_kv_mask"]) == {"text"}
        mask = extras["cross_kv_mask"]["text"]
        assert_identical(mask.int(), legacy_mask[:, 0, :].int(),
                         f"{stream.name} text slot mask")
        assert not mask[:, TEXT_LEN - 2:TEXT_LEN].any()
        assert mask[:, -1].all()

# ---- every joint layer, then the exit ----


def test_every_joint_layer_matches_in_both_streams(run, sides):
    """Layer by layer, because only the exit would say "somewhere upstream".

    Both capture points are the post-block output: the wrapper in
    ``_capturing`` returns what the source assigns to ``tokens_all[name]``
    (``mot/model.py:665``) and the engine stores the hidden state at the same
    place (``engine.py:587-589``), so layer *i* means the same thing on both
    sides and no offset correction is needed.

    The per-layer deviation is also the number that says whether the budget is
    still honest: it grows slowly down the stack, since each layer's rounding
    rides on the previous one's, and the last layer is the worst case the
    exit comparison below sees.
    """
    layers = range(sides.spec.groups[0].num_layers)
    for stream in sides.spec.streams:
        captured = run.captured[stream.name]
        assert set(captured) == set(layers)
        drift = [deviation(run.engine.captured[(stream.name, index)],
                           captured[index]) for index in layers]
        bad = [index for index, worst in enumerate(drift) if worst > _ATOL]
        if bad:
            raise AssertionError(
                f"{stream.name}: first joint layer over the "
                f"{_ATOL:.1e} budget is {bad[0]} of {len(drift)} at "
                f"{drift[bad[0]]:.3e}; {len(bad)} layers are over, worst "
                f"{max(drift):.3e}")


def test_the_engine_exit_hidden_states_match(run, sides):
    """Both streams at the exit, after ``finalize``.

    Worth asserting even with every layer already compared: ``finalize`` runs
    after the last layer, and for this family it is deliberately a no-op (the
    final norm sits in the head). A stray operation there would show up only
    here.
    """
    for stream in sides.spec.streams:
        assert_close(run.engine.states[stream.name].hidden,
                     run.tokens[stream.name],
                     f"{stream.name} exit hidden state", atol=_ATOL)


def test_the_two_flow_heads_match(run, sides):
    """``patch_flow`` against ``post_dit``, ``linear`` against ``post_dit``.

    The video head is the interesting half: its modulation is per token here,
    its exit width is ``latent_channels x prod(patch_size)``, and the
    unflatten is the one step whose failure keeps the shape right while
    stirring the time axis into the spatial axes. Going through the source's
    ``post_dit`` rather than ``head`` plus ``unpatchify`` by hand is
    deliberate -- it is the pairing the source itself ships, so the grid it
    unflattens to comes from its own ``meta``.
    """
    for name in ("video_flow", "action_flow"):
        assert_close(run.heads[name], run.legacy_heads[name],
                     f"head {name}", atol=_ATOL)
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
    free. The second claim is a tolerance too, and a wide one -- the two runs
    have to differ by more than the budget the first claim allows, or
    "different" would not be a fact about the mask.
    """
    unmasked = run_both(sides, torch.ones(BATCH, TEXT_LEN, dtype=torch.bool,
                                          device=DEVICE))
    for stream in sides.spec.streams:
        assert_close(unmasked.engine.states[stream.name].hidden,
                     unmasked.tokens[stream.name],
                     f"{stream.name} exit hidden state, no padding",
                     atol=_ATOL)
        apart = deviation(unmasked.engine.states[stream.name].hidden,
                          run.engine.states[stream.name].hidden)
        assert apart > _ATOL * 100, \
            f"{stream.name}: the text padding moved the exit by only {apart}"


# ---- the visibility matrix, against the builder fastwam uses ----

# The configured mode (``modeling_configuration_fastwam.py:96``), which
# ``infer_action`` also insists on (``mot/fastwam.py:997-999``). Written here
# rather than read off ``build_legacy``'s expert because that constructor does
# not pass it, so the expert sits at whatever ``WanVideoDiT`` defaults to.
MASK_MODE = "first_frame_causal"


def legacy_structural(spec: WorldModelSpec) -> torch.Tensor:
    """Build the ``[S, S]`` matrix fastwam itself would attend under.

    ``_build_mot_attention_mask`` is called unbound on a stand-in, because the
    video expert's own mask builder (``mot/fastwam.py:452``) is the whole of
    its dependence on ``FastWAM``. Constructing a real one would load the Wan
    components off disk.

    Args:
        spec: The spec under comparison. It supplies both sequence lengths and
            the tokens per frame, so the two sides cannot be measured at
            different sizes.

    Returns:
        The legacy mask, ``[total, total]`` bool on CPU.
    """
    expert = SimpleNamespace(video_attention_mask_mode=MASK_MODE)
    expert.build_video_to_video_mask = (
        _video_dit.WanVideoDiT.build_video_to_video_mask.__get__(expert))
    layout = spec.build_layout()
    return _fastwam.FastWAM._build_mot_attention_mask(
        SimpleNamespace(video_expert=expert),
        video_seq_len=layout.range_of("video").length,
        action_seq_len=layout.range_of("action").length,
        video_tokens_per_frame=spec.geometry.tokens_per_frame,
        device=torch.device("cpu"))


def test_the_visibility_matrix_is_the_legacy_one_minus_future_frames():
    """Against ``_build_mot_attention_mask``, cell for cell, in two parts.

    fastwam is the one provider that is **not** bit identical to its source
    here, and deliberately so: ``providers/fastwam/spec.py:67-79`` records that
    the preset named ``first_frame_causal`` and the code branch of that name
    are different shapes. The code starts all True and only closes the first
    frame's rows, so every later frame is bidirectional and may read its own
    future; the preset is per-frame causal throughout. The stricter side was
    chosen on the grounds that being too permissive cannot be detected
    numerically.

    An equality assertion would therefore fail by construction, which is why
    the divergence is pinned from both sides instead:

    1. **Never more permissive.** No cell is open here and closed there. This
       is the half that protects the numerical cases in this file -- they hand
       the legacy driver this side's own ``plan.structural`` (:407), so they
       would pass just as well under a mask that leaked.
    2. **Nothing else differs.** The cells the source grants and this side
       withholds are *exactly* the video rows past the first frame looking at
       strictly later frames. That is the recorded gap written as a set rather
       than as prose, so widening it anywhere else turns red, and closing it
       on purpose turns red too and has to be accompanied by editing this
       claim. At this fixture's geometry -- four latent frames of four
       tokens, 20 tokens with the action stream -- the gap is 48 of 400
       cells, so the set is not vacuously empty and part 1 is not carrying
       the case on its own.
    """
    spec = tiny_spec("fastwam")
    got, want = structural_mask(spec), legacy_structural(spec)
    assert_same_mask(got & want, got, "the legacy mask as an upper bound")

    layout = spec.build_layout()
    video = layout.range_of("video")
    frame = layout.labels_of("frame")
    in_video = torch.zeros(layout.total, dtype=torch.bool)
    in_video[video.start:video.stop] = True
    # The characterisation below reads frame ids as an order, which padded
    # slots (labelled -1) are not part of; this geometry has none.
    assert (frame[in_video] >= 0).all(), "padded video slots are not covered"
    later = frame[None, :] > frame[:, None]
    expected = (in_video[:, None] & in_video[None, :] & later
                & (frame[:, None] > 0))
    assert_same_mask(want & ~got, expected, "the gap to the legacy mask")


def test_the_matrix_fed_to_mot_is_the_one_compared_above(sides):
    """``model.plans[0]`` is what the case above speaks for.

    That case compiles the declaration directly, without allocating a stream;
    this one closes the gap to the object the harness really hands
    ``MoT.forward``. It also records the one shape conversion the harness has
    to get right: ``MoT.forward`` rejects anything but a 2-D square mask
    (``mot/model.py:574-577``), while ``plan.materialize("dense")`` returns
    ``[1, 1, S, S]``.

    The all/any pair stays because it fails differently: a matrix of all True
    would leave every comparison in this file saying nothing about visibility,
    and the subset claim above would not notice, since all True is a superset
    of nothing.
    """
    structural = sides.model.plans[0].structural
    total = sides.model.layout.total
    assert tuple(structural.shape) == (total, total)
    assert structural.dtype is torch.bool
    assert not structural.all()
    assert structural.any()
    assert_same_mask(structural.cpu(), structural_mask(sides.spec),
                     "plans[0] against the compiled declaration")
