# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Read-only dumps of what a world-model spec resolves to.

Three subcommands, each of which answers one question that comes up while
porting a legacy checkpoint onto the five-layer architecture:

- ``keys``: the geometry, the condition table, the head table, the per-stream
  scalars, and every ``state_dict`` key with its shape. This is the first
  thing to look at when ``load_state_dict(strict=True)`` reports a missing or
  unexpected key, because it says what the new side *wants* to be given.
- ``entry``: the shapes the entry stage produces (hidden, modulation,
  timestep embedding, RoPE table, projected cross-attention keys). Use it when
  a parity harness disagrees at the entry rather than inside the stack.
- ``timestep``: the per-token timestep tensor a stream expands a ``[B]`` draw
  into, next to the two legacy formulas it has to reproduce.

The specs are the real providers with their widths and geometry shrunk, which
is the same trick ``tests/embodied/unit/test_arch_blocks.py`` plays and for the
same reason: vpp2 at its declared size is 40 layers by 5120 width, so
allocating it to read key names off it is not worth the memory. Topology --
the condition table, the visibility rules, the head table, the clean prefix --
is whatever the provider declares and is **not** shrunk.

Usage::

    PYTHONPATH=. python -m tools.embodied.arch_probe keys internw0_delta \
        --drop-head track_align
    PYTHONPATH=. python -m tools.embodied.arch_probe entry fastwam
    PYTHONPATH=. python -m tools.embodied.arch_probe timestep fastwam
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace

import torch

from loongforge.embodied.model.arch.builder import (WorldModel,
                                                    materialize_conditions)
from loongforge.embodied.model.arch.spec.core import (build, get_provider,
                                                      provider_names)
from loongforge.embodied.model.arch.spec.world import WorldModelSpec
from loongforge.embodied.model.arch.streams.conditions import InjectionKind
from loongforge.embodied.model.arch.streams.engine import JointAttentionEngine

import loongforge.embodied.model.arch.spec.providers  # noqa: F401

BATCH = 2
TEXT_LEN = 5
CLIP_LEN = 257
ACTION_DIM = 6

# Both streams must agree on ``num_heads`` and ``head_dim`` (joint attention
# concatenates along the token axis), while ``residual_width`` is deliberately
# unequal: "residual width != attention width" is the normal case for the
# action stream. Sixteen layers rather than two because InternW0-Delta's
# intermediate heads read layer 8 and layer 14.
TINY_SCALARS = dict(
    video_residual_width=32, video_num_heads=2, video_head_dim=8,
    video_num_layers=16, video_ffn_width=32,
    action_residual_width=16, action_num_heads=2, action_head_dim=8,
    action_num_layers=16, action_ffn_width=16,
)


def tiny_spec(model_type: str) -> WorldModelSpec:
    """Return a provider's topology with its widths and geometry shrunk.

    The pixel sizes are derived from the VAE stride rather than hardcoded,
    because the three dual-stream models use different strides and a hardcoded
    size fails the divisibility check for two of them. Four latent frames
    rather than two, because InternW0-Delta declares ``condition_frames=3`` and
    its ``TokenTimestep(clean_prefix=3)`` needs a fourth frame to carry noise.

    Args:
        model_type: A registered provider name.

    Returns:
        The shrunken spec.
    """
    probe = get_provider(model_type)(action_width=ACTION_DIM, **TINY_SCALARS)
    geo = probe.geometry
    # The shrink goes through the provider's ``geometry=`` override rather
    # than ``dataclasses.replace``: visibility and the flow heads are derived
    # *inside* the provider from the resolved geometry, so replacing the
    # geometry afterwards leaves token counts and labels baked at full size.
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
        **TINY_SCALARS)


def drop_heads(spec: WorldModelSpec, names: Sequence[str]) -> WorldModelSpec:
    """Remove heads from a spec, together with the loss terms that name them.

    The loss terms have to go too: V6 requires every ``LossSpec.head`` to name
    a head that exists, so dropping only the head yields a spec that fails
    validation rather than a smaller model.

    Needed because not every declared head has an implementation yet;
    InternW0-Delta's ``track_align`` raises ``NotImplementedError`` at build
    time on purpose, so the ``keys`` and ``entry`` probes have to be told to
    leave it out.

    Args:
        spec: The spec to shrink.
        names: Head names to remove. Absent names are ignored.

    Returns:
        A copy with those heads and their loss terms gone.
    """
    dropped = frozenset(names)
    return replace(
        spec,
        heads=tuple(head for head in spec.heads if head.name not in dropped),
        diffusion=replace(spec.diffusion, losses=tuple(
            loss for loss in spec.diffusion.losses
            if loss.head not in dropped)),
    )


def world(spec: WorldModelSpec) -> WorldModel:
    """Assemble the model the way production does: spec, engine, layout.

    Args:
        spec: The world-model spec.

    Returns:
        A dense-backend model in eval mode.
    """
    layout = spec.build_layout()
    return WorldModel(spec, engine=JointAttentionEngine(backend="dense"),
                      layout=layout).eval()


def fake_batch(spec: WorldModelSpec) -> dict[str, torch.Tensor]:
    """Build one batch from the spec's declarations alone.

    Stream tensors are keyed by ``StreamSpec.batch_key`` and condition
    tensors by ``ConditionSpec.source_key``. Three shapes, chosen by what the
    consumer of each condition will do with it:

    - a condition with an encoder gets ``[B, source_width]``, because
      ``source_width`` is the encoder's *input* width while ``width`` is its
      output, and the legacy call site slices to two dimensions;
    - ``CROSS_ATTENTION`` gets ``[B, L, width]``, with the CLIP slot given
      exactly ``CLIP_LEN`` tokens because the legacy cross-attention splits
      the context at that offset;
    - ``CHANNEL_CONCAT`` gets ``[B, width, F, H, W]``, since it is
      concatenated onto the video channels rather than attended to.

    Args:
        spec: The world-model spec.

    Returns:
        The batch.
    """
    geo = spec.geometry
    frames, rows, cols = geo.latent_grid
    batch = {
        stream.batch_key: (
            torch.randn(BATCH, geo.latent_channels, frames, rows, cols)
            if stream.name == "video"
            else torch.randn(BATCH, geo.action_horizon, geo.action_dim))
        for stream in spec.streams
    }
    for condition in spec.conditions:
        if condition.encoder is not None:
            batch[condition.source_key] = torch.randn(
                BATCH, condition.source_width)
        elif condition.kind is InjectionKind.CROSS_ATTENTION:
            length = CLIP_LEN if condition.name == "clip_image" else TEXT_LEN
            batch[condition.source_key] = torch.randn(
                BATCH, length, condition.width)
        elif condition.kind is InjectionKind.CHANNEL_CONCAT:
            batch[condition.source_key] = torch.randn(
                BATCH, condition.width, frames, rows, cols)
    return batch


def probe_keys(spec: WorldModelSpec) -> None:
    """Print the spec's declarations and every ``state_dict`` key and shape.

    Args:
        spec: The world-model spec.
    """
    geo = spec.geometry
    print(f"latent_grid={geo.latent_grid} patch_size={geo.patch_size} "
          f"patch_grid={geo.patch_grid} latent_channels={geo.latent_channels} "
          f"concat_channels(video)={spec.concat_channels('video')}")
    print(f"action_dim={geo.action_dim} action_horizon={geo.action_horizon} "
          f"padded_seq_len={geo.padded_seq_len}")
    for condition in spec.conditions:
        print(f"condition {condition.name}: kind={condition.kind.name} "
              f"width={condition.width} "
              f"source_width={condition.source_width} "
              f"source_key={condition.source_key} "
              f"append_to={condition.append_to}")
    for head in spec.heads:
        print(f"head {head.name}: output_width={head.output_width} "
              f"needs_timestep={head.needs_timestep}")
    for stream in spec.streams:
        print(f"stream {stream.name}: residual_width="
              f"{stream.residual_width} ffn_width={stream.ffn_width} "
              f"num_heads={stream.num_heads} head_dim={stream.head_dim} "
              f"num_layers={stream.num_layers} "
              f"timestep_embed_width={stream.timestep_embed_width} "
              f"token_timestep={stream.token_timestep}")
    for key, value in world(spec).state_dict().items():
        print(f"  {key} {tuple(value.shape)}")


def probe_entry(spec: WorldModelSpec) -> None:
    """Print the shapes the entry stage produces, per stream.

    The RoPE table is read off the stream module rather than the state dict
    because it is deliberately not a registered buffer: it is complex128,
    which would not survive a bf16 conversion, and the key is absent from
    every existing checkpoint.

    Args:
        spec: The world-model spec.
    """
    torch.manual_seed(0)
    model = world(spec)
    batch = fake_batch(spec)
    with torch.no_grad():
        bus = materialize_conditions(spec.conditions, batch, training=False,
                                     encoders=model.condition_encoders)
        for stream in spec.streams:
            block = model.blocks[stream.name]
            state = block.prepare(batch[stream.batch_key],
                                  timestep=torch.rand(BATCH), conditions=bus)
            print(f"{stream.name}: hidden={tuple(state.hidden.shape)} "
                  f"modulation={tuple(state.modulation.shape)} "
                  f"t_embed={tuple(state.extras['t_embed'].shape)}")
            table = block._freqs
            print(f"  rope={tuple(table.shape)} {table.dtype} (not a buffer)")
            for slot, value in state.extras.get("cross_kv", {}).items():
                mask = state.extras.get("cross_kv_mask", {}).get(slot)
                print(f"  slot {slot}: {tuple(value.shape)} mask="
                      f"{None if mask is None else tuple(mask.shape)}")
    segments = [(segment.stream, segment.start, segment.stop)
                for segment in model.layout.segments]
    print(f"layout={segments}")
    print(f"noise_levels={spec.diffusion.noise_levels}")


def probe_timestep(spec: WorldModelSpec) -> None:
    """Compare the expanded per-token timestep against the legacy formulas.

    Both legacy formulas are reproduced here rather than imported, so the
    probe runs without either legacy checkout on the path. They differ only in
    where the clean prefix is zeroed: fastwam zeroes the first
    ``clean_prefix`` frames in one place, while InternW0-Delta's caller
    contributes ``clean_prefix - 1`` zero columns and its ``pre_dit``
    prepends one more.

    Args:
        spec: The world-model spec.
    """
    stream = spec.stream("video")
    scope = stream.token_timestep
    if scope is None:
        print(f"{spec.model_type}: video declares no token_timestep; the "
              "per-sample path applies and there is nothing to expand")
        return
    geo = spec.geometry
    clean = scope.clean_prefix
    frames, rows, cols = geo.patch_grid
    per_frame_tokens = rows * cols
    block = build(stream.impl, spec=stream, geometry=geo, world=spec)
    draw = torch.rand(BATCH)
    got = block._expand_timestep(draw)

    fastwam = torch.ones(BATCH, frames, per_frame_tokens) * draw.view(-1, 1, 1)
    fastwam[:, :clean, :] = 0
    fastwam = fastwam.reshape(BATCH, -1)

    caller = torch.cat([torch.zeros(BATCH, clean - 1),
                        draw[:, None].expand(BATCH, frames - clean)], dim=1)
    internw0 = torch.cat([torch.zeros(BATCH, 1), caller], dim=1)
    internw0 = internw0[:, :, None].expand(
        BATCH, frames, per_frame_tokens).reshape(BATCH, -1)

    print(f"clean_prefix={clean} shape={tuple(got.shape)} "
          f"matches_fastwam={torch.equal(got, fastwam)} "
          f"matches_internw0_delta={torch.equal(got, internw0)}")
    print(f"  first frame={got[0, :per_frame_tokens].tolist()}")
    print(f"  last frame={got[0, -per_frame_tokens:].tolist()}")
    per_frame_input = draw[:, None].expand(BATCH, frames - clean)
    print(f"  per-frame input agrees with per-sample input="
          f"{torch.equal(got, block._expand_timestep(per_frame_input))}")


PROBES = {"keys": probe_keys, "entry": probe_entry,
          "timestep": probe_timestep}


def main(argv: Sequence[str] | None = None) -> int:
    """Run one probe against one provider.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        0; every probe is read-only and reports by printing.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("probe", choices=sorted(PROBES))
    parser.add_argument("model_type", choices=sorted(provider_names()))
    parser.add_argument("--drop-head", action="append", default=[],
                        metavar="NAME",
                        help="omit this head and its loss terms; repeatable. "
                             "internw0_delta needs --drop-head track_align, "
                             "whose implementation does not exist yet")
    args = parser.parse_args(argv)
    spec = drop_heads(tiny_spec(args.model_type), args.drop_head)
    print(f"== {args.probe} {args.model_type} ==")
    PROBES[args.probe](spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
