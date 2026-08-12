# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Registry for dataset index-sampling strategies.

A map-style dataset is indexed by ``idx`` produced by the DataLoader sampler.
Different models want different mappings from ``idx`` to the *anchor* frame that
delta-timestamp decoding hangs off of:

- ``"sequential"``  — ``idx`` passed straight through (the default lerobot v3
  behaviour: deterministic flat indexing).
- ``"motus_random"`` — reproduces the source Motus sampler: a random episode and
  a random condition frame within the episode's valid range, seeded by ``idx``
  so the choice is deterministic (checkpoint/resume safe) yet uniformly covers
  every episode's valid start positions.

A strategy is a callable ``(IndexSampleContext) -> int`` returning the *global
flat frame index* to feed into ``LeRobotDataset.__getitem__``. Multi-frame video
and the action chunk are then produced natively by lerobot's ``delta_timestamps``
relative to that anchor frame.

The registry mirrors ``data/datasets/transforms/registry.py`` conventions so future
sampling strategies plug in by decorating a function, without touching the
dataset class.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple


@dataclass(frozen=True)
class IndexSampleContext:
    """Inputs available to an index-sampling strategy.

    Attributes:
        idx: The raw index handed to ``Dataset.__getitem__`` by the sampler.
        num_episodes: Number of episodes in the (possibly subset) dataset.
        episode_bounds: Per-episode ``(from_idx, to_idx)`` global flat frame
            index ranges (half-open, matching lerobot ``episode_data_index``).
        physical_chunk_size: ``action_chunk_size * global_downsample_rate`` —
            the physical frame span one sample consumes; used to keep the
            condition frame far enough from the episode end that all future
            frames exist (no boundary padding).
        base_seed: Extra seed offset for reproducible-but-varied runs.
        epoch: Reserved for future per-epoch resampling (currently always 0;
            epoch cannot reach worker processes via attribute mutation, so
            plan-A leaves this fixed — see the transplant notes).
    """

    idx: int
    num_episodes: int
    episode_bounds: Sequence[Tuple[int, int]]
    physical_chunk_size: int
    base_seed: int = 0
    epoch: int = 0


IndexSampler = Callable[[IndexSampleContext], int]

_INDEX_SAMPLER_REGISTRY: Dict[str, IndexSampler] = {}

# Large stride so that mixing ``epoch`` into the seed (future plan B) does not
# collide with the ``idx``/``base_seed`` space.
_EPOCH_SEED_STRIDE = 1_000_003


def register_index_sampler(name: str):
    """Decorator registering an index-sampling strategy under ``name``."""

    def decorator(fn: IndexSampler) -> IndexSampler:
        _INDEX_SAMPLER_REGISTRY[name] = fn
        return fn

    return decorator


def get_index_sampler(name: str) -> IndexSampler:
    """Look up a registered index sampler by name."""
    if name not in _INDEX_SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown index_sampler '{name}'. "
            f"Available: {sorted(_INDEX_SAMPLER_REGISTRY.keys())}"
        )
    return _INDEX_SAMPLER_REGISTRY[name]


@register_index_sampler("sequential")
def _sequential(ctx: IndexSampleContext) -> int:
    """Pass ``idx`` straight through (default lerobot v3 flat indexing)."""
    return ctx.idx


@register_index_sampler("motus_random")
def _motus_random(ctx: IndexSampleContext) -> int:
    """Random episode + random condition frame, matching source Motus exactly.

    Reproduces the source ``LeRobotMotusDataset.__getitem__`` +
    ``_calculate_sampling_indices`` verbatim: ``idx`` is IGNORED and both the
    episode and the condition frame are drawn from the unseeded **global**
    ``random`` module (base ``lerobot_dataset.py:529`` / ``:852``), so the anchor
    is non-deterministic w.r.t. ``idx`` — bit-for-bit matching source behaviour
    (not reproducible run-to-run, by design of the source). Draw order is
    episode first, then condition frame, exactly as base.

    The condition frame is drawn uniformly from
    ``[0, total_frames - physical_chunk_size - 1]`` so every future video/action
    frame stays inside the episode (lerobot never pads). Episode bounds come from
    ``ctx.episode_bounds`` (the transplant's correct accumulated per-episode
    lengths), NOT base's episode-id-as-frame-index bug.
    """
    # Global unseeded RNG, idx ignored — matches base random.randint calls.
    episode = random.randint(0, ctx.num_episodes - 1)
    from_idx, to_idx = ctx.episode_bounds[episode]
    total_frames = int(to_idx) - int(from_idx)

    max_condition_idx = total_frames - ctx.physical_chunk_size - 1
    if max_condition_idx < 0:
        condition_frame_idx = 0
    else:
        condition_frame_idx = random.randint(0, max_condition_idx)

    return int(from_idx) + condition_frame_idx


@register_index_sampler("motus_seeded")
def _motus_seeded(ctx: IndexSampleContext) -> int:
    """Deterministic-by-``idx`` variant of :func:`_motus_random` for loss parity.

    Uses the SAME formula and draw order as base
    ``LeRobotMotusDataset.__getitem__`` + ``_calculate_sampling_indices``
    (episode ``randint`` first, then condition-frame ``randint``, identical
    bounds), but draws from a LOCAL ``random.Random(base_seed + idx)`` instead of
    the process-global unseeded RNG. This makes the (episode, condition) anchor a
    pure function of ``idx`` + ``base_seed``, so a base run patched to seed the
    same way produces byte-identical anchors for the same ``idx``. Intended only
    for the base<->transplant parity harness (select via PARITY_DATA_SEED).
    """
    rng = random.Random(ctx.base_seed + ctx.idx)
    episode = rng.randint(0, ctx.num_episodes - 1)
    from_idx, to_idx = ctx.episode_bounds[episode]
    total_frames = int(to_idx) - int(from_idx)

    max_condition_idx = total_frames - ctx.physical_chunk_size - 1
    if max_condition_idx < 0:
        condition_frame_idx = 0
    else:
        condition_frame_idx = rng.randint(0, max_condition_idx)

    result = int(from_idx) + condition_frame_idx
    return result
