# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Motus multi-frame behaviour hooks for the generic lerobot datasets.

Instead of subclassing or branching inside the dataset classes, Motus plugs its
multi-frame geometry in through the three behaviour hooks they expose
(``delta_timestamps_fn`` / ``length_fn`` / ``index_map_fn``). All motus-specific
state is stashed on the dataset instance by :func:`motus_delta_timestamps` (which
runs first, before the dataset's ``__init__``) and consumed by the other two
hooks — the base classes stay entirely model-agnostic.

``build_motus_lerobot_dataset`` wires these hooks onto either a single
``LeRobotV3Dataset`` (``task_mode="single"``) or a ``MultiLeRobotV3Dataset``
(``task_mode="multi"``), reproducing the source Motus episode selection and the
``"motus_random"`` index sampler.
"""

from __future__ import annotations

import logging
import os
import random as _random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from loongforge.embodied.data.datasets.motus.index_samplers import (
    IndexSampleContext,
    get_index_sampler,
)
from loongforge.embodied.data.datasets.lerobot_dataset import (
    LeRobotV3Dataset,
    MultiLeRobotV3Dataset,
)

logger = logging.getLogger(__name__)


def _build_motus_delta_timestamps(
    decode_keys: List[str],
    action_key: str,
    num_video_frames: int,
    video_action_freq_ratio: int,
    global_downsample_rate: int,
    fps: int,
) -> Dict[str, list]:
    """Build delta_timestamps reproducing the source Motus sampling geometry.

    Relative to the condition (anchor) frame:
      - video/image keys: ``[0.0] + [(j+1)*vfr*gdr / fps for j in range(F)]``
        (index 0 = first_frame, the rest = the F future video-prediction frames).
      - action: ``[0.0] + [(i+1)*gdr / fps for i in range(action_chunk_size)]``
        where ``action_chunk_size = F * vfr``. Index 0 is the delta-0 anchor
        (condition-frame) action; the transform slices it off so the future
        action chunk (indices 1:) matches the source (which starts one
        downsample step after the condition frame). The anchor entry is kept
        so stateless datasets can recover the condition-frame action as the
        ``initial_state`` proxy (matches base ``item_cond`` action fallback).

    State is intentionally *not* given a delta: lerobot already returns the
    condition frame's ``observation.state`` as a single-frame value, which is
    exactly the source ``initial_state``.
    """
    vfr = video_action_freq_ratio
    gdr = global_downsample_rate
    action_chunk_size = num_video_frames * vfr

    image_deltas = [0.0] + [((j + 1) * vfr * gdr) / fps for j in range(num_video_frames)]
    # Prepend the delta-0 anchor action; the transform uses action_deltas[1:]
    # as the future chunk and action_deltas[0] as the condition-frame action.
    action_deltas = [0.0] + [((i + 1) * gdr) / fps for i in range(action_chunk_size)]

    delta_timestamps: Dict[str, list] = {k: list(image_deltas) for k in decode_keys}
    delta_timestamps[action_key] = action_deltas
    return delta_timestamps


def motus_delta_timestamps(dataset: LeRobotV3Dataset, info: Dict[str, Any], fps: int) -> Dict[str, list]:
    """``delta_timestamps_fn`` hook: expand a single anchor into a multi-frame stack.

    Runs before ``LeRobotDataset.__init__``. Resolves the camera view mode and
    stashes the geometry (``_view_mode`` / ``_decode_keys`` / ``_action_key`` /
    ``_action_chunk_size`` / ``_physical_chunk_size``) plus a ``_motus_mode``
    marker on ``dataset`` for the transform builder and the sampling hooks.
    """
    from loongforge.embodied.data.datasets.motus.transforms.views import resolve_view_mode

    kw = dataset._strategy_kwargs
    num_video_frames = int(kw["num_video_frames"])
    video_action_freq_ratio = int(kw["video_action_freq_ratio"])
    global_downsample_rate = int(kw["global_downsample_rate"])

    features = info.get("features", {})
    view_mode, decode_keys = resolve_view_mode(features.keys())
    action_key = "action" if "action" in features else (
        "actions" if "actions" in features else "action"
    )

    dataset._motus_mode = True
    dataset._view_mode = view_mode
    dataset._decode_keys = decode_keys
    dataset._action_key = action_key
    dataset._action_chunk_size = num_video_frames * video_action_freq_ratio
    dataset._physical_chunk_size = dataset._action_chunk_size * global_downsample_rate

    return _build_motus_delta_timestamps(
        decode_keys=decode_keys,
        action_key=action_key,
        num_video_frames=num_video_frames,
        video_action_freq_ratio=video_action_freq_ratio,
        global_downsample_rate=global_downsample_rate,
        fps=fps,
    )


def motus_length(dataset: LeRobotV3Dataset) -> int:
    """``length_fn`` hook: source Motus used ``num_episodes * 1000`` per epoch."""
    return _motus_runtime(dataset)["num_episodes"] * 1000


def motus_index_map(dataset: LeRobotV3Dataset, idx: int) -> int:
    """``index_map_fn`` hook: map ``idx`` to a global flat condition-frame index."""
    rt = _motus_runtime(dataset)
    ctx = IndexSampleContext(
        idx=int(idx),
        num_episodes=rt["num_episodes"],
        episode_bounds=rt["bounds"],
        physical_chunk_size=dataset._physical_chunk_size,
        base_seed=int(dataset._strategy_kwargs.get("index_sampler_seed", 0)),
        epoch=0,
    )
    return rt["sampler"](ctx)


def _motus_runtime(dataset: LeRobotV3Dataset) -> Dict[str, Any]:
    """Lazily compute + cache the post-init sampling runtime on the dataset.

    Deferred until first access (``__len__`` / ``__getitem__``) so that
    ``self.episodes`` / ``self.meta`` are already populated by the parent init.
    """
    rt = getattr(dataset, "_motus_runtime_cache", None)
    if rt is None:
        bounds = _compute_relative_episode_bounds(dataset)
        sampler_name = dataset._strategy_kwargs.get("index_sampler", "motus_random")
        rt = {
            "bounds": bounds,
            "num_episodes": len(bounds),
            "sampler": get_index_sampler(sampler_name),
        }
        dataset._motus_runtime_cache = rt
    return rt


def _compute_relative_episode_bounds(dataset) -> List[Tuple[int, int]]:
    """Per-episode ``(from, to)`` bounds in the dataset's flat index space.

    For a single :class:`LeRobotV3Dataset` these are ``hf_dataset``-relative
    bounds. For a :class:`MultiLeRobotV3Dataset` they are in the *concatenated*
    flat frame space (sub-datasets laid end-to-end in ``repo_ids`` order, by
    ``num_frames``), so a bound maps straight onto
    ``MultiLeRobotDataset.__getitem__``.

    ``meta.episodes[ep]`` stores absolute ``dataset_from_index`` /
    ``dataset_to_index``; when only a subset of episodes is loaded, absolute
    != relative, so we accumulate episode *lengths* over the selected order.
    """
    if hasattr(dataset, "_datasets"):
        # Multi: lay each sub-dataset's episodes end-to-end. The running cursor
        # doubles as the cross-sub frame offset (== sum of prior num_frames).
        bounds: List[Tuple[int, int]] = []
        cursor = 0
        for sub in dataset._datasets:
            for from_, to_ in _single_relative_bounds(sub):
                bounds.append((cursor + from_, cursor + to_))
            if bounds:
                cursor = bounds[-1][1]
        return bounds
    return _single_relative_bounds(dataset)


def _single_relative_bounds(dataset) -> List[Tuple[int, int]]:
    """Per-episode ``(from, to)`` bounds in one dataset's hf_dataset index space."""
    # Accumulate in ASCENDING episode_index order to match the filtered
    # hf_dataset physical layout (isin predicate pushdown only filters, never
    # reorders) and to match base's sorted(episode_ids) accumulation, so a
    # given drawn episode_idx maps to the same physical episode in both.
    selected = (
        sorted(int(e) for e in dataset.episodes)
        if dataset.episodes is not None
        else list(range(dataset.meta.total_episodes))
    )
    bounds: List[Tuple[int, int]] = []
    cursor = 0
    for ep in selected:
        ep_meta = dataset.meta.episodes[int(ep)]
        length = int(ep_meta["dataset_to_index"]) - int(ep_meta["dataset_from_index"])
        bounds.append((cursor, cursor + length))
        cursor += length
    return bounds


def build_motus_lerobot_dataset(model_cfg, data_cfg, training_args):
    """Build the Motus multi-frame dataset by wiring hooks onto the reused datasets.

    Sampling geometry (video frames / action stride) comes from the *model*
    config; dataset location and task selection come from the *data* config.
    Dispatches on ``data_cfg.task_mode``:

    - ``"single"``: one :class:`LeRobotV3Dataset`, reproducing the source Motus
      episode selection (shuffle with ``Random(0)`` then truncate to
      ``max_episodes``).
    - ``"multi"``: a :class:`MultiLeRobotV3Dataset` over several task repos
      (resolved from ``data_cfg.task_name``; ``None`` = every sub-directory of
      ``root``), reproducing the source multi selection (all episodes per task,
      no shuffle/truncate).

    Both enable the ``"motus_random"`` index sampler so each ``idx``
    deterministically maps to a random episode + condition frame (checkpoint/
    resume safe).
    """
    root = training_args.dataset_path or data_cfg.root or None
    if not root:
        raise ValueError("Must specify --dataset-path or data.root for motus_lerobot")

    video_backend = (
        data_cfg.video_backend or training_args.video_backend or "pyav"
    )
    # PARITY_DATA_SEED (default unset -> inert): switch to the deterministic-by-idx
    # "motus_seeded" sampler so anchors are a pure function of idx+seed, matching a
    # base run patched to seed identically. Normal runs keep "motus_random".
    _parity_seed = os.environ.get("PARITY_DATA_SEED")
    if _parity_seed is not None:
        index_sampler = "motus_seeded"
        index_sampler_seed = int(_parity_seed)
    else:
        index_sampler = "motus_random"
        index_sampler_seed = 0
    geometry = dict(
        num_video_frames=model_cfg.num_video_frames,
        video_action_freq_ratio=model_cfg.video_action_freq_ratio,
        global_downsample_rate=model_cfg.global_downsample_rate,
        index_sampler=index_sampler,
        index_sampler_seed=index_sampler_seed,
    )

    # Offline VAE-latent cache (default off): --latent-cache-dir points at a dir
    # of {flat_idx:08d}.pt fp32 latents (precompute_latent_cache.py). When set,
    # wire the read hook so __getitem__ injects data["clean_full_latent"] and the
    # trainer skips the online encode. Inert when the flag is empty/unset.
    cache_dir = getattr(training_args, "latent_cache_dir", "") or ""
    latent_cache_fn = _make_latent_cache_fn(cache_dir) if cache_dir else None

    if data_cfg.task_mode == "single":
        return _build_single(root, data_cfg, video_backend, geometry, latent_cache_fn)
    if data_cfg.task_mode == "multi":
        if latent_cache_fn is not None:
            raise NotImplementedError(
                "--latent-cache-dir is only supported for task_mode='single'; "
                "the flat-index key namespace across sub-datasets is not yet "
                "wired for 'multi'."
            )
        return _build_multi(root, data_cfg, video_backend, geometry)
    raise ValueError(
        f"Unknown task_mode '{data_cfg.task_mode}' (expected 'single' or 'multi')."
    )


def _make_latent_cache_fn(cache_dir: str):
    """Build the ``latent_cache_fn`` hook that injects a precomputed VAE latent.

    Keyed by the resolved flat frame index (the value the index sampler returns
    == ``from_idx + condition_frame_idx``), matching precompute_latent_cache.py's
    ``{flat_idx:08d}.pt`` layout. Runs in the DataLoader worker; the fp32 latent
    is loaded onto CPU and stashed under ``data["clean_full_latent"]`` so the
    collator can stack it and the trainer can skip the online encode.

    Cache MISS is graceful: if the ``{flat_idx:08d}.pt`` file is absent, the key
    is simply not set, so the collator's ``all(... clean_full_latent ...)`` guard
    turns the whole batch's latent to ``None`` and the trainer falls back to the
    online VAE encode (first_frame/video_frames are always collated, so the
    fallback has its inputs). Trade-off: a wrong/empty cache_dir silently
    degrades to all-online (correct, but loses the speedup with no hard error).

    Returns a module-level ``_LatentCacheFn`` INSTANCE (not a local closure) so
    the dataset stays picklable for ``num_workers > 0`` DataLoader workers.
    """
    return _LatentCacheFn(cache_dir)


class _LatentCacheFn:
    """Picklable ``latent_cache_fn`` hook (see :func:`_make_latent_cache_fn`).

    Must be a module-level class (not a nested closure) so it survives the pickle
    round-trip when the DataLoader forks/spawns workers. ``cache_dir`` is a plain
    str and ``_warned`` a bool, both picklable; each worker unpickles its own copy
    so the once-per-worker warning state resets per worker as before.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self._warned = False

    def __call__(self, dataset, flat_idx: int, data: Dict[str, Any]) -> Dict[str, Any]:
        import torch as _torch

        path = os.path.join(self.cache_dir, f"{int(flat_idx):08d}.pt")
        # Graceful miss: skip injecting the key so this batch falls back to the
        # online encode (see docstring). fp32 on CPU; map_location keeps it
        # host-side for the pinned H2D path.
        if os.path.exists(path):
            data["clean_full_latent"] = _torch.load(path, map_location="cpu")
        elif not self._warned:
            self._warned = True
            logger.warning(
                "[latent-cache] MISS for flat_idx=%d (%s not found); falling back "
                "to the online VAE encode for this batch. Further misses in this "
                "worker are suppressed. If unexpected, check --latent-cache-dir "
                "points at a complete precompute_latent_cache.py output.",
                int(flat_idx), path,
            )
        return data


def _build_single(root, data_cfg, video_backend, geometry, latent_cache_fn=None):
    """Single-repo path: shuffle(seed=0) + truncate to ``max_episodes``."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    repo_id = data_cfg.repo_id or Path(root).name
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    total_eps = int(meta.total_episodes)
    all_ep_ids = list(range(total_eps))
    _random.Random(0).shuffle(all_ep_ids)
    max_episodes = data_cfg.max_episodes
    if max_episodes is not None and max_episodes > 0:
        all_ep_ids = all_ep_ids[: min(max_episodes, len(all_ep_ids))]

    return LeRobotV3Dataset(
        repo_id=repo_id,
        root=str(root),
        episodes=all_ep_ids,
        video_backend=video_backend,
        tolerance_s=1e-4,
        delta_timestamps_fn=motus_delta_timestamps,
        length_fn=motus_length,
        index_map_fn=motus_index_map,
        latent_cache_fn=latent_cache_fn,
        **geometry,
    )


def _resolve_multi_repo_ids(root, task_name) -> List[str]:
    """Resolve the list of task repo_ids under ``root`` (source Motus logic).

    ``None`` = every sub-directory of ``root``; ``str`` = a single task;
    ``list`` = explicit tasks. Each resolved task must be an existing directory.
    """
    import os

    if task_name is None:
        return sorted(
            name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        )
    if isinstance(task_name, str):
        if not os.path.isdir(os.path.join(root, task_name)):
            raise ValueError(f"Task '{task_name}' not found under {root}")
        return [task_name]
    if isinstance(task_name, (list, tuple)):
        repo_ids = list(task_name)
        for name in repo_ids:
            if not os.path.isdir(os.path.join(root, name)):
                raise ValueError(f"Task '{name}' not found under {root}")
        return repo_ids
    raise ValueError(f"Invalid task_name: {task_name!r}")


def _build_multi(root, data_cfg, video_backend, geometry):
    """Multi-repo path: MultiLeRobotDataset over all episodes of each task."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    repo_ids = _resolve_multi_repo_ids(root, data_cfg.task_name)
    if not repo_ids:
        raise ValueError(f"No task repos found under {root} for multi task_mode")

    episodes = {}
    for repo_id in repo_ids:
        meta = LeRobotDatasetMetadata(repo_id, root=str(Path(root) / repo_id))
        episodes[repo_id] = list(range(int(meta.total_episodes)))

    return MultiLeRobotV3Dataset(
        repo_ids=repo_ids,
        root=str(root),
        episodes=episodes,
        video_backend=video_backend,
        delta_timestamps_fn=motus_delta_timestamps,
        length_fn=motus_length,
        index_map_fn=motus_index_map,
        **geometry,
    )
