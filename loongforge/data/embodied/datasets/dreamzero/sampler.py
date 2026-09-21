# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.
# Copyright (c) Baidu, Inc. and its affiliates. All rights reserved.
"""DreamZero sharded sampler.

Map-style implementation of DreamZero's sharded LeRobot mixture sampling for
``torch.utils.data.Sampler``.

Algorithm:

1. Group ``trajectory_ids`` into shards of approximately ``num_steps_per_shard``
   total steps each (``generate_shards``).
2. Build a global schedule by sampling shards with replacement from a numpy RNG
   seeded with ``seed`` and weighted by shard length, then shuffling (training
   mode); same on every rank.
3. Filter the schedule for this ``(rank, worker)`` via round-robin
   (``i % world_size == rank``); workers handled here at sampler level only.
4. For each shard in this rank's filtered schedule, gather every
   ``(traj_id, step_idx)`` pair from ``step_filter`` (after clipping by
   ``allow_padding_at_end`` / ``max_delta_index``), shuffle with a numpy RNG
   seeded with ``seed`` (advanced sequentially across shards within this rank),
   and emit only the first ``shard_sampling_rate * num_steps_per_shard`` entries.
5. Translate each ``(traj_id, step_idx)`` to the dataset's flat ``__getitem__``
   index using a precomputed ``all_steps`` reverse map.

Notes:
* Small schedules precompute the full per-rank index list to keep short runs
  simple. Large schedules switch to lazy per-shard expansion to avoid
  materializing hundreds of millions of step indices.
* The default map-style path lets PyTorch DataLoader workers consume a single
  shared sampler stream. ``worker_batching_mode=upstream_iterable`` is reserved
  for recipes that require DreamZero IterableDataset worker ordering.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional

import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from .dataset.datasets import DreamZeroLeRobotDataset

logger = logging.getLogger(__name__)
_PRECOMPUTE_INDEX_LIMIT = 5_000_000
_WORKER_BATCHING_NONE = "none"
_WORKER_BATCHING_UPSTREAM_ITERABLE = "upstream_iterable"


class _StatefulIndexIterator(Iterator[int]):
    """Iterator state used by torchdata's StatefulDataLoader."""

    _YIELDED = "yielded"

    def __init__(self, sampler) -> None:
        self._sampler = sampler
        self._yielded = 0

    def __iter__(self) -> "_StatefulIndexIterator":
        return self

    def __next__(self) -> int:
        if self._yielded >= len(self._sampler._indices):
            raise StopIteration
        value = self._sampler._indices[self._yielded]
        self._yielded += 1
        return value

    def state_dict(self) -> dict[str, int]:
        """Return the iterator state for checkpointing."""
        return {self._YIELDED: self._yielded}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the iterator position from a checkpoint."""
        yielded = int(state_dict.get(self._YIELDED, 0))
        if yielded < 0:
            raise ValueError(f"Cannot restore sampler iterator with negative yielded={yielded}")
        if yielded > len(self._sampler._indices):
            raise ValueError(
                f"Cannot restore sampler iterator yielded={yielded}; "
                f"current sampler only has {len(self._sampler._indices)} indices"
            )
        self._yielded = yielded


class _LazyShardIndexIterator(Iterator[int]):
    """Lazy iterator for large sharded schedules."""

    _YIELDED = "yielded"

    def __init__(self, sampler) -> None:
        self._sampler = sampler
        self._yielded = 0
        self._iterator = sampler._iter_rank_indices(num_workers=1, worker_id=0)

    def __iter__(self) -> "_LazyShardIndexIterator":
        return self

    def __next__(self) -> int:
        if self._yielded >= len(self._sampler):
            raise StopIteration
        value = next(self._iterator)
        self._yielded += 1
        return value

    def state_dict(self) -> dict[str, int]:
        """Return the iterator state for checkpointing."""
        return {self._YIELDED: self._yielded}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the iterator position from a checkpoint."""
        yielded = int(state_dict.get(self._YIELDED, 0))
        if yielded < 0:
            raise ValueError(f"Cannot restore sampler iterator with negative yielded={yielded}")
        if yielded > len(self._sampler):
            raise ValueError(
                f"Cannot restore sampler iterator yielded={yielded}; "
                f"current sampler only has {len(self._sampler)} estimated indices"
            )
        self._iterator = self._sampler._iter_rank_indices(num_workers=1, worker_id=0)
        self._yielded = 0
        for _ in range(yielded):
            next(self)


class _LazyWorkerInterleavedIndexIterator(Iterator[int]):
    """Lazy iterator for DreamZero IterableDataset-style worker ordering.

    This mirrors ``_build_worker_interleaved_indices`` without materializing the
    full rank stream for large schedules such as ``num_shards_to_sample=2**20``.
    """

    _YIELDED = "yielded"

    def __init__(self, sampler) -> None:
        self._sampler = sampler
        self._yielded = 0
        self._num_workers = max(1, int(sampler.dataloader_num_workers))
        self._worker_iters = [
            sampler._iter_rank_indices(num_workers=self._num_workers, worker_id=worker_id)
            for worker_id in range(self._num_workers)
        ]
        self._worker_exhausted = [False for _ in range(self._num_workers)]
        self._next_worker_id = 0
        self._pending_chunk: list[int] = []

    def __iter__(self) -> "_LazyWorkerInterleavedIndexIterator":
        return self

    def __next__(self) -> int:
        if self._yielded >= len(self._sampler):
            raise StopIteration
        while not self._pending_chunk:
            if all(self._worker_exhausted):
                raise StopIteration
            worker_id = self._next_worker_id
            self._next_worker_id = (self._next_worker_id + 1) % self._num_workers
            if self._worker_exhausted[worker_id]:
                continue
            for _ in range(self._sampler.micro_batch_size):
                try:
                    self._pending_chunk.append(next(self._worker_iters[worker_id]))
                except StopIteration:
                    self._worker_exhausted[worker_id] = True
                    break

        self._yielded += 1
        return self._pending_chunk.pop(0)

    def state_dict(self) -> dict[str, int]:
        """Return the iterator state for checkpointing."""
        return {self._YIELDED: self._yielded}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the iterator position from a checkpoint."""
        yielded = int(state_dict.get(self._YIELDED, 0))
        if yielded < 0:
            raise ValueError(f"Cannot restore sampler iterator with negative yielded={yielded}")
        if yielded > len(self._sampler):
            raise ValueError(
                f"Cannot restore sampler iterator yielded={yielded}; "
                f"current sampler only has {len(self._sampler)} estimated indices"
            )
        self.__init__(self._sampler)
        for _ in range(yielded):
            next(self)


class DreamZeroShardedSampler(Sampler[int]):
    """DreamZero sharded sampler for a map-style dataset.

    Args:
        dataset: A ``DreamZeroLeRobotDataset`` instance exposing ``trajectory_ids``,
            ``trajectory_lengths``, ``step_filter``, ``max_delta_index``,
            ``all_steps``, ``discard_bad_trajectories``, and ``lerobot_info_meta``.
        num_replicas: World size for DDP. If None, inferred from ``torch.distributed``.
        rank: This rank. If None, inferred.
        seed: Seed for numpy RNG.
        shard_sampling_rate: Fraction of steps per shard to keep.
        num_steps_per_shard: Target steps per shard.
        num_shards_to_sample: How many shards to draw with replacement
            (large recipes commonly use 2**20). Lower values yield finite ``__len__``.
        training: If False, deterministic sequential shard ordering.
        allow_padding_at_end: When False, steps are clipped to
            ``trajectory_length - max_delta_index``.
    """

    def __init__(
        self,
        dataset: "DreamZeroLeRobotDataset",
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 42,
        shard_sampling_rate: float = 0.1,
        num_steps_per_shard: int = 10_000,
        num_shards_to_sample: int = 1024,
        training: bool = True,
        allow_padding_at_end: bool = False,
        require_full_language_chunks: bool = False,
        worker_batching_mode: str = _WORKER_BATCHING_NONE,
        dataloader_num_workers: int = 0,
        micro_batch_size: int = 1,
    ) -> None:
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if not 0.0 <= shard_sampling_rate <= 1.0:
            raise ValueError(f"shard_sampling_rate must be in [0,1], got {shard_sampling_rate}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"Invalid rank {rank} for world_size {num_replicas}")

        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shard_sampling_rate = float(shard_sampling_rate)
        self.num_steps_per_shard = int(num_steps_per_shard)
        self.num_shards_to_sample = int(num_shards_to_sample)
        self.training = bool(training)
        self.allow_padding_at_end = bool(allow_padding_at_end)
        self.require_full_language_chunks = bool(require_full_language_chunks)
        self.worker_batching_mode = str(worker_batching_mode or _WORKER_BATCHING_NONE).lower()
        if self.worker_batching_mode not in {
            _WORKER_BATCHING_NONE,
            _WORKER_BATCHING_UPSTREAM_ITERABLE,
        }:
            raise ValueError(f"Unsupported worker_batching_mode={self.worker_batching_mode!r}")
        self.dataloader_num_workers = max(0, int(dataloader_num_workers))
        self.micro_batch_size = max(1, int(micro_batch_size))
        self.epoch = 0

        # Build (traj_id, step_idx) -> flat dataset index
        self._flat_index_map: dict[tuple[int, int], int] = {
            (int(t), int(s)): i for i, (t, s) in enumerate(dataset.all_steps)
        }

        # Group trajectories into shards.
        self._sharded_trajectories, self._shard_lengths = self._generate_shards()

        # Build full schedule (shared across ranks).
        self._schedule = self._generate_schedule()

        self._estimated_rank_indices = self._estimate_rank_indices()
        self._precompute_indices = self._should_precompute_indices()
        self._indices: list[int] | None = self._build_indices() if self._precompute_indices else None

        if self.rank == 0:
            logger.info(
                "[DreamZeroShardedSampler] num_shards=%d schedule_len=%d rank_indices=%d mode=%s "
                "shard_sampling_rate=%.3f num_steps_per_shard=%d seed=%d "
                "require_full_language_chunks=%s worker_batching_mode=%s "
                "dataloader_num_workers=%d micro_batch_size=%d",
                len(self._sharded_trajectories),
                len(self._schedule),
                len(self._indices) if self._indices is not None else self._estimated_rank_indices,
                "precompute" if self._indices is not None else "lazy",
                self.shard_sampling_rate,
                self.num_steps_per_shard,
                self.seed,
                self.require_full_language_chunks,
                self.worker_batching_mode,
                self.dataloader_num_workers,
                self.micro_batch_size,
            )

    # ------------------------------------------------------------------
    # Shard construction.
    # ------------------------------------------------------------------
    def _generate_shards(self) -> tuple[list[list[int]], np.ndarray]:
        ds = self.dataset
        discarded: list[int] = []
        if ds.discard_bad_trajectories:
            discarded = list(
                ds.lerobot_info_meta.get("discarded_episode_indices", []) or []
            )

        traj_ids = [int(t) for t in ds.trajectory_ids if int(t) not in discarded]
        if not traj_ids:
            raise RuntimeError("DreamZeroShardedSampler: no valid trajectories")

        total_steps = int(np.sum([len(ds.step_filter[t]) for t in traj_ids]))
        if total_steps <= 0:
            raise RuntimeError(
                "DreamZeroShardedSampler: step_filter removed all candidate steps; "
                f"dataset={ds.dataset_path} "
                f"valid_trajectories={len(traj_ids)}"
            )
        num_shards = int(np.ceil(total_steps / self.num_steps_per_shard))
        cutoffs = np.linspace(0, total_steps, num_shards + 1)[1:]

        sharded: list[list[int]] = [[]]
        shard_lengths: list[int] = []
        curr_steps = 0
        last_steps = 0
        curr_idx = 0
        for t in traj_ids:
            sharded[-1].append(t)
            curr_steps += len(ds.step_filter[t])
            if curr_idx < num_shards - 1 and curr_steps > cutoffs[curr_idx]:
                sharded.append([])
                shard_lengths.append(curr_steps - last_steps)
                last_steps = curr_steps
                curr_idx += 1
        shard_lengths.append(curr_steps - last_steps)
        assert len(sharded) == num_shards == len(shard_lengths)
        return sharded, np.array(shard_lengths, dtype=np.int64)

    # ------------------------------------------------------------------
    # Schedule generation (mirrors generate_shards_sample_schedule)
    # ------------------------------------------------------------------
    def _generate_schedule(self) -> list[int]:
        n_shards = len(self._sharded_trajectories)
        if self.training:
            rng = np.random.default_rng(self.seed + self.epoch)
            weights = self._shard_lengths.astype(np.float64)
            weights = weights / weights.sum()
            sampled = rng.choice(n_shards, size=self.num_shards_to_sample, p=weights)
            schedule = list(sampled.tolist())
            rng.shuffle(schedule)
            return schedule
        return [i % n_shards for i in range(self.num_shards_to_sample)]

    # ------------------------------------------------------------------
    # Per-rank index expansion (mirrors filter_shards_sample_schedule + __iter__)
    # ------------------------------------------------------------------
    def _filtered_schedule_count(self, num_workers: int, worker_id: int) -> int:
        modulus = self.num_replicas * int(num_workers)
        offset = self.rank * int(num_workers) + int(worker_id)
        total = len(self._schedule)
        if offset >= total:
            return 0
        return ((total - 1 - offset) // modulus) + 1

    def _estimate_rank_indices(self) -> int:
        per_shard_cap = int(self.num_steps_per_shard * self.shard_sampling_rate)
        if self._use_upstream_worker_batching():
            shard_count = sum(
                self._filtered_schedule_count(self.dataloader_num_workers, worker_id)
                for worker_id in range(self.dataloader_num_workers)
            )
        else:
            shard_count = self._filtered_schedule_count(1, 0)
        return int(shard_count * per_shard_cap)

    def _should_precompute_indices(self) -> bool:
        return self._estimated_rank_indices <= _PRECOMPUTE_INDEX_LIMIT

    def _build_indices(self) -> list[int]:
        if self._use_upstream_worker_batching():
            return self._build_worker_interleaved_indices()
        return self._build_rank_indices(num_workers=1, worker_id=0)

    def _use_upstream_worker_batching(self) -> bool:
        return (
            self.worker_batching_mode == _WORKER_BATCHING_UPSTREAM_ITERABLE
            and self.dataloader_num_workers > 1
        )

    def _build_worker_interleaved_indices(self) -> list[int]:
        """Mimic DreamZero IterableDataset multi-worker batch order.

        With ``num_workers > 1``, each worker filters the global shard schedule by
        ``rank * workers + worker_id`` and yields complete local micro-batches
        from its own iterator. The map-style trainer owns sampling in the main
        process, so this mode pre-interleaves those worker streams by
        ``micro_batch_size`` chunks.
        """
        worker_streams = [
            self._build_rank_indices(num_workers=self.dataloader_num_workers, worker_id=worker_id)
            for worker_id in range(self.dataloader_num_workers)
        ]
        cursors = [0 for _ in worker_streams]
        out: list[int] = []
        worker_id = 0
        exhausted_rounds = 0
        while exhausted_rounds < self.dataloader_num_workers:
            stream = worker_streams[worker_id]
            cursor = cursors[worker_id]
            end = cursor + self.micro_batch_size
            if end <= len(stream):
                out.extend(stream[cursor:end])
                cursors[worker_id] = end
                exhausted_rounds = 0
            else:
                exhausted_rounds += 1
            worker_id = (worker_id + 1) % self.dataloader_num_workers
        return out

    def _build_rank_indices(self, num_workers: int, worker_id: int) -> list[int]:
        return list(self._iter_rank_indices(num_workers=num_workers, worker_id=worker_id))

    def _iter_rank_indices(self, num_workers: int, worker_id: int) -> Iterator[int]:
        ds = self.dataset
        max_delta = int(ds.max_delta_index)
        per_shard_cap = int(self.num_steps_per_shard * self.shard_sampling_rate)

        # Round-robin filter to this rank/worker. ``num_workers=1`` keeps the
        # default map-style behavior.
        rng = np.random.default_rng(self.seed + self.epoch)
        modulus = self.num_replicas * int(num_workers)
        offset = self.rank * int(num_workers) + int(worker_id)
        for schedule_pos, shard_idx in enumerate(self._schedule):
            if schedule_pos % modulus != offset:
                continue
            traj_ids = self._sharded_trajectories[shard_idx]
            all_steps: list[tuple[int, int]] = []
            for traj_id in traj_ids:
                trajectory_index = ds.get_trajectory_index(traj_id)
                if self.allow_padding_at_end:
                    allowed_length = int(ds.trajectory_lengths[trajectory_index])
                else:
                    allowed_length = int(ds.trajectory_lengths[trajectory_index]) - max_delta
                allowed = ds.step_filter[traj_id]
                # Keep the dataset sampler's inclusive upper bound; changing
                # this changes the anchor set and requires a loss re-baseline.
                allowed = allowed[allowed <= allowed_length]
                for s in allowed:
                    step_idx = int(s)
                    if (
                        self.require_full_language_chunks
                        and ds.language_chunk_sampling
                        and not ds.has_full_language_chunks(int(traj_id), step_idx)
                    ):
                        continue
                    all_steps.append((int(traj_id), step_idx))
            if self.training:
                rng.shuffle(all_steps)
            sampled = all_steps[:per_shard_cap]
            for traj_id, step_idx in sampled:
                flat = self._flat_index_map.get((traj_id, step_idx))
                if flat is None:
                    # step wasn't in dataset.all_steps (e.g. discarded/clipped); skip
                    continue
                yield flat

    # ------------------------------------------------------------------
    # torch.utils.data.Sampler interface
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[int]:
        if self._indices is not None:
            return _StatefulIndexIterator(self)
        if self._use_upstream_worker_batching():
            return _LazyWorkerInterleavedIndexIterator(self)
        return _LazyShardIndexIterator(self)

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return self._estimated_rank_indices

    def set_epoch(self, epoch: int) -> None:
        """Re-seed for a new epoch (matches DistributedSampler convention)."""
        self.epoch = int(epoch)
        self._schedule = self._generate_schedule()
        self._estimated_rank_indices = self._estimate_rank_indices()
        self._precompute_indices = self._should_precompute_indices()
        self._indices = self._build_indices() if self._precompute_indices else None

    def state_dict(self) -> dict[str, Any]:
        """Return sampler state for checkpointing."""
        return {
            "version": 1,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore sampler state from a checkpoint."""
        self.set_epoch(int(state_dict.get("epoch", 0)))
