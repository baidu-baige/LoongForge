# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Server-side request batching (M3 plumbing)."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from loongforge.embodied.eval.servers.loongforge_policy import GenericPredictActionPolicy
from loongforge.embodied.eval.transport.batcher import AsyncBatcher


class _Policy:
    supports_batching = False

    def __init__(self, cached: set | None = None) -> None:
        self.batches: list[int] = []
        self.single_calls = 0
        self._cached = cached or set()

    def predict_action_batch(self, payloads):
        self.batches.append(len(payloads))
        return [{"actions": [[payload["i"]]], "inference_latency_ms": 1.0} for payload in payloads]

    def predict_action(self, **payload):
        self.single_calls += 1
        return {"actions": [[payload["i"]]], "inference_latency_ms": 1.0}

    def has_cached_action(self, payload) -> bool:
        return payload["i"] in self._cached


def _run(coro):
    return asyncio.run(coro)


def test_concurrent_requests_are_batched() -> None:
    async def scenario():
        policy = _Policy()
        batcher = AsyncBatcher(policy, max_batch_size=8, max_wait_ms=20)
        await batcher.start()
        results = await asyncio.gather(*[batcher.submit({"i": i}) for i in range(4)])
        await batcher.stop()
        return policy, batcher, results

    policy, batcher, results = _run(scenario())

    assert policy.batches == [4]
    assert [result["actions"][0][0] for result in results] == [0, 1, 2, 3]
    assert batcher.stats["requests"] == 4
    assert batcher.stats["batches"] == 1
    assert batcher.stats["mean_batch_size"] == 4.0


def test_max_batch_size_splits_requests() -> None:
    async def scenario():
        policy = _Policy()
        batcher = AsyncBatcher(policy, max_batch_size=2, max_wait_ms=20)
        await batcher.start()
        await asyncio.gather(*[batcher.submit({"i": i}) for i in range(4)])
        await batcher.stop()
        return policy

    policy = _run(scenario())

    assert sorted(policy.batches) == [2, 2]


def test_cache_hits_are_filtered_before_the_queue() -> None:
    async def scenario():
        policy = _Policy(cached={0})
        batcher = AsyncBatcher(policy, max_batch_size=8, max_wait_ms=20)
        await batcher.start()
        results = await asyncio.gather(*[batcher.submit({"i": i}) for i in range(4)])
        await batcher.stop()
        return policy, results

    policy, results = _run(scenario())

    assert policy.single_calls == 1
    assert policy.batches == [3]
    assert sorted(result["actions"][0][0] for result in results) == [0, 1, 2, 3]


class _BatchModel:
    """Emits the padded max_action_dim like pi05 and records the seeds."""

    def __init__(self) -> None:
        self.calls = 0
        self.seeds: list[list[int] | None] = []

    def predict_action(self, images, instructions, state=None, dataset_stats=None, seeds=None):
        self.calls += 1
        self.seeds.append(list(seeds) if seeds is not None else None)
        rows = np.zeros((len(images), 4, 32), dtype=np.float32)
        for index in range(len(images)):
            rows[index, :, 0] = float(index)
        return rows


def _policy_payload(index: int, **overrides):
    payload = {
        "images": [np.zeros((2, 2, 3), dtype=np.uint8)],
        "instructions": ["test"],
        "state": np.zeros(7, dtype=np.float32),
        "episode_id": f"episode-{index}",
        "episode_step": 0,
    }
    payload.update(overrides)
    return payload


def _batch_policy(model, **metadata):
    meta = {"action_dim": 7, "supports_batching": True, "noise_seed": 3}
    meta.update(metadata)
    return GenericPredictActionPolicy(model=model, metadata=meta, action_dim=7)


def test_batch_rows_are_trimmed_to_action_dim() -> None:
    """Batched rows match the sequential path's (horizon, action_dim) shape."""
    model = _BatchModel()
    policy = _batch_policy(model)

    responses = policy.predict_action_batch([_policy_payload(i) for i in range(3)])

    assert model.calls == 1
    assert model.seeds == [[3, 3, 3]]
    # Legacy cache mode returns one action per request, clipped to action_dim.
    assert [response["actions"].shape for response in responses] == [(1, 7)] * 3
    assert [response["actions"][0, 0] for response in responses] == [0.0, 1.0, 2.0]

    chunked = policy.predict_action_batch(
        [_policy_payload(i, return_action_chunk=True) for i in range(3)]
    )
    assert [response["actions"].shape for response in chunked] == [(4, 7)] * 3


def test_batch_rows_populate_the_server_cache() -> None:
    """Rows from a shared batch are replayable from the per-episode cache."""
    model = _BatchModel()
    policy = _batch_policy(model)

    policy.predict_action_batch([_policy_payload(0), _policy_payload(1)])

    assert policy.has_cached_action({"episode_id": "episode-0"}) is True
    played = policy.predict_action(
        **_policy_payload(0, episode_step=1, images=[np.zeros((2, 2, 3), dtype=np.uint8)])
    )
    assert model.calls == 1
    assert played["inference_latency_ms"] is None
    assert np.array_equal(played["actions"], np.array([[0.0] * 7], dtype=np.float32))


def test_mixed_cache_modes_fall_back_to_sequential() -> None:
    """Payloads that cannot share a batch still get correct per-request results."""
    model = _BatchModel()
    policy = _batch_policy(model)

    responses = policy.predict_action_batch(
        [_policy_payload(0), _policy_payload(1, return_action_chunk=True)]
    )

    assert model.calls == 2
    assert model.seeds == [None, None]
    # Sequential semantics: cache mode returns one row, chunk mode the whole chunk.
    assert [response["actions"].shape for response in responses] == [(1, 7), (4, 7)]


def test_batcher_logs_periodic_stats(caplog) -> None:
    """A SIGTERM'd server still leaves mean/max batch size in the log."""
    import logging

    async def scenario():
        policy = _Policy()
        batcher = AsyncBatcher(
            policy, max_batch_size=8, max_wait_ms=20, stats_interval_s=0.05
        )
        await batcher.start()
        await asyncio.gather(*[batcher.submit({"i": i}) for i in range(4)])
        await asyncio.sleep(0.12)
        await batcher.stop()
        return batcher

    caplog.set_level(logging.INFO, logger="loongforge.embodied.eval.transport.batcher")
    batcher = _run(scenario())
    assert batcher.stats["batches"] == 1
    heartbeats = [r.message for r in caplog.records if "Batcher stats:" in r.message]
    assert heartbeats, "no periodic heartbeat was logged"
    assert "mean_batch=" in heartbeats[-1] and "max_batch=" in heartbeats[-1]


def test_batch_without_noise_seed_samples_per_row() -> None:
    """No metadata seed -> the model samples fresh noise (official semantics)."""
    model = _BatchModel()
    policy = _batch_policy(model, noise_seed=None)

    responses = policy.predict_action_batch([_policy_payload(0), _policy_payload(1)])

    assert model.calls == 1
    assert model.seeds == [None]  # predict_action called without a seeds kwarg
    assert [response["actions"].shape for response in responses] == [(1, 7), (1, 7)]


class _ConcurrencyProbe:
    """Records how many model calls run at once."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self.active = 0
        self.max_active = 0

    def predict_action(self, images, instructions, state=None, dataset_stats=None):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self._guard:
            self.active -= 1
        return np.zeros((len(images), 2, 7), dtype=np.float32)


def test_concurrent_requests_never_enter_the_model_together() -> None:
    """Wan-VAE feat caches / streaming KV caches live on the model instance."""
    model = _ConcurrencyProbe()
    policy = GenericPredictActionPolicy(model=model, metadata={"action_dim": 7}, action_dim=7)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: policy.predict_action(**_policy_payload(index)), range(8)))

    assert model.max_active == 1
