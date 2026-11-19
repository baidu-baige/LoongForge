# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Client-side action chunks (T-14) and server-side cache ownership."""

from __future__ import annotations

import numpy as np

from loongforge.embodied.eval.orchestrator.runners._common import ChunkConsumer, EpisodePolicyStepper
from loongforge.embodied.eval.servers.loongforge_policy import GenericPredictActionPolicy


def test_chunk_is_consumed_locally() -> None:
    """A 10-action chunk costs one request and is replayed in order."""
    consumer = ChunkConsumer()
    chunk = np.arange(10 * 7, dtype=np.float32).reshape(10, 7)

    assert consumer.needs_request() is True
    consumer.load(chunk)
    assert consumer.needs_request() is False

    actions = [consumer.next_action() for _ in range(10)]

    assert consumer.requests == 1
    assert consumer.local_steps == 10
    assert all(action.shape == (1, 7) for action in actions)
    assert np.array_equal(np.concatenate(actions, axis=0), chunk)
    assert consumer.needs_request() is True


def test_limit_truncates_a_long_chunk() -> None:
    """Only the first N actions are executed when the server returns more."""
    consumer = ChunkConsumer()
    consumer.load(np.arange(32 * 7, dtype=np.float32).reshape(32, 7), limit=10)

    for _ in range(10):
        consumer.next_action()

    assert consumer.needs_request() is True


def test_disable_action_cache_requests_every_step() -> None:
    """Closed-loop models consume one action per request."""
    consumer = ChunkConsumer()
    consumer.load(np.arange(10 * 7, dtype=np.float32).reshape(10, 7))
    first = consumer.next_action(disable_action_cache=True)

    assert first.shape == (1, 7)
    assert consumer.needs_request(disable_action_cache=True) is True
    assert consumer.needs_request(disable_action_cache=False) is True


def test_reset_drops_the_chunk_and_counters() -> None:
    """A new episode must not reuse the previous episode's chunk."""
    consumer = ChunkConsumer()
    consumer.load(np.zeros((10, 7), dtype=np.float32))
    consumer.next_action()
    consumer.reset()

    assert consumer.requests == 0
    assert consumer.local_steps == 0
    assert consumer.needs_request() is True


class _CountingModel:
    def __init__(self) -> None:
        self.calls = 0

    def predict_action(self, images, instructions, state=None, dataset_stats=None):
        self.calls += 1
        return np.arange(10 * 2, dtype=np.float32).reshape(10, 2)


def _payload(**overrides):
    payload = {
        "images": [[np.zeros((2, 2, 3), dtype=np.uint8)]],
        "instructions": ["test"],
        "episode_id": "episode-0",
        "episode_step": 0,
    }
    payload.update(overrides)
    return payload


def test_return_action_chunk_bypasses_the_server_cache() -> None:
    """Client-owned chunks must always trigger a fresh inference."""
    model = _CountingModel()
    policy = GenericPredictActionPolicy(model=model, metadata={"action_dim": 2}, action_dim=2)

    first = policy.predict_action(**_payload(return_action_chunk=True))
    second = policy.predict_action(**_payload(return_action_chunk=True, episode_step=1))

    assert model.calls == 2
    assert np.array_equal(first["actions"], second["actions"])
    assert first["inference_latency_ms"] is not None
    assert second["inference_latency_ms"] is not None


def test_legacy_requests_still_use_the_server_cache() -> None:
    """Without return_action_chunk the existing replay behaviour is unchanged."""
    model = _CountingModel()
    policy = GenericPredictActionPolicy(model=model, metadata={"action_dim": 2}, action_dim=2)

    first = policy.predict_action(**_payload())
    assert policy.has_cached_action(_payload(episode_step=1)) is True
    second = policy.predict_action(**_payload(episode_step=1))

    assert model.calls == 1
    assert first["inference_latency_ms"] is not None
    assert second["inference_latency_ms"] is None
    assert np.array_equal(first["actions"], np.array([[0.0, 1.0]], dtype=np.float32))
    assert np.array_equal(second["actions"], np.array([[2.0, 3.0]], dtype=np.float32))


# --------------------------------------------------------------------------- #
# EpisodePolicyStepper: the shared runner-level request cadence
# --------------------------------------------------------------------------- #
class _FakePayloadBuilder:
    def __init__(self) -> None:
        self.disable_action_cache = False
        self.updates = 0

    def update_from_response(self, data) -> None:
        self.updates += 1


def _request_fn(chunks, calls, latencies=None):
    def request(index):
        calls.append(index)
        payload = {
            "actions": np.asarray(chunks[index], dtype=np.float32),
            "inference_latency_ms": 1.0 + index,
            "request_id": f"req-{index}",
        }
        latency = 2.0 + index if latencies is None else latencies[index]
        return payload, latency

    return request


def test_stepper_issues_one_request_per_chunk() -> None:
    """10-step chunks cost ceil(25 / 10) = 3 requests over 25 env steps."""
    chunks = [np.full((10, 7), float(i), dtype=np.float32) for i in range(3)]
    calls: list[int] = []
    builder = _FakePayloadBuilder()
    stepper = EpisodePolicyStepper(
        chunk_steps=10, return_action_chunk=True, payload_builder=builder
    )

    decisions = [stepper.next_action(_request_fn(chunks, calls)) for _ in range(25)]

    assert calls == [0, 1, 2]
    assert stepper.requests == 3
    # local_steps counts every action served from a chunk, the first included.
    assert stepper.local_steps == 25
    assert [d.issued_request for d in decisions] == [True] + [False] * 9 + [True] + [False] * 9 + [True] + [False] * 4
    assert [d.request_step for d in decisions if d.issued_request] == [0, 1, 2]
    # Cache hits keep the owning request id so traces can attribute the action.
    assert decisions[5].request_id == "req-0"
    assert decisions[12].request_id == "req-1"
    assert builder.updates == 3


def test_stepper_actions_match_the_legacy_cache_path() -> None:
    """Client-owned chunks execute exactly the sequence the server cache served."""
    chunk = np.arange(10 * 7, dtype=np.float32).reshape(10, 7)
    chunks = [chunk]

    client_actions = []
    stepper = EpisodePolicyStepper(chunk_steps=10, return_action_chunk=True)
    for _ in range(10):
        client_actions.append(stepper.next_action(_request_fn(chunks, [])).action)

    legacy_actions = []
    legacy = EpisodePolicyStepper(chunk_steps=10, return_action_chunk=False)
    for step in range(10):
        # The legacy server cache returns one row per call.
        legacy_actions.append(
            legacy.next_action(lambda _i, s=step: ({"actions": chunk[s : s + 1]}, 0.0)).action
        )

    assert np.array_equal(np.concatenate(client_actions, axis=0), chunk)
    assert np.array_equal(np.concatenate(client_actions, axis=0), np.concatenate(legacy_actions, axis=0))
    assert stepper.requests == 1
    assert legacy.requests == 10


def test_stepper_disable_action_cache_requests_every_step() -> None:
    """Closed-loop models see every frame: one request per step, first action only."""
    chunk = np.arange(10 * 7, dtype=np.float32).reshape(10, 7)
    calls: list[int] = []
    stepper = EpisodePolicyStepper(
        chunk_steps=10, return_action_chunk=True, disable_action_cache=True
    )

    decisions = [stepper.next_action(_request_fn([chunk] * 4, calls)) for _ in range(4)]

    assert calls == [0, 1, 2, 3]
    assert stepper.requests == 4
    assert stepper.local_steps == 0
    assert all(d.issued_request for d in decisions)
    assert np.array_equal(np.concatenate([d.action for d in decisions], axis=0), np.tile(chunk[0], (4, 1)))


def test_stepper_chunk_steps_one_is_the_ensemble_opt_out() -> None:
    """Ensembles keep the full returned chunk but still request every step."""
    chunk = np.arange(10 * 7, dtype=np.float32).reshape(10, 7)
    calls: list[int] = []
    stepper = EpisodePolicyStepper(chunk_steps=1, return_action_chunk=True)

    decisions = [stepper.next_action(_request_fn([chunk] * 3, calls)) for _ in range(3)]

    assert calls == [0, 1, 2]
    assert all(d.issued_request for d in decisions)
    assert all(d.raw_actions is not None and d.raw_actions.shape == (10, 7) for d in decisions)
    assert np.array_equal(np.concatenate([d.action for d in decisions], axis=0), np.tile(chunk[0], (3, 1)))


def test_stepper_begin_episode_drops_the_chunk() -> None:
    """A new episode must not consume the previous episode's buffered actions."""
    chunk = np.ones((10, 7), dtype=np.float32)
    calls: list[int] = []
    stepper = EpisodePolicyStepper(chunk_steps=10, return_action_chunk=True)

    first = stepper.next_action(_request_fn([chunk] * 2, calls))
    stepper.begin_episode()
    second = stepper.next_action(_request_fn([chunk] * 2, calls))

    assert first.issued_request and second.issued_request
    assert second.request_step == 0
    assert stepper.requests == 1
