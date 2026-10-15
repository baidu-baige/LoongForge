"""Batched RPC regression tests for GenericPredictActionPolicy."""

from __future__ import annotations

import numpy as np
import pytest

from loongforge.embodied.eval.servers.loongforge_policy import GenericPredictActionPolicy
from loongforge.embodied.eval.servers.predict_action_interface import call_predict_action


class _Model:
    def predict_action(self, images, instructions, state=None, dataset_stats=None):
        batch = len(images)
        assert len(instructions) == batch
        assert np.asarray(state).shape == (batch, 8)
        return np.arange(batch * 4 * 7, dtype=np.float32).reshape(batch, 4, 7)


def _policy():
    return GenericPredictActionPolicy(
        _Model(), metadata={"action_dim": 7}, action_dim=7
    )


def test_batch_request_preserves_batch_and_returns_first_step():
    result = _policy().predict_action(
        images=[[np.zeros((2, 2, 3), dtype=np.uint8)] for _ in range(3)],
        instructions=["a", "bb", "ccc"],
        state=np.zeros((3, 8), dtype=np.float32),
        batch_request=True,
        disable_action_cache=True,
    )
    assert result["actions"].shape == (3, 1, 7)
    np.testing.assert_array_equal(
        result["actions"][:, 0],
        np.arange(3 * 4 * 7, dtype=np.float32).reshape(3, 4, 7)[:, 0],
    )


def test_batch_request_rejects_cache_and_shape_mismatch():
    with pytest.raises(ValueError, match="disable_action_cache"):
        _policy().predict_action(
            images=[[np.zeros((2, 2, 3), dtype=np.uint8)]],
            instructions=["a"],
            state=np.zeros((1, 8), dtype=np.float32),
            batch_request=True,
            disable_action_cache=False,
        )


def test_call_predict_action_keeps_legacy_flattening_by_default():
    actions = call_predict_action(
        _Model(),
        images=[[np.zeros((2, 2, 3), dtype=np.uint8)] for _ in range(2)],
        instructions=["a", "b"],
        state=np.zeros((2, 8), dtype=np.float32),
        dataset_stats=None,
        action_dim=7,
    )
    assert actions.shape == (8, 7)
    with pytest.raises(ValueError, match="batched state"):
        _policy().predict_action(
            images=[[np.zeros((2, 2, 3), dtype=np.uint8)] for _ in range(2)],
            instructions=["a", "b"],
            state=np.zeros((1, 8), dtype=np.float32),
            batch_request=True,
            disable_action_cache=True,
        )
