# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the runner's connection-error classification (server self-healing)."""

from __future__ import annotations

from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from loongforge.embodied.eval.orchestrator.runners.libero_runner import _classify_exception


def test_dropped_websocket_is_a_connection_failure() -> None:
    """A dropped websocket must rebuild the client, not be scored as a model failure."""
    assert _classify_exception(ConnectionClosedError(None, None)) == "server_unreachable"
    assert _classify_exception(ConnectionClosedOK(None, None)) == "server_unreachable"


def test_other_failures_keep_their_classification() -> None:
    """Run test_other_failures_keep_their_classification."""
    assert _classify_exception(ConnectionRefusedError()) == "server_unreachable"
    assert _classify_exception(TimeoutError()) == "policy_timeout"
    assert _classify_exception(ValueError("boom")) == "episode_error"


def test_camera_sensors_are_swapped_only_while_disabled() -> None:
    """Intra-chunk steps must not render; the real sensor must come back intact."""
    import numpy as np

    from loongforge.embodied.eval.orchestrator.runners import libero_runner as runner

    class _Observable:
        def __init__(self, name: str) -> None:
            self._sensor = lambda cache: np.full((2, 2, 3), 7, dtype=np.uint8)
            self._data_shape = (2, 2, 3)

    class _Env:
        def __init__(self) -> None:
            self._observables = {"agentview_image": _Observable("a"), "robot0_eef_pos": _Observable("p")}

    env = _Env()
    camera = env._observables["agentview_image"]
    real = camera._sensor

    runner._set_camera_rendering(env, False)
    assert isinstance(camera._sensor, runner._CheapImage)
    assert camera._sensor({}).shape == (2, 2, 3)
    assert float(camera._sensor({}).sum()) == 0.0          # no render, but right shape
    assert env._observables["robot0_eef_pos"]._sensor is not None  # non-image untouched

    # Idempotent: disabling twice must not lose the real sensor.
    runner._set_camera_rendering(env, False)
    assert camera._sensor is not None
    runner._set_camera_rendering(env, True)
    assert camera._sensor is real
    assert camera._libero_real_sensor is None

    # Restoring an already-enabled env is a no-op.
    runner._set_camera_rendering(env, True)
    assert camera._sensor is real


def test_skip_intra_chunk_obs_is_configurable() -> None:
    """The flag must be reachable from the YAML mapping, not just the CLI."""
    from loongforge.embodied.eval.orchestrator.config import CONFIG_MAPPING

    assert CONFIG_MAPPING["benchmark.skip_intra_chunk_obs"] == "skip_intra_chunk_obs"
