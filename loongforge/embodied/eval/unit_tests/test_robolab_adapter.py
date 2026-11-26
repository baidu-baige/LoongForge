# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the RoboLab adapter, planner and Cosmos3 image packing."""

from __future__ import annotations

import numpy as np
import pytest

from loongforge.embodied.eval.adapters.robolab import ROBOLAB_ACTION_DIM, RobolabAdapter


def _obs(env_id: int = 0) -> dict:
    """Build a two-env RoboLab observation dict (nested groups of batched tensors)."""
    return {
        "image_obs": {
            "wrist_cam": np.zeros((2, 8, 16, 3), dtype=np.uint8),
            "over_shoulder_left_camera": np.full((2, 8, 16, 3), 10, dtype=np.uint8),
            "over_shoulder_right_camera": np.full((2, 8, 16, 3), 20, dtype=np.uint8),
            "head_camera": np.full((2, 8, 16, 3), 30, dtype=np.uint8),
        },
        "proprio_obs": {
            "arm_joint_pos": np.arange(14, dtype=np.float32).reshape(2, 7),
            "gripper_pos": np.asarray([[0.0], [1.0]], dtype=np.float32),
        },
    }


def test_obs_to_canonical_selects_one_env() -> None:
    """Proprio/image terms are batched; the adapter must return the requested env."""
    adapter = RobolabAdapter(task_name="BananaInBowlTask")
    canonical = adapter.obs_to_canonical(_obs(), {"instruction": "pick the banana", "env_id": 1})

    assert canonical["instruction"] == "pick the banana"
    assert canonical["state_raw"]["arm_joint_pos"].tolist() == [7, 8, 9, 10, 11, 12, 13]
    # DROID layout: 7 arm joints then the gripper.
    assert canonical["state_raw"]["joint"].shape == (ROBOLAB_ACTION_DIM,)
    assert canonical["state_raw"]["joint"][-1] == pytest.approx(1.0)
    assert canonical["images"]["wrist"].shape == (8, 16, 3)
    assert int(canonical["images"]["left"][0, 0, 0]) == 10
    assert canonical["images"]["primary"] is canonical["images"]["wrist"]
    assert canonical["meta"]["benchmark"] == "robolab"
    assert canonical["meta"]["control_hz"] == 15


def test_obs_to_canonical_requires_proprio() -> None:
    """A scene without proprio terms must fail loudly, not silently score 0."""
    adapter = RobolabAdapter(task_name="BananaInBowlTask")
    with pytest.raises(KeyError):
        adapter.obs_to_canonical({"image_obs": {}}, {})


def test_action_roundtrip() -> None:
    """Canonical actions are absolute joint targets, passed through unchanged."""
    adapter = RobolabAdapter(task_name="BananaInBowlTask")
    action = np.arange(ROBOLAB_ACTION_DIM, dtype=np.float32)
    env_action = adapter.action_from_canonical({"actions": action})
    assert env_action.dtype == np.float32
    assert env_action.tolist() == action.tolist()


def test_action_rejects_wrong_dim() -> None:
    """A 7D (gripper-less) model action must not be sent to the 8D controller."""
    adapter = RobolabAdapter(task_name="BananaInBowlTask")
    with pytest.raises(ValueError):
        adapter.action_from_canonical({"actions": np.zeros(7, dtype=np.float32)})


def test_plan_robolab_one_shard_per_task() -> None:
    """A task is one shard; episodes_per_task is split into num_envs x num_runs."""
    from loongforge.embodied.eval.orchestrator import planning

    config = {
        "benchmark": {"name": "robolab", "tasks": ["A", "B"], "num_envs": 5, "episodes_per_task": 10},
        "run": {"seed": 7},
    }
    shards = planning.plan_shards(config, total_workers=4)

    assert planning.benchmark_family(config) == "robolab"
    assert [shard["id"] for shard in shards] == ["A/seed=7", "B/seed=7"]
    assert [shard["expected"] for shard in shards] == [10, 10]
    assert shards[0]["overrides"]["benchmark"] == {
        "task_name": "A",
        "seed": 7,
        "num_envs": 5,
        "num_runs": 2,
    }


def test_plan_robolab_rejects_partial_run() -> None:
    """episodes_per_task must divide evenly into num_envs batches."""
    from loongforge.embodied.eval.orchestrator import planning

    config = {"benchmark": {"name": "robolab", "task_name": "A", "num_envs": 4, "episodes_per_task": 10}}
    with pytest.raises(ValueError):
        planning.plan_shards(config, 2)


def test_plan_robolab_honours_episode_offset() -> None:
    """episode_offset shifts the seed window (a different scene sequence)."""
    from loongforge.embodied.eval.orchestrator import planning

    config = {
        "benchmark": {"name": "robolab", "task_name": "A", "num_envs": 2, "episode_offset": 3},
        "run": {"seed": 0},
    }
    shards = planning.plan_shards(config, 2)
    assert shards[0]["overrides"]["benchmark"]["seed"] == 3
    assert shards[0]["expected"] == 2  # default episodes_per_task = num_envs


def test_cosmos3_image_mosaic_shape() -> None:
    """Cosmos3 packing is wrist (360x640) over left|right (180x320 each) -> 540x640."""
    from loongforge.embodied.eval.payload_builders.cosmos3 import Cosmos3PayloadBuilder

    builder = Cosmos3PayloadBuilder()
    observations = {
        "wrist": np.zeros((720, 1280, 3), dtype=np.uint8),
        "left": np.zeros((720, 1280, 3), dtype=np.uint8),
        "right": np.zeros((720, 1280, 3), dtype=np.uint8),
        "primary": np.zeros((720, 1280, 3), dtype=np.uint8),
        "head": None,
    }
    payload = builder.build(
        {
            "instruction": "pick the banana",
            "images": observations,
            "state_raw": {"joint": np.zeros(ROBOLAB_ACTION_DIM, dtype=np.float32)},
        },
        {},
    )

    assert builder.state_encoding == "joint_gripper"
    assert builder.action_encoding == "joint_pos"  # matches the adapter -> identity decoder
    assert payload["images"][0].shape == (540, 640, 3)
    assert payload["images"][0].dtype == np.uint8
    assert payload["instructions"] == ["pick the banana"]
    assert payload["state"].shape == (ROBOLAB_ACTION_DIM,)
