# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per-benchmark shard planners."""

from __future__ import annotations

from typing import Any

import pytest

from loongforge.embodied.eval.orchestrator import planning
from loongforge.embodied.eval.orchestrator.planning import (
    benchmark_family,
    chunk_size,
    plan_calvin,
    plan_episode_indexed,
    plan_libero,
    plan_robotwin,
    plan_shards,
)


def test_plan_libero_explicit_episode_chunk() -> None:
    """A worker can run several episodes per process to amortise startup."""
    config: dict = {
        "benchmark": {
            "name": "libero",
            "suites": ["a"],
            "task_ids": [0],
            "episodes_per_task": 10,
            "episodes_per_chunk": 4,
        }
    }
    shards = plan_libero(config, 64)
    assert [(s["overrides"]["benchmark"]["episode_offset"], s["expected"]) for s in shards] == [
        (0, 4),
        (4, 4),
        (8, 2),
    ]
    assert sum(s["expected"] for s in shards) == 10


def test_plan_libero_explicit_chunk_cannot_exceed_the_task() -> None:
    """A chunk larger than the task still yields one shard covering it."""
    config: dict = {
        "benchmark": {"name": "libero", "suites": ["a"], "task_ids": [0], "episodes_per_task": 10, "episodes_per_chunk": 32}
    }
    shards = plan_libero(config, 64)
    assert len(shards) == 1
    assert shards[0]["overrides"]["benchmark"]["episode_offset"] == 0
    assert shards[0]["expected"] == 10


def test_plan_libero_per_task_ignores_explicit_chunk() -> None:
    """per_task correctness beats the startup optimisation: one task, one shard."""
    config: dict = {
        "benchmark": {
            "name": "libero",
            "suites": ["a"],
            "task_ids": [0],
            "episodes_per_task": 10,
            "episodes_per_chunk": 4,
            "env_seed_mode": "per_task",
        }
    }
    shards = plan_libero(config, 64)
    assert len(shards) == 1
    assert shards[0]["expected"] == 10


def test_plan_libero_per_task_with_snapshots_allows_chunking() -> None:
    """RNG snapshots make episode-level sharding safe under per_task."""
    config: dict = {
        "benchmark": {
            "name": "libero",
            "suites": ["a"],
            "task_ids": [0],
            "episodes_per_task": 10,
            "episodes_per_chunk": 4,
            "env_seed_mode": "per_task",
            "scene_snapshot_dir": "/tmp/scene_snapshots",
        }
    }
    shards = plan_libero(config, 64)
    assert [(s["overrides"]["benchmark"]["episode_offset"], s["expected"]) for s in shards] == [
        (0, 4),
        (4, 4),
        (8, 2),
    ]


def test_plan_shards_allows_split_with_snapshots(monkeypatch: Any) -> None:
    """The backstop must not reject a split when snapshots are configured."""

    def split_planner(config: dict, total_workers: int) -> list:
        return [
            {
                "id": f"a/task=0/eps={offset}-{offset}",
                "overrides": {
                    "benchmark": {"suite": "a", "task_ids": [0], "episode_offset": offset, "episodes_per_task": 1}
                },
                "expected": 1,
            }
            for offset in (0, 1)
        ]

    monkeypatch.setitem(planning.PLANNERS, "libero", split_planner)
    config: dict = {
        "benchmark": {
            "name": "libero",
            "env_seed_mode": "per_task",
            "scene_snapshot_dir": "/tmp/scene_snapshots",
        }
    }
    assert len(plan_shards(config, 2)) == 2


def test_benchmark_family_normalises_libero_variants() -> None:
    """Run test_benchmark_family_normalises_libero_variants."""
    assert benchmark_family({"benchmark": {"name": "libero"}}) == "libero"
    assert benchmark_family({"benchmark": {"name": "libero_10"}}) == "libero"
    assert benchmark_family({"benchmark": {"name": "robotwin"}}) == "robotwin"


def test_chunk_size_leaves_slack_per_slot() -> None:
    """Run test_chunk_size_leaves_slack_per_slot."""
    assert chunk_size(100, 4) == 7  # ceil(100 / (4 slots x 4 units))
    assert chunk_size(10, 4) == 1
    assert chunk_size(0, 4) == 1


def test_plan_libero_covers_every_episode_exactly_once() -> None:
    """Run test_plan_libero_covers_every_episode_exactly_once."""
    config: dict = {"benchmark": {"name": "libero", "suites": ["a", "b"], "task_ids": [0, 1], "episodes_per_task": 4}}
    shards = plan_libero(config, 4)
    assert len(shards) == 16  # chunk=1 so fast slots can steal work
    assert sum(shard["expected"] for shard in shards) == 16
    ids = [shard["id"] for shard in shards]
    assert len(set(ids)) == len(ids)  # unique ids -> coverage attribution is unambiguous
    assert {shard["overrides"]["benchmark"]["suite"] for shard in shards} == {"a", "b"}


def test_plan_libero_applies_suite_overrides() -> None:
    """A per-suite override (e.g. LIBERO's max_steps) rides along in the shard patch."""
    config: dict = {
        "benchmark": {
            "name": "libero",
            "suites": ["libero_10", "libero_goal"],
            "task_ids": [0],
            "episodes_per_task": 2,
            "max_steps": 400,
            "suite_overrides": {"libero_10": {"max_steps": 700}},
        }
    }
    by_suite = {shard["overrides"]["benchmark"]["suite"]: shard["overrides"]["benchmark"] for shard in plan_libero(config, 2)}
    assert by_suite["libero_10"]["max_steps"] == 700
    assert "max_steps" not in by_suite["libero_goal"]  # inherited from the base config


def test_plan_episode_indexed_is_one_episode_per_shard() -> None:
    """SimplerEnv / ManiSkill run a single episode per invocation."""
    config: dict = {
        "benchmark": {"name": "maniskill", "task_name": "PickCube-v1", "episodes_per_task": 3, "episode_offset": 5}
    }
    shards = plan_episode_indexed(config, 4)
    assert [shard["overrides"]["benchmark"]["episode_idx"] for shard in shards] == [5, 6, 7]
    assert all(shard["expected"] == 1 for shard in shards)


def test_plan_robotwin_slices_by_seed_range() -> None:
    """Run test_plan_robotwin_slices_by_seed_range."""
    config: dict = {
        "benchmark": {"name": "robotwin", "task_name": "adjust_bottle", "task_config": "demo_clean", "episodes_per_task": 100}
    }
    shards = plan_robotwin(config, 4)
    assert sum(shard["expected"] for shard in shards) == 100
    assert shards[0]["overrides"]["benchmark"]["start_seed"] == 0
    assert all("test_num" in shard["overrides"]["benchmark"] for shard in shards)
    # chunk = ceil(100 / (4 slots x 4 units)) = 7; 15 shard blocks -> stride 105
    assert all(shard["overrides"]["benchmark"]["seed_stride"] == 105 for shard in shards)


def test_plan_calvin_is_one_shard_until_the_runner_takes_an_offset() -> None:
    """CALVIN has no sequence offset yet, so it cannot be sliced further."""
    config: dict = {"benchmark": {"name": "calvin", "task_suite_name": "task_D_D", "num_sequences": 10}}
    shards = plan_calvin(config, 4)
    assert len(shards) == 1
    assert shards[0]["overrides"]["benchmark"]["num_sequences"] == 10


def test_plan_libero_per_task_keeps_one_shard_per_task() -> None:
    """per_task seeds once per process, so a task must not be split across processes."""
    config: dict = {
        "benchmark": {
            "name": "libero",
            "suites": ["a", "b"],
            "task_ids": [0, 1],
            "episodes_per_task": 4,
            "env_seed_mode": "per_task",
        }
    }
    shards = plan_libero(config, 32)  # many workers must not induce intra-task chunks
    assert len(shards) == 4
    assert all(shard["overrides"]["benchmark"]["episode_offset"] == 0 for shard in shards)
    assert all(shard["overrides"]["benchmark"]["episodes_per_task"] == 4 for shard in shards)
    assert sum(shard["expected"] for shard in shards) == 16


def test_plan_libero_per_episode_still_splits() -> None:
    """The default per_episode seeding keeps the steal-work slicing."""
    config: dict = {"benchmark": {"name": "libero", "suites": ["a"], "task_ids": [0], "episodes_per_task": 16}}
    shards = plan_libero(config, 1)
    assert len(shards) > 1
    assert any(shard["overrides"]["benchmark"]["episode_offset"] > 0 for shard in shards)


def test_plan_libero_per_task_suite_override_is_honoured() -> None:
    """A suite-level per_task override gets the same one-shard-per-task treatment."""
    config: dict = {
        "benchmark": {
            "name": "libero",
            "suites": ["libero_10", "libero_goal"],
            "task_ids": [0],
            "episodes_per_task": 4,
            "suite_overrides": {"libero_10": {"env_seed_mode": "per_task"}},
        }
    }
    shards = plan_libero(config, 32)
    per_task = [s for s in shards if s["overrides"]["benchmark"]["suite"] == "libero_10"]
    per_episode = [s for s in shards if s["overrides"]["benchmark"]["suite"] == "libero_goal"]
    assert len(per_task) == 1
    assert len(per_episode) > 1


def test_plan_shards_rejects_intra_task_split_for_per_task(monkeypatch: Any) -> None:
    """Backstop: a planner that splits a task under per_task must fail loudly."""

    def split_planner(config: dict, total_workers: int) -> list:
        return [
            {
                "id": f"a/task=0/eps={offset}-{offset}",
                "overrides": {
                    "benchmark": {
                        "suite": "a",
                        "task_ids": [0],
                        "episode_offset": offset,
                        "episodes_per_task": 1,
                        "env_seed_mode": "per_task",
                    }
                },
                "expected": 1,
            }
            for offset in (0, 1)
        ]

    monkeypatch.setitem(planning.PLANNERS, "libero", split_planner)
    config: dict = {"benchmark": {"name": "libero", "env_seed_mode": "per_task"}}
    with pytest.raises(ValueError, match="cannot split a task"):
        plan_shards(config, 2)


def test_plan_shards_rejects_duplicate_task_for_per_task(monkeypatch: Any) -> None:
    """Two shards with the same task (offset 0) are still a split under per_task."""

    def duplicate_planner(config: dict, total_workers: int) -> list:
        return [
            {
                "id": f"a/task=0/copy{i}",
                "overrides": {
                    "benchmark": {"suite": "a", "task_ids": [0], "episode_offset": 0, "episodes_per_task": 4}
                },
                "expected": 4,
            }
            for i in range(2)
        ]

    monkeypatch.setitem(planning.PLANNERS, "libero", duplicate_planner)
    config: dict = {"benchmark": {"name": "libero", "env_seed_mode": "per_task"}}
    with pytest.raises(ValueError, match="cannot split a task"):
        plan_shards(config, 2)


def test_plan_shards_dispatches_on_the_family(monkeypatch: Any) -> None:
    """Run test_plan_shards_dispatches_on_the_family."""
    libero = plan_shards({"benchmark": {"name": "libero", "suite": "a", "task_ids": [0], "episodes_per_task": 1}}, 1)
    maniskill = plan_shards({"benchmark": {"name": "maniskill", "task_name": "t", "episodes_per_task": 1}}, 1)
    assert libero[0]["overrides"]["benchmark"]["suite"] == "a"
    assert maniskill[0]["overrides"]["benchmark"]["task_name"] == "t"


def test_plan_robotwin_fans_out_over_tasks() -> None:
    """benchmark.tasks gives every task its own seed range and stride."""
    config: dict = {
        "benchmark": {
            "name": "robotwin",
            "tasks": ["adjust_bottle", "click_bell"],
            "task_config": "demo_clean",
            "episodes_per_task": 5,
        }
    }
    shards = plan_robotwin(config, 32)
    assert len(shards) == 10  # 2 tasks x 5 single-episode units
    assert {shard["overrides"]["benchmark"]["task_name"] for shard in shards} == {
        "adjust_bottle",
        "click_bell",
    }
    for task in ("adjust_bottle", "click_bell"):
        seeds = sorted(
            shard["overrides"]["benchmark"]["start_seed"]
            for shard in shards
            if shard["overrides"]["benchmark"]["task_name"] == task
        )
        assert seeds == [0, 1, 2, 3, 4]
        assert all(
            shard["overrides"]["benchmark"]["seed_stride"] == 5
            for shard in shards
            if shard["overrides"]["benchmark"]["task_name"] == task
        )
    assert all(shard["expected"] == 1 for shard in shards)


def test_plan_robotwin_tasks_all_reads_the_official_task_set(tmp_path) -> None:
    """`tasks: all` expands to the official per-task step-limit file."""
    step_limits = tmp_path / "task_config" / "_eval_step_limit.yml"
    step_limits.parent.mkdir(parents=True)
    step_limits.write_text("adjust_bottle: 400\nclick_bell: 400\n", encoding="utf-8")
    config: dict = {
        "benchmark": {
            "name": "robotwin",
            "tasks": "all",
            "episodes_per_task": 2,
            "episodes_per_chunk": 2,
        },
        "env": {"robotwin_root": str(tmp_path)},
    }
    shards = plan_robotwin(config, 8)
    assert [shard["overrides"]["benchmark"]["task_name"] for shard in shards] == [
        "adjust_bottle",
        "click_bell",
    ]
    assert all(shard["expected"] == 2 for shard in shards)
    # One unit per task: an unstable seed advances to the next one (+1), as in the
    # legacy single-process run; a multi-unit plan uses the full block stride.
    assert all(shard["overrides"]["benchmark"]["seed_stride"] == 1 for shard in shards)


def test_plan_robotwin_rejects_tasks_all_without_a_root() -> None:
    """`all` cannot be expanded without env.robotwin_root."""
    config: dict = {"benchmark": {"name": "robotwin", "tasks": "all"}}
    with pytest.raises(ValueError):
        plan_robotwin(config, 4)


def test_plan_robotwin_order_modes(tmp_path) -> None:
    """task (default) is task-major; interleave mixes; lpt starts the longest task."""
    step_limits = tmp_path / "task_config" / "_eval_step_limit.yml"
    step_limits.parent.mkdir(parents=True)
    step_limits.write_text("fast_task: 400\nslow_task: 1500\n", encoding="utf-8")

    def plan(order: str):
        config: dict = {
            "benchmark": {
                "name": "robotwin",
                "tasks": ["fast_task", "slow_task"],
                "episodes_per_task": 4,
                "episodes_per_chunk": 1,
                "plan_order": order,
            },
            "env": {"robotwin_root": str(tmp_path)},
        }
        return [s["overrides"]["benchmark"]["task_name"] for s in plan_robotwin(config, 8)]

    assert plan("task") == ["fast_task"] * 4 + ["slow_task"] * 4
    assert plan("interleave") == ["fast_task", "slow_task"] * 4
    assert plan("lpt") == ["slow_task", "fast_task"] * 4  # longest first, then mixed

    config = {
        "benchmark": {"name": "robotwin", "task_name": "a", "plan_order": "bogus"},
    }
    with pytest.raises(ValueError):
        plan_robotwin(config, 4)

def test_plan_robotwin_lpt_long_fronts_long_tasks_with_a_finer_chunk(tmp_path, capsys) -> None:
    """lpt_long: every shard of a long task precedes all short work, and long
    tasks use their own (finer) chunk so the last unit's tail stays short."""
    limits = {
        "long_1700": 1700,
        "long_1500": 1500,
        "long_1200a": 1200,
        "long_1200b": 1200,
        "long_1200c": 1200,
        "short_a": 400,
        "short_b": 400,
        "short_c": 400,
        "short_d": 400,
        "short_e": 400,
    }
    step_limits = tmp_path / "task_config" / "_eval_step_limit.yml"
    step_limits.parent.mkdir(parents=True)
    step_limits.write_text("\n".join(f"{k}: {v}" for k, v in limits.items()) + "\n", encoding="utf-8")
    config: dict = {
        "benchmark": {
            "name": "robotwin",
            "tasks": list(limits),
            "episodes_per_task": 100,
            "episodes_per_chunk": 25,
            "long_task_steps": 900,
            "long_task_episodes_per_chunk": 10,
            "plan_order": "lpt_long",
        },
        "env": {"robotwin_root": str(tmp_path)},
    }
    queue = [
        (
            s["overrides"]["benchmark"]["task_name"],
            s["overrides"]["benchmark"]["start_seed"],
            s["overrides"]["benchmark"]["test_num"],
            s["overrides"]["benchmark"]["seed_stride"],
        )
        for s in plan_robotwin(config, 8)
    ]

    print("lpt_long queue:")
    for index, (task, seed, count, stride) in enumerate(queue):
        print(f"  [{index:2d}] {task:12s} seed={seed:3d} n={count:2d} stride={stride}")

    long_names = {name for name, limit in limits.items() if limit >= 900}
    first_short = next(i for i, (task, *_rest) in enumerate(queue) if task not in long_names)
    assert all(task in long_names for task, *_rest in queue[:first_short])
    assert all(task not in long_names for task, *_rest in queue[first_short:])

    # Round-robin inside the head: shards 0-4 are chunk 0 of every long task,
    # shards 5-9 are chunk 1, so a wave never stacks one heavy scene.
    assert {task for task, *_rest in queue[:5]} == long_names
    assert [seed for _task, seed, *_rest in queue[:5]] == [0] * 5
    assert [seed for _task, seed, *_rest in queue[5:10]] == [10] * 5

    # Long tasks: 10 shards x 10 episodes; short tasks keep 4 x 25.
    for task in long_names:
        own = [(seed, count, stride) for name, seed, count, stride in queue if name == task]
        assert [count for _seed, count, _stride in own] == [10] * 10
        assert [seed for seed, _count, _stride in own] == list(range(0, 100, 10))
        assert {stride for _seed, _count, stride in own} == {100}
    for task in set(limits) - long_names:
        own = [(seed, count, stride) for name, seed, count, stride in queue if name == task]
        assert [count for _seed, count, _stride in own] == [25] * 4
        assert [seed for seed, _count, _stride in own] == [0, 25, 50, 75]
        assert {stride for _seed, _count, stride in own} == {100}

    # Coverage is unaffected: every episode index appears exactly once per task.
    for task in limits:
        seen = [i for name, seed, count, _stride in queue if name == task for i in range(seed, seed + count)]
        assert sorted(seen) == list(range(100))

    assert "lpt_long queue:" in capsys.readouterr().out


def test_plan_robotwin_lpt_long_without_threshold_matches_interleave() -> None:
    """The new knobs are opt-in: with no threshold, lpt_long is a plain interleave."""
    base: dict = {
        "benchmark": {
            "name": "robotwin",
            "tasks": ["b", "a"],
            "episodes_per_task": 4,
            "episodes_per_chunk": 1,
        },
        "env": {"robotwin_root": "/nonexistent"},
    }

    def plan(order: str):
        config = {**base, "benchmark": {**base["benchmark"], "plan_order": order}}
        return [
            (s["overrides"]["benchmark"]["task_name"], s["overrides"]["benchmark"]["start_seed"])
            for s in plan_robotwin(config, 8)
        ]

    assert plan("lpt_long") == plan("interleave")


def test_plan_robotwin_scan_shards_interleave_candidates() -> None:
    """Sharded scan: k shards per task, residue classes start+i / stride=k."""
    from loongforge.embodied.eval.orchestrator import planning

    config = {
        "benchmark": {
            "name": "robotwin",
            "task_name": "adjust_bottle",
            "task_config": "demo_clean",
            "start_seed": 1000,
            "episodes_per_task": 10,
            "offline_expert_check_scan": "/tmp/cache",
            "offline_expert_check_scan_shards": 4,
        },
        "env": {"robotwin_root": "/nonexistent"},
    }
    shards = planning.plan_robotwin(config, total_workers=64)
    assert len(shards) == 4
    seen = []
    for i, shard in enumerate(shards):
        b = shard["overrides"]["benchmark"]
        assert b["offline_expert_check_part"] == i
        assert b["start_seed"] == 1000 + i
        assert b["seed_stride"] == 4
        assert b["test_num"] == 3          # ceil(10 / 4)
        assert shard["expected"] == 3
        seen.append(b["start_seed"])
    assert seen == [1000, 1001, 1002, 1003]


def test_plan_robotwin_consume_uses_cache_index_offset() -> None:
    """Consume mode indexes the cache; a leftover start_seed must not shift it."""
    from loongforge.embodied.eval.orchestrator import planning

    config = {
        "benchmark": {
            "name": "robotwin",
            "task_name": "adjust_bottle",
            "task_config": "demo_clean",
            "start_seed": 100000,          # a seed value; irrelevant for consume mode
            "episodes_per_task": 8,
            "episodes_per_chunk": 2,
            "offline_expert_check": "/tmp/cache",
        },
        "env": {"robotwin_root": "/nonexistent"},
    }
    shards = planning.plan_robotwin(config, total_workers=4)
    offsets = []
    for shard in shards:
        b = shard["overrides"]["benchmark"]
        assert b["test_num"] == 2
        offsets.append(b["offline_expert_check_offset"])
    assert offsets == [0, 2, 4, 6]
