# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""In-process LIBERO embodiment.

Scenes follow the LoongForge LIBERO eval protocol: the official init states,
``num_steps_wait`` no-op settling steps (not counted in the budget), and the
suite step limit from ``eval.adapters.libero.SUITE_MAX_STEPS``. Images are
returned upright (the simulator buffer flipped vertically), with metric depth
and OpenCV-convention calibration on request.
"""

from __future__ import annotations

import contextlib
import functools
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from loongforge.embodied.agent.core.types import CameraFrame, EpisodeSpec, MoveResult, Obs
from loongforge.embodied.agent.embodiments.libero.controller import MotionController
from loongforge.embodied.eval.adapters.libero import LIBERO_DUMMY_ACTION, SUITE_MAX_STEPS

CAMERA_KEYS = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}
_FLIP_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])  # MuJoCo camera axes -> OpenCV (x right, y down, z forward)

ENVIRONMENT_NOTES = (Path(__file__).with_name("environment.md")).read_text().strip()


@contextlib.contextmanager
def _permissive_torch_load():
    """LIBERO's init-state files predate ``torch.load(weights_only=True)``."""
    import torch

    original = torch.load

    @functools.wraps(original)
    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original


class LiberoEmbodiment:
    """LIBERO's Panda, one scene per task, kept across episodes of that task."""

    name = "libero"
    cameras = ("agentview", "wrist")
    environment_notes = ENVIRONMENT_NOTES

    def __init__(self, resolution: int = 512, seed: int = 7, num_steps_wait: int = 10) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        self.resolution = resolution
        self.seed = seed
        self.num_steps_wait = num_steps_wait
        self._benchmarks: Dict[str, Any] = {}
        self._scene: Any = None
        self._scene_key: Optional[Tuple[str, int]] = None
        self._raw: Dict[str, np.ndarray] = {}
        self._instruction = ""
        self._steps = 0
        self._limit = 0
        self._success = False

    # -- episode -----------------------------------------------------------
    def reset(self, spec: EpisodeSpec) -> Obs:
        """Reset to ``spec``'s init state and settle, as the LoongForge LIBERO runner does."""
        if spec.suite not in SUITE_MAX_STEPS:
            raise ValueError(f"unknown LIBERO suite {spec.suite!r}; choose one of {sorted(SUITE_MAX_STEPS)}")
        benchmark = self._benchmark(spec.suite)
        with _permissive_torch_load():
            init_states = benchmark.get_task_init_states(spec.task)
        if not 0 <= spec.init_idx < len(init_states):
            raise ValueError(f"init_idx {spec.init_idx} out of range 0..{len(init_states) - 1}")
        scene = self._open(spec.suite, spec.task)
        scene.reset()
        scene.set_init_state(init_states[spec.init_idx])
        with self._no_render():
            for _ in range(self.num_steps_wait):
                scene.step(LIBERO_DUMMY_ACTION)
        self._raw = scene.env._get_observations(force_update=True)
        self._instruction = str(benchmark.get_task(spec.task).language)
        self._steps = 0
        self._limit = SUITE_MAX_STEPS[spec.suite]
        self._success = False
        return self.observe()

    def observe(self, *, depth: bool = False) -> Obs:
        """Current observation; ``depth`` adds metric depth and calibration."""
        raw = self._raw
        frames = {name: self._frame(raw, name, depth) for name in self.cameras}
        pos, quat, gap = MotionController.tool_state(raw)
        return Obs(
            instruction=self._instruction,
            cameras=frames,
            eef_pos=pos,
            eef_quat=quat,
            gripper_gap=gap,
            success=self._success,
            steps=self._steps,
            step_limit=self._limit,
        )

    def move(
        self, pos: np.ndarray, quat: np.ndarray, opening: Optional[float], max_steps: Optional[int] = None
    ) -> MoveResult:
        """Closed-loop motion to a world pose; frames are rendered once, after it ends."""
        remaining = self._limit - self._steps
        if self._success or remaining <= 0:
            return MoveResult("success" if self._success else "budget", 0)
        budget = remaining if max_steps is None else min(remaining, max_steps)
        controller = MotionController(self._scene, self._raw, pos, quat, opening)
        with self._no_render():
            raw, actions, success, outcome = controller.move(budget)
        self._steps += len(actions)
        self._success = self._success or success
        self._raw = self._scene.env._get_observations(force_update=True)
        if outcome == "budget" and self._steps < self._limit:
            outcome = "timeout"  # the per-call cap, not the episode budget, ran out
        return MoveResult(outcome, len(actions))

    def step(self, action) -> bool:
        """One native 7-D action (OSC delta + gripper); returns the benchmark's success."""
        if self._steps >= self._limit:
            return self._success
        self._raw, _, done, _ = self._scene.step(np.asarray(action, dtype=np.float64))
        self._steps += 1
        self._success = self._success or bool(done)
        return self._success

    def base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """World position and rotation of ``robot0_base``."""
        sim = self._scene.sim
        body = sim.model.body_name2id("robot0_base")
        return sim.data.body_xpos[body].copy(), sim.data.body_xmat[body].reshape(3, 3).copy()

    def close(self) -> None:
        """Close the cached scene."""
        if self._scene is not None:
            self._scene.close()
        self._scene, self._scene_key = None, None

    # -- internals ----------------------------------------------------------
    def _benchmark(self, suite: str):
        if suite not in self._benchmarks:
            from libero.libero.benchmark import get_benchmark_dict

            self._benchmarks[suite] = get_benchmark_dict()[suite]()
        return self._benchmarks[suite]

    def _open(self, suite: str, task: int):
        if self._scene_key == (suite, task):
            return self._scene
        self.close()
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        meta = self._benchmark(suite).get_task(task)
        bddl = Path(get_libero_path("bddl_files")) / meta.problem_folder / meta.bddl_file
        self._scene = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=self.resolution,
            camera_widths=self.resolution,
            camera_depths=True,
        )
        self._scene.seed(self.seed)
        self._scene_key = (suite, task)
        return self._scene

    def _frame(self, raw: Dict[str, np.ndarray], camera: str, depth: bool) -> CameraFrame:
        key = CAMERA_KEYS[camera]
        rgb = np.ascontiguousarray(raw[f"{key}_image"][::-1])
        if not depth:
            return CameraFrame(rgb)
        from robosuite.utils import camera_utils

        sim = self._scene.sim
        metric = camera_utils.get_real_depth_map(sim, raw[f"{key}_depth"])
        metric = np.ascontiguousarray(np.asarray(metric, dtype=np.float32).reshape(rgb.shape[:2])[::-1])
        height, width = rgb.shape[:2]
        intrinsic = camera_utils.get_camera_intrinsic_matrix(sim, key, height, width)
        cam_id = sim.model.camera_name2id(key)
        cam2world = np.eye(4)
        cam2world[:3, :3] = sim.data.cam_xmat[cam_id].reshape(3, 3)
        cam2world[:3, 3] = sim.data.cam_xpos[cam_id]
        return CameraFrame(rgb, metric, np.asarray(intrinsic, dtype=np.float64), cam2world @ _FLIP_TO_OPENCV)

    @contextlib.contextmanager
    def _no_render(self):
        """Skip camera renders during controller steps; proprioception stays current."""
        sim = self._scene.sim
        render = sim.render
        blank: Dict[Tuple[int, int, bool], Any] = {}

        def no_render(width=None, height=None, *, camera_name=None, depth=False, **_):
            key = (height or 1, width or 1, bool(depth))
            if key not in blank:
                frame = np.zeros((key[0], key[1], 3), dtype=np.uint8)
                blank[key] = (frame, np.zeros(key[:2], dtype=np.float32)) if depth else frame
            return blank[key]

        sim.render = no_render
        try:
            yield
        finally:
            sim.render = render
