# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DROID LeRobot v3.0 dataset for Cosmos3 action-policy SFT.

Minimal port of cosmos-framework's ``DROIDLeRobotDataset`` with the
``action_space="joint_pos"`` + ``use_state=True`` recipe (8D action, raw
joint values, no normalization, ``concat_view`` 3-camera layout).

* Image features use the plural ``observation.images.*`` keys
  (``wrist_left`` / ``exterior_1_left`` / ``exterior_2_left``).
* ``meta/tasks.parquet`` stores task strings as the pandas index column
  (``__index_level_0__``) instead of an explicit ``"task"`` column.

Only the parts needed for the DROID smoke training are kept; ``ee_pose``,
quantile normalization, image augmentation, and keep-range filtering are
intentionally omitted.

Each ``__getitem__`` returns a raw sample dict::

    {
        "video":             Tensor [3, T, H, W] uint8 (concat_view),
        "action":            Tensor [chunk+1, 8] float32,
        "ai_caption":        str,
        "conditioning_fps":  Tensor[long] scalar,
        "mode":              "policy",
        "domain_id":         Tensor[long] scalar,
        "viewpoint":         "concat_view",
        "idle_frames":       Tensor[long] scalar,
    }

A separate :mod:`cosmos3_action_transform` is responsible for padding
``action`` to ``max_action_dim``, building the ``sequence_plan``, and
tokenizing the caption.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from lerobot.datasets.video_utils import decode_video_frames
from torch.utils.data import Dataset


_IMAGE_FEATURES: Dict[str, str] = {
    "wrist": "observation.image.wrist_image_left",
    "left": "observation.image.exterior_image_1_left",
    "right": "observation.image.exterior_image_2_left",
}

_JOINT_ACTION_FEATURE = "action.joint_position"           # [7] commanded joints
_ACTION_GRIPPER_FEATURE = "action.gripper_position"       # [1] commanded gripper
_JOINT_STATE_FEATURE = "observation.state.joint_positions"   # [7] observed joints
_GRIPPER_STATE_FEATURE = "observation.state.gripper_position"  # [1] observed gripper

_LIST_COLUMNS = {_JOINT_ACTION_FEATURE, _JOINT_STATE_FEATURE}

_CONCAT_VIEW_DESCRIPTION = (
    "The top row is from the wrist-mounted camera. "
    "The bottom row contains two horizontally concatenated third-person perspective views of the scene from opposite "
    "sides, with the robot visible."
)


# Verbatim copy of cosmos_framework/data/vfm/action/domain_utils.py's
# ``EMBODIMENT_TO_DOMAIN_ID`` registry. Keeping it inline avoids depending on
# cosmos at import time and keeps the mapping deterministic across runs (the
# previous ``hash(name) % 32`` implementation was non-deterministic because
# Python's str hash is randomized via ``PYTHONHASHSEED``).
_EMBODIMENT_TO_DOMAIN_ID: Dict[str, int] = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "fractal": 20,
}


def _domain_id_from_name(name: str) -> int:
    """Stable domain id for embodiment-aware action heads.

    Mirrors cosmos's ``get_domain_id`` (``cosmos_framework.data.vfm.action.
    domain_utils``): the embodiment name is normalized via ``lower().strip()``
    and looked up in the inline registry above. Returning a fixed integer
    keeps the value deterministic across processes (Python's built-in
    ``hash`` is salted by ``PYTHONHASHSEED``).
    """
    key = name.lower().strip()
    if key not in _EMBODIMENT_TO_DOMAIN_ID:
        raise KeyError(
            f"Unknown embodiment type: {name!r}. "
            f"Available embodiments: {sorted(_EMBODIMENT_TO_DOMAIN_ID.keys())}"
        )
    return _EMBODIMENT_TO_DOMAIN_ID[key]


def _consecutive_streaks(idle: np.ndarray, min_streak: int) -> np.ndarray:
    """Zero out idle bits not belonging to a run of ``>= min_streak`` Trues.

    Verbatim port of cosmos's ``_consecutive_streaks``
    (``cosmos_framework.data.vfm.action.pose_utils``). ``min_streak <= 1``
    is a no-op.
    """
    if min_streak <= 1:
        return idle
    out = np.zeros_like(idle)
    n = len(idle)
    i = 0
    while i < n:
        if not idle[i]:
            i += 1
            continue
        j = i
        while j < n and idle[j]:
            j += 1
        if j - i >= min_streak:
            out[i:j] = True
        i = j
    return out


def _compute_idle_frames(
    action: torch.Tensor,
    fps: float,
    eps_g: float = 1e-2,
    min_streak: int = 3,
) -> int:
    """Count idle frames for the joint_pos action layout (7 JOINT + 1 GRIPPER).

    Specialization of cosmos's ``compute_idle_frames`` (``pose_utils``) for
    the ``build_action_spec(Joint(n=7), Gripper())`` spec used by the
    DROID joint_pos recipe:

    * JOINT (dims 0..6) — frame-wise diff with ``prepend=action[:1]`` (step 0
      diff is zero), idle iff ``max |Δjoint| < joint_threshold = 5e-3 / fps``.
    * GRIPPER (dim 7) — same scheme with ``eps_g = 1e-2``.
    * A frame is idle iff BOTH groups are idle.
    * The boolean idle mask is then passed through ``_consecutive_streaks``
      with ``min_streak = 3`` to discard isolated low-motion frames.

    Returns ``int(idle.sum())`` after the streak filter.
    """
    arr = action.detach().cpu().numpy().astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"action must be 2-D (T, D); got shape {arr.shape}")
    num_frames = arr.shape[0]
    if num_frames == 0:
        return 0

    joint_threshold = 5e-3 / float(fps)

    joints = arr[:, :7]
    joint_diff = np.abs(np.diff(joints, axis=0, prepend=joints[:1]))
    idle = joint_diff.max(axis=1) < joint_threshold

    gripper = arr[:, 7:8]
    grip_diff = np.abs(np.diff(gripper, axis=0, prepend=gripper[:1]))
    idle &= grip_diff.max(axis=1) < eps_g

    if min_streak > 1:
        idle = _consecutive_streaks(idle, min_streak)

    return int(idle.sum())


class DROIDLeRobotDataset(Dataset):
    """DROID v3.0 LeRobot action dataset (joint_pos 8D + use_state).

    Args:
        root: Filesystem path to the DROID LeRobot v3.0 success split.
        fps: Frame rate (used both for video decoding and as the conditioning
            FPS surfaced to the model). Defaults to 15.0 (DROID native).
        chunk_length: Number of supervised action steps per sample. The
            window covers ``chunk_length + 1`` observation frames so the
            initial observed state can be prepended when ``use_state=True``.
        tolerance_s: Per-frame tolerance forwarded to lerobot's
            ``decode_video_frames``.
        viewpoint: Only ``"concat_view"`` is supported (wrist on top,
            L/R shoulder horizontally concatenated below).
        action_space: Only ``"joint_pos"`` is supported in this minimal port.
        use_state: When True (the canonical DROID-policy recipe) the initial
            observed joint+gripper state is prepended to the action tensor,
            yielding ``[chunk_length + 1, 8]``.
    """

    _ACTION_DIM_JOINT_POS = 8

    def __init__(
        self,
        root: str,
        fps: float = 15.0,
        chunk_length: int = 32,
        tolerance_s: float = 2e-4,
        viewpoint: str = "concat_view",
        action_space: str = "joint_pos",
        use_state: bool = True,
        domain_name: str = "droid_lerobot",
        video_backend: str = "torchcodec",
        use_image_augmentation: bool = False,
    ) -> None:
        super().__init__()
        if viewpoint != "concat_view":
            raise NotImplementedError("DROIDLeRobotDataset only supports viewpoint='concat_view'.")
        if action_space != "joint_pos":
            raise NotImplementedError("DROIDLeRobotDataset only supports action_space='joint_pos'.")
        if not use_state:
            raise NotImplementedError(
                "DROIDLeRobotDataset only supports use_state=True (matches the canonical "
                "Cosmos3 DROID-policy recipe)."
            )

        self._root = Path(root)
        self._fps = float(fps)
        self._chunk_length = int(chunk_length)
        self._tolerance_s = float(tolerance_s)
        self._viewpoint = viewpoint
        self._action_space = action_space
        self._use_state = bool(use_state)
        self._domain_id = _domain_id_from_name(domain_name)
        self._video_backend = video_backend
        self._use_image_augmentation = use_image_augmentation

        self._info = json.loads((self._root / "meta" / "info.json").read_text())

        # Episode metadata: keep one dict per episode_index for video timestamp lookups.
        self._episodes: Dict[int, Dict[str, Any]] = {
            int(row["episode_index"]): row
            for path in sorted((self._root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
            for row in pq.read_table(path).to_pylist()
        }

        self._tasks: Dict[int, str] = {
            int(row["task_index"]): str(row.get("task"))
            for row in pq.read_table(self._root / "meta" / "tasks.parquet").to_pylist()
        }

        # Compact column-oriented frame index. Materializing ~28M frames as Python
        # dicts costs tens of GB and a giant sort at construction time and forks
        # per worker; instead we keep contiguous numpy arrays which COW-share
        # across DataLoader workers.
        feature_cols = [
            _JOINT_ACTION_FEATURE,
            _ACTION_GRIPPER_FEATURE,
            _JOINT_STATE_FEATURE,
            _GRIPPER_STATE_FEATURE,
        ]
        columns = ["index", "episode_index", "task_index", "timestamp", *feature_cols]
        index_parts, episode_parts, task_parts, ts_parts = [], [], [], []
        feature_parts: Dict[str, List[np.ndarray]] = {c: [] for c in feature_cols}
        for path in sorted((self._root / "data").glob("chunk-*/file-*.parquet")):
            table = pq.read_table(path, columns=columns)
            index_parts.append(table["index"].to_numpy())
            episode_parts.append(table["episode_index"].to_numpy())
            task_parts.append(table["task_index"].to_numpy())
            ts_parts.append(table["timestamp"].to_numpy())
            for c in feature_cols:
                if c in _LIST_COLUMNS:
                    feature_parts[c].append(np.asarray(table[c].to_pylist(), dtype=np.float32))
                else:
                    feature_parts[c].append(np.asarray(table[c].to_numpy(), dtype=np.float32))

        order = np.argsort(np.concatenate(index_parts).astype(np.int64), kind="stable")
        self._row_episode = np.concatenate(episode_parts).astype(np.int64)[order]
        self._row_task = np.concatenate(task_parts).astype(np.int64)[order]
        self._row_timestamp = np.concatenate(ts_parts).astype(np.float64)[order]
        self._feat: Dict[str, np.ndarray] = {
            c: np.concatenate(feature_parts[c], axis=0).astype(np.float32)[order] for c in feature_cols
        }

        assert np.all(np.diff(self._row_episode) >= 0), (
            "episode_index is not contiguous after sort; cannot build per-episode windows."
        )
        ep_vals, ep_starts, ep_counts = np.unique(self._row_episode, return_index=True, return_counts=True)
        self._ep_vals = ep_vals.astype(np.int64)
        self._ep_starts = ep_starts.astype(np.int64)
        # ``chunk_length + 1`` observation frames are required per sample (the
        # leading frame supplies the initial state when ``use_state=True``).
        self._valid_cum = np.cumsum(np.maximum(0, ep_counts - self._chunk_length)).astype(np.int64)

    @property
    def fps(self) -> float:
        """Effective frames-per-second after subsampling."""
        return self._fps

    @property
    def chunk_length(self) -> int:
        """Number of action steps in one prediction chunk."""
        return self._chunk_length

    @property
    def domain_id(self) -> int:
        """Integer domain identifier passed through to the model."""
        return self._domain_id

    @property
    def action_dim(self) -> int:
        """Dimensionality of a single action vector."""
        return self._ACTION_DIM_JOINT_POS

    def __len__(self) -> int:
        return int(self._valid_cum[-1]) if self._valid_cum.size else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = int(idx)
        ep = int(np.searchsorted(self._valid_cum, idx, side="right"))
        prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
        start = int(self._ep_starts[ep]) + (idx - prev)
        episode_index = int(self._ep_vals[ep])
        episode = self._episodes[episode_index]

        observation_rows = self._window_rows(start, start + self._chunk_length + 1, episode_index)

        video = self._load_concat_video(episode, observation_rows)
        action = self._build_joint_action(observation_rows)
        idle_frames = _compute_idle_frames(action, fps=self._fps)

        task = self._tasks.get(int(observation_rows[0]["task_index"]), "")
        ai_caption = random.choice(task.split(" | ")) if task else ""


        # cosmos returns video as [T, 3, H, W] uint8 then permutes to [3, T, H, W].
        # Our VAE encoder expects [3, T, H, W], so do the permute here.
        formatted_video = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)

        return {
            "video": formatted_video,
            "action": action,
            "ai_caption": ai_caption,
            "additional_view_description": _CONCAT_VIEW_DESCRIPTION,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "mode": "policy",
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            "idle_frames": torch.tensor(idle_frames, dtype=torch.long),
            "dataset_index": torch.tensor(idx, dtype=torch.long),
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "start_frame": torch.tensor(idx - prev, dtype=torch.long),
            "task_index": torch.tensor(int(observation_rows[0]["task_index"]), dtype=torch.long),
        }

    def _window_rows(self, start: int, stop: int, episode_index: int) -> List[Dict[str, Any]]:
        """Reconstruct per-frame dicts over ``[start, stop)`` from the column arrays."""
        return [
            {
                "episode_index": episode_index,
                "task_index": int(self._row_task[j]),
                "timestamp": float(self._row_timestamp[j]),
                **{c: self._feat[c][j] for c in self._feat},
            }
            for j in range(start, stop)
        ]

    def _build_joint_action(self, observation_rows: List[Dict[str, Any]]) -> torch.Tensor:
        """Build [chunk+1, 8] joint_pos action with prepended initial state.

        The window is ``chunk + 1`` frames: ``rows[0]`` is the initial observed
        state (prepended when ``use_state=True``) and ``rows[1:]`` are the
        ``chunk`` commanded actions. The gripper channel is flipped (1 - g) to
        match the cosmos convention. No normalization is applied.
        """
        action_rows = observation_rows[1:]
        joints = np.asarray([r[_JOINT_ACTION_FEATURE] for r in action_rows], dtype=np.float32)  # [chunk, 7]
        gripper = np.asarray(
            [r[_ACTION_GRIPPER_FEATURE] for r in action_rows], dtype=np.float32
        ).reshape(-1, 1)
        gripper = 1.0 - gripper
        action = np.concatenate([joints, gripper], axis=-1)  # [chunk, 8]

        init = observation_rows[0]
        init_joint = np.asarray(init[_JOINT_STATE_FEATURE], dtype=np.float32)               # [7]
        init_gripper = np.asarray([1.0 - float(init[_GRIPPER_STATE_FEATURE])], dtype=np.float32)  # [1]
        initial_state = np.concatenate([init_joint, init_gripper])[None, :]                 # [1, 8]
        action = np.concatenate([initial_state, action], axis=0)                            # [chunk + 1, 8]
        return torch.from_numpy(action).float()

    def _load_concat_video(
        self,
        episode: Dict[str, Any],
        observation_rows: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """Decode the three DROID camera streams and lay them out as concat_view.

        Layout: wrist on top at full resolution, left/right shoulder
        horizontally concatenated below at half resolution.
        """
        timestamps = [float(row["timestamp"]) for row in observation_rows]
        frames_by_view = {
            name: decode_video_frames(
                    self._video_path(episode, video_key),
                    [float(episode.get(f"videos/{video_key}/from_timestamp", 0.0)) + ts for ts in timestamps],
                    self._tolerance_s,
                    backend=self._video_backend
                )
            for name, video_key in _IMAGE_FEATURES.items()
        }

        wrist = frames_by_view["wrist"]
        left = frames_by_view["left"]
        right = frames_by_view["right"]

        if self._use_image_augmentation:
            if self._image_augmentor is None:
                _, _, h, w = wrist.shape
                self._image_augmentor = T.Compose(
                    [
                        T.RandomCrop((int(h * 0.95), int(w * 0.95))),
                        T.Resize((h, w), antialias=True),
                        T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                    ]
                )
            n, m = wrist.shape[0], wrist.shape[0] + left.shape[0]
            combined = self._image_augmentor(torch.cat([wrist, left, right], dim=0))
            wrist, left, right = combined[:n], combined[n:m], combined[m:]

        _, _, h_w, w_w = wrist.shape
        half_h, half_w = h_w // 2, w_w // 2
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        bottom = torch.cat([left, right], dim=-1)
        return torch.cat([wrist, bottom], dim=-2)

    def _video_path(self, episode: Dict[str, Any], video_key: str) -> Path:
        chunk_idx = int(
            episode.get(
                f"videos/{video_key}/chunk_index",
                episode.get(f"videos/{video_key}/episode_chunk", episode.get("data/chunk_index", 0)),
            )
        )
        file_idx = int(
            episode.get(
                f"videos/{video_key}/file_index",
                episode.get(f"videos/{video_key}/episode_file", episode.get("data/file_index", 0)),
            )
        )
        rel = self._info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
            episode_chunk=chunk_idx,
            episode_file=file_idx,
        )
        return self._root / rel


def build_droid_dataset(model_cfg, data_cfg, training_args) -> DROIDLeRobotDataset:
    """Factory wired up by the dataloader_module dispatch in ``data/__init__.py``.

    Reads the chunk length from ``model_cfg.cosmos3.action_chunk_length`` (with
    a CLI override via ``args.action_chunk_length``) and falls back to 32, the
    canonical DROID action-policy length.
    """
    return DROIDLeRobotDataset(
        root=training_args.dataset_path,
        fps=data_cfg.action_fps,
        chunk_length=data_cfg.action_chunk_length,
        video_backend=data_cfg.video_backend,
        use_image_augmentation=data_cfg.use_image_augmentation
    )
