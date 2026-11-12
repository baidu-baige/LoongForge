# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.

"""Trajectory/parquet/video IO mixin for ``DreamZeroLeRobotDataset``.

Handles parquet/trajectory path resolution and lookup, padded
data-retrieval, video frame decoding, and parquet-embedded image decoding.
Mixed into ``DreamZeroLeRobotDataset``; relies on attributes assigned in that
class's ``__init__`` (``self.dataset_path``, ``self.video_backend``, etc.) via
normal Python attribute lookup at call time.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.video_utils import get_frames_by_timestamps


@dataclass
class _TrajectoryCache:
    """One-entry parquet cache owned by the dataset IO layer."""

    trajectory_id: int | None = None
    data: pd.DataFrame | None = None

    def clear(self) -> None:
        """Drop the cached trajectory before dataset handoff or serialization."""
        self.trajectory_id = None
        self.data = None


class _DreamZeroIOMixin:
    """Trajectory/parquet/video IO methods for ``DreamZeroLeRobotDataset``."""

    def get_parquet_path(self, trajectory_id: int) -> Path:
        """Get the parquet path for a trajectory."""
        chunk_index = self.get_episode_chunk(trajectory_id)
        return self.dataset_path / self.data_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id
        )

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Return trajectory parquet data through the IO-owned one-entry cache."""
        trajectory_id = int(trajectory_id)
        if (
            self._trajectory_cache.trajectory_id == trajectory_id
            and self._trajectory_cache.data is not None
        ):
            return self._trajectory_cache.data

        parquet_path = self._resolve_trajectory_parquet_path(trajectory_id)
        try:
            data = pd.read_parquet(parquet_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load trajectory {trajectory_id} from {parquet_path}"
            ) from exc
        self._trajectory_cache.trajectory_id = trajectory_id
        self._trajectory_cache.data = data
        return data

    def clear_trajectory_cache(self) -> None:
        """Clear parquet data cached during initialization or sample loading."""
        self._trajectory_cache.clear()

    def _resolve_trajectory_parquet_path(self, trajectory_id: int) -> Path:
        """Resolve a trajectory parquet path, including legacy chunk layouts."""
        parquet_path = self.get_parquet_path(trajectory_id)
        if parquet_path.exists():
            return parquet_path
        parquet_files = sorted(
            self.dataset_path.glob(f"data/*/episode_{trajectory_id:06d}.parquet")
        )
        if parquet_files:
            return parquet_files[0]
        raise FileNotFoundError(
            f"Parquet file not found for trajectory {trajectory_id}: {parquet_path}"
        )

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_index = self._trajectory_index_by_id.get(int(trajectory_id))
        if trajectory_index is None:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}; "
                f"dataset={self.dataset_path}, num_trajectories={len(self.trajectory_ids)}"
            )
        return trajectory_index

    def get_discarded_trajectory_ids(self) -> set[int]:
        """Return bad trajectory ids recorded in LeRobot metadata.

        ``discarded_episode_indices`` stores episode ids, not row positions in
        ``trajectory_ids``. Keeping that distinction matters for filtered or
        merged datasets where episode ids are not contiguous.
        """
        if not self.discard_bad_trajectories:
            return set()
        return {
            int(trajectory_id)
            for trajectory_id in self._lerobot_info_meta.get("discarded_episode_indices", []) or []
        }

    def get_flat_index_for_step(self, trajectory_id: int, base_index: int) -> int | None:
        """Return the single-dataset flat index for a trajectory/base step pair."""
        if self._all_steps_index_by_identity is None:
            self._all_steps_index_by_identity = {
                (int(traj_id), int(step_idx)): flat_index
                for flat_index, (traj_id, step_idx) in enumerate(self.all_steps)
            }
        return self._all_steps_index_by_identity.get((int(trajectory_id), int(base_index)))

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.

        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the trajectory.
            padding_strategy (str): The padding strategy, either "first_last" or "zero".
                "first_last" uses first/last step data for padding, "zero" uses zero padding.

        Returns:
            np.ndarray: The retrieved and padded data.
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, given T step_indices, the shape of the
        # retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape, dtype=array.dtype)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        """Get the video file path for a specific trajectory and video key.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The video key (without 'video.' prefix).

        Returns:
            Path: Path to the video file.
        """
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        video_filename = self.video_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
        )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        trajectory_data = self.get_trajectory_data(trajectory_id)
        le_video_meta = self.lerobot_info_meta.get("features", {}).get(original_key, {})
        if le_video_meta.get("dtype") == "image" and original_key in trajectory_data.columns:
            return np.stack(
                [
                    self._decode_parquet_image_cell(trajectory_data[original_key].iloc[int(idx)])
                    for idx in step_indices
                ],
                axis=0,
            )
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert "timestamp" in trajectory_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = trajectory_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]

        # try:
        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )
        # except:
            # self.video_backend = "torchvision_av"
            # return get_frames_by_timestamps(
            #     video_path.as_posix(),
            #     video_timestamp,
            #     video_backend=self.video_backend,
            #     video_backend_kwargs=self.video_backend_kwargs,
            # )

    @staticmethod
    def _decode_parquet_image_cell(cell) -> np.ndarray:
        """Decode a LeRobot v2 parquet image cell into an RGB numpy array."""
        from io import BytesIO
        from PIL import Image

        if isinstance(cell, dict):
            cell = cell.get("bytes", cell)
        if isinstance(cell, (bytes, bytearray)):
            return np.asarray(Image.open(BytesIO(cell)).convert("RGB"))
        return np.asarray(cell)
