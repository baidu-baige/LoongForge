# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.

"""Metadata/schema/relative-stats loading mixin for ``DreamZeroLeRobotDataset``.

Handles dataset-level metadata/schema properties, on-disk meta file
loaders (``meta/modality.json``, ``meta/info.json``, ``meta/stats.json``,
``meta/relative_stats_dreamzero.json``), on-the-fly relative-action statistics
computation, trajectory/step-index discovery, and the post-load integrity
check. Mixed into ``DreamZeroLeRobotDataset``; relies on attributes assigned in
that class's ``__init__`` (``self._dataset_path``, ``self.modality_configs``,
etc.) via normal Python attribute lookup at call time.
"""

from collections import defaultdict
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError
from tqdm import tqdm

from .constants import (
    LE_ROBOT_DATA_FILENAME,
    LE_ROBOT_DETAILED_GLOBAL_INSTRUCTION_FILENAME,
    LE_ROBOT_EPISODE_FILENAME,
    LE_ROBOT_INFO_FILENAME,
    LE_ROBOT_MODALITY_FILENAME,
    LE_ROBOT_STATS_FILENAME,
    LE_ROBOT_TASKS_FILENAME,
    LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME,
    LEROBOT_RELATIVE_STATS_FILE_NAME,
    METADATA_DIR,
    METADATA_LANG_KEYS,
    STEP_FILTER_FILENAME,
)
from ..schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)

logger = logging.getLogger(__name__)


def _calculate_dataset_statistics(
    parquet_paths: list[Path],
    features: list[str] | None = None,
) -> dict[str, DatasetStatisticalValues]:
    """Calculate low-dimensional statistics from LeRobot parquet files."""
    frames = [
        pd.read_parquet(parquet_path)
        for parquet_path in tqdm(
            sorted(parquet_paths),
            desc="Collecting all parquet files...",
        )
    ]
    all_low_dim_data = pd.concat(frames, axis=0)
    feature_names = list(all_low_dim_data.columns) if features is None else features
    statistics = {}
    for feature_name in feature_names:
        logger.info("Computing statistics for %s...", feature_name)
        values = np.vstack(
            [
                np.asarray(value, dtype=np.float32)
                for value in all_low_dim_data[feature_name]
            ]
        )
        statistics[feature_name] = DatasetStatisticalValues(
            mean=np.mean(values, axis=0).tolist(),
            std=np.std(values, axis=0).tolist(),
            min=np.min(values, axis=0).tolist(),
            max=np.max(values, axis=0).tolist(),
            q01=np.quantile(values, 0.01, axis=0).tolist(),
            q99=np.quantile(values, 0.99, axis=0).tolist(),
        )
    return statistics


class _DreamZeroMetaMixin:
    """Metadata/schema/stats loading methods for ``DreamZeroLeRobotDataset``."""

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the
        values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset.

        The keys are the modality.key, and the values are the delta indices for
        each modality.key.
        """
        return self._delta_indices

    def _get_max_delta_index(self) -> int:
        """Calculate the maximum delta index across all modalities.

        Returns:
            int: The maximum delta index value.
        """
        max_delta_index = 0
        for delta_index in self.delta_indices.values():
            max_delta_index = max(max_delta_index, delta_index.max())
        return max_delta_index

    @property
    def max_delta_index(self) -> int:
        """The maximum delta index across all modalities."""
        return self._max_delta_index

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def lerobot_stats_meta(self) -> dict[str, DatasetStatisticalValues]:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_stats_meta

    @property
    def lerobot_relative_stats_meta(self) -> dict[str, DatasetStatisticalValues]:
        """The relative action stats metadata for the LeRobot dataset."""
        return self._lerobot_relative_stats_meta

    @property
    def lerobot_relative_horizon_stats_meta(self) -> dict[str, dict[str, list]]:
        """The per-horizon relative action stats metadata for the LeRobot dataset.

        Format: {action_key: {stat_name: [[h0_vals], [h1_vals], ...]}}
        """
        return self._lerobot_relative_horizon_stats_meta

    @property
    def step_filter(self) -> dict[int, np.ndarray]:
        """The step filter for the dataset."""
        return self._step_filter

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        if self.use_global_metadata:
            assert (
                self.metadata_version is not None
            ), "metadata_version must be provided if use_global_metadata is True"
            modality_meta_path = (
                METADATA_DIR
                / self.tag.value
                / self.metadata_version
                / Path(LE_ROBOT_MODALITY_FILENAME).name
            )
            assert modality_meta_path.exists(), (
                f"Please provide a {Path(LE_ROBOT_MODALITY_FILENAME).name} file in "
                f"{METADATA_DIR / self.tag.value / self.metadata_version}"
            )
            with open(modality_meta_path, "r") as f:
                modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
            return modality_meta
        else:
            modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
            assert (
                modality_meta_path.exists()
            ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
            with open(modality_meta_path, "r") as f:
                modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
            return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_lerobot_stats_meta(self) -> dict[str, DatasetStatisticalValues]:
        """Get the metadata for the LeRobot dataset."""
        if self.use_global_metadata:
            assert (
                self.metadata_version is not None
            ), "metadata_version must be provided if use_global_metadata is True"
            stats_path = (
                METADATA_DIR
                / self.tag.value
                / self.metadata_version
                / Path(LE_ROBOT_STATS_FILENAME).name
            )
        else:
            stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        try:
            with open(stats_path, "r") as f:
                stats: dict = json.load(f)
            for name in ["num_trajectories", "total_trajectory_length"]:
                stats.pop(name, None)
            for name, stat in stats.items():
                stats[name] = DatasetStatisticalValues.model_validate(stat)
            return stats
        except (FileNotFoundError, ValidationError) as e:
            if self.use_global_metadata:
                raise ValueError(
                    f"{e}: Please provide a {Path(LE_ROBOT_STATS_FILENAME).name} file in {stats_path}"
                    " and ensure the metadata format is correct."
                )
            logger.warning("Failed to load dataset statistics: %s", e)
            logger.info("Calculating dataset statistics for %s", self.dataset_name)
            # Get all parquet files in the dataset paths
            parquet_files = list((self.dataset_path).glob(LE_ROBOT_DATA_FILENAME))
            lowdim_features = []
            le_features = self.lerobot_info_meta["features"]
            for feature in le_features:
                if "float" in le_features[feature]["dtype"]:
                    lowdim_features.append(feature)

            stats = _calculate_dataset_statistics(parquet_files, lowdim_features)
            stats_serialized = {k: v.model_dump(mode="json") for k, v in stats.items()}
            with open(stats_path, "w") as f:
                json.dump(stats_serialized, f, indent=4)
            return stats

    def _get_lerobot_relative_stats_meta(self) -> dict[str, DatasetStatisticalValues]:
        """Get the relative action stats metadata for the LeRobot dataset.

        Returns:
            dict[str, DatasetStatisticalValues]: Dictionary mapping action keys to their relative stats.
        """
        # Determine the path for relative stats file
        if self.use_global_metadata:
            assert (
                self.metadata_version is not None
            ), "metadata_version must be provided if use_global_metadata is True"
            stats_path = (
                METADATA_DIR
                / self.tag.value
                / self.metadata_version
                / Path(LEROBOT_RELATIVE_STATS_FILE_NAME).name
            )
            assert stats_path.exists(), (
                f"Please provide a {Path(LEROBOT_RELATIVE_STATS_FILE_NAME).name} file in "
                f"{METADATA_DIR / self.tag.value / self.metadata_version}"
            )
        else:
            stats_path = self.dataset_path / LEROBOT_RELATIVE_STATS_FILE_NAME

        # Try to load existing relative stats
        if stats_path.exists():
            logger.info("Loading relative action stats from %s", stats_path)
            with open(stats_path, "r") as f:
                stats: dict = json.load(f)
            for name, stat in stats.items():
                stats[name] = DatasetStatisticalValues.model_validate(stat)
            return stats

        # Calculate relative stats if file doesn't exist
        logger.warning("Relative stats file not found at %s", stats_path)
        logger.info("Calculating relative action stats for %s", self.dataset_name)

        # Get action keys from modality configs, filtered by relative_action_keys
        action_config = self.modality_configs.get("action")
        all_action_keys = action_config.modality_keys if action_config is not None else []
        if not all_action_keys:
            logger.warning(
                "No action keys found in modality configs, skipping relative stats calculation"
            )
            return {}

        # Filter to only the keys that should use relative action
        action_keys_to_process = []
        for key in all_action_keys:
            subkey = key.replace("action.", "")
            if self.relative_action_keys is None or subkey in self.relative_action_keys:
                action_keys_to_process.append(subkey)

        if not action_keys_to_process:
            logger.warning("No action keys to process for relative stats")
            return {}

        logger.info("Will calculate relative stats for: %s", action_keys_to_process)

        stats = {}
        for action_key in action_keys_to_process:
            logger.info("Calculating relative stats for action key: %s", action_key)
            try:
                relative_stats = self._calculate_relative_stats_for_key(action_key)
                stats[action_key] = relative_stats
            except Exception as e:
                logger.warning("Failed to calculate relative stats for %s: %s", action_key, e)
                continue

        if stats:
            # Save the calculated stats
            stats_serialized = {k: v.model_dump(mode="json") for k, v in stats.items()}
            # Only save to dataset path (not global metadata path)
            save_path = self.dataset_path / LEROBOT_RELATIVE_STATS_FILE_NAME
            logger.info("Saving relative action stats to %s", save_path)
            with open(save_path, "w") as f:
                json.dump(stats_serialized, f, indent=4)

        return stats

    def _get_lerobot_relative_horizon_stats_meta(self) -> dict[str, dict[str, list]]:
        """Get the per-horizon relative action stats metadata for the LeRobot dataset.

        Similar to _get_lerobot_relative_stats_meta but calculates separate stats for each
        action horizon index. Will load from file if exists, otherwise calculate and save.

        Returns:
            dict[str, dict[str, list]]: Nested dictionary where:
                - Outer key is the action key (e.g., 'joint_position')
                - Inner key is the stat name (e.g., 'max', 'min', 'mean', 'std', 'q01', 'q99')
                - Value is a list of stat values per horizon index
        """
        # Determine the path for per-horizon relative stats file
        if self.use_global_metadata:
            assert (
                self.metadata_version is not None
            ), "metadata_version must be provided if use_global_metadata is True"
            stats_path = (
                METADATA_DIR
                / self.tag.value
                / self.metadata_version
                / Path(LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME).name
            )
            assert stats_path.exists(), (
                f"Please provide a {Path(LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME).name} "
                f"file in {METADATA_DIR / self.tag.value / self.metadata_version}"
            )
        else:
            stats_path = self.dataset_path / LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME

        # Try to load existing per-horizon relative stats
        if stats_path.exists():
            logger.info("Loading per-horizon relative action stats from %s", stats_path)
            with open(stats_path, "r") as f:
                stats: dict = json.load(f)
            return stats

        # Calculate per-horizon relative stats if file doesn't exist
        logger.warning("Per-horizon relative stats file not found at %s", stats_path)
        logger.info("Calculating per-horizon relative action stats for %s", self.dataset_name)

        # Get action keys from modality configs, filtered by relative_action_keys
        action_config = self.modality_configs.get("action")
        all_action_keys = action_config.modality_keys if action_config is not None else []
        if not all_action_keys:
            logger.warning(
                "No action keys found in modality configs, "
                "skipping per-horizon relative stats calculation"
            )
            return {}

        # Filter to only the keys that should use relative action
        action_keys_to_process = []
        for key in all_action_keys:
            subkey = key.replace("action.", "")
            if self.relative_action_keys is None or subkey in self.relative_action_keys:
                action_keys_to_process.append(subkey)

        if not action_keys_to_process:
            logger.warning("No action keys to process for per-horizon relative stats")
            return {}

        logger.info("Will calculate per-horizon relative stats for: %s", action_keys_to_process)

        stats = {}
        for action_key in action_keys_to_process:
            logger.info("Calculating per-horizon relative stats for action key: %s", action_key)
            try:
                relative_stats = self._calculate_relative_stats_for_key_per_horizon(action_key)
                stats[action_key] = relative_stats
            except Exception as e:
                logger.warning("Failed to calculate per-horizon relative stats for %s: %s", action_key, e)
                continue

        if stats:
            # Only save to dataset path (not global metadata path)
            save_path = self.dataset_path / LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME
            logger.info("Saving per-horizon relative action stats to %s", save_path)
            with open(save_path, "w") as f:
                json.dump(stats, f, indent=4)

        return stats

    def _calculate_relative_stats_for_key(self, action_key: str) -> DatasetStatisticalValues:
        """Calculate relative action statistics for a specific action key.

        Args:
            action_key: The action key to calculate stats for (e.g., 'joint_position')

        Returns:
            DatasetStatisticalValues: The calculated statistics for the relative action.
        """
        # Get state and action metadata from lerobot modality config
        state_key = action_key  # Assume state key matches action key

        # Get the modality metadata to find original column names and indices
        state_meta = self.lerobot_modality_meta.state.get(state_key)
        action_meta = self.lerobot_modality_meta.action.get(action_key)

        if state_meta is None:
            raise ValueError(f"State key '{state_key}' not found in modality metadata")
        if action_meta is None:
            raise ValueError(f"Action key '{action_key}' not found in modality metadata")

        # Get the original column names (e.g., 'observation.state', 'action')
        state_original_key = state_meta.original_key
        action_original_key = action_meta.original_key

        # Get the indices to slice from the concatenated vectors
        state_start, state_end = state_meta.start, state_meta.end
        action_start, action_end = action_meta.start, action_meta.end

        state_config = self.modality_configs.get("state")
        state_delta_indices = (
            state_config.delta_indices if state_config is not None else [0]
        )
        action_delta_indices = self.modality_configs["action"].delta_indices

        logger.info(
            "Calculating relative stats for %s: state=%s[%s:%s], action=%s[%s:%s]",
            action_key,
            state_original_key,
            state_start,
            state_end,
            action_original_key,
            action_start,
            action_end,
        )

        # # Calculate relative actions for all trajectories
        all_relative_actions = []

        # for traj_id in tqdm(self.trajectory_ids, desc=f"Calculating relative stats for {action_key}"):
        max_trajs_for_stats = 10000
        traj_ids_to_process = self.trajectory_ids
        if len(traj_ids_to_process) > max_trajs_for_stats:
            # Randomly sample 500 trajectories
            rng = np.random.default_rng(seed=42)
            sampled_indices = rng.choice(len(traj_ids_to_process), size=max_trajs_for_stats, replace=False)
            traj_ids_to_process = traj_ids_to_process[sampled_indices]
            logger.info(
                "Sampling %s trajectories out of %s for stats calculation",
                max_trajs_for_stats,
                len(self.trajectory_ids),
            )

        # Calculate relative actions for sampled trajectories
        all_relative_actions = []

        for traj_id in tqdm(traj_ids_to_process, desc=f"Calculating relative stats for {action_key}"):
            try:
                traj_data = self.get_trajectory_data(traj_id)

                # Check if columns exist
                if state_original_key not in traj_data.columns or action_original_key not in traj_data.columns:
                    logger.warning(
                        "Missing columns: state=%r exists=%s, action=%r exists=%s",
                        state_original_key,
                        state_original_key in traj_data.columns,
                        action_original_key,
                        action_original_key in traj_data.columns,
                    )
                    continue

                # Load full state and action arrays, then slice to get the specific component
                full_state_data = np.stack(traj_data[state_original_key].values)
                full_action_data = np.stack(traj_data[action_original_key].values)

                # Slice to get just the component we care about (e.g., joint_position)
                state_data = full_state_data[:, state_start:state_end]
                action_data = full_action_data[:, action_start:action_end]

                # Calculate usable length based on action delta indices
                usable_length = len(traj_data) - max(action_delta_indices)

                for i in range(usable_length):
                    # Get reference state (last state before action chunk)
                    ref_state_idx = state_delta_indices[-1] + i
                    if ref_state_idx >= len(state_data):
                        continue
                    ref_state = state_data[ref_state_idx]

                    # Get action chunk
                    action_indices = [idx + i for idx in action_delta_indices]
                    if max(action_indices) >= len(action_data):
                        continue
                    actions = action_data[action_indices]


                    # Calculate relative actions (action - reference state)
                    relative_actions = actions - ref_state
                    all_relative_actions.extend(relative_actions)

            except Exception as e:
                logger.warning("Error processing trajectory %s: %s", traj_id, e)
                continue

        if not all_relative_actions:
            raise ValueError(f"No relative actions calculated for {action_key}")

        all_relative_actions = np.array(all_relative_actions)
        logger.info("Collected %s relative action samples for %s", len(all_relative_actions), action_key)

        return DatasetStatisticalValues(
            max=np.max(all_relative_actions, axis=0).tolist(),
            min=np.min(all_relative_actions, axis=0).tolist(),
            mean=np.mean(all_relative_actions, axis=0).tolist(),
            std=np.std(all_relative_actions, axis=0).tolist(),
            q01=np.quantile(all_relative_actions, 0.01, axis=0).tolist(),
            q99=np.quantile(all_relative_actions, 0.99, axis=0).tolist(),
        )

    def _calculate_relative_stats_for_key_per_horizon(
        self, action_key: str
    ) -> dict[str, list]:
        """Calculate relative action statistics for each delta index (horizon step) separately.

        Unlike `_calculate_relative_stats_for_key` which pools all horizon steps together,
        this method calculates separate statistics for each action horizon index.

        Args:
            action_key: The action key to calculate stats for (e.g., 'joint_position')

        Returns:
            dict[str, list]: Dictionary where keys are stat names (max, min, mean, std, q01, q99)
                and values are lists of stat values per horizon index.
                Format: {"max": [[h0_vals], [h1_vals], ...], "min": [...], ...}
        """
        # Get state and action metadata from lerobot modality config
        state_key = action_key  # Assume state key matches action key

        # Get the modality metadata to find original column names and indices
        state_meta = self.lerobot_modality_meta.state.get(state_key)
        action_meta = self.lerobot_modality_meta.action.get(action_key)

        if state_meta is None:
            raise ValueError(f"State key '{state_key}' not found in modality metadata")
        if action_meta is None:
            raise ValueError(f"Action key '{action_key}' not found in modality metadata")

        # Get the original column names (e.g., 'observation.state', 'action')
        state_original_key = state_meta.original_key
        action_original_key = action_meta.original_key

        # Get the indices to slice from the concatenated vectors
        state_start, state_end = state_meta.start, state_meta.end
        action_start, action_end = action_meta.start, action_meta.end

        state_config = self.modality_configs.get("state")
        state_delta_indices = (
            state_config.delta_indices if state_config is not None else [0]
        )
        action_delta_indices = self.modality_configs["action"].delta_indices

        logger.info(
            "Calculating per-horizon relative stats for %s: state=%s[%s:%s], action=%s[%s:%s], "
            "action_delta_indices=%s",
            action_key,
            state_original_key,
            state_start,
            state_end,
            action_original_key,
            action_start,
            action_end,
            action_delta_indices,
        )

        # Initialize separate lists for each horizon index
        all_relative_actions_per_horizon: dict[int, list] = {
            delta_idx: [] for delta_idx in action_delta_indices
        }

        max_trajs_for_stats = 10000
        traj_ids_to_process = self.trajectory_ids
        if len(traj_ids_to_process) > max_trajs_for_stats:
            # Randomly sample trajectories
            rng = np.random.default_rng(seed=42)
            sampled_indices = rng.choice(len(traj_ids_to_process), size=max_trajs_for_stats, replace=False)
            traj_ids_to_process = traj_ids_to_process[sampled_indices]
            logger.info(
                "Sampling %s trajectories out of %s for stats calculation",
                max_trajs_for_stats,
                len(self.trajectory_ids),
            )

        for traj_id in tqdm(traj_ids_to_process, desc=f"Calculating per-horizon relative stats for {action_key}"):
            try:
                traj_data = self.get_trajectory_data(traj_id)

                # Check if columns exist
                if state_original_key not in traj_data.columns or action_original_key not in traj_data.columns:
                    continue

                # Load full state and action arrays, then slice to get the specific component
                full_state_data = np.stack(traj_data[state_original_key].values)
                full_action_data = np.stack(traj_data[action_original_key].values)

                # Slice to get just the component we care about (e.g., joint_position)
                state_data = full_state_data[:, state_start:state_end]
                action_data = full_action_data[:, action_start:action_end]

                # Calculate usable length based on action delta indices
                usable_length = len(traj_data) - max(action_delta_indices)

                for i in range(usable_length):
                    # Get reference state (last state before action chunk)
                    ref_state_idx = state_delta_indices[-1] + i
                    if ref_state_idx >= len(state_data):
                        continue
                    ref_state = state_data[ref_state_idx]

                    # Get action for each horizon index separately
                    for delta_idx in action_delta_indices:
                        action_idx = delta_idx + i
                        if action_idx >= len(action_data):
                            continue
                        action = action_data[action_idx]

                        # Calculate relative action (action - reference state)
                        relative_action = action - ref_state
                        all_relative_actions_per_horizon[delta_idx].append(relative_action)

            except Exception as e:
                logger.warning("Error processing trajectory %s: %s", traj_id, e)
                continue

        # Calculate stats for each horizon index and organize by stat name
        stat_names = ["max", "min", "mean", "std", "q01", "q99"]
        stats_by_name: dict[str, list] = {name: [] for name in stat_names}

        for delta_idx in action_delta_indices:
            relative_actions = all_relative_actions_per_horizon[delta_idx]
            if not relative_actions:
                logger.warning("No relative actions calculated for %s at horizon index %s", action_key, delta_idx)
                # Add empty/placeholder values
                for name in stat_names:
                    stats_by_name[name].append([])
                continue

            relative_actions_array = np.array(relative_actions)
            logger.info(
                "Collected %s relative action samples for %s at horizon %s",
                len(relative_actions_array),
                action_key,
                delta_idx,
            )

            stats_by_name["max"].append(np.max(relative_actions_array, axis=0).tolist())
            stats_by_name["min"].append(np.min(relative_actions_array, axis=0).tolist())
            stats_by_name["mean"].append(np.mean(relative_actions_array, axis=0).tolist())
            stats_by_name["std"].append(np.std(relative_actions_array, axis=0).tolist())
            stats_by_name["q01"].append(np.quantile(relative_actions_array, 0.01, axis=0).tolist())
            stats_by_name["q99"].append(np.quantile(relative_actions_array, 0.99, axis=0).tolist())

        return stats_by_name

    def _get_step_filter(self) -> dict[int, np.ndarray]:
        """Get the step filter for the dataset."""
        step_filter_path = self.dataset_path / STEP_FILTER_FILENAME
        step_filter = {}
        if step_filter_path.exists():
            with open(step_filter_path, "r") as f:
                for line in f:
                    episode_step_filter = json.loads(line)
                    trajectory_id = int(episode_step_filter["episode_index"])
                    trajectory_index = self.get_trajectory_index(trajectory_id)
                    all_indices = np.arange(self.trajectory_lengths[trajectory_index].item())
                    indices_to_filter = np.array(episode_step_filter["step_indices"])
                    step_filter[trajectory_id] = np.setdiff1d(all_indices, indices_to_filter)
        else:
            for trajectory_index, trajectory_id in enumerate(self.trajectory_ids):
                step_filter[int(trajectory_id)] = np.arange(
                    self.trajectory_lengths[trajectory_index].item()
                )
        return step_filter

    def _get_metadata(self) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                self.lerobot_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in self.lerobot_modality_meta.video:
            original_key = self.lerobot_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            names = le_video_meta.get("names") or []
            video_info = le_video_meta.get("video_info") or le_video_meta.get("info") or {}
            if "video.channels" in video_info:
                channels = video_info["video.channels"]
            elif "channel" in names:
                channels = le_video_meta["shape"][names.index("channel")]
            elif "channels" in names:
                channels = le_video_meta["shape"][names.index("channels")]
            else:
                channels = le_video_meta["shape"][-1]
            fps = video_info.get("video.fps", self.fps or le_info.get("fps", 10))
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics
        dataset_statistics = {}
        le_statistics = {k: v.model_dump() for k, v in self.lerobot_stats_meta.items()}
        # Prepare relative stats if available
        relative_stats = {}
        if self.relative_action:
            relative_stats = {k: v.model_dump() for k, v in self._lerobot_relative_stats_meta.items()}

        # Prepare per-horizon relative stats if available
        per_horizon_stats = {}
        if self.relative_action_per_horizon:
            per_horizon_stats = self._lerobot_relative_horizon_stats_meta

        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = self.lerobot_modality_meta.get_key_meta(
                    f"{our_modality}.{subkey}"
                )
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)

                # Check if we should use per-horizon relative stats for this action key
                should_use_per_horizon = (
                    our_modality == "action"
                    and self.relative_action_per_horizon
                    and subkey in per_horizon_stats
                    and (self.relative_action_keys is None or subkey in self.relative_action_keys)
                )

                # Use relative stats for action modality if relative_action is enabled and stats are available
                # Also check if this subkey is in the list of keys that should use relative action
                should_use_relative = (
                    our_modality == "action"
                    and self.relative_action
                    and subkey in relative_stats
                    and (self.relative_action_keys is None or subkey in self.relative_action_keys)
                )

                if should_use_per_horizon:
                    # Use per-horizon relative action stats (format: {stat_name: [[h0_vals], [h1_vals], ...]})
                    for stat_name in per_horizon_stats[subkey]:
                        dataset_statistics[our_modality][subkey][stat_name] = per_horizon_stats[subkey][stat_name]
                    logger.info("Using per-horizon relative stats for %s", subkey)
                elif should_use_relative:
                    # Use relative action stats directly
                    for stat_name in relative_stats[subkey]:
                        dataset_statistics[our_modality][subkey][stat_name] = relative_stats[subkey][stat_name]
                    logger.info("Using relative stats for %s", subkey)
                else:
                    # Use original absolute stats
                    le_modality = state_action_meta.original_key
                    for stat_name in le_statistics[le_modality]:
                        indices = np.arange(
                            state_action_meta.start,
                            state_action_meta.end,
                        )
                        stat = np.array(le_statistics[le_modality][stat_name])
                        dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=self.tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]
        trajectory_ids = []
        trajectory_lengths = []
        for episode in episode_metadata:
            trajectory_ids.append(episode["episode_index"])
            trajectory_lengths.append(episode["length"])
        return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[int, int]]: A list of (trajectory_id, base_index) tuples.

        Example:
            self.trajectory_ids: [0, 1, 2]
            self.step_filter: {
                0: [0, 1, 2],
                1: [0, 1],
                2: [0, 2, 3]
            }
            return: [
                (0, 0), (0, 1), (0, 2),
                (1, 0), (1, 1),
                (2, 0), (2, 2), (2, 3)
            ]
        """
        all_steps: list[tuple[int, int]] = []
        # All steps is used in single dataset, so we need to discard bad trajectories
        # Mixture dataset directly use trajectory_ids, so we handle it by changing the sampling weights
        discarded_episode_indices = self.get_discarded_trajectory_ids()

        for trajectory_id in self.trajectory_ids:
            trajectory_id = int(trajectory_id)
            if trajectory_id in discarded_episode_indices:
                continue
            for base_index in self.step_filter[trajectory_id]:
                all_steps.append((trajectory_id, int(base_index)))
        return all_steps

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.

        Returns:
            dict: Dictionary mapping modality names to their keys.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
        with open(tasks_path, "r") as f:
            tasks = [json.loads(line) for line in f]
        df = pd.DataFrame(tasks)
        return df.set_index("task_index")

    def _get_detailed_global_instructions(self) -> dict[int, dict]:
        """Get the detailed global instructions for the dataset.

        Loads from episodes_detail_global_instruction.jsonl if it exists.

        Returns:
            dict[int, dict]: Mapping from episode_index to detailed instruction dict.
        """
        detailed_instruction_path = self.dataset_path / LE_ROBOT_DETAILED_GLOBAL_INSTRUCTION_FILENAME
        if not detailed_instruction_path.exists():
            return {}
        with open(detailed_instruction_path, "r") as f:
            instructions_list = [json.loads(line) for line in f]
        return {entry["episode_index"]: entry for entry in instructions_list}

    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality, modality_config in self.modality_configs.items():
            if modality in ["lapa_action", "dream_actions", "rl_info", "task_embedding"]:
                continue
            for key in modality_config.modality_keys:

                if key == "action.task_progress":
                    continue
                # Skip metadata-based language keys (they don't need modality metadata)
                if modality == "language" and key.startswith("annotation."):
                    lang_subkey = key.replace("annotation.", "")
                    if lang_subkey in METADATA_LANG_KEYS:
                        continue
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )
