# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.

"""LeRobot-format dataset classes for DreamZero embodied training.

Provides ``DreamZeroLeRobotDataset`` for loading a single LeRobot-format dataset
(parquet state/action tables plus videos) and ``DreamZeroLeRobotMixtureDataset`` for
sampling across multiple weighted datasets during training/evaluation.
"""

from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import logging
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from loongforge.embodied.model.dreamzero.precomputed_cache import (
    DreamZeroPrecomputedCacheConfig,
)
from loongforge.embodied.model.dreamzero.precomputed_cache.artifact import (
    DreamZeroPrecomputedFeatureArtifact,
)

from ..schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    EmbodimentTag,
)
from ..transforms.base import ComposedModalityTransform
from ._io_mixin import _DreamZeroIOMixin, _TrajectoryCache
from ._language_action_mixin import _DreamZeroLanguageActionMixin
from ._meta_mixin import _DreamZeroMetaMixin
from ._precomputed_cache_mixin import _DreamZeroPrecomputedCacheMixin
from .modality_configs import ModalityConfig

logger = logging.getLogger(__name__)


@contextmanager
def _temporary_transform_rng(seed: int):
    """Temporarily seed torch/numpy/python RNGs, restoring the prior state on exit."""
    torch_rng_state = torch.random.get_rng_state()
    numpy_rng_state = np.random.get_state()
    python_rng_state = random.getstate()
    try:
        seed = int(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        random.seed(seed)
        yield
    finally:
        torch.random.set_rng_state(torch_rng_state)
        np.random.set_state(numpy_rng_state)
        random.setstate(python_rng_state)


def _worker_rng_state_dict() -> dict:
    """Capture transform RNG streams for StatefulDataLoader checkpoints."""
    return {
        "torch": torch.random.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _load_worker_rng_state_dict(state_dict: dict) -> None:
    """Restore transform RNG streams in a StatefulDataLoader worker."""
    torch.random.set_rng_state(state_dict["torch"])
    np.random.set_state(_lists_to_tuples(state_dict["numpy"]))
    random.setstate(_lists_to_tuples(state_dict["python"]))


def _lists_to_tuples(value):
    """Undo torchdata's list normalization for RNG tuple states."""
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    return value


class DreamZeroLeRobotDataset(
    Dataset,
    _DreamZeroMetaMixin,
    _DreamZeroPrecomputedCacheMixin,
    _DreamZeroIOMixin,
    _DreamZeroLanguageActionMixin,
):
    """
    Base dataset class for LeRobot that supports sharding.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        use_global_metadata: bool = True,
        metadata_version: str | None = None,
        video_backend: str = "ffmpeg",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        discard_bad_trajectories: bool = True,
        fps: float = None,
        max_chunk_size: int = None,
        relative_action: bool = False,
        relative_action_keys: list[str] | None = None,
        relative_action_per_horizon: bool = False,
        language_chunk_sampling: bool = False,
        use_sample_transform_seed: bool = False,
        sample_transform_seed: int = 0,
        precomputed_cache_config: DreamZeroPrecomputedCacheConfig | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality.
                The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            use_global_metadata (bool): Whether to use global metadata for normalization.
            metadata_version (str): The version of the metadata, if `use_global_metadata` is True.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when
                initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset.
                e.g. define it as "new_embodiment"
            relative_action (bool): Whether to use relative action stats for normalization.
                If True, will load or calculate relative action stats from
                relative_stats_dreamzero.json. If the file doesn't exist, stats will be
                calculated.
            relative_action_keys (list[str] | None): List of action keys to apply relative
                action to (e.g., ['joint_position']). If None and relative_action is True,
                applies to all action keys except those containing 'gripper'.
            relative_action_per_horizon (bool): Whether to use per-horizon relative action
                stats. If True, will load or calculate separate stats for each action
                horizon index from relative_horizon_stats_dreamzero.json.
            language_chunk_sampling (bool): Whether to mirror DreamZero's
                ShardedLeRobotSubLangSingleActionChunkDatasetDROID sampling:
                video/state/action chunks are expanded around the anchor while
                language stays unchanged.
            use_sample_transform_seed (bool): Whether to reset torch/numpy/python
                RNG per sample before random transforms. Used by alignment recipes
                that require transform randomness to be independent of worker/order.
            sample_transform_seed (int): Base seed for per-sample transform RNG. The
                effective sample seed is ``sample_transform_seed + index``.
        """
        # first check if the path directory exists
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        self.modality_configs = modality_configs
        self.use_global_metadata = use_global_metadata
        self.metadata_version = metadata_version
        self.video_backend = video_backend
        self.video_backend_kwargs = (
            video_backend_kwargs if video_backend_kwargs is not None else {}
        )
        self.fps = fps
        self.max_chunk_size = max_chunk_size
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )
        self.discard_bad_trajectories = discard_bad_trajectories
        self.relative_action = relative_action
        self.relative_action_per_horizon = relative_action_per_horizon
        self.language_chunk_sampling = language_chunk_sampling
        self.use_sample_transform_seed = bool(use_sample_transform_seed)
        self.sample_transform_seed = int(sample_transform_seed)
        self.precomputed_cache_config = (
            precomputed_cache_config
            if precomputed_cache_config is not None
            else DreamZeroPrecomputedCacheConfig()
        )
        self._current_num_chunks: dict[int, int] = {}
        self._full_language_chunk_cache: dict[tuple[int, int], bool] = {}
        # Determine which action keys should use relative action
        if relative_action_keys is not None:
            self.relative_action_keys = relative_action_keys
        else:
            # Default: apply to all action keys except those containing 'gripper'
            self.relative_action_keys = None  # Will be set after modality_configs is available
        self._relative_action_keys_input = relative_action_keys  # Store original input
        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        self._trajectory_cache = _TrajectoryCache()
        self._dreamzero_precomputed_artifact_checked = False
        self._dreamzero_precomputed_artifact: DreamZeroPrecomputedFeatureArtifact | None = None
        self.tag = EmbodimentTag(embodiment_tag)
        # For dream and lapa, we use the global metadata since the lapa_actions and dream_actions are already normalized
        if self.tag == EmbodimentTag.DREAM or self.tag == EmbodimentTag.LAPA:
            self.use_global_metadata = True
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        # Notice: We also include discarded trajectories in stats for larger state
        # coverage, for questions please ask @Fengyuan Hu @Yuqi Xie
        self._lerobot_stats_meta = self._get_lerobot_stats_meta()

        # Initialize trajectory info and chunk size early (needed for relative stats calculation)
        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._trajectory_index_by_id = {
            int(trajectory_id): trajectory_index
            for trajectory_index, trajectory_id in enumerate(self._trajectory_ids)
        }
        self._data_path_pattern = self._get_data_path_pattern()
        self._chunk_size = self._get_chunk_size()

        # Set default relative_action_keys if not provided
        if self.relative_action and self._relative_action_keys_input is None:
            # Default: apply to all action keys except those containing 'gripper'
            default_action_config = ModalityConfig(delta_indices=[0], modality_keys=[])
            action_keys = self.modality_configs.get("action", default_action_config).modality_keys
            self.relative_action_keys = [
                k.replace("action.", "") for k in action_keys
                if "gripper" not in k.lower()
            ]
            logger.info("Relative action will be applied to keys: %s", self.relative_action_keys)
        # Load relative action stats if relative_action is enabled
        self._lerobot_relative_stats_meta = (
            self._get_lerobot_relative_stats_meta() if self.relative_action else {}
        )
        # Load per-horizon relative action stats if relative_action_per_horizon is enabled
        self._lerobot_relative_horizon_stats_meta = (
            self._get_lerobot_relative_horizon_stats_meta()
            if self.relative_action_per_horizon
            else {}
        )
        self._metadata = self._get_metadata()
        self._step_filter = self._get_step_filter()
        self._all_steps = self._get_all_steps()
        self._all_steps_index_by_identity = None
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._max_delta_index = self._get_max_delta_index()
        self._dataset_name = self._dataset_path.name

        # NOTE(YL): method to predict the task progress
        if "action.task_progress" in self._modality_keys["action"]:
            from .schema import StateActionMetadata

            logger.info("Adding task progress to the action modality")
            self._modality_keys["action"].append("action.task_progress")
            self._metadata.modalities.action["task_progress"] = StateActionMetadata(
                absolute=True, rotation_type=None, shape=(1,), continuous=True
            )
            # assume the task progress is uniformly distributed between 0 and 1
            self._metadata.statistics.action["task_progress"] = DatasetStatisticalValues(
                max=[1.0], min=[0.0], mean=[0.5], std=[0.2887], q01=[0.01], q99=[0.99]
            )

        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        logger.info("Initialized dataset %s with %s", self.dataset_name, embodiment_tag)

        # LeRobot-specific config (some already initialized above for relative stats)
        self._video_path_pattern = self._get_video_path_pattern()
        self._tasks = self._get_tasks()
        self._detailed_global_instructions = self._get_detailed_global_instructions()

        # Check if the dataset is valid
        self._check_integrity()
        self.clear_trajectory_cache()

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms.

        This is useful for transforms that need to know the metadata, such as the
        normalization values.
        """
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def state_dict(self) -> dict:
        """Capture worker RNG state for exact dataloader resume."""
        return _worker_rng_state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore worker RNG state after dataloader resume."""
        _load_worker_rng_state_dict(state_dict)

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        indices = {
            key: delta_indices + base_index for key, delta_indices in self.delta_indices.items()
        }
        data = self.get_step_data(trajectory_id, indices)
        if self.use_sample_transform_seed:
            seed = self.sample_transform_seed + int(index)
            with _temporary_transform_rng(seed):
                transformed = self.transforms(data)
                return self._maybe_attach_precomputed_features(
                    transformed,
                    index=index,
                    trajectory_id=trajectory_id,
                    base_index=base_index,
                )
        transformed = self.transforms(data)
        return self._maybe_attach_precomputed_features(
            transformed,
            index=index,
            trajectory_id=trajectory_id,
            base_index=base_index,
        )

    def get_step_data(self, trajectory_id: int, indices: dict[str, np.ndarray]) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            indices (dict[str, np.ndarray]): The indices for each modality.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                # Only load the data if the key is in the indices
                if key in indices:
                    data[key] = self.get_data_by_modality(
                        trajectory_id, modality, key, indices[key]
                    )
        return data


def safe_hash(input_tuple):
    """Generate a safe hash from an input tuple.

    Creates a deterministic hash using SHA256 and returns the lower 128 bits.
    This is used for deterministic random seed generation.

    Args:
        input_tuple: The tuple to hash.

    Returns:
        int: A 128-bit hash value.
    """
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class DreamZeroLeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the
    dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[DreamZeroLeRobotDataset, float]],
        training: bool,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        allow_padding_at_end: bool = False,
        use_sample_transform_seed: bool = False,
        sample_transform_seed: int = 0,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[DreamZeroLeRobotDataset, float]]): Datasets and their
                corresponding weights.
            training (bool): If True, __getitem__ will return different samples every
                epoch; if False, __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be
                multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a
                dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
            allow_padding_at_end (bool): If True, allow padding at the end of the dataset.
            use_sample_transform_seed (bool): Whether to reset torch/numpy/python RNG
                per sampled index before random transforms.
            sample_transform_seed (int): Base seed for per-sample transform RNG. The
                effective sample seed is ``sample_transform_seed + index``.
        """
        datasets: list[DreamZeroLeRobotDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)
        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.training = training
        self.allow_padding_at_end = allow_padding_at_end
        self.use_sample_transform_seed = bool(use_sample_transform_seed)
        self.sample_transform_seed = int(sample_transform_seed)

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])

        # 2. Dataset sampling weights
        if not self.datasets:
            raise ValueError("DreamZeroLeRobotMixtureDataset requires at least one dataset")
        self._dataset_sampling_weights = np.array(dataset_sampling_weights, dtype=np.float64)
        if not np.all(np.isfinite(self._dataset_sampling_weights)):
            raise ValueError(
                f"dataset sampling weights must be finite, got {dataset_sampling_weights}"
            )
        if np.any(self._dataset_sampling_weights < 0):
            raise ValueError(
                f"dataset sampling weights must be non-negative, got {dataset_sampling_weights}"
            )
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        dataset_weight_sum = float(self._dataset_sampling_weights.sum())
        if dataset_weight_sum <= 0.0:
            raise ValueError(
                "dataset sampling weights sum to zero after balancing; "
                f"dataset_lengths={self._dataset_lengths.tolist()} "
                f"raw_weights={list(dataset_sampling_weights)}"
            )
        self._dataset_sampling_weights /= dataset_weight_sum

        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for dataset in self.datasets:
            trajectory_sampling_weights = np.ones(len(dataset.trajectory_ids))
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= np.array(
                    [
                        len(dataset.step_filter[trajectory_id])
                        for trajectory_id in dataset.trajectory_ids
                    ]
                )

            if dataset.discard_bad_trajectories:
                discarded_trajectory_ids = dataset.get_discarded_trajectory_ids()
                bad_trajectory_indices = [
                    trajectory_index
                    for trajectory_index, trajectory_id in enumerate(dataset.trajectory_ids)
                    if int(trajectory_id) in discarded_trajectory_ids
                ]
                trajectory_sampling_weights[bad_trajectory_indices] = 0.0

            if trajectory_sampling_weights.sum() == 0:
                raise ValueError(f"No valid trajectories found for dataset {dataset}")

            trajectory_sampling_weights /= trajectory_sampling_weights.sum()
            self._trajectory_sampling_weights.append(trajectory_sampling_weights)
            if not self.allow_padding_at_end:
                min_trajectory_length = int(dataset.max_delta_index) + 1
                trajectory_lengths = np.asarray(dataset.trajectory_lengths, dtype=np.int64)
                eligible_weight_sum = float(
                    trajectory_sampling_weights[trajectory_lengths >= min_trajectory_length].sum()
                )
                if eligible_weight_sum <= 0.0:
                    max_trajectory_length = (
                        int(trajectory_lengths.max()) if len(trajectory_lengths) > 0 else 0
                    )
                    raise ValueError(
                        "allow_padding_at_end=False requires at least one sampled "
                        f"trajectory with length >= max_delta_index + 1 "
                        f"({min_trajectory_length}); dataset={dataset.dataset_path} "
                        f"max_trajectory_length={max_trajectory_length}"
                    )

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        # Create a merged metadata for the mixture dataset (we don't need this in the
        # future as eval will directly use `get_metadata`)
        self.update_metadata(metadata_config)

        # Set the transforms to training or evaluation mode
        if self.training:
            for dataset in self.datasets:
                dataset.transforms.train()
        else:
            for dataset in self.datasets:
                dataset.transforms.eval()

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    def __str__(self) -> str:
        """Return a string representation of the mixture dataset with weights."""
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return yaml.dump({"Mixture dataset": dataset_descriptions})

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch
        # self.sampled_steps = self.sample_epoch()

    def state_dict(self) -> dict:
        """Capture worker RNG state for exact dataloader resume."""
        return _worker_rng_state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore worker RNG state after dataloader resume."""
        _load_worker_rng_state_dict(state_dict)

    def sample_step(self, index: int) -> tuple[DreamZeroLeRobotDataset, int, int]:
        """Sample a single step from the mixture dataset.

        Args:
            index (int): The index to sample (used for deterministic sampling).

        Returns:
            tuple[DreamZeroLeRobotDataset, int, int]: A tuple of (dataset, trajectory_id, step_index).
        """
        # return self.sampled_steps[index]

        # Set seed
        if self.training:
            seed = safe_hash((self.epoch, index, self.seed))
            rng = np.random.default_rng(seed)

            # Sample dataset
            dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
            dataset = self.datasets[dataset_index]

            if self.allow_padding_at_end:
                # Sample trajectory
                trajectory_index = rng.choice(
                    len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index]
                )
                trajectory_id = dataset.trajectory_ids[trajectory_index]

                allowed_length = dataset.trajectory_lengths[trajectory_index]
            else:
                # Avoid padding at the end of the trajectory
                max_delta_index = dataset.max_delta_index
                trajectory_length = 0
                trajectory_id = None
                while trajectory_length < max_delta_index + 1:
                    # Sample trajectory
                    trajectory_index = rng.choice(
                        len(dataset.trajectory_ids),
                        p=self.trajectory_sampling_weights[dataset_index],
                    )
                    trajectory_id = dataset.trajectory_ids[trajectory_index]
                    trajectory_length = dataset.trajectory_lengths[trajectory_index]
                assert trajectory_id is not None

                # Sample step
                assert (
                    trajectory_length >= max_delta_index + 1
                ), f"{trajectory_length=}, {max_delta_index=}"
                allowed_length = trajectory_length - max_delta_index
            # Get the allowed indices from the step filter
            allowed_indices = dataset.step_filter[trajectory_id]
            # Remove indices that are too large
            # Keep the inclusive upper bound; changing this alters sampled
            # anchors and requires revalidating loss.
            allowed_indices = allowed_indices[allowed_indices <= allowed_length]
            step_index = rng.choice(allowed_indices)
            return dataset, trajectory_id, step_index
        else:
            length_cumsum = np.cumsum(self.dataset_lengths)
            dataset_index = np.searchsorted(length_cumsum, index)
            dataset = self.datasets[dataset_index]
            assert (
                len(dataset.lerobot_info_meta.get("discarded_episode_indices", [])) == 0
            ), f"Find discarded episode indices in evaluation dataset {dataset.dataset_path}"
            trajectory_id, step_index = dataset.all_steps[index - length_cumsum[dataset_index]]
            return dataset, trajectory_id, step_index

    @staticmethod
    def _maybe_attach_dataset_precomputed_features(
        dataset: DreamZeroLeRobotDataset,
        data: dict,
        *,
        index: int,
        trajectory_id: int,
        base_index: int,
    ) -> dict:
        """Attach precomputed features to ``data`` using the dataset's flat sample index."""
        flat_index = dataset.get_flat_index_for_step(trajectory_id, base_index)
        if flat_index is None:
            flat_index = int(index)
        return dataset._maybe_attach_precomputed_features(
            data,
            index=int(flat_index),
            trajectory_id=int(trajectory_id),
            base_index=int(base_index),
        )

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """
        dataset, trajectory_id, step_index = self.sample_step(index)
        indices = {
            key: delta_indices + step_index for key, delta_indices in dataset.delta_indices.items()
        }
        if self.use_sample_transform_seed:
            seed = self.sample_transform_seed + int(index)
            with _temporary_transform_rng(seed):
                data = dataset.get_step_data(trajectory_id, indices)
                transformed = dataset.transforms(data)
                return self._maybe_attach_dataset_precomputed_features(
                    dataset,
                    transformed,
                    index=int(index),
                    trajectory_id=int(trajectory_id),
                    base_index=int(step_index),
                )
        data = dataset.get_step_data(trajectory_id, indices)
        transformed = dataset.transforms(data)
        return self._maybe_attach_dataset_precomputed_features(
            dataset,
            transformed,
            index=int(index),
            trajectory_id=int(trajectory_id),
            base_index=int(step_index),
        )

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        if self.training:
            return int((self.dataset_lengths * self.dataset_sampling_weights).sum())
        else:
            return int(self.dataset_lengths.sum())

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys
        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            # Check if stats are per-horizon (2D) by examining the first task's mean
            first_mean = np.array(per_task_stats[0][modality]["mean"])
            is_per_horizon = first_mean.ndim == 2  # Shape (horizon_len, action_dim)

            if is_per_horizon:
                # Handle per-horizon stats (2D arrays)
                stats_shape = first_mean.shape  # (horizon_len, action_dim)

                # Initialize accumulators for means and variances
                weighted_means = np.zeros(stats_shape)
                weighted_squares = np.zeros(stats_shape)

                # Collect min, max, q01, q99 from all tasks
                min_list = []
                max_list = []
                q01_list = []
                q99_list = []

                for task_idx, task_stats in enumerate(per_task_stats):
                    w_i = normalized_weights[task_idx]
                    stats = task_stats[modality]
                    means = np.array(stats["mean"])
                    stds = np.array(stats["std"])

                    # Update weighted sums for mean and variance
                    weighted_means += w_i * means
                    weighted_squares += w_i * (stds**2 + means**2)

                    # Collect min, max, q01, q99
                    min_list.append(np.array(stats["min"]))
                    max_list.append(np.array(stats["max"]))
                    q01_list.append(np.array(stats["q01"]))
                    q99_list.append(np.array(stats["q99"]))

                # Compute overall mean
                overall_mean = weighted_means.tolist()

                # Compute overall variance and std deviation
                overall_variance = weighted_squares - weighted_means**2
                overall_std = np.sqrt(np.maximum(overall_variance, 0)).tolist()

                # Compute overall min and max per dimension
                # Stack along new axis: (num_tasks, horizon_len, action_dim)
                overall_min = np.min(np.stack(min_list, axis=0), axis=0).tolist()
                overall_max = np.max(np.stack(max_list, axis=0), axis=0).tolist()

                # Compute overall q01 and q99 per dimension
                q01_array = np.stack(q01_list, axis=0)  # (num_tasks, horizon_len, action_dim)
                q99_array = np.stack(q99_list, axis=0)
                if percentile_mixing_method == "weighted_average":
                    # Weighted average along task axis
                    weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                    weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                elif percentile_mixing_method == "min_max":
                    weighted_q01 = np.min(q01_array, axis=0).tolist()
                    weighted_q99 = np.max(q99_array, axis=0).tolist()
                else:
                    raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")
            else:
                # Handle regular stats (1D arrays)
                num_dims = len(first_mean)

                # Initialize accumulators for means and variances
                weighted_means = np.zeros(num_dims)
                weighted_squares = np.zeros(num_dims)

                # Collect min, max, q01, q99 from all tasks
                min_list = []
                max_list = []
                q01_list = []
                q99_list = []

                for task_idx, task_stats in enumerate(per_task_stats):
                    w_i = normalized_weights[task_idx]
                    stats = task_stats[modality]
                    means = np.array(stats["mean"])
                    stds = np.array(stats["std"])

                    # Update weighted sums for mean and variance
                    weighted_means += w_i * means
                    weighted_squares += w_i * (stds**2 + means**2)

                    # Collect min, max, q01, q99
                    min_list.append(stats["min"])
                    max_list.append(stats["max"])
                    q01_list.append(stats["q01"])
                    q99_list.append(stats["q99"])

                # Compute overall mean
                overall_mean = weighted_means.tolist()

                # Compute overall variance and std deviation
                overall_variance = weighted_squares - weighted_means**2
                overall_std = np.sqrt(np.maximum(overall_variance, 0)).tolist()

                # Compute overall min and max per dimension
                overall_min = np.min(np.array(min_list), axis=0).tolist()
                overall_max = np.max(np.array(max_list), axis=0).tolist()

                # Compute overall q01 and q99 per dimension
                # Use weighted average of per-task quantiles
                q01_array = np.array(q01_list)
                q99_array = np.array(q99_list)
                if percentile_mixing_method == "weighted_average":
                    weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                    weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                elif percentile_mixing_method == "min_max":
                    weighted_q01 = np.min(q01_array, axis=0).tolist()
                    weighted_q99 = np.max(q99_array, axis=0).tolist()
                else:
                    raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = DreamZeroLeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = DreamZeroLeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            # Check that all modality configs correspond to the same tag matches
            assert (
                len(configs) == 1
            ), f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict) -> None:
        """Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either
                    "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles
                        using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th
                        percentile.
        """

        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # Group metadata by tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        for dataset in self.datasets:
            if dataset.tag.value not in all_metadatas:
                all_metadatas[dataset.tag.value] = []
            all_metadatas[dataset.tag.value].append(dataset.metadata)
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=self.dataset_sampling_weights.tolist(),
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag.value])

    def get_initial_actions(self):
        """Collect initial actions from every underlying dataset in the mixture."""
        initial_actions = []
        for dataset in self.datasets:
            initial_actions.extend(dataset.get_initial_actions())
        return initial_actions
