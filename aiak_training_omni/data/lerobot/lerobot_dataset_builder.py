"""LeRobot dataset builder utilities for PI0.5 (Pi05)."""

from __future__ import annotations

from typing import Any
import torch

from lerobot_dataset_config import LeRobotDatasetConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig
from lerobot.configs.policies import PreTrainedConfig

ACTION = "action"
REWARD = "next.reward"
OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def build_lerobot_dataset(
    cfg: LeRobotDatasetConfig,
    policy: PreTrainedConfig | None = None
) -> LeRobotDataset:
    """Instantiate a `LeRobotDataset`.

    Args:
        cfg (LeRobotDatasetConfig): The dataset configuration.
        policy (PreTrainedConfig | None): Optional policy configuration to resolve delta timestamps.

    Returns:
        LeRobotDataset: The instantiated dataset.

    Notes:
        LeRobotDataset itself does not have a built-in train/val/test split arg.
        Splitting should be done by selecting episode indices (cfg.episodes) or
        by wrapping with `torch.utils.data.Subset`.
    """
    image_transforms = (
        ImageTransforms(cfg.image_transforms) if cfg.image_transforms.enable else None
    )
    ds_meta = LeRobotDatasetMetadata(
        cfg.repo_id, root=cfg.root, revision=cfg.revision
    )

    delta_timestamps = None
    if policy is not None:
        delta_timestamps = resolve_delta_timestamps(policy, ds_meta)

    dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        root=cfg.root,
        episodes=cfg.episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=cfg.revision,
        video_backend=cfg.video_backend,
        tolerance_s=cfg.tolerance_s,
    )

    if cfg.use_imagenet_stats:
        # 当使用在 ImageNet 上预训练的模型（例如 ResNet、ViT）的均值和标准差覆盖数据集统计数据。
        # 该方法使输入分布与模型期望的分布保持一致。
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset



def get_lerobot_dataset_stats(dataset: Any) -> dict[str, Any] | None:
    """Return LeRobot dataset stats if available."""

    meta = getattr(dataset, "meta", None)
    stats = getattr(meta, "stats", None) if meta is not None else None
    return stats
