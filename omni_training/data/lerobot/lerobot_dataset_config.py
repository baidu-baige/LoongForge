"""LeRobot dataset config for PI0.5 (Pi05) training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.datasets.video_utils import get_safe_default_codec

@dataclass
class LeRobotDatasetConfig:
    """Minimal dataset config for LeRobot-backed training.
    Attributes:
        repo_id (str): The repository ID for the dataset.
        root (str | None): The root directory for the dataset.
        episodes (list[int] | None): List of episode indices to include in the dataset.
        image_transforms (ImageTransformsConfig): Configuration for image transformations.
        revision (str | None): The revision of the dataset repository.
        use_imagenet_stats (bool): Whether to use ImageNet statistics for normalization.
        video_backend (str): The video backend to use for loading videos.
        streaming (bool): Whether to use streaming mode for dataset loading.    
    """
    repo_id: str
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    streaming: bool = False
    tolerance_s: float = 1e-4

