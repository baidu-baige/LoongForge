# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Wall-OSS-0.5 DataConfig."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class WallOss05DataConfig:
    """Data-processing config for Wall-OSS-0.5 LeRobot training."""

    episodes: Optional[List[int]] = field(
        default_factory=lambda: list(range(20))
    )
    key_mappings: Dict = field(
        default_factory=lambda: {
            "camera": {
                "observation.images.image": "face_view",
                "observation.images.image2": "right_wrist_view",
            },
            "state": "observation.state",
            "action": "action",
        }
    )
    norm_stats_path: str = "/workspace/libero_norm_stats.json"
    train_test_split: float = 0.95
    max_length: int = 1024
    resolution: Dict[str, int] = field(
        default_factory=lambda: {
            "face_view": 256,
            "right_wrist_view": 256,
        }
    )
    padding_side: str = "left"
    use_fast_tokenizer: bool = False
    action_tokenizer_path: Optional[str] = None
    noise_scheduler: Dict = field(default_factory=dict)
    priority_order: Optional[Dict[str, float]] = None
    camera_name_mapping: Optional[Dict[str, str]] = None
    generate_subtask_ratio: float = 0.0
    video_backend: str = "pyav"
    state_bins: int = 256
