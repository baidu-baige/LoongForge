# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingbotVLA-V2 DataConfig — data-processing parameters (YAML ``data:`` section).

Only data-pipeline fields live here. Fields that influence the model structure
(action_dim, chunk_size, tokenizer_max_length, ...) are defined once in
``LingbotVLAV2ModelConfig`` and read from ``model_cfg`` on the data side.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class LingbotVLAV2DataConfig:
    """lingbot_vla_v2 data-processing config (maps 1:1 to YAML ``data:`` section)."""

    # MultiVLADataset (LeRobot-based) selection
    data_name: str = "multi"
    robot_config_root: str = ""
    prompt_type: str = "global"

    # Joint layout / normalization — entries are python-literal strings,
    # e.g. "{'arm.position': 14}", matching the benchmark YAML format.
    joints: List[str] = field(
        default_factory=lambda: [
            "{'arm.position': 14}",
            "{'end.position': 14}",
            "{'effector.position': 2}",
        ]
    )
    cameras: List[str] = field(
        default_factory=lambda: [
            "camera_top",
            "camera_wrist_left",
            "camera_wrist_right",
        ]
    )
    norm_type: List[str] = field(
        default_factory=lambda: [
            "{'arm.position': 'bounds_99_woclip'}",
            "{'end.position': 'bounds_99_woclip'}",
            "{'effector.position': 'bounds_99_woclip'}",
        ]
    )
    state_norm_type: Optional[str] = None

    # Image pipeline
    img_size: int = 256
    image_augment: bool = False
    use_future_image: bool = True

    # Video decode backend for LeRobot datasets
    video_backend: str = "torchcodec"
