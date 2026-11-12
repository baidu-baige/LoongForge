# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""On-disk layout constants for DreamZero's LeRobot-format datasets.

Filenames follow the LeRobot dataset layout (``meta/*`` + ``data/*/*.parquet``)
plus DreamZero/GEAR-specific extensions (modality/relative-stats/step-filter/
detailed-instruction files).
"""

from pathlib import Path

LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_DETAILED_GLOBAL_INSTRUCTION_FILENAME = "meta/episodes_detail_global_instruction.jsonl"
INITIAL_ACTIONS_FILENAME = "meta/initial_actions.npz"
METADATA_DIR = Path(__file__).resolve().parent / "metadata"
STEP_FILTER_FILENAME = "meta/step_filter.jsonl"
LEROBOT_RELATIVE_STATS_FILE_NAME = "meta/relative_stats_dreamzero.json"
LEROBOT_RELATIVE_HORIZON_STATS_FILE_NAME = "meta/relative_horizon_stats_dreamzero.json"

# Special language keys that load from metadata files instead of parquet columns
METADATA_LANG_KEYS = ["detailed_global_instruction_medium", "detailed_global_instruction_concise"]
