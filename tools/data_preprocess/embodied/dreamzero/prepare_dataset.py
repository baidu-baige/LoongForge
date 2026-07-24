# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.

"""Prepare a LeRobot v2 dataset for LoongForge DreamZero training.

The tool validates the source LeRobot layout and writes only DreamZero/GEAR
metadata under ``meta/``. It does not copy, transcode, or modify parquet and
image/video payloads. Frozen-model feature cache generation remains a separate,
optional step handled by ``precompute_features.py``.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


logger = logging.getLogger(__name__)
_STAT_NAMES = ("max", "min", "mean", "std", "q01", "q99")


@dataclass(frozen=True)
class VectorField:
    """One named slice in a packed LeRobot state or action column."""

    name: str
    original_key: str
    start: int
    end: int


@dataclass(frozen=True)
class DreamZeroDatasetPreset:
    """Expected LeRobot schema for one supported DreamZero embodiment."""

    state: tuple[VectorField, ...]
    action: tuple[VectorField, ...]
    video: tuple[tuple[str, str], ...]
    annotation: tuple[tuple[str, str], ...]
    relative_action_keys: tuple[str, ...] = ()
    action_horizon: int = 24


PRESETS = {
    "oxe_droid": DreamZeroDatasetPreset(
        state=(
            VectorField("joint_position", "observation.state", 7, 14),
            VectorField("gripper_position", "observation.state", 6, 7),
        ),
        action=(
            VectorField("joint_position", "action", 14, 21),
            VectorField("gripper_position", "action", 12, 13),
        ),
        video=(
            ("exterior_image_1_left", "observation.images.exterior_image_1_left"),
            ("exterior_image_2_left", "observation.images.exterior_image_2_left"),
            ("wrist_image_left", "observation.images.wrist_image_left"),
        ),
        annotation=(
            (
                "language.language_instruction",
                "annotation.language.language_instruction",
            ),
            (
                "language.language_instruction_2",
                "annotation.language.language_instruction_2",
            ),
            (
                "language.language_instruction_3",
                "annotation.language.language_instruction_3",
            ),
        ),
        relative_action_keys=("joint_position",),
    ),
    "libero_sim": DreamZeroDatasetPreset(
        state=(VectorField("state", "state", 0, 8),),
        action=(VectorField("actions", "actions", 0, 7),),
        video=(("image", "image"), ("wrist_image", "wrist_image")),
        annotation=(("task", "task_index"),),
        action_horizon=16,
    ),
    "agibot": DreamZeroDatasetPreset(
        state=(
            VectorField("left_arm_joint_position", "observation.state", 0, 7),
            VectorField("right_arm_joint_position", "observation.state", 7, 14),
            VectorField("left_effector_position", "observation.state", 14, 15),
            VectorField("right_effector_position", "observation.state", 15, 16),
            VectorField("head_position", "observation.state", 16, 18),
            VectorField("waist_position", "observation.state", 18, 20),
        ),
        action=(
            VectorField("left_arm_joint_position", "action", 0, 7),
            VectorField("right_arm_joint_position", "action", 7, 14),
            VectorField("left_effector_position", "action", 14, 15),
            VectorField("right_effector_position", "action", 15, 16),
            VectorField("head_position", "action", 16, 18),
            VectorField("waist_position", "action", 18, 20),
            VectorField("robot_velocity", "action", 20, 22),
        ),
        video=(
            ("top_head", "observation.images.top_head"),
            ("hand_left", "observation.images.hand_left"),
            ("hand_right", "observation.images.hand_right"),
        ),
        annotation=(("language.action_text", "task_index"),),
        relative_action_keys=(
            "left_arm_joint_position",
            "right_arm_joint_position",
            "left_effector_position",
            "right_effector_position",
            "head_position",
            "waist_position",
        ),
    ),
    "yam": DreamZeroDatasetPreset(
        state=(
            VectorField("left_joint_pos", "observation.state", 34, 40),
            VectorField("left_gripper_pos", "observation.state", 32, 33),
            VectorField("right_joint_pos", "observation.state", 40, 46),
            VectorField("right_gripper_pos", "observation.state", 33, 34),
        ),
        action=(
            VectorField("left_joint_pos", "action", 34, 40),
            VectorField("left_gripper_pos", "action", 32, 33),
            VectorField("right_joint_pos", "action", 40, 46),
            VectorField("right_gripper_pos", "action", 33, 34),
        ),
        video=(
            ("top_camera-images-rgb", "observation.images.top_camera-images-rgb"),
            ("left_camera-images-rgb", "observation.images.left_camera-images-rgb"),
            ("right_camera-images-rgb", "observation.images.right_camera-images-rgb"),
        ),
        annotation=(("task", "task_index"),),
        relative_action_keys=(
            "left_joint_pos",
            "left_gripper_pos",
            "right_joint_pos",
            "right_gripper_pos",
        ),
    ),
}


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=4)
        file.write("\n")


def _dataset_parquet_paths(dataset_path: Path) -> list[Path]:
    return sorted((dataset_path / "data").glob("**/*.parquet"))


def _feature_dimension(feature: dict) -> int | None:
    shape = feature.get("shape")
    if isinstance(shape, list) and shape and isinstance(shape[0], int):
        return shape[0]
    return None


def _validate_source_dataset(
    dataset_path: Path,
    info: dict,
    preset: DreamZeroDatasetPreset,
) -> tuple[list[Path], bool]:
    required_meta = ("info.json", "tasks.jsonl", "episodes.jsonl")
    missing_meta = [name for name in required_meta if not (dataset_path / "meta" / name).is_file()]
    if missing_meta:
        raise FileNotFoundError(
            f"LeRobot metadata is incomplete under {dataset_path / 'meta'}: {missing_meta}"
        )

    parquet_paths = _dataset_parquet_paths(dataset_path)
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_path / 'data'}")
    declared_episodes = info.get("total_episodes")
    if not isinstance(declared_episodes, int) or declared_episodes <= 0:
        raise ValueError("meta/info.json must contain a positive total_episodes")
    dataset_complete = len(parquet_paths) == declared_episodes
    if not dataset_complete:
        logger.warning(
            "Found %d parquet files, but info.json declares %d episodes; "
            "existing metadata can be validated, but statistics will not be regenerated",
            len(parquet_paths),
            declared_episodes,
        )

    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("meta/info.json must contain a features mapping")
    codebase_version = str(info.get("codebase_version", ""))
    if not codebase_version.startswith("v2"):
        raise ValueError(
            f"DreamZero preparation currently requires LeRobot v2, got "
            f"codebase_version={codebase_version!r}"
        )
    if not info.get("fps"):
        raise ValueError("meta/info.json must contain a positive fps")

    for field in preset.state + preset.action:
        feature = features.get(field.original_key)
        if feature is None:
            raise ValueError(
                f"Preset requires feature {field.original_key!r}, but it is absent from info.json"
            )
        dimension = _feature_dimension(feature)
        if (
            dimension is None
            or field.start < 0
            or field.start >= field.end
            or field.end > dimension
        ):
            raise ValueError(
                f"Invalid preset slice {field.name}="
                f"{field.original_key}[{field.start}:{field.end}] "
                f"for feature shape {feature.get('shape')}"
            )

    for _, original_key in preset.video + preset.annotation:
        if original_key not in features:
            raise ValueError(
                f"Preset requires feature {original_key!r}, but it is absent from info.json"
            )

    video_features = [features[original_key] for _, original_key in preset.video]
    if any(feature.get("dtype") == "video" for feature in video_features):
        if not (dataset_path / "videos").is_dir():
            raise FileNotFoundError(
                f"Video-backed dataset is missing directory: {dataset_path / 'videos'}"
            )

    return parquet_paths, dataset_complete


def _vector_metadata(field: VectorField, info: dict) -> dict:
    feature = info["features"][field.original_key]
    return {
        "original_key": field.original_key,
        "start": field.start,
        "end": field.end,
        "rotation_type": None,
        "absolute": True,
        "dtype": feature.get("dtype", "float32"),
        "range": None,
    }


def _build_modality_metadata(info: dict, preset: DreamZeroDatasetPreset) -> dict:
    return {
        "state": {field.name: _vector_metadata(field, info) for field in preset.state},
        "action": {field.name: _vector_metadata(field, info) for field in preset.action},
        "video": {
            name: {"original_key": original_key}
            for name, original_key in preset.video
        },
        "annotation": {
            name: {"original_key": original_key}
            for name, original_key in preset.annotation
        },
    }


def _effective_original_key(modality: str, name: str, metadata: dict) -> str:
    original_key = metadata.get("original_key")
    if original_key is not None:
        return original_key
    if modality == "state":
        return "observation.state"
    if modality == "action":
        return "action"
    return f"{modality}.{name}"


def _validate_existing_modality(actual: dict, expected: dict) -> None:
    for modality, expected_fields in expected.items():
        actual_fields = actual.get(modality, {})
        for name, expected_metadata in expected_fields.items():
            if name not in actual_fields:
                raise ValueError(f"modality.json is missing {modality}.{name}")
            actual_metadata = actual_fields[name]
            actual_original_key = _effective_original_key(modality, name, actual_metadata)
            if actual_original_key != expected_metadata["original_key"]:
                raise ValueError(
                    f"modality.json {modality}.{name}.original_key={actual_original_key!r}; "
                    f"expected {expected_metadata['original_key']!r}"
                )
            if modality in {"state", "action"}:
                for bound in ("start", "end"):
                    if actual_metadata.get(bound) != expected_metadata[bound]:
                        raise ValueError(
                            f"modality.json {modality}.{name}.{bound}="
                            f"{actual_metadata.get(bound)!r}; expected {expected_metadata[bound]}"
                        )


def _prepare_modality_metadata(
    meta_dir: Path,
    expected: dict,
    force: bool,
) -> None:
    path = meta_dir / "modality.json"
    if path.exists() and not force:
        _validate_existing_modality(_load_json(path), expected)
        logger.info("Validated existing %s", path)
        return
    _write_json(path, expected)
    logger.info("Wrote %s", path)


def _prepare_embodiment_metadata(meta_dir: Path, embodiment_tag: str, force: bool) -> None:
    path = meta_dir / "embodiment.json"
    expected = {"robot_type": embodiment_tag, "embodiment_tag": embodiment_tag}
    if path.exists() and not force:
        actual = _load_json(path)
        if actual.get("embodiment_tag") != embodiment_tag:
            raise ValueError(
                f"{path} uses embodiment_tag={actual.get('embodiment_tag')!r}; "
                f"expected {embodiment_tag!r}. Use --force to replace it."
            )
        logger.info("Validated existing %s", path)
        return
    _write_json(path, expected)
    logger.info("Wrote %s", path)


def _statistics_are_usable(statistics: dict, required_columns: tuple[str, ...]) -> bool:
    if not all(column in statistics for column in required_columns):
        return False
    return all(
        isinstance(values, dict) and all(name in values for name in _STAT_NAMES)
        for key, values in statistics.items()
        if key not in {"num_trajectories", "total_trajectory_length"}
    )


def _summarize(values: np.ndarray) -> dict[str, list]:
    return {
        "max": np.max(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _compute_dataset_statistics(
    parquet_paths: list[Path],
    columns: tuple[str, ...],
) -> dict:
    collected: dict[str, list[np.ndarray]] = {column: [] for column in columns}
    for parquet_path in tqdm(parquet_paths, desc="Collecting dataset statistics"):
        frame = pd.read_parquet(parquet_path, columns=list(columns))
        for column in columns:
            if column not in frame:
                raise ValueError(f"Parquet file {parquet_path} is missing column {column!r}")
            values = np.stack(frame[column].to_numpy()).astype(np.float32, copy=False)
            if values.ndim == 1:
                values = values[:, None]
            collected[column].append(values)
    return {
        column: _summarize(np.concatenate(column_values, axis=0))
        for column, column_values in collected.items()
    }


def _prepare_dataset_statistics(
    meta_dir: Path,
    parquet_paths: list[Path],
    preset: DreamZeroDatasetPreset,
    force: bool,
    dataset_complete: bool,
    allow_partial_statistics: bool,
) -> None:
    path = meta_dir / "stats.json"
    columns = tuple(
        dict.fromkeys(field.original_key for field in preset.state + preset.action)
    )
    if path.exists() and not force:
        statistics = _load_json(path)
        if _statistics_are_usable(statistics, columns):
            logger.info("Validated existing %s", path)
            return
        logger.info("Existing %s lacks DreamZero percentile statistics; regenerating", path)
    if not dataset_complete and not allow_partial_statistics:
        raise ValueError(
            "Refusing to generate stats.json from a partial dataset. Complete the dataset "
            "download or pass --allow-partial-statistics for an intentional subset."
        )
    statistics = _compute_dataset_statistics(parquet_paths, columns)
    _write_json(path, statistics)
    logger.info("Wrote %s", path)


def _relative_statistics_are_usable(statistics: dict, keys: tuple[str, ...]) -> bool:
    return all(
        key in statistics
        and isinstance(statistics[key], dict)
        and all(name in statistics[key] for name in _STAT_NAMES)
        for key in keys
    )


def _sample_relative_paths(parquet_paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(parquet_paths) <= limit:
        return parquet_paths
    generator = np.random.default_rng(seed=42)
    indices = sorted(generator.choice(len(parquet_paths), size=limit, replace=False).tolist())
    logger.info(
        "Sampling %d of %d episodes for relative-action statistics",
        len(indices),
        len(parquet_paths),
    )
    return [parquet_paths[index] for index in indices]


def _compute_relative_statistics(
    parquet_paths: list[Path],
    preset: DreamZeroDatasetPreset,
) -> dict:
    state_fields = {field.name: field for field in preset.state}
    action_fields = {field.name: field for field in preset.action}
    relative_values: dict[str, list[np.ndarray]] = {
        key: [] for key in preset.relative_action_keys
    }
    columns = sorted(
        {
            field.original_key
            for key in preset.relative_action_keys
            for field in (state_fields[key], action_fields[key])
        }
    )

    for parquet_path in tqdm(parquet_paths, desc="Collecting relative-action statistics"):
        frame = pd.read_parquet(parquet_path, columns=columns)
        arrays = {
            column: np.stack(frame[column].to_numpy()).astype(np.float32, copy=False)
            for column in columns
        }
        for key in preset.relative_action_keys:
            state_field = state_fields[key]
            action_field = action_fields[key]
            state = arrays[state_field.original_key][
                :, state_field.start : state_field.end
            ]
            action = arrays[action_field.original_key][
                :, action_field.start : action_field.end
            ]
            if state.shape[1] != action.shape[1]:
                raise ValueError(
                    f"Relative-action dimensions differ for {key}: "
                    f"state={state.shape[1]}, action={action.shape[1]}"
                )
            usable_length = len(frame) - preset.action_horizon + 1
            if usable_length <= 0:
                continue
            reference_state = state[:usable_length]
            for offset in range(preset.action_horizon):
                relative_values[key].append(
                    action[offset : offset + usable_length] - reference_state
                )

    missing_keys = [key for key, values in relative_values.items() if not values]
    if missing_keys:
        raise ValueError(f"No relative-action samples were produced for keys: {missing_keys}")
    return {
        key: _summarize(np.concatenate(values, axis=0))
        for key, values in relative_values.items()
    }


def _prepare_relative_statistics(
    meta_dir: Path,
    parquet_paths: list[Path],
    preset: DreamZeroDatasetPreset,
    force: bool,
    max_episodes: int,
    dataset_complete: bool,
    allow_partial_statistics: bool,
) -> None:
    if not preset.relative_action_keys:
        return
    path = meta_dir / "relative_stats_dreamzero.json"
    if path.exists() and not force:
        statistics = _load_json(path)
        if _relative_statistics_are_usable(statistics, preset.relative_action_keys):
            logger.info("Validated existing %s", path)
            return
    if not dataset_complete and not allow_partial_statistics:
        raise ValueError(
            "Refusing to generate relative-action statistics from a partial dataset. "
            "Complete the dataset download or pass --allow-partial-statistics for an "
            "intentional subset."
        )
    selected_paths = _sample_relative_paths(parquet_paths, max_episodes)
    statistics = _compute_relative_statistics(selected_paths, preset)
    _write_json(path, statistics)
    logger.info("Wrote %s", path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate LeRobot data and generate DreamZero/GEAR metadata without "
            "modifying parquet or image/video payloads."
        )
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--embodiment-tag", choices=sorted(PRESETS), required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing modality, embodiment, and statistics metadata.",
    )
    parser.add_argument(
        "--skip-statistics",
        action="store_true",
        help="Generate schema metadata only; training may calculate missing statistics.",
    )
    parser.add_argument(
        "--max-relative-stat-episodes",
        type=int,
        default=10_000,
        help="Maximum sampled episodes for relative-action statistics; <=0 uses all.",
    )
    parser.add_argument(
        "--allow-partial-statistics",
        action="store_true",
        help="Allow statistics generation when parquet count differs from total_episodes.",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare one supported LeRobot dataset for DreamZero training."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset_path = args.dataset_path.expanduser().resolve()
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot info metadata not found: {info_path}")

    preset = PRESETS[args.embodiment_tag]
    info = _load_json(info_path)
    parquet_paths, dataset_complete = _validate_source_dataset(dataset_path, info, preset)
    meta_dir = dataset_path / "meta"

    expected_modality = _build_modality_metadata(info, preset)
    _prepare_modality_metadata(meta_dir, expected_modality, args.force)
    _prepare_embodiment_metadata(meta_dir, args.embodiment_tag, args.force)
    if not args.skip_statistics:
        _prepare_dataset_statistics(
            meta_dir,
            parquet_paths,
            preset,
            args.force,
            dataset_complete,
            args.allow_partial_statistics,
        )
        _prepare_relative_statistics(
            meta_dir,
            parquet_paths,
            preset,
            args.force,
            args.max_relative_stat_episodes,
            dataset_complete,
            args.allow_partial_statistics,
        )

    logger.info(
        "DreamZero dataset preparation complete: path=%s, embodiment=%s, episodes=%d",
        dataset_path,
        args.embodiment_tag,
        len(parquet_paths),
    )


if __name__ == "__main__":
    main()
