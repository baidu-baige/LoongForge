# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Adapted from MIT-licensed VPP2.
import io
import os
import traceback
from collections import OrderedDict, defaultdict
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Optional

import av
import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as transforms_F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from .transforms.image import (
    CenterCrop,
    Normalize,
    ResizeSmallestSideAspectPreserving,
)
from .transforms.normalizer import (
    load_dataset_stats_from_json,
    save_dataset_stats_to_json,
)

logger = get_logger(__name__)
_work_dir = Path.cwd()


def register_work_dir(path):
    global _work_dir
    _work_dir = Path(path)


def get_work_dir():
    return str(_work_dir)


def _is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


class VideoEventDataset(torch.utils.data.Dataset):
    """Dataset for event-level robot clips stored as CSV + videos + parquet arrays."""

    def __init__(
        self,
        metadata_path: str | list[str] = "data/train.csv",
        video_root: Optional[str | list[str]] = None,
        sample_ratio: Optional[list[float]] = None,
        balance_column: Optional[str] = None,
        balance_weight_overrides: Optional[dict[str, float]] = None,
        segment_sampling_mode: str = "step_uniform",
        min_valid_action_steps: int = 12,
        shape_meta: Optional[dict[str, Any]] = None,
        num_frames: int = 33,
        action_horizon: Optional[int] = None,
        video_size: list[int] = [240, 416],
        video_resize_mode: str = "center_crop",
        processor: Optional[Any] = None,
        context_len: int = 512,
        text_embedding_cache_dir: Optional[str] = None,
        text_embedding_cache_tag: str = "wan22ti2v5b",
        allow_missing_text_embeddings: bool = True,
        pretrained_norm_stats: Optional[Any] = None,
        pretrained_norm_stats_relative: Optional[Any] = None,
        use_relative_action: bool = False,
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        seed: int = 42,
        global_sample_stride: int = 1,
        action_time_offset: int = 0,
        action_video_freq_ratio: int = 1,
        decode_only_first_video_frame: bool = False,
        decode_sampled_video_frames: bool = False,
        condition_history_frames: int = 0,
        condition_history_stride: Optional[int] = None,
        condition_history_stride_choices: Optional[list[int]] = None,
        condition_history_stride_probs: Optional[list[float]] = None,
        condition_history_jitter_probability: float = 0.0,
        condition_history_jitter_max_native_steps: int = 1,
        condition_history_jitter_oldest_frame: bool = False,
        condition_history_random_span_probability: float = 0.0,
        condition_history_random_span_native_steps: Optional[int] = None,
        condition_include_episode_first: bool = False,
        condition_history_target_prefix: bool = False,
        condition_observation_separate: bool = False,
        condition_observation_repeats: int = 4,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        override_instruction: Optional[str] = None,
        video_path_column: str | list[str] = "video_path",
        video_hdf5_path_column: Optional[str] = None,
        video_hdf5_camera_keys: Optional[list[str]] = None,
        concat_multi_camera: Optional[str] = None,
        per_camera_video_size: Optional[list[int]] = None,
        parquet_path_column: str = "parquet_path",
        text_embedding_path_column: Optional[str] = None,
        instruction_column: str = "gemini_caption",
        fallback_instruction_column: str = "high_level_instruction",
        action_column: str = "action",
        proprio_column: str = "qpos",
        trim_start_column: str = "trim_start",
        trim_end_column: str = "trim_end",
        video_trim_start_column: Optional[str] = None,
        video_trim_end_column: Optional[str] = None,
        table_trim_start_column: Optional[str] = None,
        table_trim_end_column: Optional[str] = None,
        supervise_terminal_action_padding: bool = False,
    ):
        if shape_meta is None:
            raise ValueError("`shape_meta` must be provided.")

        metadata_is_list = isinstance(metadata_path, (list, tuple)) or OmegaConf.is_list(metadata_path)
        metadata_values = list(metadata_path) if metadata_is_list else [metadata_path]
        if not metadata_values:
            raise ValueError("`metadata_path` must contain at least one metadata file or dataset directory.")
        self.is_multi_source = metadata_is_list

        if isinstance(video_root, (list, tuple)) or OmegaConf.is_list(video_root):
            if len(video_root) != len(metadata_values):
                raise ValueError(
                    "When `video_root` is a list, it must have the same length as `metadata_path`: "
                    f"{len(video_root)} != {len(metadata_values)}"
                )
            video_root_values = list(video_root)
        elif video_root is None or self.is_multi_source:
            # Multi-source datasets infer each root independently from its metadata location.
            video_root_values = [None] * len(metadata_values)
        else:
            video_root_values = [video_root]

        self.metadata_paths: list[Path] = []
        self.video_roots: list[Path] = []
        for metadata_value, root_value in zip(metadata_values, video_root_values):
            metadata_file = Path(str(metadata_value)).expanduser().resolve()
            if metadata_file.is_dir():
                metadata_file = metadata_file / "metadata_split.csv"
            if not metadata_file.is_file():
                raise FileNotFoundError(f"Metadata file does not exist: {metadata_file}")
            source_root = metadata_file.parent if root_value is None else Path(str(root_value)).expanduser().resolve()
            self.metadata_paths.append(metadata_file)
            self.video_roots.append(source_root)

        self.metadata_path = self.metadata_paths if self.is_multi_source else self.metadata_paths[0]
        self.video_root = self.video_roots if self.is_multi_source else self.video_roots[0]
        if sample_ratio is None:
            ratios = np.ones(len(self.metadata_paths), dtype=np.float64)
        else:
            if len(sample_ratio) != len(self.metadata_paths):
                raise ValueError(
                    "`sample_ratio` must have the same length as `metadata_path`: "
                    f"{len(sample_ratio)} != {len(self.metadata_paths)}"
                )
            ratios = np.asarray(list(sample_ratio), dtype=np.float64)
        if not np.all(np.isfinite(ratios)) or np.any(ratios < 0) or float(ratios.sum()) <= 0:
            raise ValueError(
                f"`sample_ratio` must contain finite non-negative values with a positive sum, got {ratios}"
            )
        self.sample_ratio = ratios / ratios.sum()
        self.balance_column = (
            None if balance_column is None or not str(balance_column).strip() else str(balance_column).strip()
        )
        if balance_weight_overrides is None:
            self.balance_weight_overrides: dict[str, float] = {}
        else:
            if isinstance(balance_weight_overrides, DictConfig):
                balance_weight_overrides = OmegaConf.to_container(
                    balance_weight_overrides,
                    resolve=True,
                )
            if not isinstance(balance_weight_overrides, dict):
                raise TypeError(
                    "`balance_weight_overrides` must be a mapping from balance "
                    f"group to positive weight, got {type(balance_weight_overrides)}"
                )
            self.balance_weight_overrides = {str(key): float(value) for key, value in balance_weight_overrides.items()}
        invalid_balance_weights = {
            key: value for key, value in self.balance_weight_overrides.items() if not np.isfinite(value) or value <= 0
        }
        if invalid_balance_weights:
            raise ValueError(
                f"`balance_weight_overrides` values must be finite and positive, got {invalid_balance_weights}"
            )
        if self.balance_weight_overrides and self.balance_column is None:
            raise ValueError("`balance_weight_overrides` requires `balance_column` to be set.")
        if self.balance_column is not None and self.is_multi_source:
            raise ValueError(
                "`balance_column` currently requires a single metadata source; "
                "use sample_ratio for multi-source balancing."
            )
        self.segment_sampling_mode = str(segment_sampling_mode).strip().lower()
        if self.segment_sampling_mode not in {"step_uniform", "segment_uniform"}:
            raise ValueError(
                f"`segment_sampling_mode` must be one of: step_uniform, segment_uniform; got {segment_sampling_mode!r}"
            )
        # `segment_uniform` and `balance_column` compose: the balance column picks
        # the group (weighted), then every segment inside that group is equally
        # likely regardless of its duration. Without a balance column the same
        # mode spreads the probability over all segments uniformly.
        self.sampling_epoch = 0
        if isinstance(shape_meta, DictConfig):
            self.shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        else:
            self.shape_meta = shape_meta
        self.num_frames = int(num_frames)
        self.action_horizon = self.num_frames - 1 if action_horizon is None else int(action_horizon)
        if self.action_horizon <= 0 or self.action_horizon > self.num_frames - 1:
            raise ValueError(
                "`action_horizon` must be in [1, num_frames-1], got "
                f"action_horizon={self.action_horizon}, num_frames={self.num_frames}"
            )
        self.min_valid_action_steps = int(min_valid_action_steps)
        if self.min_valid_action_steps <= 0:
            raise ValueError(f"`min_valid_action_steps` must be positive, got {self.min_valid_action_steps}")
        if self.segment_sampling_mode == "segment_uniform" and self.min_valid_action_steps > self.action_horizon:
            raise ValueError(
                "`min_valid_action_steps` cannot exceed `action_horizon` when "
                "using segment-uniform sampling, got "
                f"{self.min_valid_action_steps} > {self.action_horizon}"
            )
        self.video_size = video_size
        self.video_resize_mode = str(video_resize_mode).strip().lower()
        if self.video_resize_mode not in {"center_crop", "direct"}:
            raise ValueError(f"`video_resize_mode` must be one of: center_crop, direct; got {video_resize_mode}")
        self.context_len = int(context_len)
        self.text_embedding_cache_dir = (
            None if text_embedding_cache_dir is None else Path(str(text_embedding_cache_dir)).expanduser().resolve()
        )
        self.text_embedding_cache_tag = str(text_embedding_cache_tag)
        self.allow_missing_text_embeddings = bool(allow_missing_text_embeddings)
        self.global_sample_stride = int(global_sample_stride)
        self.action_time_offset = int(action_time_offset)
        self.supervise_terminal_action_padding = bool(supervise_terminal_action_padding)
        if self.action_time_offset < 0:
            raise ValueError(f"`action_time_offset` must be non-negative, got {self.action_time_offset}")
        # This span is used only by segment-uniform sampling. It guarantees
        # that the first N action targets are real samples, independent of the
        # evaluator's eventual replanning interval. Later targets may still be
        # padded and excluded by action_is_pad.
        self.min_valid_action_span = 1 + (self.min_valid_action_steps - 1) * self.global_sample_stride
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.decode_only_first_video_frame = bool(decode_only_first_video_frame)
        self.decode_sampled_video_frames = bool(decode_sampled_video_frames)
        if self.decode_only_first_video_frame and self.decode_sampled_video_frames:
            raise ValueError("`decode_only_first_video_frame` and `decode_sampled_video_frames` are mutually exclusive")
        self.condition_history_frames = int(condition_history_frames)
        self.condition_history_stride = int(
            self.action_video_freq_ratio if condition_history_stride is None else condition_history_stride
        )
        if condition_history_stride_choices is None:
            if condition_history_stride_probs is not None:
                raise ValueError("`condition_history_stride_probs` requires `condition_history_stride_choices`.")
            self.condition_history_stride_choices = [self.condition_history_stride]
            self.condition_history_stride_probs = np.asarray([1.0], dtype=np.float64)
        else:
            self.condition_history_stride_choices = [int(value) for value in condition_history_stride_choices]
            if not self.condition_history_stride_choices or any(
                value <= 0 for value in self.condition_history_stride_choices
            ):
                raise ValueError(
                    "`condition_history_stride_choices` must contain positive "
                    f"integers, got {self.condition_history_stride_choices}."
                )
            if condition_history_stride_probs is None:
                raise ValueError(
                    "`condition_history_stride_probs` must be provided with `condition_history_stride_choices`."
                )
            probabilities = np.asarray(condition_history_stride_probs, dtype=np.float64)
            if probabilities.ndim != 1 or len(probabilities) != len(self.condition_history_stride_choices):
                raise ValueError(
                    "History stride choices/probabilities must have the same "
                    "length, got "
                    f"{len(self.condition_history_stride_choices)} and "
                    f"{probabilities.shape}."
                )
            if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or float(probabilities.sum()) <= 0:
                raise ValueError(
                    "`condition_history_stride_probs` must be finite, "
                    f"non-negative, and sum to a positive value; got {probabilities}."
                )
            self.condition_history_stride_probs = probabilities / probabilities.sum()
        self.condition_history_jitter_probability = float(condition_history_jitter_probability)
        self.condition_history_jitter_max_native_steps = int(condition_history_jitter_max_native_steps)
        self.condition_history_jitter_oldest_frame = bool(condition_history_jitter_oldest_frame)
        self.condition_history_random_span_probability = float(condition_history_random_span_probability)
        if not 0.0 <= self.condition_history_random_span_probability <= 1.0:
            raise ValueError(
                "`condition_history_random_span_probability` must be in [0, 1], got "
                f"{self.condition_history_random_span_probability}."
            )
        default_random_span = self.condition_history_stride * max(self.condition_history_frames - 1, 1)
        self.condition_history_random_span_native_steps = int(
            default_random_span
            if condition_history_random_span_native_steps is None
            else condition_history_random_span_native_steps
        )
        if self.condition_history_random_span_native_steps <= 0:
            raise ValueError(
                "`condition_history_random_span_native_steps` must be positive, got "
                f"{self.condition_history_random_span_native_steps}."
            )
        if self.condition_history_random_span_probability > 0 and self.condition_history_random_span_native_steps < max(
            self.condition_history_frames - 1, 1
        ):
            raise ValueError(
                "Random history span must contain enough native steps for distinct "
                "history offsets, got span="
                f"{self.condition_history_random_span_native_steps}, frames="
                f"{self.condition_history_frames}."
            )
        self.condition_include_episode_first = bool(condition_include_episode_first)
        self.condition_history_target_prefix = bool(condition_history_target_prefix)
        self.condition_observation_separate = bool(condition_observation_separate)
        self.condition_observation_repeats = int(condition_observation_repeats)
        if self.condition_observation_repeats <= 0:
            raise ValueError(
                f"`condition_observation_repeats` must be positive, got {self.condition_observation_repeats}."
            )
        if self.condition_observation_separate and (
            self.condition_history_frames <= 0
            or not self.condition_include_episode_first
            or not self.condition_history_target_prefix
        ):
            raise ValueError(
                "`condition_observation_separate=true` requires a history "
                "condition with the episode anchor and target-prefix mode."
            )
        if self.condition_observation_separate and self.decode_only_first_video_frame:
            raise ValueError(
                "`condition_observation_separate=true` requires the current "
                "and future sampled video frames; it cannot use the first-frame "
                "decode fast path."
            )
        if self.condition_history_frames < 0:
            raise ValueError(f"`condition_history_frames` must be non-negative, got {self.condition_history_frames}.")
        if self.condition_history_stride <= 0:
            raise ValueError(f"`condition_history_stride` must be positive, got {self.condition_history_stride}.")
        if not 0.0 <= self.condition_history_jitter_probability <= 1.0:
            raise ValueError(
                "`condition_history_jitter_probability` must be in [0, 1], got "
                f"{self.condition_history_jitter_probability}."
            )
        if self.condition_history_jitter_max_native_steps < 0:
            raise ValueError(
                "`condition_history_jitter_max_native_steps` must be "
                "non-negative, got "
                f"{self.condition_history_jitter_max_native_steps}."
            )
        if self.condition_history_jitter_probability > 0 and self.condition_history_jitter_max_native_steps == 0:
            raise ValueError("Positive history jitter probability requires a positive jitter magnitude.")
        if (
            self.condition_history_jitter_probability > 0
            and self.condition_history_stride_choices
            and min(self.condition_history_stride_choices) <= 2 * self.condition_history_jitter_max_native_steps
        ):
            raise ValueError(
                "To keep independently jittered history indices strictly "
                "increasing, every stride choice must be greater than twice "
                "the jitter magnitude; got choices="
                f"{self.condition_history_stride_choices}, jitter="
                f"{self.condition_history_jitter_max_native_steps}."
            )
        condition_frames = self.condition_history_frames + int(self.condition_include_episode_first)
        if condition_frames and (condition_frames <= 1 or condition_frames % 4 != 1):
            raise ValueError(
                f"History condition RGB length must be > 1 and satisfy T % 4 == 1, got T={condition_frames}."
            )
        if self.condition_history_target_prefix and not condition_frames:
            raise ValueError("`condition_history_target_prefix=true` requires a non-empty history condition.")
        self.skip_padding_as_possible = bool(skip_padding_as_possible)
        self.max_padding_retry = int(max_padding_retry)
        self.override_instruction = override_instruction
        self.use_relative_action = bool(use_relative_action)
        self.seed = int(seed)

        video_path_is_list = isinstance(video_path_column, (list, tuple)) or OmegaConf.is_list(video_path_column)
        self.video_path_columns = (
            [str(value) for value in video_path_column] if video_path_is_list else [str(video_path_column)]
        )
        if not self.video_path_columns or any(not value for value in self.video_path_columns):
            raise ValueError("`video_path_column` must contain at least one non-empty column name.")
        # Retain the historical scalar attribute for downstream callers that
        # inspect the single-camera configuration directly.
        self.video_path_column = self.video_path_columns if video_path_is_list else self.video_path_columns[0]
        self.video_hdf5_path_column = (
            None
            if video_hdf5_path_column is None or not str(video_hdf5_path_column).strip()
            else str(video_hdf5_path_column).strip()
        )
        if video_hdf5_camera_keys is None:
            self.video_hdf5_camera_keys: list[str] = []
        else:
            self.video_hdf5_camera_keys = [str(value) for value in video_hdf5_camera_keys]
        if bool(self.video_hdf5_path_column) != bool(self.video_hdf5_camera_keys):
            raise ValueError("`video_hdf5_path_column` and `video_hdf5_camera_keys` must be set together.")
        if self.video_hdf5_camera_keys and len(self.video_hdf5_camera_keys) != len(self.video_path_columns):
            raise ValueError(
                "HDF5 camera count must match video path column count: "
                f"{len(self.video_hdf5_camera_keys)} != {len(self.video_path_columns)}"
            )
        self.concat_multi_camera = (
            None
            if concat_multi_camera is None or not str(concat_multi_camera).strip()
            else str(concat_multi_camera).strip().lower()
        )
        if len(self.video_path_columns) > 1 and self.concat_multi_camera != "horizontal":
            raise ValueError("Multi-camera VideoEventDataset currently requires `concat_multi_camera=horizontal`.")
        self.per_camera_video_size = (
            None if per_camera_video_size is None else [int(value) for value in per_camera_video_size]
        )
        if self.per_camera_video_size is not None:
            if len(self.per_camera_video_size) != 2 or any(value <= 0 for value in self.per_camera_video_size):
                raise ValueError(
                    "`per_camera_video_size` must be [height, width] with positive values, "
                    f"got {self.per_camera_video_size}."
                )
            if len(self.video_path_columns) == 1:
                raise ValueError("`per_camera_video_size` is only valid for multi-camera input.")
        self.parquet_path_column = parquet_path_column
        self.instruction_column = instruction_column
        self.fallback_instruction_column = fallback_instruction_column
        self.action_column = action_column
        self.proprio_column = proprio_column
        self.trim_start_column = trim_start_column
        self.trim_end_column = trim_end_column
        # A metadata row may point either to an already-split clip or to the
        # original full episode.  Keep the video and action/state coordinate
        # systems explicit so a split video can be paired with a full-episode
        # parquet without silently shifting supervision back to frame zero.
        self.video_trim_start_column = video_trim_start_column or trim_start_column
        self.video_trim_end_column = video_trim_end_column or trim_end_column
        self.table_trim_start_column = table_trim_start_column or trim_start_column
        self.table_trim_end_column = table_trim_end_column or trim_end_column
        self._require_video_trim_start_column = video_trim_start_column is not None
        self._require_table_trim_start_column = table_trim_start_column is not None

        if (self.num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                "num_frames-1 must be divisible by action_video_freq_ratio, got "
                f"{self.num_frames - 1} and {self.action_video_freq_ratio}"
            )
        if ((self.num_frames - 1) // self.action_video_freq_ratio) % 4 != 0:
            raise ValueError(
                "video frames must be divisible by 4 for tokenization, got "
                f"{(self.num_frames - 1) // self.action_video_freq_ratio}"
            )

        self.obs_offsets = np.arange(self.num_frames, dtype=np.int64) * self.global_sample_stride
        self.action_offsets = (
            np.arange(self.action_horizon, dtype=np.int64) * self.global_sample_stride + self.action_time_offset
        )
        self.full_video_sample_indices = list(range(0, self.num_frames, self.action_video_freq_ratio))
        self.logical_num_video_frames = len(self.full_video_sample_indices)
        self.video_sample_indices = [0] if self.decode_only_first_video_frame else self.full_video_sample_indices
        self.decoded_video_sample_indices = (
            list(range(len(self.video_sample_indices)))
            if self.decode_only_first_video_frame or self.decode_sampled_video_frames
            else self.video_sample_indices
        )
        history_condition_rgb_frames = self.condition_history_frames + int(self.condition_include_episode_first)
        self.target_logical_num_video_frames = self.logical_num_video_frames
        if self.condition_observation_separate:
            # The ordinary sampled video contains the current frame plus the
            # 16 future RGB samples.  The target sequence is rebuilt as
            # [history/anchor (5), current observation repeated 4 times,
            #  future (16)] = 25 RGB frames -> 7 VAE latents.  Repeating the
            # observation only constructs a clean temporal boundary; the
            # condition observation latent itself is encoded separately in
            # the model.
            self.target_logical_num_video_frames = (
                history_condition_rgb_frames + self.condition_observation_repeats + (self.logical_num_video_frames - 1)
            )
            if self.target_logical_num_video_frames <= 1 or (self.target_logical_num_video_frames - 1) % 4 != 0:
                raise ValueError(
                    "Separate-observation target length must satisfy T=4k+1, "
                    f"got {self.target_logical_num_video_frames}."
                )

        self.image_meta = self.shape_meta["images"]
        self.state_meta = self.shape_meta["state"]
        self.action_meta = self.shape_meta["action"]

        source_frames = []
        for source_idx, (metadata_file, source_root) in enumerate(zip(self.metadata_paths, self.video_roots)):
            source_df = pd.read_csv(metadata_file)
            source_df["_vpp2_source_idx"] = source_idx
            source_df["_vpp2_source_root"] = str(source_root)
            source_frames.append(source_df)
        df = pd.concat(source_frames, axis=0, ignore_index=True)
        required = {
            *self.video_path_columns,
            self.parquet_path_column,
            self.instruction_column,
            self.video_trim_end_column,
            self.table_trim_end_column,
        }
        if self.video_hdf5_path_column is not None:
            required.add(self.video_hdf5_path_column)
        if self._require_video_trim_start_column:
            required.add(self.video_trim_start_column)
        if self._require_table_trim_start_column:
            required.add(self.table_trim_start_column)
        missing = sorted(col for col in required if col not in df.columns)
        if missing:
            raise ValueError(f"Metadata missing required columns: {missing}")

        self.text_embedding_path_column = text_embedding_path_column
        if self.text_embedding_path_column is None:
            for candidate in ("t5_emb", "t5_emb_path", "text_embedding_path"):
                if candidate in df.columns:
                    self.text_embedding_path_column = candidate
                    break
        if self.text_embedding_path_column is None and self.text_embedding_cache_dir is None:
            raise ValueError("Metadata must contain one of: t5_emb, t5_emb_path, text_embedding_path.")

        df = df.reset_index(drop=False).rename(columns={"index": "metadata_index"})
        if val_set_proportion > 1e-6:
            split_frames = []
            for source_idx in range(len(self.metadata_paths)):
                source_df = df[df["_vpp2_source_idx"] == source_idx]
                rng = np.random.default_rng(seed + source_idx)
                order = np.arange(len(source_df))
                rng.shuffle(order)
                split_idx = int(len(order) * (1 - val_set_proportion))
                selected = order[:split_idx] if is_training_set else order[split_idx:]
                split_frames.append(source_df.iloc[np.sort(selected)])
            df = pd.concat(split_frames, axis=0, ignore_index=True)
        else:
            df = df.reset_index(drop=True)
        self.metadata = df
        self.is_training_set = bool(is_training_set)
        self._build_step_index()

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(args={"img_w": self.video_size[1], "img_h": self.video_size[0]})
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        self._parquet_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._parquet_cache_size = 32

        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            expected_image_steps = (
                len(self.video_sample_indices)
                if self.decode_only_first_video_frame or self.decode_sampled_video_frames
                else self.num_frames
            )
            actual_image_steps = int(getattr(processor, "num_image_obs_steps", processor.num_obs_steps))
            if actual_image_steps != expected_image_steps:
                raise ValueError(
                    "Processor image horizon does not match the dataset decode horizon: "
                    f"expected {expected_image_steps}, got {actual_image_steps}."
                )
            norm_stats_source = pretrained_norm_stats_relative if self.use_relative_action else pretrained_norm_stats
            if OmegaConf.is_config(norm_stats_source):
                norm_stats_source = OmegaConf.to_container(norm_stats_source, resolve=True)
            if not norm_stats_source:
                if not is_training_set:
                    raise ValueError(
                        "Normalization stats must be provided for validation/test sets since "
                        "we don't want to calculate stats on them. For relative-action datasets, "
                        "set `pretrained_norm_stats_relative` or let runtime pass the freshly "
                        "computed training stats."
                    )
                dist_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
                is_main = _is_main_process()
                if is_main or not dist_ready:
                    if not is_main and not dist_ready:
                        logger.warning(
                            "Distributed process group is not initialized while building dataset stats on a "
                            "non-main rank; calculating stats locally to avoid using empty normalization stats."
                        )
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.get_dataset_stats(processor)
                    if is_main:
                        work_dir = get_work_dir()
                        save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if dist_ready:
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                if isinstance(norm_stats_source, (dict, defaultdict)):
                    dataset_stats = norm_stats_source
                    logger.info("Using in-memory dataset stats.")
                else:
                    dataset_stats = load_dataset_stats_from_json(norm_stats_source)
                    logger.info(f"Using dataset stats: {norm_stats_source}")
                if _is_main_process():
                    work_dir = get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.processor = processor.train() if self.is_training_set else processor.eval()
        else:
            self.processor = None

    def __len__(self):
        return int(self.total_steps)

    def set_epoch(self, epoch: int):
        """Advance deterministic random sampling without sharing mutable worker RNGs."""

        self.sampling_epoch = int(epoch)

    def _sampling_rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, self.sampling_epoch, int(idx)]))

    def _sample_condition_history_offsets(
        self,
        *,
        dataset_idx: int,
        trajectory_idx: int,
        step_idx: int,
        num_frames: int,
    ) -> tuple[np.ndarray, int, bool]:
        """Sample one monotonic history grid while keeping the current frame fixed."""

        base_stride = int(
            getattr(
                self,
                "condition_history_stride",
                getattr(self, "action_video_freq_ratio", 1),
            )
        )
        stride_choices = list(getattr(self, "condition_history_stride_choices", [base_stride]))
        stride_probabilities = np.asarray(
            getattr(self, "condition_history_stride_probs", [1.0]),
            dtype=np.float64,
        )
        augment = bool(getattr(self, "is_training_set", False))
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    int(getattr(self, "seed", 42)),
                    int(getattr(self, "sampling_epoch", 0)),
                    int(dataset_idx),
                    int(trajectory_idx),
                    int(step_idx),
                    0x48495354,  # "HIST": keep this stream separate from sampling.
                ]
            )
        )
        random_span_probability = float(getattr(self, "condition_history_random_span_probability", 0.0))
        random_span_native_steps = int(
            getattr(
                self,
                "condition_history_random_span_native_steps",
                base_stride * max(num_frames - 1, 1),
            )
        )
        if augment and random_span_probability > 0.0 and float(rng.random()) < random_span_probability:
            # Sample distinct past positions uniformly within the configured
            # temporal window and keep them ordered with the current frame at
            # offset zero. This branch deliberately does not apply jitter;
            # jitter belongs to the alternate variable-stride branch below.
            sampled_past = np.sort(
                rng.choice(
                    np.arange(1, random_span_native_steps + 1, dtype=np.int64),
                    size=max(num_frames - 1, 0),
                    replace=False,
                )
            )
            history_offsets = np.concatenate([(-sampled_past)[::-1], np.asarray([0], dtype=np.int64)])
            if history_offsets.shape[0] != num_frames or np.any(np.diff(history_offsets) <= 0):
                raise RuntimeError(
                    "Random history span sampling must produce a strictly "
                    f"increasing grid ending at zero, got {history_offsets.tolist()}."
                )
            # Zero denotes an irregular/random span in the existing metadata
            # convention used by the history evaluation utilities.
            return history_offsets, 0, False
        history_stride = (
            int(rng.choice(stride_choices, p=stride_probabilities))
            if augment and len(stride_choices) > 1
            else base_stride
        )
        history_offsets = (np.arange(num_frames, dtype=np.int64) - (num_frames - 1)) * history_stride

        jitter_probability = float(getattr(self, "condition_history_jitter_probability", 0.0))
        jitter_max = int(getattr(self, "condition_history_jitter_max_native_steps", 1))
        history_was_jittered = False
        jitter_oldest = bool(getattr(self, "condition_history_jitter_oldest_frame", False))
        jitter_start = 0 if jitter_oldest else 1
        jitter_count = max(0, (num_frames - 1) - jitter_start)
        if augment and jitter_probability > 0 and jitter_count > 0:
            # Jitter eligible non-current history frames independently.  The
            # current observation (the last offset, exactly zero) is never
            # jittered.  Existing profiles retain their fixed oldest endpoint;
            # the aligned-history profile opts into jittering it as well.
            jitter_mask = rng.random(jitter_count) < jitter_probability
            if bool(jitter_mask.any()):
                magnitudes = rng.integers(
                    1,
                    jitter_max + 1,
                    size=jitter_count,
                    dtype=np.int64,
                )
                signs = rng.choice(
                    np.asarray([-1, 1], dtype=np.int64),
                    size=jitter_count,
                )
                history_offsets[jitter_start:-1] += jitter_mask.astype(np.int64) * magnitudes * signs
                history_was_jittered = True

        if history_offsets[-1] != 0 or np.any(np.diff(history_offsets) <= 0):
            raise RuntimeError(
                "History temporal augmentation must preserve a strictly "
                f"increasing grid ending at zero, got {history_offsets.tolist()}."
            )
        return history_offsets, history_stride, history_was_jittered

    @staticmethod
    def _build_condition_prefixed_video_target(
        *,
        video: torch.Tensor,
        image_is_pad: torch.Tensor,
        condition_video: torch.Tensor,
        condition_image_is_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replace the target prefix with the exact condition RGB sequence.

        ``video`` is the ordinary current-plus-future target after temporal
        subsampling.  Its current frame is replaced by the complete condition
        sequence, and enough tail frames are removed to preserve the original
        logical target length.  This makes the VAE's condition latents an exact
        causal prefix of the target latents.
        """

        if video.ndim != 4 or condition_video.ndim != 4:
            raise ValueError(
                f"Video and condition must be [C,T,H,W], got {tuple(video.shape)} and {tuple(condition_video.shape)}."
            )
        if video.shape[0] != condition_video.shape[0] or video.shape[2:] != condition_video.shape[2:]:
            raise ValueError(
                "Video and condition channel/spatial shapes must match, got "
                f"{tuple(video.shape)} and {tuple(condition_video.shape)}."
            )
        target_frames = int(video.shape[1])
        condition_frames = int(condition_video.shape[1])
        if condition_frames <= 1 or condition_frames >= target_frames:
            raise ValueError(
                "Condition prefix must contain between 2 and T-1 frames, got "
                f"condition={condition_frames}, target={target_frames}."
            )
        if tuple(image_is_pad.shape) != (target_frames,):
            raise ValueError(f"Target padding must have shape ({target_frames},), got {tuple(image_is_pad.shape)}.")
        condition_image_is_pad = torch.as_tensor(
            condition_image_is_pad,
            dtype=torch.bool,
            device=image_is_pad.device,
        )
        if tuple(condition_image_is_pad.shape) != (condition_frames,):
            raise ValueError(
                "Condition padding must match its temporal length, got "
                f"{tuple(condition_image_is_pad.shape)} vs ({condition_frames},)."
            )
        future_frames = target_frames - condition_frames
        target = torch.cat(
            [condition_video, video[:, 1 : 1 + future_frames]],
            dim=1,
        )
        target_is_pad = torch.cat(
            [condition_image_is_pad, image_is_pad[1 : 1 + future_frames]],
            dim=0,
        )
        # Return a view of the constructed target so callers/tests can assert
        # numerical prefix identity, not merely matching source indices.
        condition = target[:, :condition_frames]
        return target, target_is_pad, condition

    @staticmethod
    def _build_separate_observation_video_target(
        *,
        video: torch.Tensor,
        image_is_pad: torch.Tensor,
        condition_video: torch.Tensor,
        condition_image_is_pad: torch.Tensor,
        observation_repeats: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a 7-latent target with a separately encoded observation.

        ``video`` is the ordinary logical sequence ``[current, future...]``
        and ``condition_video`` is the five-frame anchor/history prefix.  The
        current observation is repeated four times in the *target RGB
        sequence* so that the VAE temporal grid has an exact three-latent
        clean prefix (two history latents plus one observation latent), while
        the observation condition itself is encoded separately by the model.
        The repeated frames are never exposed as additional observations to
        the dataset sampler.
        """

        if video.ndim != 4 or condition_video.ndim != 4:
            raise ValueError(
                f"Video and condition must be [C,T,H,W], got {tuple(video.shape)} and {tuple(condition_video.shape)}."
            )
        if video.shape[0] != condition_video.shape[0] or video.shape[2:] != condition_video.shape[2:]:
            raise ValueError(
                "Video and condition channel/spatial shapes must match, got "
                f"{tuple(video.shape)} and {tuple(condition_video.shape)}."
            )
        if video.shape[1] < 2:
            raise ValueError(
                f"Separate-observation target requires current plus future frames, got T={video.shape[1]}."
            )
        condition_frames = int(condition_video.shape[1])
        if condition_frames <= 1 or condition_frames % 4 != 1:
            raise ValueError(
                f"History condition must contain 5-style RGB frames satisfying T % 4 == 1, got T={condition_frames}."
            )
        observation_repeats = int(observation_repeats)
        if observation_repeats <= 0:
            raise ValueError(f"observation_repeats must be positive, got {observation_repeats}.")
        if tuple(image_is_pad.shape) != (int(video.shape[1]),):
            raise ValueError(
                "Target padding must match the ordinary logical video length, "
                f"got {tuple(image_is_pad.shape)} vs ({video.shape[1]},)."
            )
        condition_image_is_pad = torch.as_tensor(
            condition_image_is_pad,
            dtype=torch.bool,
            device=image_is_pad.device,
        )
        if tuple(condition_image_is_pad.shape) != (condition_frames,):
            raise ValueError(
                "Condition padding must match its temporal length, got "
                f"{tuple(condition_image_is_pad.shape)} vs ({condition_frames},)."
            )
        current = video[:, :1]
        current_pad = image_is_pad[:1].repeat(observation_repeats)
        target = torch.cat(
            [condition_video, current.repeat(1, observation_repeats, 1, 1), video[:, 1:]],
            dim=1,
        )
        target_is_pad = torch.cat(
            [condition_image_is_pad, current_pad, image_is_pad[1:]],
            dim=0,
        )
        return target, target_is_pad, current

    def _build_step_index(self):
        if self.video_trim_start_column in self.metadata.columns:
            video_start = self.metadata[self.video_trim_start_column].fillna(0).to_numpy(dtype=np.int64)
        else:
            video_start = np.zeros(len(self.metadata), dtype=np.int64)
        video_end = self.metadata[self.video_trim_end_column].fillna(0).to_numpy(dtype=np.int64)
        if self.table_trim_start_column in self.metadata.columns:
            table_start = self.metadata[self.table_trim_start_column].fillna(0).to_numpy(dtype=np.int64)
        else:
            table_start = np.zeros(len(self.metadata), dtype=np.int64)
        table_end = self.metadata[self.table_trim_end_column].fillna(0).to_numpy(dtype=np.int64)
        video_lengths = video_end - video_start
        table_lengths = table_end - table_start
        invalid = (video_start < 0) | (table_start < 0) | (video_lengths <= 0) | (table_lengths <= 0)
        if np.any(invalid):
            bad = np.nonzero(invalid)[0][:10].tolist()
            raise ValueError(f"Metadata contains invalid video/table trim bounds at rows: {bad}")
        mismatched = video_lengths != table_lengths
        if np.any(mismatched):
            bad = np.nonzero(mismatched)[0][:10].tolist()
            details = [
                {
                    "row": int(idx),
                    "video_length": int(video_lengths[idx]),
                    "table_length": int(table_lengths[idx]),
                }
                for idx in bad
            ]
            raise ValueError(f"Video and action/state segment lengths must match; mismatched rows: {details}")

        self.video_start_steps = video_start.astype(np.int64)
        self.table_start_steps = table_start.astype(np.int64)
        self.trajectory_lengths = table_lengths.astype(np.int64)
        if self.segment_sampling_mode == "segment_uniform":
            too_short = self.trajectory_lengths < self.min_valid_action_span
            if np.any(too_short):
                bad = np.nonzero(too_short)[0][:10].tolist()
                raise ValueError(
                    "Segment-uniform sampling cannot satisfy "
                    f"min_valid_action_steps={self.min_valid_action_steps} with "
                    f"global_sample_stride={self.global_sample_stride}; segments "
                    f"at rows {bad} are shorter than the required "
                    f"{self.min_valid_action_span} source steps."
                )
        # `absolute_step_idx` is defined in the action/state episode timeline.
        self.trajectory_start_steps = self.table_start_steps
        self.cumulative_steps = np.cumsum(self.trajectory_lengths, dtype=np.int64)
        self.num_trajectories = int(len(self.metadata))
        self.balance_values: list[str] = []
        self.balance_trajectory_indices: list[np.ndarray] = []
        self.balance_cumulative_steps: list[np.ndarray] = []
        self.balance_total_steps: list[int] = []
        self.balance_probabilities = np.asarray([], dtype=np.float64)
        if self.balance_column is not None:
            if self.balance_column not in self.metadata.columns:
                raise ValueError(f"balance_column `{self.balance_column}` not found in metadata columns")
            balance_series = self.metadata[self.balance_column].astype(str).to_numpy()
            self.balance_values = sorted(set(balance_series.tolist()))
            for value in self.balance_values:
                trajectory_indices = np.flatnonzero(balance_series == value)
                group_lengths = self.trajectory_lengths[trajectory_indices]
                group_cumulative = np.cumsum(group_lengths, dtype=np.int64)
                group_total = int(group_cumulative[-1]) if len(group_cumulative) else 0
                if group_total <= 0:
                    raise ValueError(f"balance group {value!r} has no samples after splitting")
                self.balance_trajectory_indices.append(trajectory_indices)
                self.balance_cumulative_steps.append(group_cumulative)
                self.balance_total_steps.append(group_total)
            weight_overrides = getattr(self, "balance_weight_overrides", {})
            unknown_weight_groups = sorted(set(weight_overrides) - set(self.balance_values))
            if unknown_weight_groups:
                raise ValueError(
                    "`balance_weight_overrides` contains values absent from "
                    f"metadata column {self.balance_column!r}: {unknown_weight_groups}"
                )
            balance_weights = np.asarray(
                [weight_overrides.get(value, 1.0) for value in self.balance_values],
                dtype=np.float64,
            )
            self.balance_probabilities = balance_weights / balance_weights.sum()
        self.source_trajectory_indices: list[np.ndarray] = []
        self.source_cumulative_steps: list[np.ndarray] = []
        self.source_total_steps: list[int] = []
        source_ids = self.metadata["_vpp2_source_idx"].to_numpy(dtype=np.int64)
        for source_idx in range(len(self.metadata_paths)):
            trajectory_indices = np.flatnonzero(source_ids == source_idx)
            source_lengths = self.trajectory_lengths[trajectory_indices]
            source_cumulative = np.cumsum(source_lengths, dtype=np.int64)
            source_total = int(source_cumulative[-1]) if len(source_cumulative) else 0
            if source_total <= 0:
                raise ValueError(f"Dataset source {self.metadata_paths[source_idx]} has no samples after splitting.")
            self.source_trajectory_indices.append(trajectory_indices)
            self.source_cumulative_steps.append(source_cumulative)
            self.source_total_steps.append(source_total)

        if self.is_multi_source:
            self.total_steps = max(self.source_total_steps)
        else:
            self.total_steps = self.source_total_steps[0]
        logger.info(
            "Built step-level VideoEventDataset index: trajectories=%d total_steps=%d "
            "source_steps=%s sample_ratio=%s balance_column=%s balance_groups=%d "
            "balance_weight_overrides=%s segment_sampling_mode=%s "
            "min_valid_action_steps=%d",
            self.num_trajectories,
            self.total_steps,
            self.source_total_steps,
            self.sample_ratio.tolist(),
            self.balance_column,
            len(self.balance_values),
            getattr(self, "balance_weight_overrides", {}),
            self.segment_sampling_mode,
            self.min_valid_action_steps,
        )

    def _sample_segment_local_step(
        self,
        rng: np.random.Generator,
        trajectory_idx: int,
    ) -> int:
        """Sample a start while keeping the configured action prefix valid."""

        max_local_step = int(self.trajectory_lengths[trajectory_idx] - self.min_valid_action_span)
        return int(rng.integers(max_local_step + 1))

    def _resolve_step_index(self, idx: int) -> tuple[int, int, int]:
        if self.total_steps <= 0:
            raise ValueError("Cannot sample from an empty step-level dataset.")
        idx = int(idx)
        if idx < 0:
            idx += self.total_steps
        if idx < 0 or idx >= self.total_steps:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.total_steps}")
        if self.balance_values:
            rng = self._sampling_rng(idx)
            group_idx = int(
                rng.choice(
                    len(self.balance_values),
                    p=self.balance_probabilities,
                )
            )
            trajectory_indices = self.balance_trajectory_indices[group_idx]
            if self.segment_sampling_mode == "segment_uniform":
                traj_idx = int(trajectory_indices[int(rng.integers(len(trajectory_indices)))])
                local_step_idx = self._sample_segment_local_step(rng, traj_idx)
                absolute_step_idx = int(self.trajectory_start_steps[traj_idx] + local_step_idx)
                return traj_idx, local_step_idx, absolute_step_idx
            group_step_idx = int(rng.integers(self.balance_total_steps[group_idx]))
            group_cumulative = self.balance_cumulative_steps[group_idx]
            group_traj_idx = int(np.searchsorted(group_cumulative, group_step_idx, side="right"))
            prev = int(group_cumulative[group_traj_idx - 1]) if group_traj_idx > 0 else 0
            traj_idx = int(trajectory_indices[group_traj_idx])
            local_step_idx = int(group_step_idx - prev)
            absolute_step_idx = int(self.trajectory_start_steps[traj_idx] + local_step_idx)
            return traj_idx, local_step_idx, absolute_step_idx
        if self.is_multi_source:
            # Make source selection reproducible across workers/ranks without sharing RNG state.
            rng = self._sampling_rng(idx)
            source_idx = int(rng.choice(len(self.metadata_paths), p=self.sample_ratio))
            if self.segment_sampling_mode == "segment_uniform":
                trajectory_indices = self.source_trajectory_indices[source_idx]
                traj_idx = int(trajectory_indices[int(rng.integers(len(trajectory_indices)))])
                local_step_idx = self._sample_segment_local_step(rng, traj_idx)
                absolute_step_idx = int(self.trajectory_start_steps[traj_idx] + local_step_idx)
                return traj_idx, local_step_idx, absolute_step_idx
            source_step_idx = int(rng.integers(self.source_total_steps[source_idx]))
        else:
            source_idx = 0
            if self.segment_sampling_mode == "segment_uniform":
                rng = self._sampling_rng(idx)
                trajectory_indices = self.source_trajectory_indices[source_idx]
                traj_idx = int(trajectory_indices[int(rng.integers(len(trajectory_indices)))])
                local_step_idx = self._sample_segment_local_step(rng, traj_idx)
                absolute_step_idx = int(self.trajectory_start_steps[traj_idx] + local_step_idx)
                return traj_idx, local_step_idx, absolute_step_idx
            source_step_idx = idx

        source_cumulative = self.source_cumulative_steps[source_idx]
        source_traj_idx = int(np.searchsorted(source_cumulative, source_step_idx, side="right"))
        prev = int(source_cumulative[source_traj_idx - 1]) if source_traj_idx > 0 else 0
        traj_idx = int(self.source_trajectory_indices[source_idx][source_traj_idx])
        local_step_idx = int(source_step_idx - prev)
        absolute_step_idx = int(self.trajectory_start_steps[traj_idx] + local_step_idx)
        return traj_idx, local_step_idx, absolute_step_idx

    def _resolve_path(self, value: str, source_root: Optional[str | Path] = None) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            if source_root is None:
                if self.is_multi_source:
                    raise ValueError("`source_root` is required to resolve a relative path in multi-source mode.")
                source_root = self.video_roots[0]
            path = Path(source_root) / path
        return path.resolve()

    def _resolve_row_path(self, row: pd.Series, column: str) -> Path:
        return self._resolve_path(row[column], source_root=row["_vpp2_source_root"])

    def _load_parquet(self, path: Path) -> pd.DataFrame:
        key = str(path)
        if key in self._parquet_cache:
            self._parquet_cache.move_to_end(key)
            return self._parquet_cache[key]
        table = pd.read_parquet(path)
        self._parquet_cache[key] = table
        if len(self._parquet_cache) > self._parquet_cache_size:
            self._parquet_cache.popitem(last=False)
        return table

    @staticmethod
    def _series_to_tensor(series: pd.Series, column: str) -> torch.Tensor:
        values = [np.asarray(item, dtype=np.float32) for item in series.to_list()]
        if len(values) == 0:
            raise ValueError(f"Column `{column}` produced an empty slice.")
        return torch.from_numpy(np.stack(values, axis=0))

    @staticmethod
    def _pad_indices(indices: np.ndarray, length: int) -> tuple[np.ndarray, torch.Tensor]:
        is_pad = indices >= length
        if length <= 0:
            raise ValueError("Cannot sample from an empty sequence.")
        indices = np.clip(indices, 0, length - 1)
        return indices, torch.as_tensor(is_pad, dtype=torch.bool)

    @staticmethod
    def _row_bound(row: pd.Series, column: str, *, default: Optional[int] = None) -> int:
        value = row.get(column, default)
        if value is None or pd.isna(value):
            if default is None:
                raise ValueError(f"Invalid metadata bound `{column}`: {value}")
            value = default
        return int(value)

    def _get_segment_bounds(
        self,
        row: pd.Series,
        *,
        table_len: int,
    ) -> tuple[int, int, int]:
        video_start = self._row_bound(row, self.video_trim_start_column, default=0)
        video_end = self._row_bound(row, self.video_trim_end_column)
        table_start = self._row_bound(row, self.table_trim_start_column, default=0)
        table_end = self._row_bound(row, self.table_trim_end_column)
        video_length = video_end - video_start
        table_length = table_end - table_start
        if video_start < 0 or table_start < 0 or video_length <= 0 or table_length <= 0:
            raise ValueError(
                "Invalid video/table segment bounds: "
                f"video=[{video_start},{video_end}), table=[{table_start},{table_end})"
            )
        if video_length != table_length:
            raise ValueError(
                f"Video and action/state segment lengths must match, got video={video_length}, table={table_length}"
            )
        if table_end > int(table_len):
            raise ValueError(
                "Action/state segment exceeds parquet length: "
                f"table=[{table_start},{table_end}), parquet_length={table_len}"
            )
        return video_start, table_start, table_length

    def _decode_video_frames(self, video_path: Path, indices: np.ndarray) -> torch.Tensor:
        selected = set(int(i) for i in indices)
        frames: dict[int, torch.Tensor] = {}
        last_frame = None
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx in selected:
                    arr = frame.to_ndarray(format="rgb24")
                    last_frame = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
                    frames[frame_idx] = last_frame
                elif frame_idx <= int(indices.max()):
                    arr = frame.to_ndarray(format="rgb24")
                    last_frame = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
                if frame_idx >= int(indices.max()) and len(frames) == len(selected):
                    break

        if last_frame is None:
            raise ValueError(f"No frames decoded from video: {video_path}")
        return torch.stack([frames.get(int(i), last_frame) for i in indices], dim=0).to(torch.uint8)

    @staticmethod
    def _has_metadata_value(value: Any) -> bool:
        return value is not None and not pd.isna(value) and str(value).strip() != ""

    @staticmethod
    def _decode_hdf5_image(value: Any, *, path: Path, key: str, index: int) -> torch.Tensor:
        encoded = np.asarray(value)
        if encoded.ndim == 3 and encoded.shape[-1] in (3, 4):
            array = encoded[..., :3].astype(np.uint8, copy=False)
        else:
            try:
                payload = bytes(value)
            except Exception as err:
                raise ValueError(
                    f"Unsupported HDF5 image value at {path}:{key}[{index}]: "
                    f"shape={encoded.shape} dtype={encoded.dtype}"
                ) from err
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            except Exception as err:
                raise ValueError(f"Failed to decode HDF5 image at {path}:{key}[{index}]") from err
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _decode_hdf5_video_frames(
        self,
        hdf5_path: Path,
        camera_keys: list[str],
        indices: np.ndarray,
    ) -> list[torch.Tensor]:
        selected = sorted(set(int(index) for index in indices))
        cameras: list[torch.Tensor] = []
        with h5py.File(hdf5_path, "r") as handle:
            for key in camera_keys:
                if key not in handle:
                    raise KeyError(f"Missing HDF5 camera `{key}` in {hdf5_path}")
                dataset = handle[key]
                if int(indices.max()) >= len(dataset):
                    raise ValueError(
                        f"HDF5 camera index exceeds `{key}` length in {hdf5_path}: "
                        f"max_index={int(indices.max())}, length={len(dataset)}"
                    )
                decoded = {
                    index: self._decode_hdf5_image(dataset[index], path=hdf5_path, key=key, index=index)
                    for index in selected
                }
                cameras.append(torch.stack([decoded[int(index)] for index in indices], dim=0))
        return cameras

    def _decode_row_video_frames(self, row: pd.Series, indices: np.ndarray) -> torch.Tensor:
        hdf5_column = getattr(self, "video_hdf5_path_column", None)
        video_columns = getattr(self, "video_path_columns", None)
        if video_columns is None:
            legacy_column = getattr(self, "video_path_column", "video_path")
            video_columns = list(legacy_column) if isinstance(legacy_column, (list, tuple)) else [str(legacy_column)]
        hdf5_value = row.get(hdf5_column) if hdf5_column is not None else None
        if self._has_metadata_value(hdf5_value):
            hdf5_path = self._resolve_row_path(row, hdf5_column)
            cameras = self._decode_hdf5_video_frames(
                hdf5_path,
                getattr(self, "video_hdf5_camera_keys", []),
                indices,
            )
        else:
            cameras = []
            for column in video_columns:
                value = row.get(column)
                if not self._has_metadata_value(value):
                    raise ValueError(
                        f"Metadata row has neither an HDF5 source nor `{column}`: "
                        f"metadata_index={row.get('metadata_index')}"
                    )
                cameras.append(self._decode_video_frames(self._resolve_row_path(row, column), indices))

        if len(cameras) == 1:
            return cameras[0]
        if len(cameras) != len(video_columns):
            raise ValueError(f"Decoded {len(cameras)} cameras, expected {len(video_columns)}.")

        per_camera_video_size = getattr(self, "per_camera_video_size", None)
        if per_camera_video_size is not None:
            target_size = [
                int(per_camera_video_size[0]),
                int(per_camera_video_size[1]),
            ]
            cameras = [
                transforms_F.resize(
                    camera,
                    size=target_size,
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                ).to(torch.uint8)
                for camera in cameras
            ]
        camera_shapes = {tuple(camera.shape) for camera in cameras}
        if len(camera_shapes) != 1:
            raise ValueError(
                "All camera tensors must share [T,C,H,W] before composition, got "
                f"{sorted(camera_shapes)}. Set `per_camera_video_size`."
            )
        concat_multi_camera = getattr(self, "concat_multi_camera", None)
        if concat_multi_camera == "horizontal":
            return torch.cat(cameras, dim=-1)
        raise ValueError(f"Unsupported multi-camera composition: {concat_multi_camera!r}")

    def _get_instruction(self, row: pd.Series) -> str:
        if self.override_instruction is not None:
            task = self.override_instruction
        else:
            task = row.get(self.instruction_column)
            if not isinstance(task, str) or task.strip() == "":
                task = row.get(self.fallback_instruction_column, "")
            if not isinstance(task, str) or task.strip() == "":
                task = "robot manipulation"
        return task

    def _load_text_context(self, row: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_embedding_path_column is not None:
            path = self.metadata_paths[int(row["_vpp2_source_idx"])].parent / str(row[self.text_embedding_path_column])
        else:
            raise ValueError("Set text_embedding_path in the metadata; run vpp2 prepare and vpp2 text-cache first.")

        if not path.exists():
            if not self.allow_missing_text_embeddings:
                raise FileNotFoundError(f"Missing text embedding cache: {path}")
            logger.warning(f"Missing text embedding cache: {path}. Returning zero context for now.")
            context = torch.zeros(self.context_len, 4096, dtype=torch.float32)
            context_mask = torch.zeros(self.context_len, dtype=torch.bool)
            return context, context_mask

        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            context = payload["context"]
            context_mask = payload.get("mask", payload.get("context_mask"))
        else:
            context = payload
            context_mask = torch.ones(context.shape[0], dtype=torch.bool)
        if context.ndim != 2:
            raise ValueError(f"Text context must be 2D [L,D], got {tuple(context.shape)} in {path}")
        if context_mask is None:
            context_mask = torch.ones(context.shape[0], dtype=torch.bool)
        context_mask = context_mask.bool()
        if context_mask.ndim != 1:
            raise ValueError(f"Text context mask must be 1D [L], got {tuple(context_mask.shape)} in {path}")
        if context_mask.shape[0] != context.shape[0]:
            raise ValueError(
                f"Text context mask length must match context length, got {context_mask.shape[0]} vs {context.shape[0]} in {path}"
            )

        length, width = context.shape
        if length > self.context_len:
            logger.warning(
                f"Text embedding cache is longer than context_len ({length} > {self.context_len}) in {path}. Truncating."
            )
            context = context[: self.context_len]
            context_mask = context_mask[: self.context_len]
        elif length < self.context_len:
            padded_context = context.new_zeros((self.context_len, width))
            padded_mask = context_mask.new_zeros((self.context_len,))
            padded_context[:length] = context
            padded_mask[:length] = context_mask
            context = padded_context
            context_mask = padded_mask

        return context, context_mask

    def _make_action_relative(self, sample: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
        if not self.use_relative_action or "action" not in sample:
            return sample

        for meta in self.action_meta:
            key = meta["key"]
            if key not in sample["state"]:
                raise ValueError(f"Relative action requires matching state key `{key}`.")
            action = sample["action"][key]
            state = sample["state"][key]
            if action.ndim != 2 or state.ndim != 2:
                raise ValueError(
                    f"Relative action expects 2D action/state tensors for `{key}`, "
                    f"got {tuple(action.shape)} and {tuple(state.shape)}."
                )
            if action.shape[-1] != state.shape[-1]:
                raise ValueError(
                    f"Relative action dim mismatch for `{key}`: action={action.shape[-1]} state={state.shape[-1]}."
                )
            sample["action"][key] = action - state[:1]

        return sample

    def _build_raw_sample(
        self, idx: int, step_idx: Optional[int] = None, dataset_idx: Optional[int] = None
    ) -> dict[str, Any]:
        row = self.metadata.iloc[idx]
        parquet_path = self._resolve_row_path(row, self.parquet_path_column)
        table = self._load_parquet(parquet_path)

        video_start, table_start, clip_len = self._get_segment_bounds(
            row,
            table_len=len(table),
        )
        start = int(step_idx) if step_idx is not None else 0
        if start < 0:
            raise ValueError(f"step_idx must be non-negative, got {start}")
        if start >= clip_len:
            logger.warning(
                "Step index %d exceeds loaded clip length %d for trajectory %d; padding from last frame.",
                start,
                clip_len,
                idx,
            )

        video_local_indices, image_is_pad = self._pad_indices(start + self.obs_offsets, clip_len)
        if getattr(self, "decode_only_first_video_frame", False):
            # Action-only VPP2 consumes only the current RGB frame to build
            # the condition VAE latent and CLIP feature. Preserve the logical
            # video length separately while avoiding decode/resize work for
            # the future frames that are replaced by pure noise in the model.
            video_local_indices = video_local_indices[:1]
            image_is_pad = image_is_pad[:1]
        elif getattr(self, "decode_sampled_video_frames", False):
            # Preserve the 81-step native state/action window while asking the
            # decoder and image processor only for [0,4,...,80].  The returned
            # video is already in logical 21-frame order, so _get() uses the
            # corresponding dense decoded indices rather than subsampling it
            # a second time.
            video_local_indices = video_local_indices[self.full_video_sample_indices]
            image_is_pad = image_is_pad[self.full_video_sample_indices]
        obs_local_indices, state_is_pad = self._pad_indices(start + self.obs_offsets, clip_len)
        action_local_indices, action_is_pad = self._pad_indices(start + self.action_offsets, clip_len)
        video_indices = video_start + video_local_indices
        obs_indices = table_start + obs_local_indices
        action_indices = table_start + action_local_indices

        condition_video = None
        condition_observation_video = None
        condition_observation_image_is_pad = None
        condition_image_is_pad = None
        condition_history_offsets = None
        condition_history_stride_used = None
        condition_history_was_jittered = False
        condition_history_frames = int(getattr(self, "condition_history_frames", 0))
        condition_include_episode_first = bool(getattr(self, "condition_include_episode_first", False))
        if condition_history_frames > 0:
            (
                condition_history_offsets,
                condition_history_stride_used,
                condition_history_was_jittered,
            ) = self._sample_condition_history_offsets(
                dataset_idx=(int(dataset_idx) if dataset_idx is not None else int(idx)),
                trajectory_idx=int(idx),
                step_idx=start,
                num_frames=condition_history_frames,
            )
            # The history window is ordered oldest -> newest and includes the
            # current observation.  Negative positions near the beginning of
            # an episode are clamped to its first frame.
            requested_history_indices = start + condition_history_offsets
            history_is_pad = np.logical_or(
                requested_history_indices < 0,
                requested_history_indices >= clip_len,
            )
            history_local_indices = np.clip(
                requested_history_indices,
                0,
                clip_len - 1,
            )
            condition_local_indices = history_local_indices
            condition_image_is_pad = history_is_pad
            if condition_include_episode_first:
                condition_local_indices = np.concatenate([np.asarray([0], dtype=np.int64), condition_local_indices])
                condition_image_is_pad = np.concatenate([np.asarray([False], dtype=np.bool_), condition_image_is_pad])
            condition_indices = video_start + condition_local_indices
            requested_indices = np.concatenate([video_indices, condition_indices])
            decoded = self._decode_row_video_frames(row, requested_indices)
            video = decoded[: len(video_indices)]
            condition_video = decoded[len(video_indices) :]
            if getattr(self, "condition_observation_separate", False):
                # Keep the current observation as a one-frame tensor so the
                # model can VAE-encode it independently from the five-frame
                # anchor/history prefix.
                condition_observation_video = video[:1]
                condition_observation_image_is_pad = image_is_pad[:1]
        else:
            video = self._decode_row_video_frames(row, video_indices)
        state = self._series_to_tensor(table.iloc[obs_indices][self.proprio_column], self.proprio_column)
        action = self._series_to_tensor(table.iloc[action_indices][self.action_column], self.action_column)

        sample = {
            "idx": int(dataset_idx) if dataset_idx is not None else int(idx),
            "trajectory_idx": int(idx),
            "metadata_index": int(row.get("metadata_index", idx)),
            "step_idx": int(start),
            "task": self._get_instruction(row),
            "action": {},
            "state": {},
            "images": {},
            "action_is_pad": action_is_pad,
            "state_is_pad": state_is_pad,
            "image_is_pad": image_is_pad,
        }
        for meta in self.action_meta:
            key = meta["key"]
            if action.shape[-1] != meta["raw_shape"]:
                raise ValueError(
                    f"Action raw dim mismatch for {key}: got {action.shape[-1]}, expected {meta['raw_shape']}"
                )
            sample["action"][key] = action.float()
        for meta in self.state_meta:
            key = meta["key"]
            if state.shape[-1] != meta["raw_shape"]:
                raise ValueError(
                    f"State raw dim mismatch for {key}: got {state.shape[-1]}, expected {meta['raw_shape']}"
                )
            sample["state"][key] = state.float()
        for meta in self.image_meta:
            sample["images"][meta["key"]] = video
        if condition_video is not None:
            sample["condition_images"] = {meta["key"]: condition_video for meta in self.image_meta}
            sample["condition_image_is_pad"] = torch.as_tensor(
                condition_image_is_pad,
                dtype=torch.bool,
            )
            sample["condition_history_offsets"] = torch.as_tensor(
                condition_history_offsets,
                dtype=torch.long,
            )
            sample["condition_history_stride_used"] = int(condition_history_stride_used)
            sample["condition_history_was_jittered"] = bool(condition_history_was_jittered)
            if condition_observation_video is not None:
                sample["condition_observation_images"] = {
                    meta["key"]: condition_observation_video for meta in self.image_meta
                }
                sample["condition_observation_image_is_pad"] = torch.as_tensor(
                    condition_observation_image_is_pad,
                    dtype=torch.bool,
                )
        sample = self._make_action_relative(sample)
        return sample

    def _build_action_state_sample(self, idx: int) -> dict[str, dict[str, torch.Tensor]]:
        row = self.metadata.iloc[idx]
        parquet_path = self._resolve_row_path(row, self.parquet_path_column)
        table = self._load_parquet(parquet_path)

        _, table_start, clip_len = self._get_segment_bounds(row, table_len=len(table))
        table_slice = table.iloc[table_start : table_start + clip_len]
        state = self._series_to_tensor(table_slice[self.proprio_column], self.proprio_column)
        action = self._series_to_tensor(table_slice[self.action_column], self.action_column)

        sample = {"action": {}, "state": {}}
        for meta in self.action_meta:
            key = meta["key"]
            if action.shape[-1] != meta["raw_shape"]:
                raise ValueError(
                    f"Action raw dim mismatch for {key}: got {action.shape[-1]}, expected {meta['raw_shape']}"
                )
            sample["action"][key] = action.float()
        for meta in self.state_meta:
            key = meta["key"]
            if state.shape[-1] != meta["raw_shape"]:
                raise ValueError(
                    f"State raw dim mismatch for {key}: got {state.shape[-1]}, expected {meta['raw_shape']}"
                )
            sample["state"][key] = state.float()
        return sample

    def _get_processed_sample(self, idx: int) -> dict[str, Any]:
        trajectory_idx, step_idx, absolute_step_idx = self._resolve_step_index(idx)
        raw_sample = self._build_raw_sample(trajectory_idx, step_idx=step_idx, dataset_idx=idx)
        raw_sample["absolute_step_idx"] = absolute_step_idx
        sample_meta = {
            "trajectory_idx": int(raw_sample["trajectory_idx"]),
            "metadata_index": int(raw_sample["metadata_index"]),
            "step_idx": int(raw_sample["step_idx"]),
            "absolute_step_idx": int(raw_sample["absolute_step_idx"]),
        }
        for key in (
            "condition_history_offsets",
            "condition_history_stride_used",
            "condition_history_was_jittered",
        ):
            if key in raw_sample:
                sample_meta[key] = raw_sample[key]
        condition_images = raw_sample.get("condition_images")
        condition_image_is_pad = raw_sample.get("condition_image_is_pad")
        condition_observation_images = raw_sample.get("condition_observation_images")
        condition_observation_image_is_pad = raw_sample.get("condition_observation_image_is_pad")
        sample = raw_sample
        if self.processor is not None:
            sample = self.processor.preprocess(sample)
            sample.update(sample_meta)
            if condition_images is not None:
                sample["condition_images"] = condition_images
                sample["condition_image_is_pad"] = condition_image_is_pad
            if condition_observation_images is not None:
                sample["condition_observation_images"] = condition_observation_images
                sample["condition_observation_image_is_pad"] = condition_observation_image_is_pad
        return sample

    def _terminal_action_mask(
        self,
        row: pd.Series,
        step_idx: int,
        action_is_pad: torch.Tensor,
    ) -> torch.Tensor:
        """Select only tail padding of explicitly terminal semantic segments.

        The original action labels already clamp to the last valid command.
        Keep padding provenance intact; this mask only enables its loss.
        """
        mask = torch.zeros_like(action_is_pad, dtype=torch.bool)
        terminal = row.get("segment_terminal")
        if not self._has_metadata_value(terminal):
            return mask
        if str(terminal).strip().lower() not in {"true", "1"}:
            return mask
        start_column = getattr(self, "trim_start_column", "trim_start")
        end_column = getattr(self, "trim_end_column", "trim_end")
        clip_len = int(row[end_column]) - int(row[start_column])
        tail = torch.as_tensor(
            step_idx + self.action_offsets >= clip_len,
            device=action_is_pad.device,
            dtype=torch.bool,
        )
        return action_is_pad.bool() & tail

    def _get(self, idx: int) -> dict[str, Any]:
        sample_idx = idx
        sample = None
        trajectory_idx = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self._get_processed_sample(sample_idx)
            trajectory_idx = int(sample.get("trajectory_idx", 0))
            if not self.skip_padding_as_possible:
                break
            has_pad = (
                bool(sample["action_is_pad"].any().item())
                or bool(sample["image_is_pad"].any().item())
                or bool(sample["state_is_pad" if self.processor is None else "proprio_is_pad"].any().item())
            )
            if not has_pad or attempt >= self.max_padding_retry:
                break
            sample_idx = int(np.random.randint(len(self)))

        image_is_pad = sample["image_is_pad"]
        condition_video = None
        condition_image_is_pad = sample.get("condition_image_is_pad")
        condition_observation_video = None
        condition_observation_image_is_pad = sample.get("condition_observation_image_is_pad")
        if "condition_images" in sample:
            if len(self.image_meta) != 1:
                raise ValueError("History conditioning currently requires exactly one composed image stream.")
            condition_video = sample["condition_images"][self.image_meta[0]["key"]].float().div(255.0)
        if "condition_observation_images" in sample:
            if len(self.image_meta) != 1:
                raise ValueError(
                    "Separate observation conditioning currently requires exactly one composed image stream."
                )
            condition_observation_video = (
                sample["condition_observation_images"][self.image_meta[0]["key"]].float().div(255.0)
            )

        if self.processor is None:
            if len(self.image_meta) != 1 or len(self.action_meta) != 1 or len(self.state_meta) != 1:
                raise ValueError("Processor-free VideoEventDataset requires exactly one image, action, and state key.")
            video = sample["images"][self.image_meta[0]["key"]].float().div(255.0)
            action = sample["action"][self.action_meta[0]["key"]]
            proprio = sample["state"][self.state_meta[0]["key"]]
            action_is_pad = sample["action_is_pad"]
            proprio_is_pad = sample["state_is_pad"]
        else:
            video = sample["pixel_values"]
            if video.ndim == 5:
                if video.shape[0] != 1:
                    raise ValueError(f"VideoEventDataset expects exactly one camera, got {video.shape[0]}")
                video = video[0]
            action = sample["action"]
            proprio = sample["proprio"]
            action_is_pad = sample["action_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
        if video.ndim == 4:
            video = video[self.decoded_video_sample_indices, :, :, :]
        else:
            raise ValueError(f"Expected video shape [T,C,H,W] or [1,T,C,H,W], got {tuple(video.shape)}")

        image_is_pad = image_is_pad[self.decoded_video_sample_indices]

        if self.video_resize_mode == "direct":
            video = transforms_F.resize(
                video,
                size=[int(self.video_size[0]), int(self.video_size[1])],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
        else:
            video = self.resize_transform(video)
            video = self.crop_transform(video)
        video = self.normalize_transform(video)
        video = video.permute(1, 0, 2, 3)

        if condition_video is not None:
            if self.video_resize_mode == "direct":
                condition_video = transforms_F.resize(
                    condition_video,
                    size=[int(self.video_size[0]), int(self.video_size[1])],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )
            else:
                condition_video = self.resize_transform(condition_video)
                condition_video = self.crop_transform(condition_video)
            condition_video = self.normalize_transform(condition_video)
            condition_video = condition_video.permute(1, 0, 2, 3)

        if condition_observation_video is not None:
            if self.video_resize_mode == "direct":
                condition_observation_video = transforms_F.resize(
                    condition_observation_video,
                    size=[int(self.video_size[0]), int(self.video_size[1])],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )
            else:
                condition_observation_video = self.resize_transform(condition_observation_video)
                condition_observation_video = self.crop_transform(condition_observation_video)
            condition_observation_video = self.normalize_transform(condition_observation_video)
            condition_observation_video = condition_observation_video.permute(1, 0, 2, 3)

        if self.condition_history_target_prefix and not self.decode_only_first_video_frame:
            if self.condition_observation_separate:
                video, image_is_pad, _ = self._build_separate_observation_video_target(
                    video=video,
                    image_is_pad=image_is_pad,
                    condition_video=condition_video,
                    condition_image_is_pad=condition_image_is_pad,
                    observation_repeats=self.condition_observation_repeats,
                )
            else:
                video, image_is_pad, condition_video = self._build_condition_prefixed_video_target(
                    video=video,
                    image_is_pad=image_is_pad,
                    condition_video=condition_video,
                    condition_image_is_pad=condition_image_is_pad,
                )

        proprio = proprio[:1, :]
        if video.shape[1] <= 1 and not getattr(self, "decode_only_first_video_frame", False):
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")

        if trajectory_idx is None:
            trajectory_idx, _, _ = self._resolve_step_index(sample_idx)
        row = self.metadata.iloc[trajectory_idx]
        instruction = self._get_instruction(row)
        context, context_mask = self._load_text_context(row)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        result = {
            "video": video,
            "video_num_frames": torch.tensor(
                int(
                    getattr(
                        self,
                        "target_logical_num_video_frames",
                        video.shape[1],
                    )
                ),
                dtype=torch.long,
            ),
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "proprio_is_pad": proprio_is_pad[:1],
            "idx": torch.tensor(sample_idx, dtype=torch.long),
            "dataset_idx": torch.tensor(sample_idx, dtype=torch.long),
            "trajectory_idx": torch.tensor(sample["trajectory_idx"], dtype=torch.long),
            "metadata_index": torch.tensor(sample["metadata_index"], dtype=torch.long),
            "source_idx": torch.tensor(
                int(self.metadata.iloc[int(sample["trajectory_idx"])]["_vpp2_source_idx"]),
                dtype=torch.long,
            ),
            "step_idx": torch.tensor(sample["step_idx"], dtype=torch.long),
            "absolute_step_idx": torch.tensor(sample["absolute_step_idx"], dtype=torch.long),
        }
        if getattr(self, "supervise_terminal_action_padding", False):
            result["action_terminal_mask"] = self._terminal_action_mask(
                row,
                int(sample["step_idx"]),
                action_is_pad,
            )
        if condition_video is not None:
            result["condition_video"] = condition_video
            result["condition_image_is_pad"] = torch.as_tensor(
                condition_image_is_pad,
                dtype=torch.bool,
            )
            result["condition_history_offsets"] = torch.as_tensor(
                sample["condition_history_offsets"],
                dtype=torch.long,
            )
            result["condition_history_stride_used"] = torch.tensor(
                int(sample["condition_history_stride_used"]),
                dtype=torch.long,
            )
            result["condition_history_was_jittered"] = torch.tensor(
                bool(sample["condition_history_was_jittered"]),
                dtype=torch.bool,
            )
        if condition_observation_video is not None:
            result["condition_observation_video"] = condition_observation_video
            result["condition_observation_image_is_pad"] = torch.as_tensor(
                condition_observation_image_is_pad,
                dtype=torch.bool,
            )
        return result

    def _get_episode_data(self, idx: int) -> dict[str, dict[str, torch.Tensor]]:
        return self._build_action_state_sample(idx)

    def get_dataset_stats(self, preprocessor):
        state_values = defaultdict(list)
        action_values = defaultdict(list)
        action_count = defaultdict(int)
        action_sum = {}
        action_sumsq = {}
        action_min = {}
        action_max = {}
        action_traj_min = defaultdict(list)
        action_traj_max = defaultdict(list)

        for idx in tqdm(range(self.num_trajectories), desc="Iterating trajectories to get normalization"):
            batch = self._get_episode_data(idx)
            if self.use_relative_action:
                for meta in self.action_meta:
                    key = meta["key"]
                    action = batch["action"][key]
                    state = batch["state"][key]
                    if action.ndim != 2 or state.ndim != 2:
                        raise ValueError(
                            f"Relative action stats expect 2D action/state tensors for `{key}`, "
                            f"got {tuple(action.shape)} and {tuple(state.shape)}."
                        )
                    if action.shape[-1] != state.shape[-1]:
                        raise ValueError(
                            f"Relative action stats dim mismatch for `{key}`: "
                            f"action={action.shape[-1]} state={state.shape[-1]}."
                        )
                    starts = torch.arange(action.shape[0], dtype=torch.long)
                    offsets = torch.as_tensor(self.action_offsets, dtype=torch.long)
                    action_indices = torch.clamp(starts[:, None] + offsets[None, :], max=action.shape[0] - 1)
                    relative_action = action[action_indices.reshape(-1)].view(
                        action.shape[0], len(self.action_offsets), action.shape[-1]
                    )
                    relative_action = relative_action - state[:, None, :]
                    batch["action"][key] = relative_action.reshape(-1, action.shape[-1])
            batch = preprocessor.action_state_transform(batch)
            for meta in self.state_meta:
                state_values[meta["key"]].append(batch["state"][meta["key"]])
            for meta in self.action_meta:
                key = meta["key"]
                cur_action = batch["action"][key]
                if self.use_relative_action:
                    cur_action64 = cur_action.to(torch.float64)
                    cur_min = cur_action.amin(0)
                    cur_max = cur_action.amax(0)
                    if key not in action_sum:
                        action_sum[key] = cur_action64.sum(0)
                        action_sumsq[key] = cur_action64.square().sum(0)
                        action_min[key] = cur_min
                        action_max[key] = cur_max
                    else:
                        action_sum[key] += cur_action64.sum(0)
                        action_sumsq[key] += cur_action64.square().sum(0)
                        action_min[key] = torch.minimum(action_min[key], cur_min)
                        action_max[key] = torch.maximum(action_max[key], cur_max)
                    action_count[key] += int(cur_action.shape[0])
                    action_traj_min[key].append(cur_min)
                    action_traj_max[key].append(cur_max)
                else:
                    action_values[key].append(cur_action)

        stats = {
            "state": defaultdict(dict),
            "action": defaultdict(dict),
            "num_episodes": self.num_trajectories,
            "num_transition": int(self.trajectory_lengths.sum()),
        }
        for meta in self.state_meta:
            key = meta["key"]
            x = torch.cat(state_values[key], dim=0)
            stats["state"][key]["global_min"] = x.amin(0)
            stats["state"][key]["global_max"] = x.amax(0)
            stats["state"][key]["global_mean"] = x.mean(0)
            stats["state"][key]["global_std"] = x.std(0)
            stats["state"][key]["global_q01"] = torch.quantile(x, 0.01, dim=0)
            stats["state"][key]["global_q99"] = torch.quantile(x, 0.99, dim=0)
        for meta in self.action_meta:
            key = meta["key"]
            if self.use_relative_action:
                count = action_count[key]
                if count <= 0:
                    raise ValueError(f"No action values collected for `{key}`.")
                global_min = action_min[key]
                global_max = action_max[key]
                global_mean = (action_sum[key] / count).to(torch.float32)
                if count > 1:
                    variance = (action_sumsq[key] - action_sum[key].square() / count) / (count - 1)
                else:
                    variance = torch.zeros_like(action_sum[key])
                global_std = torch.sqrt(torch.clamp(variance, min=0.0)).to(torch.float32)
                global_q01 = torch.quantile(torch.stack(action_traj_min[key], dim=0), 0.01, dim=0)
                global_q99 = torch.quantile(torch.stack(action_traj_max[key], dim=0), 0.99, dim=0)
            else:
                x = torch.cat(action_values[key], dim=0)
                global_min = x.amin(0)
                global_max = x.amax(0)
                global_mean = x.mean(0)
                global_std = x.std(0)
                global_q01 = torch.quantile(x, 0.01, dim=0)
                global_q99 = torch.quantile(x, 0.99, dim=0)
            stats["action"][key]["global_min"] = global_min
            stats["action"][key]["global_max"] = global_max
            stats["action"][key]["global_mean"] = global_mean
            stats["action"][key]["global_std"] = global_std
            stats["action"][key]["global_q01"] = global_q01
            stats["action"][key]["global_q99"] = global_q99

            action_horizon = self.action_horizon
            stats["action"][key]["stepwise_min"] = global_min.unsqueeze(0).repeat(action_horizon, 1)
            stats["action"][key]["stepwise_max"] = global_max.unsqueeze(0).repeat(action_horizon, 1)
            stats["action"][key]["stepwise_mean"] = global_mean.unsqueeze(0).repeat(action_horizon, 1)
            stats["action"][key]["stepwise_std"] = global_std.unsqueeze(0).repeat(action_horizon, 1)
            stats["action"][key]["stepwise_q01"] = global_q01.unsqueeze(0).repeat(action_horizon, 1)
            stats["action"][key]["stepwise_q99"] = global_q99.unsqueeze(0).repeat(action_horizon, 1)
        return stats

    def __getitem__(self, idx: int):
        try:
            return self._get(idx)
        except Exception as err:
            print(f"Error processing sample idx {idx}: {err}. Returning a random sample instead.")
            print(traceback.format_exc())
            random_idx = int(np.random.randint(len(self)))
            return self._get(random_idx)
