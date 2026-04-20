"""Data collator for Gr00tN1d6.

Copyright 2024 NVIDIA. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import re
import shutil
import warnings
import random
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal

from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature

from .utils import (
    ALBUMENTATIONS_AVAILABLE,
    EMBODIMENT_STAT_CONFIGS,
    EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    MODALITY_CONFIGS,
    ActionRepresentation,
    EmbodimentTag,
    ModalityConfig,
    apply_sin_cos_encoding,
    apply_with_replay,
    build_image_transformations,
    build_image_transformations_albumentations,
    compute_relative_action_stats,
    convert_lerobot_stats_to_processor_format,
    ensure_eagle_cache_ready,
    nested_dict_to_numpy,
    normalize_values_meanstd,
    normalize_values_minmax,
    parse_modality_configs,
    unnormalize_values_meanstd,
    unnormalize_values_minmax,
)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")


_PROCESSOR_CACHE: dict[tuple[str, tuple[tuple[str, object], ...]], ProcessorMixin] = {}
_PROCESSOR_LOG_ONCE: set[tuple[str, tuple[tuple[str, object], ...]]] = set()


def build_processor(
    model_name: str,
    tokenizer_assets_repo: str | None = None,
    transformers_loading_kwargs: dict | None = None,
) -> ProcessorMixin:
    """Build the Eagle processor for the model."""
    expected_repo_id = "aravindhs-NV/eagle3-processor-groot-n1d6"
    allowed_models = {"nvidia/Eagle-Block2A-2B-v2", "aravindhs-NV/eagle3-processor-groot-n1d6"}
    if model_name not in allowed_models:
        raise AssertionError(f"Processor for {model_name} not supported")

    local_assets_path = None
    if tokenizer_assets_repo and Path(tokenizer_assets_repo).exists():
        local_assets_path = Path(tokenizer_assets_repo)

    transformers_loading_kwargs = dict(transformers_loading_kwargs or {})
    offline_mode = (
        str(os.environ.get("HF_HUB_OFFLINE", "")).lower() in {"1", "true", "yes"}
        or str(os.environ.get("TRANSFORMERS_OFFLINE", "")).lower() in {"1", "true", "yes"}
        or bool(transformers_loading_kwargs.get("local_files_only", False))
    )
    disable_hf_fallback = str(os.environ.get("EAGLE_DISABLE_HF_FALLBACK", "1")).lower() in {
        "1",
        "true",
        "yes",
    }
    verbose_logs = str(os.environ.get("EAGLE_PROCESSOR_VERBOSE", "0")).lower() in {
        "1",
        "true",
        "yes",
    }
    if not verbose_logs:
        try:
            from transformers.utils import logging as hf_logging  # type: ignore

            hf_logging.set_verbosity_error()
        except Exception:
            pass

    # Try to find Eagle processor assets in local paths first.
    vendor_eagle_path = (
        Path(__file__).parent.parent
        / "vendor"
        / "gr00t"
        / "model"
        / "modules"
        / "nvidia"
        / "Eagle-Block2A-2B-v2"
    )

    env_local_path = os.environ.get("EAGLE_LOCAL_PATH")
    hardcoded_local_path = f"/workspace/huggingface.co/{expected_repo_id}"
    cache_root = os.environ.get("TRANSFORMERS_CACHE", "")

    def _is_processor_dir(path: Path) -> bool:
        if not path or not path.exists() or not path.is_dir():
            return False
        required_any = (
            "processor_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
        )
        return any((path / filename).exists() for filename in required_any)

    repo_org, repo_name = expected_repo_id.split("/", 1)
    hf_cache_dirname = f"models--{repo_org}--{repo_name}"

    base_candidates = [
        local_assets_path,
        Path(env_local_path) if env_local_path else None,
        vendor_eagle_path,
        Path(hardcoded_local_path),
        Path(hardcoded_local_path) / "tree" / "main",
        Path(cache_root) if cache_root else None,
        (Path(cache_root) / "Eagle-Block2A-2B-v2") if cache_root else None,
        (Path(cache_root) / "eagle3-processor-groot-n1d6") if cache_root else None,
        (Path(cache_root) / hf_cache_dirname) if cache_root else None,
        (Path(cache_root) / hf_cache_dirname / "snapshots") if cache_root else None,
        Path(model_name) if model_name and not model_name.startswith(("nvidia/", "aravindhs-NV/")) else None,
    ]

    snapshots_dir = (Path(cache_root) / hf_cache_dirname / "snapshots") if cache_root else None
    snapshot_dirs = []
    if snapshots_dir and snapshots_dir.exists() and snapshots_dir.is_dir():
        for sub in sorted(snapshots_dir.iterdir(), reverse=True):
            if sub.is_dir():
                snapshot_dirs.append(sub)

    def _expand_candidates(paths: list[Path | None]) -> list[Path]:
        expanded: list[Path] = []
        for p in paths:
            if p is None:
                continue
            if p.is_file():
                continue
            expanded.append(p)
            snap = p / "snapshots"
            if snap.exists() and snap.is_dir():
                for sub in sorted(snap.iterdir(), reverse=True):
                    if sub.is_dir():
                        expanded.append(sub)
        return expanded

    candidate_paths = _expand_candidates(base_candidates) + snapshot_dirs

    local_candidates = []
    for candidate in candidate_paths:
        if candidate is not None and _is_processor_dir(candidate):
            local_candidates.append(candidate)

    cache_key = (model_name, tuple(sorted((transformers_loading_kwargs or {}).items())))
    if cache_key in _PROCESSOR_CACHE:
        return _PROCESSOR_CACHE[cache_key]

    _PROCESSOR_LOG_ONCE.add(cache_key)

    load_errors = []
    base_kwargs = dict(transformers_loading_kwargs)
    if offline_mode:
        base_kwargs["local_files_only"] = True
    # Eagle3_VLImageProcessorFast requires fast=True; ensure fast path is present (we inject if missing)
    base_kwargs["use_fast"] = True

    bundled_dir = Path(__file__).parent / "eagle3_model"
    bundled_processing = bundled_dir / "processing_eagle3_vl.py"
    bundled_image_processing_fast = bundled_dir / "image_processing_eagle3_vl_fast.py"

    for local_path in local_candidates:
        try:
            processing_target = local_path / "processing_eagle3_vl.py"
            if not processing_target.exists() and bundled_processing.exists():
                try:
                    shutil.copy(bundled_processing, processing_target)
                    if verbose_logs:
                        print(f"Injected missing processing_eagle3_vl.py into {local_path}")
                except Exception as copy_exc:
                    if verbose_logs:
                        print(
                            f"Warning: failed to inject processing_eagle3_vl.py into {local_path}: {copy_exc}"
                        )

            image_processing_fast_target = local_path / "image_processing_eagle3_vl_fast.py"
            if not image_processing_fast_target.exists() and bundled_image_processing_fast.exists():
                try:
                    shutil.copy(bundled_image_processing_fast, image_processing_fast_target)
                    if verbose_logs:
                        print(f"Injected missing image_processing_eagle3_vl_fast.py into {local_path}")
                except Exception as copy_exc:
                    if verbose_logs:
                        print(
                            "Warning: failed to inject image_processing_eagle3_vl_fast.py "
                            f"into {local_path}: {copy_exc}"
                        )

            processor = AutoProcessor.from_pretrained(str(local_path), **base_kwargs)
            if cache_key not in _PROCESSOR_CACHE and verbose_logs:
                print(f"Using Eagle processor from local path: {local_path}")
            _PROCESSOR_CACHE[cache_key] = processor
            return processor
        except Exception as exc:
            load_errors.append(f"{local_path}: {exc}")
            if verbose_logs:
                print(f"Warning: failed loading Eagle processor from {local_path}: {exc}")

    if offline_mode:
        raise FileNotFoundError(
            "Offline mode is enabled and no usable local Eagle processor was found. "
            f"Checked candidates={ [str(p) for p in local_candidates] }. "
            f"Errors={load_errors}"
        )

    if disable_hf_fallback:
        raise FileNotFoundError(
            "No usable local Eagle processor found and HF fallback is disabled "
            "(EAGLE_DISABLE_HF_FALLBACK=1). "
            f"Checked candidates={ [str(p) for p in local_candidates] }. "
            f"Errors={load_errors}"
        )

    if tokenizer_assets_repo:
        try:
            vendor_dir = Path(__file__).parent / "eagle3_model"
            cache_dir = ensure_eagle_cache_ready(vendor_dir=vendor_dir, assets_repo=tokenizer_assets_repo)
            processor = AutoProcessor.from_pretrained(str(cache_dir), **base_kwargs)
            if cache_key not in _PROCESSOR_CACHE and verbose_logs:
                print(f"Using Eagle processor from tokenizer assets: {cache_dir}")
            _PROCESSOR_CACHE[cache_key] = processor
            return processor
        except Exception as exc:
            if verbose_logs:
                print(f"Warning: failed loading Eagle processor from tokenizer assets: {exc}")

    processor = AutoProcessor.from_pretrained(expected_repo_id, **base_kwargs)
    if cache_key not in _PROCESSOR_CACHE and verbose_logs:
        print(f"Using Eagle processor from HF Hub: {expected_repo_id}")
    _PROCESSOR_CACHE[cache_key] = processor
    return processor


def make_gr00t_n1d6_pre_post_processors(policy_cfg=None, dataset_stats=None, dataset=None):
    """Build preprocessors aligned with `lerobot` GR00T-N1.6 baseline."""

    if policy_cfg is None:
        raise ValueError("policy_cfg (Gr00tN1d6Config) is required to build processors")

    BaselineGr00tProcessor = Gr00tN1d6Processor

    embodiment_tag = str(getattr(policy_cfg, "embodiment_tag", "libero_panda") or "libero_panda")

    if embodiment_tag in MODALITY_CONFIGS:
        modality_configs = {embodiment_tag: MODALITY_CONFIGS[embodiment_tag]}
    else:
        modality_configs = {
            embodiment_tag: {
                "state": ModalityConfig(delta_indices=[0], modality_keys=["state"]),
                "action": ModalityConfig(
                    delta_indices=list(range(int(getattr(policy_cfg, "action_horizon", 16) or 16))),
                    modality_keys=["action"],
                ),
                "video": ModalityConfig(delta_indices=[0], modality_keys=["image"]),
            }
        }

    dataset_stats_local = dict(dataset_stats or {})
    if embodiment_tag in EMBODIMENT_STAT_CONFIGS:
        action_cfg = EMBODIMENT_STAT_CONFIGS[embodiment_tag]["modality_config"]["action"]
        needs_relative_stats = any(
            cfg.rep.value == "relative" for cfg in (action_cfg.action_configs or [])
        )
        if needs_relative_stats and "relative_action" not in dataset_stats_local:
            if dataset is None:
                raise ValueError(
                    "Missing 'relative_action' stats for GR00T baseline preprocessing and dataset is None"
                )
            dataset_stats_local["relative_action"] = compute_relative_action_stats(dataset, embodiment_tag)

    statistics = None
    if dataset_stats_local:
        statistics = convert_lerobot_stats_to_processor_format(dataset_stats_local, embodiment_tag)

    baseline_processor = BaselineGr00tProcessor(
        modality_configs=modality_configs,
        statistics=statistics,
        formalize_language=bool(getattr(policy_cfg, "formalize_language", True)),
        model_name="nvidia/Eagle-Block2A-2B-v2",
        tokenizer_assets_repo=getattr(
            policy_cfg,
            "tokenizer_assets_repo",
            "aravindhs-NV/eagle3-processor-groot-n1d6",
        ),
        max_state_dim=int(getattr(policy_cfg, "max_state_dim", 29) or 29),
        max_action_dim=int(getattr(policy_cfg, "max_action_dim", 29) or 29),
        max_action_horizon=int(getattr(policy_cfg, "action_horizon", 16) or 16),
        use_albumentations=bool(getattr(policy_cfg, "use_albumentations_transforms", False)),
        use_relative_action=bool(getattr(policy_cfg, "use_relative_action", True)),
        apply_sincos_state_encoding=bool(getattr(policy_cfg, "apply_sincos_state_encoding", False)),
        embodiment_id_mapping={
            embodiment_tag: EMBODIMENT_TAG_TO_PROJECTOR_INDEX.get(embodiment_tag, 10)
        },
        image_target_size=(
            list(getattr(policy_cfg, "image_target_size", None))
            if getattr(policy_cfg, "image_target_size", None)
            else [224, 224]
        ),
        image_crop_size=(
            list(getattr(policy_cfg, "image_crop_size", None))
            if getattr(policy_cfg, "image_crop_size", None)
            else [224, 224]
        ),
        shortest_image_edge=(
            getattr(policy_cfg, "shortest_image_edge", None) or 256
        ),
        crop_fraction=(
            getattr(policy_cfg, "crop_fraction", None) or 0.95
        ),
        random_rotation_angle=getattr(policy_cfg, "random_rotation_angle", None),
        color_jitter_params=getattr(policy_cfg, "color_jitter_params", None),
        use_processor_image_size=False,
    )
    baseline_processor.train()

    modality_meta = (
        EMBODIMENT_STAT_CONFIGS[embodiment_tag]["modality_meta"]
        if embodiment_tag in EMBODIMENT_STAT_CONFIGS
        else None
    )

    def _to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)

    def _to_uint8_hwc(frame: np.ndarray) -> np.ndarray:
        arr = frame
        if arr.ndim != 3:
            raise ValueError(f"Expected image frame ndim=3, got {arr.ndim}")
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected image channel size 3, got shape={arr.shape}")
        if arr.dtype != np.uint8:
            finite_vals = arr[np.isfinite(arr)]
            if finite_vals.size and finite_vals.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    def _extract_frames(image_array: np.ndarray) -> list[np.ndarray]:
        arr = image_array
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4:
            raise ValueError(f"Unsupported image shape {arr.shape}")
        return [_to_uint8_hwc(frame) for frame in arr]

    def _slice_modalities(values: np.ndarray, modality: str) -> dict[str, np.ndarray]:
        if values.ndim == 1:
            values = values[None, :]
        if modality_meta is None:
            return {modality: values}

        grouped: dict[str, np.ndarray] = {}
        for key in modality_configs[embodiment_tag][modality].modality_keys:
            start_idx = modality_meta[modality][key]["start"]
            end_idx = modality_meta[modality][key]["end"]
            grouped[key] = values[..., start_idx:end_idx]
        return grouped

    def _normalize_text(text_value: Any) -> str:
        value = text_value
        if torch.is_tensor(value):
            if value.numel() == 1:
                value = value.item()
            else:
                value = value.tolist()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return value if isinstance(value, str) else str(value)

    try:
        embodiment_enum = EmbodimentTag(embodiment_tag)
    except ValueError:
        embodiment_enum = EmbodimentTag.NEW_EMBODIMENT

    def _build_raw_state(state_tensor: Any) -> dict[str, np.ndarray] | None:
        if modality_meta is None:
            return None
        if state_tensor is None:
            return None
        state_np = _to_numpy(state_tensor)
        if state_np.ndim == 1:
            state_np = state_np[None, :]
        raw_state: dict[str, np.ndarray] = {}
        for key, meta in modality_meta.get("state", {}).items():
            start_idx = meta["start"]
            end_idx = meta["end"]
            raw_state[key] = state_np[:, start_idx:end_idx]
        return raw_state

    def _preprocessor(batch: dict[str, Any]):
        bs = None
        for v in batch.values():
            if torch.is_tensor(v) or isinstance(v, np.ndarray):
                bs = v.shape[0]
                break
        if bs is None:
            raise ValueError("Cannot infer batch size from input batch")

        image_keys = sorted(
            k for k in batch if k.startswith("observation.images.")
        )
        state_tensor = batch.get("observation.state")
        action_tensor = batch.get("action")
        language_tensor = batch.get("task")

        if state_tensor is None:
            raise KeyError("Missing required key 'observation.state' in batch")
        if not image_keys:
            raise KeyError("Missing required observation image keys in batch")

        features: list[dict[str, Any]] = []
        for idx in range(bs):
            images: dict[str, list[np.ndarray]] = {}
            for key in image_keys:
                view_name = key.split("observation.images.", 1)[-1]
                image_arr = _to_numpy(batch[key][idx])
                images[view_name] = _extract_frames(image_arr)

            state_arr = _to_numpy(state_tensor[idx])
            states = _slice_modalities(state_arr, "state")

            actions: dict[str, np.ndarray] = {}
            if action_tensor is not None:
                action_arr = _to_numpy(action_tensor[idx])
                if action_arr.ndim == 1:
                    action_arr = action_arr[None, :]

                expected_action_steps = len(
                    modality_configs[embodiment_tag]["action"].delta_indices
                )

                if action_arr.shape[0] > expected_action_steps:
                    action_arr = action_arr[:expected_action_steps]

                actions = _slice_modalities(action_arr, "action")

            text = ""
            if language_tensor is not None:
                text = _normalize_text(language_tensor[idx])

            vla_step = VLAStepData(
                images=images,
                states=states,
                actions=actions,
                text=text,
                embodiment=embodiment_enum,
            )

            transformed_inputs = baseline_processor([{"role": "user", "content": vla_step}])
            features.append(transformed_inputs)

        batch_feature = baseline_processor.collator(features)
        output = dict(batch_feature.data["inputs"])
        passthrough = [
            "task",
            "info",
            "index",
            "task_index",
            "episode_index",
            "frame_index",
            "timestamp",
            "next.reward",
            "next.done",
            "next.truncated",
            "action_is_pad",
        ]
        for key in passthrough:
            if key in batch:
                output[key] = batch[key]

        if "action_is_pad" in output and "action" in output:
            action_is_pad = output["action_is_pad"]
            action_tensor = output["action"]
            if torch.is_tensor(action_is_pad) and torch.is_tensor(action_tensor):
                if action_is_pad.ndim == 2 and action_tensor.ndim == 3:
                    target_horizon = action_tensor.shape[1]
                    if action_is_pad.shape[1] > target_horizon:
                        output["action_is_pad"] = action_is_pad[:, :target_horizon]
                    elif action_is_pad.shape[1] < target_horizon:
                        pad = torch.ones(
                            action_is_pad.shape[0],
                            target_horizon - action_is_pad.shape[1],
                            device=action_is_pad.device,
                            dtype=action_is_pad.dtype,
                        )
                        output["action_is_pad"] = torch.cat([action_is_pad, pad], dim=1)

        if "action_is_pad" not in output and "action_mask" in output:
            mask = output["action_mask"]
            if torch.is_tensor(mask):
                output["action_is_pad"] = mask.sum(dim=-1) == 0
        raw_state = _build_raw_state(state_tensor)
        if raw_state is not None:
            output["raw_state"] = raw_state
        output["_gr00t_processor"] = baseline_processor
        return output

    def _postprocessor(output):
        """Postprocess the output."""
        return output

    _preprocessor._gr00t_processor = baseline_processor
    return _preprocessor, _postprocessor


class StateActionProcessor(object):
    """Normalize and denormalize state/action signals for embodied training.

    This processor manages per-embodiment normalization statistics and applies
    consistent preprocessing for both states and actions according to
    ``modality_configs``. It supports:

    - Min-max or mean-std normalization based on modality configuration.
    - Optional percentile-based min/max bounds and outlier clipping.
    - Optional sine/cosine encoding for selected state keys.
    - Relative-action representation conversion (apply and reverse).

    The class is used by ``Gr00tN1d6Processor`` to prepare model inputs during
    training/evaluation and to decode actions back to physical scale.
    """
    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        apply_sincos_state_encoding: bool = False,
        use_relative_action: bool = True,
    ):
        """Initialize the state action processor."""
        self.modality_configs = parse_modality_configs(modality_configs)
        self.statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action
        self.norm_params: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}

        if statistics is not None:
            self.set_statistics(statistics)

        self.train()

    def train(self):
        """Set the processor to training mode."""
        self.training = True

    def eval(self):
        """Set the processor to evaluation mode."""
        self.training = False

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """Set the statistics for normalization."""
        for key in statistics:
            if key not in self.statistics or override:
                self.statistics[key] = deepcopy(statistics[key])
            else:
                print(f"Embodiment tag {key} already in statistics, skipping updating")
        self._compute_normalization_parameters()

    def _compute_normalization_parameters(self) -> None:
        """Compute normalization parameters."""
        for embodiment_tag in self.statistics:
            self.norm_params[embodiment_tag] = {}

            for modality in ["state", "action"]:
                if modality not in self.statistics[embodiment_tag]:
                    continue

                self.norm_params[embodiment_tag][modality] = {}

                for joint_group, stats in self.statistics[embodiment_tag][modality].items():
                    if self.use_percentiles:
                        min_vals = np.array(stats["q01"])
                        max_vals = np.array(stats["q99"])
                    else:
                        min_vals = np.array(stats["min"])
                        max_vals = np.array(stats["max"])

                    mean_vals = np.array(stats["mean"])
                    std_vals = np.array(stats["std"])

                    range_vals = max_vals - min_vals
                    range_vals = np.maximum(range_vals, 1e-8)

                    self.norm_params[embodiment_tag][modality][joint_group] = {
                        "min": min_vals,
                        "max": max_vals,
                        "dim": np.array(range_vals.shape[-1]),
                        "mean": mean_vals,
                        "std": std_vals,
                    }

            if "action" in self.modality_configs[embodiment_tag]:
                modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
                action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

                if action_configs is not None:
                    for key, action_config in zip(modality_keys, action_configs, strict=True):
                        if action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                            if "relative_action" not in self.statistics[embodiment_tag]:
                                raise ValueError(
                                    f"Relative action statistics required for embodiment '{embodiment_tag}' "
                                    f"but 'relative_action' not found"
                                )
                            if key not in self.statistics[embodiment_tag]["relative_action"]:
                                raise ValueError(
                                    f"Relative action statistics required for key '{key}' "
                                    f"in embodiment '{embodiment_tag}' but not found"
                                )
                            action_dim = self.norm_params[embodiment_tag]["action"][key]["dim"]
                            self.norm_params[embodiment_tag]["action"][key] = nested_dict_to_numpy(
                                self.statistics[embodiment_tag]["relative_action"][key]
                            )
                            self.norm_params[embodiment_tag]["action"][key]["dim"] = action_dim

    def apply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """Apply state normalization."""
        normalized_values = {}
        state = deepcopy(state)

        sin_cos_keys = None
        if self.apply_sincos_state_encoding:
            state_config = self.modality_configs[embodiment_tag].get("state")
            if state_config and hasattr(state_config, "sin_cos_embedding_keys"):
                sin_cos_keys = state_config.sin_cos_embedding_keys

        for joint_group in self.modality_configs[embodiment_tag]["state"].modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            if sin_cos_keys and joint_group in sin_cos_keys:
                normalized_values[joint_group] = apply_sin_cos_encoding(state[joint_group])
            elif (
                hasattr(self.modality_configs[embodiment_tag]["state"], "mean_std_embedding_keys")
                and self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
                and joint_group in self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
            ):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_meanstd(state[joint_group], params)
                normalized_values[joint_group] = normalized
            else:
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_minmax(state[joint_group], params)

                if self.clip_outliers:
                    normalized = np.clip(normalized, -1.0, 1.0)

                normalized_values[joint_group] = normalized

        return normalized_values

    def apply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Apply action to the given state and return normalized values."""
        action = deepcopy(action)

        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

        if action_configs is not None and self.use_relative_action:
            for key, action_config in zip(modality_keys, action_configs, strict=True):
                if action_config.rep == ActionRepresentation.RELATIVE:
                    if state is None:
                        raise ValueError(f"State dict required for relative action processing of key '{key}'")

                    state_key = action_config.state_key if action_config.state_key else key
                    if state_key not in state:
                        raise KeyError(f"Reference state key '{state_key}' not found in state dict")

                    reference_state = state[state_key][-1]
                    action[key] = action[key] - reference_state

        normalized_values = {}
        for joint_group in modality_keys:
            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )

            params = self.norm_params[embodiment_tag]["action"][joint_group]
            if (
                self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys is not None
                and joint_group in self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys
            ):
                normalized = normalize_values_meanstd(action[joint_group], params)
            else:
                normalized = normalize_values_minmax(action[joint_group], params)

            if self.clip_outliers:
                normalized = np.clip(normalized, -1.0, 1.0)

            normalized_values[joint_group] = normalized

        return normalized_values

    def unapply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Unapply action from the given state and return unnormalized values."""
        unnormalized_values = {}
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys

        for joint_group in modality_keys:
            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )

            params = self.norm_params[embodiment_tag]["action"][joint_group]
            group_values = action[joint_group]

            if (
                self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys is not None
                and joint_group in self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys
            ):
                unnormalized = unnormalize_values_meanstd(group_values, params)
            else:
                unnormalized = unnormalize_values_minmax(group_values, params)

            unnormalized_values[joint_group] = unnormalized

        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

        if action_configs is not None and self.use_relative_action:
            for key, action_config in zip(modality_keys, action_configs, strict=True):
                if action_config.rep == ActionRepresentation.RELATIVE:
                    if state is None:
                        warnings.warn(
                            f"State dict required for relative->absolute conversion of key '{key}', "
                            "but state is None. Returning unnormalized relative actions.",
                            stacklevel=2,
                        )
                        continue

                    state_key = action_config.state_key if action_config.state_key else key

                    if state_key not in state:
                        available_keys = list(state.keys())
                        if len(available_keys) == 1:
                            state_key = available_keys[0]
                        elif "state" in state:
                            state_key = "state"
                        else:
                            continue

                    relative_action = unnormalized_values[key]
                    reference_state = state[state_key]
                    action_dim = relative_action.shape[-1]

                    if reference_state.ndim == 2:
                        ref_state_slice = (
                            reference_state[-1, :action_dim]
                            if reference_state.shape[-1] >= action_dim
                            else reference_state[-1]
                        )
                        if ref_state_slice.shape[-1] < action_dim:
                            padding = np.zeros(action_dim - ref_state_slice.shape[-1])
                            ref_state_slice = np.concatenate([ref_state_slice, padding])
                        unnormalized_values[key] = relative_action + ref_state_slice
                    elif reference_state.ndim == 3:
                        ref_state_slice = (
                            reference_state[:, -1:, :action_dim]
                            if reference_state.shape[-1] >= action_dim
                            else reference_state[:, -1:]
                        )
                        if ref_state_slice.shape[-1] < action_dim:
                            padding = np.zeros(
                                (ref_state_slice.shape[0], 1, action_dim - ref_state_slice.shape[-1])
                            )
                            ref_state_slice = np.concatenate([ref_state_slice, padding], axis=-1)
                        unnormalized_values[key] = relative_action + ref_state_slice
                    elif reference_state.ndim == 1:
                        ref_state_slice = (
                            reference_state[:action_dim]
                            if reference_state.shape[-1] >= action_dim
                            else reference_state
                        )
                        if ref_state_slice.shape[-1] < action_dim:
                            padding = np.zeros(action_dim - ref_state_slice.shape[-1])
                            ref_state_slice = np.concatenate([ref_state_slice, padding])
                        unnormalized_values[key] = relative_action + ref_state_slice

        return unnormalized_values

    def apply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Apply processing to state and action."""
        processed_state = self.apply_state(state, embodiment_tag)
        if action:
            processed_action = self.apply_action(action, embodiment_tag, state=state)
        else:
            assert not self.training, "Action is required in training mode"
            processed_action = {}
        return processed_state, processed_action

    def get_action_dim(self, embodiment_tag: str) -> int:
        """Get action dimension for given embodiment tag."""
        total_dim = 0
        for joint_group in self.modality_configs[embodiment_tag]["action"].modality_keys:
            total_dim += self.norm_params[embodiment_tag]["action"][joint_group]["dim"].item()
        return total_dim


class Gr00tN1d6DataCollator:
    """Data collator for Gr00tN1d6 model."""
    def __init__(
        self,
        model_name: str,
        tokenizer_assets_repo: str = "lerobot/eagle3-processor-groot-n1d6",
        model_type: Literal["eagle"] = "eagle",
        transformers_loading_kwargs: dict | None = None,
    ):
        """Initialize data collator."""
        if transformers_loading_kwargs is None:
            transformers_loading_kwargs = {}
        self.processor = build_processor(model_name, tokenizer_assets_repo, transformers_loading_kwargs)
        self.processor.tokenizer.padding_side = "left"
        self.model_type = model_type
        self.model_name = model_name

    def __call__(self, features: list[dict[str, Any]]) -> BatchFeature:
        """Process features into batch."""
        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))

        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            if key == "vlm_content":
                text_list = []
                image_inputs = []
                for v in values:
                    text_list += [v["text"]]
                    image_inputs += v["images"]

                if self.model_type == "eagle":
                    image_inputs, _ = self.processor.process_vision_info([v["conversation"] for v in values])
                vlm_inputs = self.processor(text=text_list, images=image_inputs, return_tensors="pt", padding=True)
                for k, v in vlm_inputs.items():
                    batch[k] = v
            elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
                raise Exception("Not implemented")
            else:
                batch[key] = torch.from_numpy(np.stack(values))
        return BatchFeature(data={"inputs": batch})

    def __str__(self):
        """String representation of data collator."""
        return f"Gr00tN1d6DataCollator(model_name={self.model_name}, model_type={self.model_type})"


@dataclass
class VLAStepData:
    """VLAStep data class."""
    images: dict[str, list[np.ndarray]]
    states: dict[str, np.ndarray]
    actions: dict[str, np.ndarray]
    text: str | None = None
    embodiment: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    is_demonstration: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class Gr00tN1d6Processor:
    """Gr00tN1d6 processor class."""
    data_collator_class = Gr00tN1d6DataCollator

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        image_crop_size: list[int] | None = None,
        image_target_size: list[int] | None = None,
        shortest_image_edge: int = 512,
        crop_fraction: float = 0.95,
        random_rotation_angle: int | None = None,
        color_jitter_params: dict[str, float] | None = None,
        formalize_language: bool = True,
        model_name: str = "nvidia/Eagle-Block2A-2B-v2",
        tokenizer_assets_repo: str = "lerobot/eagle3-processor-groot-n1d6",
        model_type: Literal["eagle"] = "eagle",
        max_state_dim: int = 29,
        max_action_dim: int = 29,
        apply_sincos_state_encoding: bool = False,
        max_action_horizon: int = 16,
        use_albumentations: bool = False,
        use_relative_action: bool = True,
        embodiment_id_mapping: dict[str, int] | None = None,
        transformers_loading_kwargs: dict | None = None,
        use_processor_image_size: bool = False,
    ):
        """Initialize Gr00tN1d6 processor."""
        if transformers_loading_kwargs is None:
            transformers_loading_kwargs = {"trust_remote_code": True}

        self.modality_configs = parse_modality_configs(modality_configs)

        self.state_action_processor = StateActionProcessor(
            modality_configs=modality_configs,
            statistics=statistics,
            use_percentiles=use_percentiles,
            clip_outliers=clip_outliers,
            apply_sincos_state_encoding=apply_sincos_state_encoding,
            use_relative_action=use_relative_action,
        )

        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action

        self.formalize_language = formalize_language
        self.model_name = model_name
        self.tokenizer_assets_repo = tokenizer_assets_repo
        self.model_type = model_type

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_action_horizon = max_action_horizon
        self.image_crop_size = image_crop_size
        self.image_target_size = image_target_size
        self.random_rotation_angle = random_rotation_angle
        self.color_jitter_params = color_jitter_params
        self.processor = build_processor(model_name, tokenizer_assets_repo, transformers_loading_kwargs)
        self.processor.tokenizer.padding_side = "left"
        self.embodiment_id_mapping = embodiment_id_mapping or EMBODIMENT_TAG_TO_PROJECTOR_INDEX.copy()
        for k, v in EMBODIMENT_TAG_TO_PROJECTOR_INDEX.items():
            if k not in self.embodiment_id_mapping:
                self.embodiment_id_mapping[k] = v
        self.shortest_image_edge = shortest_image_edge
        self.crop_fraction = crop_fraction
        self.use_processor_image_size = use_processor_image_size

        if use_albumentations and not ALBUMENTATIONS_AVAILABLE:
            warnings.warn(
                "use_albumentations_transforms=True but albumentations is not installed.",
                UserWarning,
                stacklevel=2,
            )
            use_albumentations = False

        self.use_albumentations = use_albumentations
        if use_albumentations:
            self.train_image_transform, self.eval_image_transform = (
                build_image_transformations_albumentations(
                    image_target_size,
                    image_crop_size,
                    random_rotation_angle,
                    color_jitter_params,
                    shortest_image_edge,
                    crop_fraction,
                )
            )
        else:
            self.train_image_transform, self.eval_image_transform = build_image_transformations(
                image_target_size,
                image_crop_size,
                random_rotation_angle,
                color_jitter_params,
                shortest_image_edge,
                crop_fraction,
            )
        self._collator = self.data_collator_class(
            model_name=model_name,
            tokenizer_assets_repo=tokenizer_assets_repo,
            model_type=model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )
        self.training = True
        self._cached_raw_state = None

    @property
    def collator(self):
        """
        Returns the collator instance.
        """
        return self._collator

    def train(self):
        """
        Sets the training mode to True.
        """
        self.training = True
        self.state_action_processor.train()

    def eval(self):
        """
        Sets the training mode to False.
        """
        self.training = False
        self.state_action_processor.eval()

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """
        Set statistics for the state action processor.

        Args:
            statistics: A nested dictionary containing statistics.
            override: Whether to override existing statistics.
        """
        self.state_action_processor.set_statistics(statistics, override=override)
        self.action_dim = {}
        for embodiment_tag in self.state_action_processor.statistics:
            self.action_dim[embodiment_tag] = self.state_action_processor.get_action_dim(embodiment_tag)

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ):
        """
        Decodes the action into a dictionary of joint positions.

        Args:
            action: The action array to decode.
            embodiment_tag: The embodiment tag.
            state: The state dictionary (optional).
        """
        out_dict = {}
        start_idx = 0
        joint_groups = self.modality_configs[embodiment_tag.value]["action"].modality_keys
        action_horizon = len(self.modality_configs[embodiment_tag.value]["action"].delta_indices)
        for key in joint_groups:
            joint_dim = self.state_action_processor.norm_params[embodiment_tag.value]["action"][key]["dim"].item()
            out_dict[key] = action[..., :action_horizon, start_idx : start_idx + joint_dim]
            start_idx += joint_dim

        return self.state_action_processor.unapply_action(out_dict, embodiment_tag.value, state=state)

    def _apply_vlm_processing(self, images: np.ndarray, language: str) -> dict:
        pil_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in images]
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": language},
                    *[{"type": "image", "image": img} for img in pil_images],
                ],
            }
        ]
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        return {"vlm_content": {"text": text, "images": pil_images, "conversation": conversation}}

    def __call__(self, messages: list[dict[str, Any]]):
        assert len(messages) == 1
        content = messages[0]["content"]
        embodiment_tag = content.embodiment
        action_data = content.actions
        state_data = content.states

        normalized_states, normalized_actions = self.state_action_processor.apply(
            state=state_data,
            action=action_data,
            embodiment_tag=embodiment_tag.value,
        )

        if normalized_actions:
            action_keys = self.modality_configs[embodiment_tag.value]["action"].modality_keys
            action_tensors = []
            for key in action_keys:
                arr = normalized_actions[key]
                arr_tensor = torch.from_numpy(arr)
                if arr_tensor.ndim == 1:
                    arr_tensor = arr_tensor.unsqueeze(0)
                elif arr_tensor.ndim == 3:
                    if arr_tensor.shape[0] == 1:
                        arr_tensor = arr_tensor.squeeze(0)
                    elif arr_tensor.shape[1] == 1:
                        arr_tensor = arr_tensor.squeeze(1)
                    else:
                        arr_tensor = arr_tensor[0]
                action_tensors.append(arr_tensor)
            normalized_actions = torch.cat(action_tensors, dim=-1)
            action_dim = normalized_actions.shape[1]
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        normalized_actions.shape[0],
                        self.max_action_dim - normalized_actions.shape[1],
                    ),
                ],
                dim=-1,
            )
            action_horizon = normalized_actions.shape[0]
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        self.max_action_horizon - normalized_actions.shape[0],
                        self.max_action_dim,
                    ),
                ],
                dim=0,
            )
            action_mask = torch.ones_like(normalized_actions)
            action_mask[action_horizon:] = 0
            action_mask[:, action_dim:] = 0
        else:
            assert not self.training, "Action is required in training mode"
            normalized_actions = None
            action_mask = None

        state_keys = self.modality_configs[embodiment_tag.value]["state"].modality_keys
        normalized_states = torch.cat(
            [torch.from_numpy(normalized_states[key]) for key in state_keys], dim=-1
        )
        normalized_states = torch.cat(
            [
                normalized_states,
                torch.zeros(normalized_states.shape[0], self.max_state_dim - normalized_states.shape[1]),
            ],
            dim=-1,
        )

        image_transform = None if self.use_processor_image_size else (
            self.train_image_transform if self.training else self.eval_image_transform
        )
        if content.images:
            image_keys = list(content.images.keys())
        else:
            image_keys = self.modality_configs[embodiment_tag.value]["video"].modality_keys

        if self.formalize_language:
            language = content.text.lower()
            language = re.sub(r"[^\w\s]", "", language)
        else:
            language = content.text

        vlm_inputs = self._get_vlm_inputs(
            image_keys=image_keys,
            images=content.images,
            image_transform=image_transform,
            language=language,
        )

        transformed_inputs = {"state": normalized_states.to(torch.get_default_dtype())}
        if normalized_actions is not None:
            transformed_inputs["action"] = normalized_actions.to(torch.get_default_dtype())
        transformed_inputs.update(vlm_inputs)
        if action_mask is not None:
            transformed_inputs["action_mask"] = action_mask
        transformed_inputs["embodiment_id"] = self.embodiment_id_mapping[embodiment_tag.value]
        return transformed_inputs

    def _get_vlm_inputs(self, image_keys: list[str], images: dict[str, list], image_transform, language: str):
        temporal_stacked_images = {}

        if not getattr(self, "_gr00t_pre_transform_printed", False) and image_keys:
            first_view = image_keys[0]
            if first_view in images and images[first_view]:
                raw_img = images[first_view][0]
                raw_arr = np.array(raw_img)
                checksum = int(raw_arr.sum()) if raw_arr.size else None
                print(
                    "[GR00T-N1.6][Omni][debug_once] "
                    f"pre_transform image_keys={image_keys} first_view={first_view} "
                    f"raw shape={raw_arr.shape} dtype={raw_arr.dtype} "
                    f"min={raw_arr.min() if raw_arr.size else None} "
                    f"max={raw_arr.max() if raw_arr.size else None} "
                    f"sum={checksum}"
                )
            self._gr00t_pre_transform_printed = True
        
        if self.use_albumentations:
            replay = None
            for view in image_keys:
                assert view in images, f"{view} not in {images}"
                transformed_images, replay = apply_with_replay(image_transform, images[view], replay)
              
                temporal_stacked_images[view] = torch.stack(transformed_images)
        else:
            for view in image_keys:
                assert view in images, f"{view} not in {images}"
                temporal_stacked_images[view] = torch.stack([image_transform(img) for img in images[view]])

        for k, v in temporal_stacked_images.items():
            assert isinstance(k, str), f"{k} is not a string"
            assert isinstance(v, torch.Tensor), f"{v} is not a torch tensor"
            assert v.ndim == 4, f"{v} is not a 4D tensor"
            assert v.dtype == torch.uint8, f"{v} is not a uint8 tensor"
            assert v.shape[1] == 3, f"{v} is not a 3 channel tensor"

        stacked_images = (
            torch.stack([temporal_stacked_images[view] for view in image_keys], dim=1).flatten(0, 1).numpy()
        )


        if not getattr(self, "_gr00t_debug_printed", False):
            img_min = stacked_images.min() if stacked_images.size else None
            img_max = stacked_images.max() if stacked_images.size else None
            image_processor = getattr(self.processor, "image_processor", None)
            print(
                "[GR00T-N1.6][Omni][debug_once] "
                f"stacked_images dtype={stacked_images.dtype} min={img_min} max={img_max} shape={stacked_images.shape}"
            )
            if image_keys:
                first_view = image_keys[0]
                first_img = temporal_stacked_images[first_view][0]
                first_arr = first_img.detach().cpu().numpy() if torch.is_tensor(first_img) else np.array(first_img)
                checksum = int(first_arr.sum()) if first_arr.size else None
                print(
                    "[GR00T-N1.6][Omni][debug_once] "
                    f"image_keys={image_keys} first_view={first_view} "
                    f"first_img shape={first_arr.shape} dtype={first_arr.dtype} "
                    f"min={first_arr.min() if first_arr.size else None} "
                    f"max={first_arr.max() if first_arr.size else None} "
                    f"sum={checksum}"
                )
            if image_processor is not None:
                print(
                    "[GR00T-N1.6][Omni][debug_once] "
                    f"image_mean={getattr(image_processor, 'image_mean', None)} "
                    f"image_std={getattr(image_processor, 'image_std', None)} "
                    f"do_rescale={getattr(image_processor, 'do_rescale', None)} "
                    f"rescale_factor={getattr(image_processor, 'rescale_factor', None)} "
                    f"size={getattr(image_processor, 'size', None)}"
                )
            self._gr00t_debug_printed = True

        vlm_inputs = self._apply_vlm_processing(stacked_images, language)
        return vlm_inputs


__all__ = [
    "Gr00tN1d6Processor",
    "Gr00tN1d6DataCollator",
    "VLAStepData",
    "make_gr00t_n1d6_pre_post_processors",
]
