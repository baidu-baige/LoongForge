# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from NVIDIA GR00T under the Apache-2.0 License.

"""Per-sample GR00T-N1.7 transforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
import torch

from loongforge.data.embodied.transforms.base import BaseTransform
from loongforge.data.embodied.groot_n1_6.groot_n1_6_transform import (
    _as_size_list,
    _extract_frames,
    _normalize_text,
    _to_numpy,
)
from loongforge.data.embodied.groot_n1_6.processing_groot_n1_6 import (
    StateActionProcessor,
)
from loongforge.data.embodied.registry import (
    TransformBuilderContext,
    register_transform_builder,
)
from loongforge.models.embodied.groot_n1_6.embodiment_configs import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from loongforge.models.embodied.groot_n1_7.embodiment_configs import (
    EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    MODALITY_CONFIGS,
    convert_lerobot_stats_to_groot_n1d7_format,
)
from loongforge.data.embodied.groot_n1_7.image_augmentations import (
    apply_with_replay,
    build_image_transformations,
    build_image_transformations_albumentations,
)
from loongforge.data.embodied.groot_n1_7.data_configuration_groot_n1_7 import (
    GrootN1d7DataConfig,
)
from loongforge.data.embodied.groot_n1_7.groot_n1_7_dataset import (
    _read_json_if_exists,
    _resolve_checkpoint_path,
    _resolve_data_action_horizon,
)
from loongforge.models.embodied.groot_n1_7.configuration_groot_n1_7 import GrootN1d7Config


@dataclass
class GrootN1d7RuntimeSemantics:
    """Runtime modality semantics resolved from a LeRobot dataset."""

    embodiment_tag: str
    modality_meta: dict[str, dict[str, dict[str, int]]]
    modality_config: dict[str, ModalityConfig]
    processor_stats: dict[str, Any] | None
    video_key_mapping: dict[str, str]
    embodiment_id: int
    data_action_horizon: int


def _load_relative_action_stats(dataset: Any) -> dict[str, Any]:
    dataset_path = dataset.dataset_path or dataset.root
    if dataset_path is None:
        return {}
    relative_stats = _read_json_if_exists(Path(dataset_path) / "meta" / "relative_stats.json")
    relative_stats.pop("__fingerprints__", None)
    return relative_stats


def _resolve_runtime_semantics(
    model_cfg: Any,
    data_cfg: Any,
    dataset_stats: Optional[Dict[str, Any]],
    dataset: Any,
) -> GrootN1d7RuntimeSemantics:
    policy_cfg = GrootN1d7Config.from_config(model_cfg)
    data_cfg = GrootN1d7DataConfig() if data_cfg is None else data_cfg
    embodiment_tag = data_cfg.embodiment_tag
    data_action_horizon = _resolve_data_action_horizon(policy_cfg, data_cfg)
    features = _get_dataset_features(dataset)
    modality_json = dataset.modality
    if not isinstance(modality_json, dict):
        root = _get_dataset_root(dataset)
        modality_json = _read_json_if_exists(root / "meta" / "modality.json") if root else {}

    modality_meta = _resolve_modality_meta(features, modality_json)
    modality_config = _resolve_modality_config(
        features=features,
        modality_json=modality_json,
        modality_meta=modality_meta,
        action_horizon=data_action_horizon,
        fallback_config=MODALITY_CONFIGS.get(embodiment_tag),
    )
    processor_stats = _build_runtime_processor_stats(
        dataset_stats,
        dataset,
        embodiment_tag,
        modality_meta,
        modality_config,
    )
    video_key_mapping = _resolve_video_key_mapping(
        features=features,
        modality_json=modality_json,
        modality_config=modality_config,
    )
    embodiment_id = _resolve_embodiment_id(policy_cfg, data_cfg)
    return GrootN1d7RuntimeSemantics(
        embodiment_tag=embodiment_tag,
        modality_meta=modality_meta,
        modality_config=modality_config,
        processor_stats=processor_stats,
        video_key_mapping=video_key_mapping,
        embodiment_id=embodiment_id,
        data_action_horizon=data_action_horizon,
    )


def _get_dataset_root(dataset: Any) -> Path | None:
    root = dataset.root or dataset.dataset_path
    return Path(root) if root is not None else None


def _get_dataset_features(dataset: Any) -> dict[str, Any]:
    info = dataset.info
    if isinstance(info, dict):
        features = info["features"]
        if isinstance(features, dict):
            return features
    root = _get_dataset_root(dataset)
    if root is None:
        return {}
    info = _read_json_if_exists(root / "meta" / "info.json")
    features = info["features"]
    return features if isinstance(features, dict) else {}


def _feature_shape(features: dict[str, Any], key: str) -> list[int]:
    feature = features.get(key, {})
    shape = feature.get("shape", []) if isinstance(feature, dict) else []
    return list(shape) if isinstance(shape, (list, tuple)) else []


def _feature_dim(features: dict[str, Any], key: str, default: int = 0) -> int:
    shape = _feature_shape(features, key)
    if shape:
        return int(shape[0])
    return default


def _state_action_names(features: dict[str, Any], key: str, dim: int) -> list[str]:
    feature = features.get(key, {})
    names = feature.get("names") if isinstance(feature, dict) else None
    if isinstance(names, list) and len(names) >= dim:
        return [str(name) for name in names[:dim]]
    return []


def _resolve_modality_meta(
    features: dict[str, Any],
    modality_json: dict[str, Any],
) -> dict[str, dict[str, dict[str, int]]]:
    meta: dict[str, dict[str, dict[str, int]]] = {"state": {}, "action": {}}
    for modality, feature_key in (("state", "observation.state"), ("action", "action")):
        json_groups = modality_json.get(modality, {}) if isinstance(modality_json, dict) else {}
        if isinstance(json_groups, dict) and json_groups:
            for key, value in json_groups.items():
                if not isinstance(value, dict) or "start" not in value or "end" not in value:
                    continue
                meta[modality][str(key)] = {
                    "start": int(value["start"]),
                    "end": int(value["end"]),
                }
            if meta[modality]:
                continue

        dim = _feature_dim(features, feature_key)
        names = _state_action_names(features, feature_key, dim)
        if names:
            for index, name in enumerate(names):
                meta[modality][_feature_group_key(name)] = {"start": index, "end": index + 1}
            continue
        if dim > 0:
            meta[modality][modality] = {"start": 0, "end": dim}
            continue
    return meta


def _feature_group_key(name: str) -> str:
    base = str(name).removesuffix(".pos").split(".")[-1]
    return base.replace(" ", "_") or "value"


def _resolve_video_modality_keys(
    features: dict[str, Any],
    modality_json: dict[str, Any],
    fallback_config: dict[str, ModalityConfig] | None,
) -> list[str]:
    json_video = modality_json.get("video", {}) if isinstance(modality_json, dict) else {}
    if isinstance(json_video, dict) and json_video:
        return [str(key) for key in json_video]

    keys: list[str] = []
    for key, value in features.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        dtype = value.get("dtype")
        feature_type = value.get("type")
        is_visual = dtype in {"image", "video"} or str(feature_type).upper().endswith("VISUAL")
        if not is_visual:
            continue
        keys.append(key.split("observation.images.", 1)[-1] if key.startswith("observation.images.") else key)
    if keys:
        return keys
    if fallback_config is not None and "video" in fallback_config:
        return list(fallback_config["video"].modality_keys)
    return ["image"]


def _resolve_modality_config(
    *,
    features: dict[str, Any],
    modality_json: dict[str, Any],
    modality_meta: dict[str, dict[str, dict[str, int]]],
    action_horizon: int,
    fallback_config: dict[str, ModalityConfig] | None,
) -> dict[str, ModalityConfig]:
    state_keys = list(modality_meta.get("state", {}))
    action_keys = list(modality_meta.get("action", {}))
    if not state_keys and fallback_config is not None and "state" in fallback_config:
        state_keys = list(fallback_config["state"].modality_keys)
    if not action_keys and fallback_config is not None and "action" in fallback_config:
        action_keys = list(fallback_config["action"].modality_keys)
    if not state_keys:
        state_keys = ["state"]
    if not action_keys:
        action_keys = ["action"]

    video_keys = _resolve_video_modality_keys(features, modality_json, fallback_config)
    action_configs = _resolve_action_configs(action_keys, fallback_config)
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=video_keys),
        "state": ModalityConfig(delta_indices=[0], modality_keys=state_keys),
        "action": ModalityConfig(
            delta_indices=list(range(action_horizon)),
            modality_keys=action_keys,
            action_configs=action_configs,
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }


def _resolve_action_configs(
    action_keys: list[str],
    fallback_config: dict[str, ModalityConfig] | None,
) -> list[ActionConfig]:
    if fallback_config is not None and "action" in fallback_config:
        fallback_action = fallback_config["action"]
        if fallback_action.action_configs is not None and len(fallback_action.action_configs) == len(action_keys):
            return list(fallback_action.action_configs)
    return [
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        )
        for _ in action_keys
    ]


def _resolve_video_key_mapping(
    *,
    features: dict[str, Any],
    modality_json: dict[str, Any],
    modality_config: dict[str, ModalityConfig],
) -> dict[str, str]:
    dataset_video_keys = _resolve_video_modality_keys(features, modality_json, None)
    config_video_keys = list(modality_config["video"].modality_keys)
    if all(key in dataset_video_keys for key in config_video_keys):
        return {key: key for key in config_video_keys}
    if len(config_video_keys) != len(dataset_video_keys):
        return {}
    return {
        dataset_key: config_key
        for config_key, dataset_key in zip(config_video_keys, dataset_video_keys)
    }


def _resolve_embodiment_id(policy_cfg: GrootN1d7Config, data_cfg: Any) -> int:
    checkpoint_path = _resolve_checkpoint_path(policy_cfg)
    if checkpoint_path is not None:
        mapping = _read_json_if_exists(checkpoint_path / "embodiment_id.json")
        value = mapping.get(data_cfg.embodiment_tag)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return EMBODIMENT_TAG_TO_PROJECTOR_INDEX.get(data_cfg.embodiment_tag, 10)


def _build_runtime_processor_stats(
    dataset_stats: Optional[Dict[str, Any]],
    dataset: Any,
    embodiment_tag: str,
    modality_meta: dict[str, dict[str, dict[str, int]]],
    modality_config: dict[str, ModalityConfig],
) -> Optional[dict[str, Any]]:
    dataset_stats_local = {
        key: value
        for key, value in dict(dataset_stats or {}).items()
        if not str(key).startswith("__")
    }
    if not dataset_stats_local:
        return None
    relative_stats = _load_relative_action_stats(dataset)
    if relative_stats:
        dataset_stats_local["relative_action"] = relative_stats
    return convert_lerobot_stats_to_groot_n1d7_format(
        dataset_stats_local,
        embodiment_tag,
        modality_meta=modality_meta,
        modality_config=modality_config,
    )


class GrootN1d7FeatureTransform(BaseTransform):
    """Build GR00T-N1.7 sample features from LoongForge LeRobot samples."""

    def __init__(
        self,
        model_cfg: Any,
        data_cfg: Any = None,
        dataset_stats: Optional[Dict[str, Any]] = None,
        dataset: Any = None,
        training_args: Any = None,
        training: bool = True,
    ):
        super().__init__(apply_to=[], training=training)
        self.policy_cfg = GrootN1d7Config.from_config(model_cfg)
        self.data_cfg = GrootN1d7DataConfig() if data_cfg is None else data_cfg
        self.embodiment_tag = self.data_cfg.embodiment_tag
        self.max_state_dim = int(self.policy_cfg.max_state_dim)
        self.max_action_dim = int(self.policy_cfg.max_action_dim)
        self.max_action_horizon = self.policy_cfg.action_horizon
        self.formalize_language = self.data_cfg.formalize_language
        self.use_albumentations = bool(self.data_cfg.use_albumentations_transforms)
        self.image_target_size = _as_size_list(self.data_cfg.image_target_size, [256, 256])
        self.image_crop_size = _as_size_list(self.data_cfg.image_crop_size, [230, 230])
        self.shortest_image_edge = self.data_cfg.shortest_image_edge
        self.crop_fraction = self.data_cfg.crop_fraction
        self.random_rotation_angle = self.data_cfg.random_rotation_angle
        self.color_jitter_params = self.data_cfg.color_jitter_params
        self.image_augmentation_seed = int(training_args.seed) if training_args is not None else 42
        self._replay_image_shape = _resolve_replay_image_shape(dataset)

        runtime_semantics = _resolve_runtime_semantics(model_cfg, self.data_cfg, dataset_stats, dataset)
        self.modality_configs = {self.embodiment_tag: runtime_semantics.modality_config}
        self._video_key_mapping = runtime_semantics.video_key_mapping
        self.modality_meta = runtime_semantics.modality_meta
        self.embodiment_id = runtime_semantics.embodiment_id
        self.data_action_horizon = runtime_semantics.data_action_horizon

        self.state_action_processor = StateActionProcessor(
            modality_configs=self.modality_configs,
            statistics=runtime_semantics.processor_stats,
            use_percentiles=self.data_cfg.use_percentiles,
            clip_outliers=self.data_cfg.clip_outliers,
            apply_sincos_state_encoding=self.data_cfg.apply_sincos_state_encoding,
            use_relative_action=self.data_cfg.use_relative_action,
        )
        self.state_action_processor.train()

        self.train_image_transform, self.eval_image_transform = self._build_image_transforms()

    def prepare_replay(self, data: Dict[str, Any] | None = None) -> dict[str, Any]:
        """Bind legacy worker-global random draws to one sample.

        Shard workers call this method in the same serial order used before
        parallel prefetching. The expensive deterministic part of the transform
        can then run concurrently without touching worker-global RNG state.
        """
        drop_state = bool(self.data_cfg.exclude_state)
        if not drop_state and self.data_cfg.state_dropout_prob > 0:
            drop_state = (
                random.random() < self.data_cfg.state_dropout_prob
                and self.training
            )

        image_replay = None
        if self.training:
            if not self.use_albumentations:
                raise RuntimeError(
                    "Deterministic shard prefetch requires replayable "
                    "Albumentations transforms"
                )
            image_transform = self.train_image_transform
            sample_replay_for_shape = getattr(
                image_transform,
                "sample_replay_for_shape",
                None,
            )
            if sample_replay_for_shape is None:
                raise RuntimeError(
                    "The configured image transform cannot sample a deterministic replay"
                )
            if getattr(image_transform, "mask_transforms", []):
                raise RuntimeError(
                    "Deterministic shard prefetch does not support random mask transforms"
                )

            image_shape = self._replay_image_shape
            if data is not None:
                image_keys = sorted(
                    key for key in data if key.startswith("observation.images.")
                )
                if not image_keys:
                    raise KeyError("Missing required observation image keys for GR00T-N1.7")
                first_frames = _extract_frames(_to_numpy(data[image_keys[0]]))
                if not first_frames:
                    raise ValueError("GR00T-N1.7 image sequence is empty")
                image_shape = first_frames[0].shape[:2]
            if image_shape is None:
                raise RuntimeError(
                    "Cannot prepare deterministic image replay without image geometry"
                )
            image_replay = sample_replay_for_shape(image_shape)

        return {
            "drop_state": drop_state,
            "image_replay": image_replay,
        }

    def apply_with_replay(
        self,
        data: Dict[str, Any],
        replay: dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply a sample transform using pre-bound random parameters."""
        return self._apply(data, replay=replay)

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply GR00T-N1.7 sample transform."""
        return self._apply(data, replay=None)

    def _apply(
        self,
        data: Dict[str, Any],
        *,
        replay: dict[str, Any] | None,
    ) -> Dict[str, Any]:
        if "observation.state" not in data:
            raise KeyError("Missing required GR00T-N1.7 state key: observation.state")
        if self.training and "action" not in data:
            raise KeyError("Missing required GR00T-N1.7 action key: action")

        image_keys = sorted(key for key in data if key.startswith("observation.images."))
        if not image_keys:
            raise KeyError("Missing required observation image keys for GR00T-N1.7")

        images = {}
        for key in image_keys:
            dataset_view = key.split("observation.images.", 1)[-1]
            config_view = self._video_key_mapping.get(dataset_view, dataset_view)
            images[config_view] = _extract_frames(_to_numpy(data[key]))
        masks = {}
        for key in sorted(key for key in data if key.startswith("observation.masks.")):
            dataset_view = key.split("observation.masks.", 1)[-1]
            config_view = self._video_key_mapping.get(dataset_view, dataset_view)
            masks[config_view] = _extract_masks(_to_numpy(data[key]))
        state_values = _to_numpy(data["observation.state"])
        states = self._slice_modalities(state_values, "state")

        actions: dict[str, np.ndarray] = {}
        if "action" in data and data["action"] is not None:
            action_values = _to_numpy(data["action"])
            if action_values.ndim == 1:
                action_values = action_values[None, :]
            expected_steps = len(self.modality_configs[self.embodiment_tag]["action"].delta_indices)
            if action_values.shape[0] > expected_steps:
                action_values = action_values[:expected_steps]
            actions = self._slice_modalities(action_values, "action")

        normalized_states, normalized_actions = self.state_action_processor.apply(
            state=states,
            action=actions,
            embodiment_tag=self.embodiment_tag,
        )

        if replay is None:
            drop_state = self.data_cfg.exclude_state or (
                self.data_cfg.state_dropout_prob > 0
                and random.random() < self.data_cfg.state_dropout_prob
                and self.training
            )
        else:
            drop_state = bool(replay["drop_state"])
        if drop_state:
            normalized_states = {
                key: np.zeros_like(value)
                for key, value in normalized_states.items()
            }

        result: Dict[str, Any] = {"state": self._pack_state(normalized_states)}
        if normalized_actions:
            action, action_mask = self._pack_action(normalized_actions)
            result["action"] = action
            result["action_mask"] = action_mask
            result["action_is_pad"] = self._build_action_is_pad(data.get("action_is_pad"), action_mask)

        language = _normalize_text(data.get("task", ""))
        if self.formalize_language:
            language = re.sub(r"[^\w\s]", "", language.lower())
        result["vlm_content"] = self._build_vlm_content(
            images,
            language,
            masks or None,
            image_replay=None if replay is None else replay["image_replay"],
            replay_prepared=replay is not None,
        )
        result["embodiment_id"] = np.array(self.embodiment_id, dtype=np.int64)
        return result

    def train(self) -> None:
        """Switch to training mode."""
        self.training = True
        self.state_action_processor.train()

    def eval(self) -> None:
        """Switch to eval mode."""
        self.training = False
        self.state_action_processor.eval()

    def _build_image_transforms(self):
        if self.use_albumentations:
            return build_image_transformations_albumentations(
                self.image_target_size,
                self.image_crop_size,
                self.random_rotation_angle,
                self.color_jitter_params,
                self.shortest_image_edge,
                self.crop_fraction,
                extra_augmentation_config=self.data_cfg.extra_augmentation_config,
                seed=self.image_augmentation_seed,
            )
        return build_image_transformations(
            self.image_target_size,
            self.image_crop_size,
            self.random_rotation_angle,
            self.color_jitter_params,
            self.shortest_image_edge,
            self.crop_fraction,
        )

    def _slice_modalities(self, values: np.ndarray, modality: str) -> dict[str, np.ndarray]:
        if values.ndim == 1:
            values = values[None, :]
        if self.modality_meta is None:
            return {modality: values}
        grouped: dict[str, np.ndarray] = {}
        for key in self.modality_configs[self.embodiment_tag][modality].modality_keys:
            start_idx = self.modality_meta[modality][key]["start"]
            end_idx = self.modality_meta[modality][key]["end"]
            grouped[key] = values[..., start_idx:end_idx]
        return grouped

    def _pack_state(self, normalized_states: dict[str, np.ndarray]) -> np.ndarray:
        state_keys = self.modality_configs[self.embodiment_tag]["state"].modality_keys
        state_tensors = []
        for key in state_keys:
            arr_tensor = torch.from_numpy(normalized_states[key])
            if arr_tensor.ndim == 1:
                arr_tensor = arr_tensor.unsqueeze(0)
            elif arr_tensor.ndim == 3:
                if arr_tensor.shape[0] == 1:
                    arr_tensor = arr_tensor.squeeze(0)
                elif arr_tensor.shape[1] == 1:
                    arr_tensor = arr_tensor.squeeze(1)
                else:
                    arr_tensor = arr_tensor[0]
            state_tensors.append(arr_tensor)
        state = torch.cat(state_tensors, dim=-1)
        state_dim = state.shape[-1]
        if state_dim < self.max_state_dim:
            state = torch.cat(
                [state, torch.zeros(state.shape[0], self.max_state_dim - state_dim)],
                dim=-1,
            )
        elif state_dim > self.max_state_dim:
            state = state[..., : self.max_state_dim]
        return state.to(torch.get_default_dtype()).numpy()

    def _pack_action(
        self,
        normalized_actions: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        action_keys = self.modality_configs[self.embodiment_tag]["action"].modality_keys
        action_tensors = []
        for key in action_keys:
            arr_tensor = torch.from_numpy(normalized_actions[key])
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
        action = torch.cat(action_tensors, dim=-1)

        action_dim = min(action.shape[-1], self.max_action_dim)
        if action.shape[-1] > self.max_action_dim:
            action = action[:, : self.max_action_dim]
        elif action.shape[-1] < self.max_action_dim:
            action = torch.cat(
                [
                    action,
                    torch.zeros(action.shape[0], self.max_action_dim - action.shape[-1]),
                ],
                dim=-1,
            )

        action_horizon = min(action.shape[0], self.max_action_horizon)
        if action.shape[0] > self.max_action_horizon:
            action = action[: self.max_action_horizon]
        elif action.shape[0] < self.max_action_horizon:
            action = torch.cat(
                [
                    action,
                    torch.zeros(self.max_action_horizon - action.shape[0], self.max_action_dim),
                ],
                dim=0,
            )

        action_mask = torch.ones_like(action)
        action_mask[action_horizon:] = 0
        action_mask[:, action_dim:] = 0
        return action.to(torch.get_default_dtype()).numpy(), action_mask.to(torch.float64).numpy()

    def _build_action_is_pad(self, source: Any, action_mask: np.ndarray) -> np.ndarray:
        target_horizon = action_mask.shape[0]
        pad_from_mask = action_mask.sum(axis=-1) == 0
        if source is None:
            return pad_from_mask
        value = _to_numpy(source).astype(bool)
        if value.ndim > 1:
            value = value.reshape(-1)
        if value.shape[0] > target_horizon:
            value = value[:target_horizon]
        elif value.shape[0] < target_horizon:
            value = np.concatenate(
                [value, np.ones(target_horizon - value.shape[0], dtype=bool)],
                axis=0,
            )
        return np.logical_or(value, pad_from_mask)

    def _build_vlm_content(
        self,
        images: dict[str, list[np.ndarray]],
        language: str,
        masks: dict[str, list[np.ndarray]] | None = None,
        image_replay: dict[str, Any] | None = None,
        replay_prepared: bool = False,
    ) -> dict[str, Any]:
        image_keys = list(images.keys())
        image_transform = self.train_image_transform if self.training else self.eval_image_transform

        temporal_stacked_images = {}
        if self.use_albumentations:
            replay = image_replay if replay_prepared else None
            for view in image_keys:
                transformed_images, replay = apply_with_replay(
                    image_transform,
                    images[view],
                    masks.get(view) if masks else None,
                    replay,
                )
                temporal_stacked_images[view] = torch.stack(transformed_images)
        else:
            if masks is not None:
                raise ValueError(
                    "GR00T-N1.7 mask transforms require albumentations image transforms"
                )
            for view in image_keys:
                temporal_stacked_images[view] = torch.stack([image_transform(img) for img in images[view]])

        stacked = torch.stack([temporal_stacked_images[view] for view in image_keys], dim=1)
        stacked_images = stacked.flatten(0, 1).numpy()
        pil_images = [Image.fromarray(np.transpose(frame, (1, 2, 0))) for frame in stacked_images]
        conversation = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in pil_images],
                    {"type": "text", "text": language},
                ],
            }
        ]
        return {
            "images": pil_images,
            "conversation": conversation,
        }


def _resolve_replay_image_shape(dataset: Any) -> tuple[int, int] | None:
    info = getattr(dataset, "info", None)
    if not isinstance(info, dict):
        return None
    features = info.get("features", {})
    if not isinstance(features, dict):
        return None
    for key in sorted(features):
        feature = features[key]
        if not key.startswith("observation.images.") or not isinstance(feature, dict):
            continue
        shape = feature.get("shape")
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            return int(shape[0]), int(shape[1])
    return None


def _extract_masks(mask_array: np.ndarray) -> list[np.ndarray]:
    arr = mask_array
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported mask shape {arr.shape}")
    return [np.ascontiguousarray(frame) for frame in arr]


@register_transform_builder("Gr00tN1d7")
def build_groot_n1_7_transforms(ctx: TransformBuilderContext):
    """Build GR00T-N1.7-specific per-sample transforms."""
    if ctx.dataset.already_transformed:
        return []
    return [
        GrootN1d7FeatureTransform(
            model_cfg=ctx.model_cfg,
            data_cfg=ctx.data_cfg,
            dataset_stats=ctx.dataset_stats,
            dataset=ctx.dataset,
            training_args=ctx.training_args,
        )
    ]
