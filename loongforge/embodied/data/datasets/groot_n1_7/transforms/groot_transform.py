# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from NVIDIA GR00T under the Apache-2.0 License.

"""Per-sample GR00T-N1.7 transforms."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
import torch
import warnings

from loongforge.embodied.data.datasets.transforms.base import BaseTransform
from loongforge.embodied.data.datasets.groot_n1_6.transforms.processor_groot_n1_6 import (
    StateActionProcessor,
)
from loongforge.embodied.data.datasets.transforms.registry import (
    TransformBuilderContext,
    register_transform_builder,
)
from loongforge.embodied.data.datasets.groot_n1_6.transforms.utils import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from loongforge.embodied.data.datasets.groot_n1_7.transforms.image_augmentations import (
    apply_with_replay,
    build_image_transformations,
    build_image_transformations_albumentations,
)
from loongforge.embodied.data.datasets.groot_n1_7.transforms.data_configuration_groot_n1_7 import (
    GrootN1d7DataConfig,
)
from loongforge.embodied.model.groot_n1_7.model_configuration_groot_n1_7 import GrootN1d7Config


EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    "oxe_droid_relative_eef_relative_joint": 24,
    "xdof_relative_eef_relative_joint": 27,
    "xdof_relative_eef_relative_joint_subtask": 27,
    "real_g1_relative_eef_relative_joints": 25,
    "real_r1_pro_sharpa_relative_eef": 26,
    "real_r1_pro_sharpa_relative_eef_human": 26,
    "real_r1_pro_sharpa_relative_eef_maxinsights": 26,
    "real_r1_pro_sharpa_relative_eef_mecka": 26,
    "unitree_g1_full_body_with_waist_height_nav_cmd": 25,
    "simpler_env_google": 0,
    "simpler_env_widowx": 1,
    "libero_sim": 2,
    "new_embodiment": 10,
}


LIBERO_SIM_MODALITY_META = {
    "state": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 7, "end": 8},
    },
    "action": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 7},
    },
}


LIBERO_SIM_MODALITY_CONFIG = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["image", "wrist_image"]),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
        ]
        * 7,
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}


NEW_EMBODIMENT_MODALITY_META = {
    "state": {
        "single_arm": {"start": 0, "end": 5},
        "gripper": {"start": 5, "end": 6},
    },
    "action": {
        "single_arm": {"start": 0, "end": 5},
        "gripper": {"start": 5, "end": 6},
    },
}


NEW_EMBODIMENT_MODALITY_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["exterior_1_left", "wrist_left"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.language.language_instruction"],
    ),
}


MODALITY_CONFIGS = {
    "libero_sim": LIBERO_SIM_MODALITY_CONFIG,
    "new_embodiment": NEW_EMBODIMENT_MODALITY_CONFIG,
}


EMBODIMENT_STAT_CONFIGS = {
    "libero_sim": {
        "modality_meta": LIBERO_SIM_MODALITY_META,
        "modality_config": LIBERO_SIM_MODALITY_CONFIG,
    },
    "new_embodiment": {
        "modality_meta": NEW_EMBODIMENT_MODALITY_META,
        "modality_config": NEW_EMBODIMENT_MODALITY_CONFIG,
    },
}


NEW_EMBODIMENT_GROUP_STAT_KEYS = {
    "state": {
        "single_arm": "observation.state.single_arm",
        "gripper": "observation.state.gripper",
    },
    "action": {
        "single_arm": "action.single_arm",
        "gripper": "action.gripper",
    },
}


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


def _resolve_data_action_horizon(model_cfg: Any, data_cfg: Any) -> int:
    configured = data_cfg.preprocess_action_horizon
    if configured is not None:
        return int(configured)
    return _resolve_checkpoint_action_horizon(model_cfg, data_cfg) or int(model_cfg.action_horizon)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open(encoding="utf-8") as file_obj:
            return json.load(file_obj)
    return {}


def _resolve_checkpoint_path(model_cfg: Any) -> Path | None:
    path_value = model_cfg.base_model_path or model_cfg.pretrained_checkpoint
    if path_value is None:
        return None
    path = Path(str(path_value))
    return path if path.exists() else None


def _resolve_checkpoint_action_horizon(model_cfg: Any, data_cfg: Any) -> int | None:
    checkpoint_path = _resolve_checkpoint_path(model_cfg)
    if checkpoint_path is None:
        return None
    processor_config = _read_json_if_exists(checkpoint_path / "processor_config.json")
    processor_kwargs = processor_config.get("processor_kwargs", {})
    if not isinstance(processor_kwargs, dict):
        return None
    modality_configs = processor_kwargs.get("modality_configs", {})
    if not isinstance(modality_configs, dict):
        return None
    embodiment_tag = data_cfg.embodiment_tag
    embodiment_config = modality_configs.get(embodiment_tag, {})
    if not isinstance(embodiment_config, dict):
        return None
    action_config = embodiment_config.get("action", {})
    if not isinstance(action_config, dict):
        return None
    delta_indices = action_config.get("delta_indices", [])
    if not isinstance(delta_indices, list) or not delta_indices:
        return None
    return len(delta_indices)


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

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply GR00T-N1.7 sample transform."""
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

        if self.data_cfg.exclude_state or (
            self.data_cfg.state_dropout_prob > 0
            and random.random() < self.data_cfg.state_dropout_prob
            and self.training
        ):
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
        result["vlm_content"] = self._build_vlm_content(images, language, masks or None)
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
    ) -> dict[str, Any]:
        image_keys = list(images.keys())
        image_transform = self.train_image_transform if self.training else self.eval_image_transform

        temporal_stacked_images = {}
        if self.use_albumentations:
            replay = None
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


def _as_size_list(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, int):
        return [value, value]
    return list(value)


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.array(value)


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


def convert_lerobot_stats_to_groot_n1d7_format(
    dataset_stats: Dict[str, Any],
    embodiment_tag: str = "libero_sim",
    *,
    modality_meta: dict[str, dict[str, dict[str, int]]] | None = None,
    modality_config: dict[str, ModalityConfig] | None = None,
) -> dict:
    """Convert flat LoongForge LeRobot stats to processor-style GR00T stats."""
    if "statistics" in dataset_stats and isinstance(dataset_stats["statistics"], dict):
        dataset_stats = dataset_stats["statistics"]
    if modality_meta is None or modality_config is None:
        if embodiment_tag not in EMBODIMENT_STAT_CONFIGS:
            raise ValueError(f"Unsupported GR00T-N1.7 embodiment tag: {embodiment_tag}")
        modality_meta = EMBODIMENT_STAT_CONFIGS[embodiment_tag]["modality_meta"]
        modality_config = EMBODIMENT_STAT_CONFIGS[embodiment_tag]["modality_config"]
    if not modality_meta or not modality_config:
        raise ValueError(f"Unsupported GR00T-N1.7 embodiment tag: {embodiment_tag}")
    statistics = {embodiment_tag: {}}

    stats_key_map = {"state": "observation.state", "action": "action"}
    for modality in ("state", "action"):
        source_key = stats_key_map[modality]
        if source_key not in dataset_stats:
            raise KeyError(f"Missing dataset statistics key '{source_key}'")
        source_stats = dataset_stats[source_key]
        statistics[embodiment_tag][modality] = {}
        for joint_group in modality_config[modality].modality_keys:
            group_source_stats = _get_group_source_stats(
                dataset_stats,
                embodiment_tag,
                modality,
                joint_group,
            )
            if group_source_stats is not None:
                statistics[embodiment_tag][modality][joint_group] = _copy_stats(group_source_stats)
            else:
                meta = modality_meta[modality][joint_group]
                statistics[embodiment_tag][modality][joint_group] = _slice_stats(
                    source_stats,
                    meta["start"],
                    meta["end"],
                )

    if "relative_action" in dataset_stats:
        statistics[embodiment_tag]["relative_action"] = deepcopy(dataset_stats["relative_action"])
    return statistics


def get_groot_n1d7_statistics(dataset_stats: Dict[str, Any], embodiment_tag: str = "libero_sim") -> dict:
    """Backward-compatible alias for GR00T-N1.7 statistics conversion."""
    return convert_lerobot_stats_to_groot_n1d7_format(dataset_stats, embodiment_tag)


def _slice_stats(stats_dict: dict[str, Any], start_idx: int, end_idx: int) -> dict[str, list]:
    sliced = {}
    for stat_type, values in stats_dict.items():
        arr = _to_numpy(values)
        sliced[stat_type] = arr[start_idx:end_idx].tolist()
    return sliced


def _copy_stats(stats_dict: dict[str, Any]) -> dict[str, list]:
    copied = {}
    for stat_type, values in stats_dict.items():
        arr = _to_numpy(values)
        copied[stat_type] = arr.tolist()
    return copied


def _get_group_source_stats(
    dataset_stats: dict[str, Any],
    embodiment_tag: str,
    modality: str,
    joint_group: str,
) -> dict[str, Any] | None:
    if embodiment_tag != "new_embodiment":
        return None
    source_key = NEW_EMBODIMENT_GROUP_STAT_KEYS.get(modality, {}).get(joint_group)
    if source_key is None:
        return None
    value = dataset_stats.get(source_key)
    return value if isinstance(value, dict) else None


def _ensure_new_embodiment_group_stats(
    dataset_stats: dict[str, Any],
    dataset: Any,
) -> dict[str, Any]:
    root = dataset.root or dataset.dataset_path
    if root is None:
        return dataset_stats
    root = Path(root)
    required_keys = {
        source_key
        for modality_sources in NEW_EMBODIMENT_GROUP_STAT_KEYS.values()
        for source_key in modality_sources.values()
    }
    missing_keys = [key for key in sorted(required_keys) if key not in dataset_stats]
    if not missing_keys:
        return dataset_stats
    computed = _compute_dataset_feature_stats(root, missing_keys)
    merged = dict(dataset_stats)
    merged.update(computed)
    return merged


def _compute_dataset_feature_stats(root: Path, feature_keys: list[str]) -> dict[str, dict[str, list]]:
    import pandas as pd

    parquet_files = sorted((root / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    values_by_key: dict[str, list[np.ndarray]] = {key: [] for key in feature_keys}
    missing_by_key: dict[str, bool] = {key: True for key in feature_keys}
    for parquet_file in parquet_files:
        frame = pd.read_parquet(parquet_file, columns=feature_keys)
        for key in feature_keys:
            if key not in frame.columns:
                continue
            missing_by_key[key] = False
            values_by_key[key].extend(
                np.asarray(value, dtype=np.float32).reshape(-1)
                for value in frame[key].to_numpy()
            )

    missing = [key for key, is_missing in missing_by_key.items() if is_missing]
    if missing:
        raise KeyError(f"Missing parquet columns required for new_embodiment stats: {missing}")

    stats: dict[str, dict[str, list]] = {}
    for key, values in values_by_key.items():
        data = np.vstack(values)
        stats[key] = {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }
    return stats


def _apply_qwen_chat_template(conversation: list[dict[str, Any]]) -> str:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            from transformers.utils.chat_template_utils import render_jinja_template

            rendered, _ = render_jinja_template(
                conversations=[conversation],
                tools=None,
                documents=None,
                chat_template=_QWEN_CHAT_TEMPLATE,
                add_generation_prompt=False,
                continue_final_message=False,
                return_assistant_tokens_mask=False,
                template_kwargs={},
                return_dict=False,
            )
            return rendered[0]
        except Exception:
            pieces = []
            for message in conversation:
                content = message.get("content", [])
                if isinstance(content, str):
                    pieces.append(content)
                    continue
                for item in content:
                    if item.get("type") == "image":
                        pieces.append("<|vision_start|><|image_pad|><|vision_end|>")
                    elif item.get("type") == "text":
                        pieces.append(str(item.get("text", "")))
            return "\n".join(pieces)


_QWEN_CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}"
    "{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "{% endif %}<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
    "{% else %}{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}{% endif %}"
    "{% endfor %}<|im_end|>\n"
    "{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% endif %}"
)


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
