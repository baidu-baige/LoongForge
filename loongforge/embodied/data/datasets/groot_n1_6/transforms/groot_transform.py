# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Per-sample GR00T-N1.6 transforms."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
import torch

from loongforge.embodied.data.datasets.transforms.base import BaseTransform
from loongforge.embodied.data.datasets.transforms.registry import (
    TransformBuilderContext,
    register_transform_builder,
)
from loongforge.embodied.data.datasets.groot_n1_6.transforms.processor_groot_n1_6 import (
    StateActionProcessor,
)
from loongforge.embodied.data.datasets.groot_n1_6.transforms.utils import (
    ALBUMENTATIONS_AVAILABLE,
    EMBODIMENT_STAT_CONFIGS,
    EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    MODALITY_CONFIGS,
    EmbodimentTag,
    ModalityConfig,
    apply_with_replay,
    build_image_transformations,
    build_image_transformations_albumentations,
    compute_relative_action_stats,
    convert_lerobot_stats_to_processor_format,
)


class GrootPromptTransform(BaseTransform):
    """Ensure every sample has a task prompt string."""

    def __init__(
        self,
        task_key: str = "task",
        prompt_key: str = "task",
        default_prompt: str = "perform the task",
        training: bool = True,
    ):
        super().__init__(apply_to=[prompt_key], training=training)
        self.task_key = task_key
        self.prompt_key = prompt_key
        self.default_prompt = default_prompt

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply prompt normalization transform to data."""
        task = data[self.task_key] if self.task_key in data else None
        if task is None or task == "":
            task = self.default_prompt
        data[self.prompt_key] = _normalize_text(task)
        return data


class GrootStateActionTransform(BaseTransform):
    """Validate state/action presence for GR00T samples."""

    def __init__(
        self,
        state_key: str = "observation.state",
        action_key: str = "action",
        training: bool = True,
    ):
        super().__init__(apply_to=[state_key, action_key], training=training)
        self.state_key = state_key
        self.action_key = action_key

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate state/action keys presence in data dict."""
        if self.state_key not in data:
            raise KeyError(f"Missing required GR00T state key: {self.state_key}")
        if self.training and self.action_key not in data:
            raise KeyError(f"Missing required GR00T action key: {self.action_key}")
        return data


class GrootN1d6FeatureTransform(BaseTransform):
    """Build model-specific GR00T single-sample features.

    This transform owns GR00T modality slicing, state/action normalization,
    relative-action handling, embodiment IDs and VLM conversation assembly.
    The batch collator then only tokenizes/pads text/images and stacks numeric
    fields.
    """

    def __init__(
        self,
        model_cfg: Any,
        data_cfg: Any,
        dataset_stats: Optional[Dict[str, Any]] = None,
        dataset: Any = None,
        training: bool = True,
    ):
        super().__init__(apply_to=[], training=training)
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        # Data-processing fields → DataConfig; shared dims → ModelConfig fallback.
        self.embodiment_tag = str(data_cfg.embodiment_tag or "libero_panda")
        self.max_state_dim = int(
            data_cfg.preprocess_max_state_dim
            if data_cfg.preprocess_max_state_dim is not None
            else model_cfg.max_state_dim
        )
        self.max_action_dim = int(
            data_cfg.preprocess_max_action_dim
            if data_cfg.preprocess_max_action_dim is not None
            else model_cfg.max_action_dim
        )
        self.max_action_horizon = int(
            data_cfg.preprocess_action_horizon
            if data_cfg.preprocess_action_horizon is not None
            else model_cfg.action_horizon
        )
        self.formalize_language = bool(data_cfg.formalize_language)
        self.apply_sincos_state_encoding = bool(data_cfg.apply_sincos_state_encoding)
        self.use_relative_action = bool(data_cfg.use_relative_action)
        self.use_albumentations = bool(data_cfg.use_albumentations_transforms)
        self.use_processor_image_size = bool(data_cfg.use_processor_image_size)
        self.image_target_size = _as_size_list(data_cfg.image_target_size, [224, 224])
        self.image_crop_size = _as_size_list(data_cfg.image_crop_size, [224, 224])
        self.shortest_image_edge = data_cfg.shortest_image_edge or 256
        self.crop_fraction = data_cfg.crop_fraction or 0.95
        self.random_rotation_angle = data_cfg.random_rotation_angle
        self.color_jitter_params = data_cfg.color_jitter_params

        self.modality_configs = self._build_modality_configs()
        self.modality_meta = (
            EMBODIMENT_STAT_CONFIGS[self.embodiment_tag]["modality_meta"]
            if self.embodiment_tag in EMBODIMENT_STAT_CONFIGS
            else None
        )
        self.embodiment_enum = self._build_embodiment_enum()
        self.embodiment_id = EMBODIMENT_TAG_TO_PROJECTOR_INDEX.get(self.embodiment_tag, 10)

        processor_stats = self._build_processor_stats(dataset_stats, dataset)
        self.state_action_processor = StateActionProcessor(
            modality_configs=self.modality_configs,
            statistics=processor_stats,
            apply_sincos_state_encoding=self.apply_sincos_state_encoding,
            use_relative_action=self.use_relative_action,
        )
        self.state_action_processor.train()

        self.train_image_transform = None
        self.eval_image_transform = None
        if not self.use_processor_image_size:
            self.train_image_transform, self.eval_image_transform = self._build_image_transforms()

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply GR00T N1.6 feature transform to a single sample."""
        GrootStateActionTransform(training=self.training).apply(data)

        image_keys = sorted(k for k in data if k.startswith("observation.images."))
        if not image_keys:
            raise KeyError("Missing required observation image keys for GR00T")

        images = {
            key.split("observation.images.", 1)[-1]: _extract_frames(
                _to_numpy(data[key]),
                normalize_mode="identity",
            )
            for key in image_keys
        }
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

        result: Dict[str, Any] = {}
        result["state"] = self._pack_state(normalized_states)
        if normalized_actions:
            action, action_mask = self._pack_action(normalized_actions)
            result["action"] = action
            result["action_mask"] = action_mask
            result["action_is_pad"] = self._build_action_is_pad(data.get("action_is_pad"), action_mask)

        language = _normalize_text(data.get("task", ""))
        if self.formalize_language:
            language = re.sub(r"[^\w\s]", "", language.lower())
        result["vlm_content"] = self._build_vlm_content(images, language)
        result["embodiment_id"] = np.array(self.embodiment_id, dtype=np.int64)
        return result

    def train(self):
        """Switch transform to training mode."""
        self.training = True
        self.state_action_processor.train()

    def eval(self):
        """Switch transform to evaluation mode."""
        self.training = False
        self.state_action_processor.eval()

    def _build_modality_configs(self) -> dict[str, dict[str, ModalityConfig]]:
        if self.embodiment_tag in MODALITY_CONFIGS:
            return {self.embodiment_tag: MODALITY_CONFIGS[self.embodiment_tag]}
        return {
            self.embodiment_tag: {
                "state": ModalityConfig(delta_indices=[0], modality_keys=["state"]),
                "action": ModalityConfig(
                    delta_indices=list(range(self.max_action_horizon)),
                    modality_keys=["action"],
                ),
                "video": ModalityConfig(delta_indices=[0], modality_keys=["image"]),
            }
        }

    def _build_processor_stats(
        self,
        dataset_stats: Optional[Dict[str, Any]],
        dataset: Any,
    ) -> Optional[dict[str, Any]]:
        dataset_stats_local = dict(dataset_stats or {})
        if self.embodiment_tag in EMBODIMENT_STAT_CONFIGS:
            action_cfg = EMBODIMENT_STAT_CONFIGS[self.embodiment_tag]["modality_config"]["action"]
            needs_relative_stats = any(
                cfg.rep.value == "relative"
                for cfg in (action_cfg.action_configs or [])
            )
            if needs_relative_stats and "relative_action" not in dataset_stats_local:
                if dataset is None:
                    raise ValueError(
                        "Missing 'relative_action' stats for GR00T preprocessing and dataset is None"
                    )
                dataset_stats_local["relative_action"] = compute_relative_action_stats(
                    dataset,
                    self.embodiment_tag,
                )
        if not dataset_stats_local:
            return None
        return convert_lerobot_stats_to_processor_format(dataset_stats_local, self.embodiment_tag)

    def _build_embodiment_enum(self) -> EmbodimentTag:
        try:
            return EmbodimentTag(self.embodiment_tag)
        except ValueError:
            return EmbodimentTag.NEW_EMBODIMENT

    def _build_image_transforms(self):
        if self.use_albumentations and not ALBUMENTATIONS_AVAILABLE:
            self.use_albumentations = False
        if self.use_albumentations:
            return build_image_transformations_albumentations(
                self.image_target_size,
                self.image_crop_size,
                self.random_rotation_angle,
                self.color_jitter_params,
                self.shortest_image_edge,
                self.crop_fraction,
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
        state = torch.cat([torch.from_numpy(normalized_states[key]) for key in state_keys], dim=-1)
        state_dim = state.shape[-1]
        if state_dim < self.max_state_dim:
            state = torch.cat(
                [state, torch.zeros(state.shape[0], self.max_state_dim - state_dim)],
                dim=-1,
            )
        elif state_dim > self.max_state_dim:
            state = state[..., :self.max_state_dim]
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
        valid_dim = min(action.shape[-1], self.max_action_dim)
        valid_horizon = min(action.shape[0], self.max_action_horizon)

        padded_action = torch.zeros(self.max_action_horizon, self.max_action_dim)
        padded_action[:valid_horizon, :valid_dim] = action[:valid_horizon, :valid_dim]

        action_mask = torch.zeros_like(padded_action)
        action_mask[:valid_horizon, :valid_dim] = 1
        return (
            padded_action.to(torch.get_default_dtype()).numpy(),
            action_mask.to(torch.get_default_dtype()).numpy(),
        )

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
    ) -> dict[str, Any]:
        image_keys = list(images.keys()) or self.modality_configs[self.embodiment_tag]["video"].modality_keys
        image_transform = None
        if not self.use_processor_image_size:
            image_transform = self.train_image_transform if self.training else self.eval_image_transform

        temporal_stacked_images = {}
        if self.use_albumentations and image_transform is not None:
            replay = None
            for view in image_keys:
                if view not in images:
                    raise KeyError(f"Missing GR00T image view '{view}'")
                transformed_images, replay = apply_with_replay(image_transform, images[view], replay)
                temporal_stacked_images[view] = torch.stack(transformed_images)
        elif image_transform is not None:
            for view in image_keys:
                if view not in images:
                    raise KeyError(f"Missing GR00T image view '{view}'")
                temporal_stacked_images[view] = torch.stack([image_transform(img) for img in images[view]])
        else:
            for view in image_keys:
                if view not in images:
                    raise KeyError(f"Missing GR00T image view '{view}'")
                temporal_stacked_images[view] = torch.stack(
                    [torch.from_numpy(frame).permute(2, 0, 1) for frame in images[view]]
                )

        for view, tensor in temporal_stacked_images.items():
            if tensor.ndim != 4 or tensor.shape[1] != 3:
                raise ValueError(f"GR00T image view '{view}' must be [T, 3, H, W], got {tuple(tensor.shape)}")
            if tensor.dtype != torch.uint8:
                tensor = tensor.clamp(0, 255).to(torch.uint8)
                temporal_stacked_images[view] = tensor

        stacked = torch.stack([temporal_stacked_images[view] for view in image_keys], dim=1)
        stacked_images = stacked.flatten(0, 1).numpy()
        pil_images = [Image.fromarray(np.transpose(frame, (1, 2, 0))) for frame in stacked_images]
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": language},
                    *[{"type": "image", "image": img} for img in pil_images],
                ],
            }
        ]
        return {"text": None, "images": pil_images, "conversation": conversation}


class GrootBatchTransform(BaseTransform):
    """No-op marker transform for GR00T-specific pipeline registration."""

    def __init__(self, apply_to: List[str] | None = None, training: bool = True):
        super().__init__(apply_to=apply_to or [], training=training)

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Pass data through unchanged."""
        return data


def _as_size_list(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, int):
        return [value, value]
    return list(value)


def _to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def _to_uint8_hwc(frame: np.ndarray, normalize_mode: str = "identity") -> np.ndarray:
    arr = frame
    if arr.ndim != 3:
        raise ValueError(f"Expected image frame ndim=3, got {arr.ndim}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected image channel size 3, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        if normalize_mode != "identity":
            raise ValueError(
                "GR00T expects image preprocessing to keep identity normalization. "
                f"Got {normalize_mode!r}; Eagle processor owns VLM image rescale/normalize."
            )
        finite_vals = arr[np.isfinite(arr)]
        if finite_vals.size and finite_vals.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _extract_frames(image_array: np.ndarray, normalize_mode: str = "identity") -> list[np.ndarray]:
    arr = image_array
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Unsupported image shape {arr.shape}")
    return [_to_uint8_hwc(frame, normalize_mode=normalize_mode) for frame in arr]


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


@register_transform_builder("Gr00tN1d6")
def build_groot_n1_6_transforms(ctx: TransformBuilderContext):
    """Build GR00T-N1.6-specific per-sample transforms."""
    preprocess_mode = ctx.data_cfg.groot_preprocess_mode
    if preprocess_mode != "sample":
        raise ValueError(
            f"Unsupported GR00T-N1.6 preprocess mode for per-sample transforms: {preprocess_mode!r}"
        )

    return [
        GrootPromptTransform(),
        GrootN1d6FeatureTransform(
            model_cfg=ctx.model_cfg,
            data_cfg=ctx.data_cfg,
            dataset_stats=ctx.dataset_stats,
            dataset=ctx.dataset,
        ),
    ]
