# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from NVIDIA GR00T under the Apache-2.0 License.

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

import json
import logging
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from transformers import AutoTokenizer, ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature

from loongforge.embodied.data.datasets.groot_n1_6.transforms.eagle3_model.image_processing_eagle3_vl_fast import (
    Eagle3VLImageProcessorFast,
)
from loongforge.embodied.data.datasets.groot_n1_6.transforms.eagle3_model.processing_eagle3_vl import (
    Eagle3VLProcessor,
)
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
    nested_dict_to_numpy,
    normalize_values_meanstd,
    normalize_values_minmax,
    parse_modality_configs,
    unnormalize_values_meanstd,
    unnormalize_values_minmax,
)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")

logger = logging.getLogger(__name__)


_PROCESSOR_CACHE: dict[tuple[str, tuple[tuple[str, object], ...]], ProcessorMixin] = {}
_PROCESSOR_LOG_ONCE: set[tuple[str, tuple[tuple[str, object], ...]]] = set()


def build_processor(
    model_name: str,
    vlm_tokenizer_path: str | None = None,
    transformers_loading_kwargs: dict | None = None,
) -> ProcessorMixin:
    """Build the Eagle processor from repo code plus tokenizer/assets resources.

    ``vlm_tokenizer_path`` is treated strictly as a resource location
    containing tokenizer/config JSON files. Python classes are imported from
    this package, so loading does not depend on remote-code files copied into
    the assets directory.
    """
    transformers_loading_kwargs = dict(transformers_loading_kwargs or {})
    asset_id = vlm_tokenizer_path or model_name or "aravindhs-NV/eagle3-processor-groot-n1d6"
    asset_kwargs = _asset_loading_kwargs(transformers_loading_kwargs)
    offline_mode = _offline_mode(asset_kwargs)
    if offline_mode:
        asset_kwargs["local_files_only"] = True

    cache_key = (
        str(asset_id),
        tuple(sorted((key, repr(value)) for key, value in asset_kwargs.items())),
    )
    if cache_key in _PROCESSOR_CACHE:
        return _PROCESSOR_CACHE[cache_key]

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

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            asset_id,
            use_fast=True,
            trust_remote_code=False,
            **asset_kwargs,
        )
        image_processor = Eagle3VLImageProcessorFast.from_pretrained(
            asset_id,
            **asset_kwargs,
        )
    except Exception as exc:
        raise FileNotFoundError(
            "Failed to load Eagle tokenizer/assets. Provide a local vlm_tokenizer_path "
            "or enable normal HuggingFace asset loading; processor Python code is loaded "
            f"from the repository package. asset_id={asset_id!r}"
        ) from exc

    processor_cfg = _load_local_json(asset_id, "processor_config.json")
    chat_template = _load_chat_template(asset_id)
    processor = Eagle3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=chat_template or getattr(tokenizer, "chat_template", None),
        **processor_cfg,
    )
    if cache_key not in _PROCESSOR_CACHE and verbose_logs:
        logger.info("Using Eagle processor assets from: %s", asset_id)
    _PROCESSOR_CACHE[cache_key] = processor
    return processor


def _asset_loading_kwargs(transformers_loading_kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "cache_dir",
        "force_download",
        "local_files_only",
        "proxies",
        "revision",
        "subfolder",
        "token",
        "use_auth_token",
    }
    return {
        key: value
        for key, value in transformers_loading_kwargs.items()
        if key in allowed_keys and value is not None
    }


def _offline_mode(asset_kwargs: dict[str, Any]) -> bool:
    return (
        str(os.environ.get("HF_HUB_OFFLINE", "")).lower() in {"1", "true", "yes"}
        or str(os.environ.get("TRANSFORMERS_OFFLINE", "")).lower() in {"1", "true", "yes"}
        or bool(asset_kwargs.get("local_files_only", False))
    )


def _load_local_json(asset_id: str | Path, filename: str) -> dict[str, Any]:
    path = Path(asset_id)
    if not path.exists():
        return {}
    config_path = path / filename
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _load_chat_template(asset_id: str | Path) -> str | None:
    config = _load_local_json(asset_id, "chat_template.json")
    value = config.get("chat_template")
    return value if isinstance(value, str) else None


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
                logger.info("Embodiment tag %s already in statistics, skipping updating", key)
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
        vlm_tokenizer_path: str = "lerobot/eagle3-processor-groot-n1d6",
        model_type: Literal["eagle"] = "eagle",
        transformers_loading_kwargs: dict | None = None,
        max_length: int | None = None,
    ):
        """Initialize data collator.

        Args:
            max_length: If set, pad all sequences to this fixed length using
                ``padding="max_length"`` and ``truncation=True``. This ensures
                all batches have identical tensor shapes, which is required for
                full-iteration CUDA graph (``--cuda-graph-scope=full_iteration``).
                When ``None`` (default), uses dynamic padding (pad to longest
                sequence in the batch).
        """
        if transformers_loading_kwargs is None:
            transformers_loading_kwargs = {}
        self.processor = build_processor(model_name, vlm_tokenizer_path, transformers_loading_kwargs)
        self.processor.tokenizer.padding_side = "left"
        self.model_type = model_type
        self.model_name = model_name
        self.max_length = max_length
        self._truncation_warned = False

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
                    text = v.get("text")
                    if text is None:
                        text = self.processor.apply_chat_template(
                            v["conversation"],
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                    text_list += [text]
                    image_inputs += v["images"]

                if self.model_type == "eagle":
                    image_inputs, _ = self.processor.process_vision_info([v["conversation"] for v in values])
                vlm_inputs = self.processor(
                    text=text_list, images=image_inputs, return_tensors="pt",
                    padding="max_length" if self.max_length else True,
                    max_length=self.max_length,
                    truncation=self.max_length is not None,
                )
                # Detect truncation: if any sample has attention_mask all-1 with
                # max_length set, it was truncated (no pad tokens added).
                if (self.max_length is not None and not self._truncation_warned
                        and "attention_mask" in vlm_inputs):
                    attn_mask = vlm_inputs["attention_mask"]
                    num_truncated = int((attn_mask.sum(dim=-1) == attn_mask.shape[-1]).sum())
                    if num_truncated > 0:
                        self._truncation_warned = True
                        warnings.warn(
                            f"{num_truncated}/{attn_mask.shape[0]} samples in this batch "
                            f"have no padding tokens (sequence length == max_length="
                            f"{self.max_length}), which likely means they were truncated. "
                            "Consider increasing --cuda-graph-pad-length if this is unexpected.",
                            UserWarning,
                            stacklevel=2,
                        )
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


__all__ = [
    "Gr00tN1d6DataCollator",
]
