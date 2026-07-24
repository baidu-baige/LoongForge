# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Pi05 special per-sample transforms: state discretization, image collation, fallback prompt, tokenization."""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from loongforge.embodied.data.datasets.transforms.base import BaseTransform
from loongforge.embodied.data.datasets.transforms.registry import (
    TransformBuilderContext,
    register_transform_builder,
)
from loongforge.embodied.data.datasets.transforms.utils import (
    build_action_transform,
    build_image_transform,
    convert_stats,
)
from loongforge.embodied.data.datasets.transforms.utils.normalizer import Normalizer
from loongforge.embodied.data.datasets.pi05.transforms.pi05_collator import tokenize_prompts


class StateDiscretizationTransform(BaseTransform):
    """Discretize state and embed into language prompt.

    Converts:
        state (raw) -> normalize -> discretize (N bins, no clip) -> embed in prompt
    """

    def __init__(
        self,
        apply_to: List[str] = None,
        state_key: str = "observation.state",
        task_key: str = "lang",
        num_bins: int = 256,
        max_state_dim: Optional[int] = None,
        prompt_template: str = "Task: {task}, State: {state};\nAction: ",
        normalization_mode: str = "q99",
        statistics: Optional[Dict[str, Any]] = None,
        training: bool = True,
    ):
        super().__init__(apply_to=apply_to or ["lang"], training=training)
        self.state_key = state_key
        self.task_key = task_key
        self.num_bins = num_bins
        self.max_state_dim = max_state_dim
        self.prompt_template = prompt_template

        self.normalizer = None
        if statistics is not None:
            self.normalizer = Normalizer(
                mode=normalization_mode,
                statistics=statistics,
            )

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Discretize state vector and embed into a formatted language prompt.

        If state is missing, produces a simple task-only prompt as fallback.
        Otherwise: normalize -> pad/truncate to fixed dim -> digitize into N bins -> format.
        """
        state = data.get(self.state_key)
        task = data.get(self.task_key, "perform the task")

        # Fallback when no state is available
        if state is None:
            data[self.apply_to[0]] = f"Task: {task.strip()};\nAction: "
            return data

        # Convert to float tensor and flatten
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        state = state.float().flatten()

        # Normalize state using dataset statistics
        if self.normalizer is not None:
            state = self.normalizer.forward(state)

        # Pad or truncate to fixed dimension
        actual_dim = state.shape[0]
        effective_dim = self.max_state_dim if self.max_state_dim else actual_dim

        if actual_dim < effective_dim:
            state = torch.nn.functional.pad(state, (0, effective_dim - actual_dim))
        elif actual_dim > effective_dim:
            state = state[:effective_dim]

        # Discretize normalized state into bin indices
        state_np = state.cpu().numpy()
        bins = np.linspace(-1, 1, self.num_bins + 1)[:-1]
        discretized = np.digitize(state_np, bins=bins) - 1

        # Format the prompt with task description and discretized state
        cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
        state_str = " ".join(map(str, discretized))
        data[self.apply_to[0]] = self.prompt_template.format(
            task=cleaned_text, state=state_str
        )

        return data


class Pi05CollateImagesTransform(BaseTransform):
    """Collate observation images into stacked per-view tensors with masks.

    Reads keys matching `observation.images.*`, produces:
        - "images_list": List of (3, H, W) tensors per view
        - "img_masks": List of bool values per view
    """

    def __init__(
        self,
        image_size: int = 224,
        num_images: int = 2,
        image_mask: Optional[List[bool]] = None,
        training: bool = True,
    ):
        super().__init__(apply_to=["images_list", "img_masks"], training=training)
        self.image_size = image_size
        self.num_images = num_images
        self.image_mask = image_mask or [True] * num_images

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect per-view images from observation keys into a list.

        For each view slot up to num_images, extracts the corresponding image tensor.
        Missing views are filled with a -1.0 placeholder tensor.
        """
        image_keys = sorted(k for k in data.keys() if k.startswith("observation.images."))

        images_list = []
        img_masks = []
        for view_idx in range(self.num_images):
            has_view = view_idx < len(image_keys)
            if has_view:
                # Use available image, convert to tensor if needed
                img = data[image_keys[view_idx]]
                if not isinstance(img, torch.Tensor):
                    img = torch.as_tensor(img, dtype=torch.float32)
            else:
                # Fill missing view with placeholder
                img = torch.zeros(3, self.image_size, self.image_size) - 1.0
            images_list.append(img)
            # Mask is True only when view exists and config enables it
            mask_val = (
                has_view
                and view_idx < len(self.image_mask)
                and self.image_mask[view_idx]
            )
            img_masks.append(mask_val)

        data["images_list"] = images_list
        data["img_masks"] = img_masks
        return data


class Pi05FallbackPromptTransform(BaseTransform):
    """Ensure a 'prompt' key exists in the data dict.

    If 'prompt' is missing, generates a fallback from the 'task' key.
    """

    def __init__(
        self,
        task_key: str = "task",
        prompt_key: str = "prompt",
        training: bool = True,
    ):
        super().__init__(apply_to=[prompt_key], training=training)
        self.task_key = task_key
        self.prompt_key = prompt_key

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a fallback prompt from task description if prompt key is absent."""
        if self.prompt_key not in data or data[self.prompt_key] is None:
            task = data.get(self.task_key, "perform the task")
            data[self.prompt_key] = f"Task: {task.strip()};\nAction: "
        return data


class Pi05TokenizeTransform(BaseTransform):
    """Tokenize the 'prompt' field into 'input_ids' and 'attention_mask'.

    Uses the PaliGemma tokenizer with configurable max length.
    """

    def __init__(
        self,
        tokenizer_path: str = "",
        max_token_len: int = 200,
        prompt_key: str = "prompt",
        training: bool = True,
    ):
        super().__init__(apply_to=["input_ids", "attention_mask"], training=training)
        self.tokenizer_path = tokenizer_path
        self.max_token_len = max_token_len
        self.prompt_key = prompt_key
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer on first access (once per DataLoader worker)."""
        if self._tokenizer is None:
            path = self.tokenizer_path or os.environ.get("TOKENIZER_PATH", "")
            if not path:
                raise ValueError(
                    "Tokenizer path not set. Pass tokenizer_path or set TOKENIZER_PATH env."
                )
            self._tokenizer = AutoTokenizer.from_pretrained(path)
        return self._tokenizer

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize the prompt string into padded input_ids and attention_mask tensors."""
        prompt = data[self.prompt_key]
        tok_out = tokenize_prompts(
            [prompt], self.tokenizer, max_length=self.max_token_len
        )
        data["input_ids"] = tok_out["input_ids"].squeeze(0)
        data["attention_mask"] = tok_out["attention_mask"].squeeze(0).bool()
        return data


@register_transform_builder("pi05")
def build_pi05_transforms(ctx: TransformBuilderContext):
    """Build Pi05-specific per-sample transforms."""
    transforms = []
    model_cfg = ctx.model_cfg
    data_cfg = ctx.data_cfg
    image_keys = _discover_image_keys(ctx.dataset)
    image_size = data_cfg.image_size
    action_horizon = model_cfg.action_horizon
    max_action_dim = model_cfg.max_action_dim
    normalization_mode = data_cfg.normalization_mode

    image_transform = build_image_transform(data_cfg, image_keys, image_size)
    if image_transform is not None:
        transforms.append(image_transform)

    action_transform = build_action_transform(
        data_cfg,
        ctx.dataset_stats,
        action_horizon,
        max_action_dim,
        normalization_mode,
    )
    if action_transform is not None:
        transforms.append(action_transform)

    state_stats = (
        convert_stats(ctx.dataset_stats["observation.state"])
        if ctx.dataset_stats and "observation.state" in ctx.dataset_stats
        else None
    )
    transforms.append(
        StateDiscretizationTransform(
            apply_to=["prompt"],
            state_key="observation.state",
            task_key="task",
            num_bins=256,
            max_state_dim=None,
            normalization_mode=normalization_mode,
            statistics=state_stats,
        )
    )

    num_images = data_cfg.num_images
    image_mask = data_cfg.image_mask or [True] * num_images
    max_token_len = data_cfg.max_token_len
    tokenizer_path = ctx.training_args.tokenizer_path or os.environ.get("TOKENIZER_PATH", "")

    transforms.append(
        Pi05CollateImagesTransform(
            image_size=image_size,
            num_images=num_images,
            image_mask=image_mask,
        )
    )
    transforms.append(Pi05FallbackPromptTransform())
    transforms.append(Pi05TokenizeTransform(tokenizer_path=tokenizer_path, max_token_len=max_token_len))
    return transforms


def _discover_image_keys(dataset) -> list[str]:
    """Discover image keys from the first sample for Pi05 image transforms."""
    try:
        first_sample = dataset[0]
    except Exception:
        return []
    return sorted(k for k in first_sample.keys() if k.startswith("observation.images."))
