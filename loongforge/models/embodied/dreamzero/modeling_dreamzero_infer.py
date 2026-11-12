# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Inference-only facade for the shared DreamZero action head.

The DROID camera, prompt, state, and action semantics stay in the evaluation
factory. This module owns only the model-facing inference boundary so eval
does not need to call the training model directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


class DreamZeroInferenceModel:
    """Run the shared DreamZero model through an inference-only interface."""

    def __init__(
        self,
        model: Any,
        tokenizer_path: str,
        text_len: int,
        embodiment_id: int,
        device: Any,
    ):
        """Initialize the facade.

        Args:
            model: Constructed DreamZero model exposing ``predict_action_chunk``.
            tokenizer_path: Local UMT5 tokenizer path.
            text_len: Maximum token sequence length.
            embodiment_id: DreamZero embodiment identifier.
            device: Device receiving model inputs.
        """
        self._model = model
        self._tokenizer_path = tokenizer_path
        self._text_len = int(text_len)
        self._embodiment_id = int(embodiment_id)
        self._device = device
        self._tokenizer = None

    def _load_tokenizer(self):
        """Load the local tokenizer once for the inference process."""
        if self._tokenizer is None:
            if not self._tokenizer_path:
                raise ValueError(
                    "DreamZero eval requires a tokenizer path for inference"
                )
            from transformers import AutoTokenizer

            load_kwargs = (
                {"local_files_only": True}
                if Path(self._tokenizer_path).is_dir()
                else {}
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_path, **load_kwargs
            )
        return self._tokenizer

    @staticmethod
    def _eval_image_transform(video: np.ndarray) -> np.ndarray:
        """Apply the official eval image transform to a video grid.

        Mirrors RLinf ``libero_sim.py`` (VideoCrop(scale=0.95) → VideoResize,
        linear) applied per view before the horizontal grid concat: a center
        crop of the concatenated exterior|wrist grid removes the same 5%
        border from each view, so cropping the grid is equivalent. Training
        VideoColorJitter augmentation is intentionally omitted at eval.
        Callers pass the env-native grid; this is the model's input
        convention (same placement as GR00T-N1.6's
        ``_predict_action_eval_image_transform``).
        """
        import torch
        from torchvision.transforms.v2 import functional as tvf

        t, h, w, c = video.shape
        half = w // 2
        out = np.empty_like(video)
        for start in (0, half):  # crop each view independently (grid = ext|wrist)
            view = torch.from_numpy(video[:, :, start:start + half]).permute(0, 3, 1, 2).float()
            view = tvf.center_crop(view, [int(h * 0.95), int(half * 0.95)])
            view = tvf.resize(
                view,
                [h, half],
                interpolation=tvf.InterpolationMode.BILINEAR,
                antialias=True,
            )
            out[:, :, start:start + half] = view.permute(0, 2, 3, 1).to(torch.uint8).numpy()
        return out

    def predict(
        self,
        video: np.ndarray,
        prompt: str,
        negative_prompt: str,
        state: np.ndarray,
    ) -> np.ndarray:
        """Tokenize a prepared prompt and run the action head.

        Args:
            video: Model video grid in uint8 THWC format — env-native pixels;
                the official eval image transform (95% center crop + resize)
                is applied here as part of the model's input convention.
            prompt: Positive prompt already formatted by the eval adapter.
            negative_prompt: Model classifier-free negative prompt.
            state: Prepared normalized state vector.

        Returns:
            Action predictions as a CPU NumPy array with batch dimension.
        """
        import torch

        video = self._eval_image_transform(np.asarray(video))
        tokenizer = self._load_tokenizer()
        tokens = tokenizer(
            [prompt], return_tensors="pt", padding="max_length", truncation=True,
            max_length=self._text_len,
        )
        negative_tokens = tokenizer(
            [negative_prompt], return_tensors="pt", padding="max_length",
            truncation=True, max_length=self._text_len,
        )
        batch = {
            "images": torch.from_numpy(video)[None].to(self._device),
            "text": tokens.input_ids.to(self._device),
            "text_attention_mask": tokens.attention_mask.to(self._device),
            "text_negative": negative_tokens.input_ids.to(self._device),
            "text_attention_mask_negative": negative_tokens.attention_mask.to(self._device),
            "state": torch.from_numpy(state)[None, None].to(self._device),
            "embodiment_id": torch.tensor(
                [self._embodiment_id], dtype=torch.long, device=self._device
            ),
        }
        return self.predict_action_chunk(batch)

    def predict_action_chunk(self, batch: Mapping[str, Any]) -> np.ndarray:
        """Run one action-head inference call and return a CPU float array.

        Args:
            batch: Model-ready DreamZero tensors.

        Returns:
            Action predictions as a NumPy array; batch dimension is preserved.
        """
        import torch

        with torch.no_grad():
            result = self._model.predict_action_chunk(batch)
        if hasattr(result, "detach"):
            return result.detach().float().cpu().numpy()
        return np.asarray(result, dtype=np.float32)

    def reset(self) -> None:
        """Reset temporal state when the wrapped model exposes it."""
        reset = getattr(self._model, "reset", None)
        if callable(reset):
            reset()
