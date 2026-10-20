import logging
from typing import Any, Iterable

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModelWithProjection


class AestheticScoreEvaluator:
    """Compute LAION-style Aesthetic Score using a CLIP image tower and MLP
    head."""

    def __init__(
        self,
        clip_model_path: str,
        aesthetic_model_path: str,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        local_files_only: bool = True,
    ) -> None:
        """Initialize evaluator with paths and runtime options.

        Args:
            clip_model_path: HF model id or local path for CLIP vision tower.
            aesthetic_model_path: Path to a linear head checkpoint (state_dict).
            device: Device to place models on.
            dtype: Optional dtype to cast model parameters to.
            local_files_only: If True, do not attempt to download from internet.
        """
        self.clip_model_path = clip_model_path
        self.aesthetic_model_path = aesthetic_model_path
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.processor = None
        self.clip_model = None
        self.aesthetic_model = None

    def load_model(self) -> None:
        """Lazy-load processor, CLIP vision tower and linear head."""
        if self.clip_model is None:
            processor = CLIPProcessor.from_pretrained(
                self.clip_model_path,
                local_files_only=self.local_files_only,
            )
            clip_model = CLIPVisionModelWithProjection.from_pretrained(
                self.clip_model_path,
                local_files_only=self.local_files_only,
            )
            state_dict = torch.load(self.aesthetic_model_path, map_location='cpu')
            in_channels = state_dict['weight'].shape[1]
            aesthetic_model = nn.Linear(in_channels, 1)
            aesthetic_model.load_state_dict(state_dict)
            clip_model.eval()
            aesthetic_model.eval()
            if self.device is not None:
                clip_model.to(self.device)
                aesthetic_model.to(self.device)
            if self.dtype is not None:
                clip_model.to(self.dtype)
                aesthetic_model.to(self.dtype)
            self.processor = processor
            self.clip_model = clip_model
            self.aesthetic_model = aesthetic_model

    def __call__(self, results: Iterable[dict[str, Any]]):
        """Evaluate mean aesthetic score over a list of results with key
        ``image``."""
        self.load_model()
        aesthetic_score_list = []
        for i, result in enumerate(results):
            image = result['image']
            aesthetic_score = calculate_aesthetic_score(
                image,
                self.processor,
                self.clip_model,
                self.aesthetic_model,
            )
            aesthetic_score_list.append(aesthetic_score)
        aesthetic_score = sum(aesthetic_score_list) / len(aesthetic_score_list)
        logging.info('Aesthetic Score: {}'.format(aesthetic_score))
        return dict(aesthetic_score=aesthetic_score)


def calculate_aesthetic_score(
    image: Any,
    processor: CLIPProcessor,
    clip_model: CLIPVisionModelWithProjection,
    aesthetic_model: nn.Linear,
) -> float:
    """Compute aesthetic score for a single image.

    Returns:
        Scalar aesthetic score.
    """
    inputs = processor(images=image, return_tensors='pt')
    pixel_values = inputs.pixel_values.to(clip_model.device, dtype=clip_model.dtype)
    with torch.no_grad():
        image_features = clip_model(pixel_values)[0]
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        aesthetic_score = aesthetic_model(image_features)
        aesthetic_score = aesthetic_score.reshape(-1).item()
    return aesthetic_score
