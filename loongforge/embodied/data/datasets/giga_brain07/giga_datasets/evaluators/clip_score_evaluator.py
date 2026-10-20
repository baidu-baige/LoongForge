import logging
from typing import Any, Iterable

import torch
from transformers import CLIPModel, CLIPProcessor


class CLIPScoreEvaluator:
    """Compute CLIP Score given pairs of prompt and image."""

    def __init__(
        self,
        model_path: str,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        local_files_only: bool = True,
    ) -> None:
        """Initialize the evaluator and optionally configure device/dtype.

        Args:
            model_path: HuggingFace model identifier or local path.
            device: Device to run the model on. If ``None``, uses model defaults.
            dtype: Cast model parameters to this dtype if provided.
            local_files_only: If True, do not attempt to download models from the internet.
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.processor = None
        self.model = None

    def load_model(self) -> None:
        """Lazy-load CLIP processor and model, move to desired device/dtype."""
        if self.model is None:
            processor = CLIPProcessor.from_pretrained(
                self.model_path,
                local_files_only=self.local_files_only,
            )
            model = CLIPModel.from_pretrained(
                self.model_path,
                local_files_only=self.local_files_only,
            )
            model.eval()
            if self.device is not None:
                model.to(self.device)
            if self.dtype is not None:
                model.to(self.dtype)
            self.processor = processor
            self.model = model

    def __call__(self, results: Iterable[dict[str, Any]]) -> dict[str, float]:
        """Evaluate CLIP score over a list of results with keys ``image`` and
        ``prompt``.

        Args:
            results: Iterable of dicts each containing 'prompt' (str) and 'image' (PIL.Image or numpy array).

        Returns:
            dict[str, float]: {'clip_score': mean_score}
        """
        self.load_model()
        clip_score_list = []
        for i, result in enumerate(results):
            image = result['image']
            prompt = result['prompt']
            clip_score = calculate_clip_score(prompt, image, self.processor, self.model)
            clip_score_list.append(clip_score)
        clip_score = sum(clip_score_list) / len(clip_score_list)
        logging.info('CLIP Score: {}'.format(clip_score))
        return dict(clip_score=clip_score)


def calculate_clip_score(prompt: str, image: Any, processor: CLIPProcessor, model: CLIPModel) -> float:
    """Compute CLIP score for a single (prompt, image) pair.

    Returns:
        Scalar CLIP score (logits_per_image).
    """
    # Tokenize text and preprocess image as batched tensors
    inputs = processor(
        text=prompt,
        images=image,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    input_ids = inputs.input_ids
    pixel_values = inputs.pixel_values
    # Move to model device and dtype to avoid implicit casts
    input_ids = input_ids.to(model.device)
    pixel_values = pixel_values.to(model.device, dtype=model.dtype)
    with torch.no_grad():
        # Forward pass returns similarities; we read image->text logits
        outputs = model(input_ids, pixel_values)
        clip_score = outputs.logits_per_image.reshape(-1).item()
    return clip_score
