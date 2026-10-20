from typing import Any, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from ...pipeline import BasePipeline
from .palette import get_palette


class UperNetForSemanticSegmentationPipeline(BasePipeline):
    """UPerNet semantic segmentation pipeline with optional colorization."""

    def __init__(self, model_path: str) -> None:
        """Load processor, model and color palette.

        Args:
            model_path: HuggingFace model id or local path.
        """
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = UperNetForSemanticSegmentation.from_pretrained(model_path)
        self.palette = get_palette(rgb=True)
        self.device = 'cpu'

    def to(self, device: str):
        self.device = device
        self.model.to(device)
        return self

    def __call__(self, image: Any, return_image: bool = True) -> Union[Image.Image, np.ndarray]:
        inputs = self.processor(image, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        seg = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[(image.height, image.width)])[0]
        seg = seg.cpu().numpy()
        if return_image:
            color_image = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(self.palette):
                color_image[seg == label, :] = color
            color_image = color_image.astype(np.uint8)
            color_image = Image.fromarray(color_image)
            return color_image
        else:
            return seg
