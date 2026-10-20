from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor

from ...pipeline import BasePipeline


class DPTForDepthEstimationPipeline(BasePipeline):
    """Depth estimation inference helper powered by DPT models.

    Args:
        model_path: Pretrained model identifier or local path.
    """

    def __init__(self, model_path: str) -> None:
        self.image_processor = DPTImageProcessor.from_pretrained(model_path)
        self.model = DPTForDepthEstimation.from_pretrained(model_path)
        self.device = 'cpu'

    def to(self, device: Union[str, torch.device]):
        self.device = device
        self.model.to(device)
        return self

    def __call__(self, image: Image.Image) -> Image.Image:
        """Run depth estimation on a PIL image and return an RGB depth image.

        Args:
            image: Input PIL image.

        Returns:
            A PIL Image in RGB mode where intensity encodes normalized depth.
        """
        inputs = self.image_processor(images=image, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            depth_map = outputs.predicted_depth
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(image.height, image.width),
            mode='bicubic',
            align_corners=False,
        )
        depth_map = depth_map.squeeze().unsqueeze(-1)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8) * 255.0
        depth_image = torch.cat([depth_map] * 3, dim=-1)
        depth_image = depth_image.cpu().numpy().clip(0, 255).astype(np.uint8)
        depth_image = Image.fromarray(depth_image)
        return depth_image
