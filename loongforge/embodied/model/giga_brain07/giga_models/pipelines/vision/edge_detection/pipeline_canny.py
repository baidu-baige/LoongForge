import cv2
import numpy as np
from PIL import Image

from ...pipeline import BasePipeline


class CannyPipeline(BasePipeline):
    """Thin wrapper around OpenCV Canny for consistency with other pipelines."""

    def __init__(self) -> None:
        pass

    def __call__(self, image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
        """Run Canny edge detection and return a 3-channel PIL image.

        Args:
            image: Input PIL Image.
            low_threshold: Lower hysteresis threshold.
            high_threshold: Upper hysteresis threshold.

        Returns:
            A PIL Image with edges replicated across RGB channels.
        """
        image = np.array(image)
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image
