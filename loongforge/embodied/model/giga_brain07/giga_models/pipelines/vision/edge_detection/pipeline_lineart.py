import os
from typing import Any, Union

from controlnet_aux import LineartDetector
from PIL import Image

from .... import utils
from ...pipeline import BasePipeline


class LineartPipeline(BasePipeline):
    """Line art extraction pipeline."""

    def __init__(self, model_path: Union[str, os.PathLike]) -> None:
        """Initialize the pipeline with a pretrained Lineart model.

        Args:
            model_path: Path or identifier for
                ``LineartDetector.from_pretrained``.
        """
        self.model = utils.wrap_call(LineartDetector.from_pretrained)(model_path)
        self.device = 'cpu'

    def to(self, device: Any):
        self.device = device
        self.model.to(device)
        return self

    def __call__(self, image: Image.Image, **kwargs: Any) -> Image.Image:
        """Run line art detection and return an image matching input size.

        Args:
            image: Input PIL image.
            **kwargs: Additional arguments forwarded to the underlying detector.

        Returns:
            Line art map as a PIL image resized to the input ``image.size``.
        """
        return self.model(image, **kwargs).resize(image.size)
