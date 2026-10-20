import os
from typing import Any, Union

from controlnet_aux import OpenposeDetector
from PIL import Image

from .... import utils
from ...pipeline import BasePipeline


class OpenPosePipeline(BasePipeline):
    """Pipeline for running OpenPose detection."""

    def __init__(self, model_path: Union[str, os.PathLike]) -> None:
        """Initialize the pipeline with a pretrained checkpoint.

        Args:
            model_path: Path or identifier passed to
                ``OpenposeDetector.from_pretrained``.
        """
        self.model = utils.wrap_call(OpenposeDetector.from_pretrained)(model_path)
        self.device = 'cpu'

    def to(self, device: Any):
        self.device = device
        self.model.to(device)
        return self

    def __call__(self, image: Image.Image, **kwargs: Any) -> Image.Image:
        """Run OpenPose and resize output to input size."""
        return self.model(image, **kwargs).resize(image.size)
