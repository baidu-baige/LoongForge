import os
from typing import Any, Union

import torch
from controlnet_aux import PidiNetDetector

from .... import utils
from ...pipeline import BasePipeline


class PidiNetPipeline(BasePipeline):
    """Pipeline for running PidiNet edge detection.

    Attributes:
        model: Underlying ``PidiNetDetector`` instance.
        device: Current device identifier (e.g. "cpu", "cuda").
    """

    def __init__(self, model_path: Union[str, os.PathLike]) -> None:
        """Initialize the pipeline with a pretrained model.

        Args:
            model_path: Path or identifier passed to
                ``PidiNetDetector.from_pretrained``.
        """
        self.model = utils.wrap_call(PidiNetDetector.from_pretrained)(model_path)
        self.device = 'cpu'

    def to(self, device: Union[str, torch.device]):
        self.device = device
        self.model.to(device)
        return self

    def __call__(self, image: Any, **kwargs: Any) -> Any:
        """Run edge detection on the input image.

        Args:
            image: Input PIL image.
            **kwargs: Additional arguments forwarded to ``PidiNetDetector.__call__``
                (e.g., thresholds or other detector-specific options).

        Returns:
            A PIL image containing the edge map, resized to match the input size.
        """
        return self.model(image, **kwargs).resize(image.size)
