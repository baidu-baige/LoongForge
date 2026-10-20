import os
from typing import List, Sequence, Tuple, Union

import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.util.inference import load_model, predict
from PIL import Image
from torchvision.ops import box_convert

from ....utils import download_from_huggingface
from ...pipeline import BasePipeline


class GroundingDINOPipeline(BasePipeline):
    """Text-conditioned detection with GroundingDINO."""

    def __init__(self, config_path: str, model_path: str):
        self.download([config_path, model_path])
        self.device = 'cpu'
        self.model = load_model(config_path, model_path, device=self.device)
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def download(self, file_paths: Sequence[str]) -> None:
        """Ensure required files exist locally, fetching from HF Hub if needed."""
        for file_path in file_paths:
            if os.path.exists(file_path):
                continue
            file_name = os.path.basename(file_path)
            if file_name == 'GroundingDINO_SwinT_OGC.py':
                file_name = 'GroundingDINO_SwinT_OGC.cfg.py'
            local_path = download_from_huggingface(
                repo_id='ShilongLiu/GroundingDINO',
                filenames=[file_name],
                local_dir=os.path.dirname(file_path),
            )
            if local_path != file_path:
                os.rename(local_path, file_path)

    def to(self, device: Union[str, torch.device]):
        self.device = device
        self.model.to(self.device)
        return self

    def get_grounding_boxes(
        self,
        image: torch.Tensor,
        image_shape: Tuple[int, int],
        det_labels: Union[str, Sequence[str]],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Run GroundingDINO prediction on a transformed image tensor.

        Args:
            image: Transformed image tensor as expected by GroundingDINO.
            image_shape: (H, W) of the original image for box rescaling.
            det_labels: Category names to condition detection (str or list of str).
            box_threshold: Box score threshold for filtering.
            text_threshold: Text score threshold for filtering.

        Returns:
            Tuple of (boxes_xyxy, phrases, scores). Boxes and scores are numpy arrays.
        """
        if not isinstance(det_labels, list):
            det_labels = [det_labels]
        caption = ''
        for det_label in det_labels:
            det_label = det_label.lower().strip()
            caption += '{}.'.format(det_label)
        boxes, scores, phrases = predict(
            model=self.model,
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        if len(boxes) > 0:
            h, w = image_shape[0], image_shape[1]
            boxes = boxes * torch.Tensor([w, h, w, h])
            boxes = box_convert(boxes=boxes, in_fmt='cxcywh', out_fmt='xyxy').numpy()
        scores = scores.numpy()
        return boxes, phrases, scores

    def __call__(
        self,
        image: Image.Image,
        det_labels: Union[str, Sequence[str]],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Run the end-to-end detection pipeline on a PIL image.

        Args:
            image: Input PIL Image.
            det_labels: Category names to detect (str or list[str]).
            box_threshold: Box score threshold for filtering.
            text_threshold: Text score threshold for filtering.

        Returns:
            Tuple of (boxes_xyxy, labels, scores) as numpy arrays/list.
        """
        image_shape = (image.height, image.width)
        image = self.transform(image, None)[0]
        pred_boxes, pred_labels, pred_scores = self.get_grounding_boxes(
            image, image_shape, det_labels, box_threshold=box_threshold, text_threshold=text_threshold
        )
        return pred_boxes, pred_labels, pred_scores
