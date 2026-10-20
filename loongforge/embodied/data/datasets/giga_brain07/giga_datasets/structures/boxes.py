from typing import Any

import torch

from .base_structure import BaseStructure
from .utils import boxes_utils


class Boxes(BaseStructure):
    """Axis-aligned 2D bounding boxes with helper geometry ops."""

    def __init__(self, tensor: torch.Tensor | list | tuple, offset: int = 0):
        """Initialize bounding boxes.

        Args:
            tensor: Tensor of shape (..., 4) storing boxes in (x1, y1, x2, y2) or (x, y, w, h) depending on ``offset``.
            offset: 0 for (x1, y1, x2, y2) and 1 for 1-based indexing compatibility in certain datasets.
        """
        super(Boxes, self).__init__(tensor, offset=offset)
        assert self.tensor.shape[-1] == 4
        self.tensor = self.tensor.to(torch.float32)
        self.offset = offset

    @property
    def width(self):
        return boxes_utils.get_width(self.tensor, self.offset)

    @property
    def height(self):
        return boxes_utils.get_height(self.tensor, self.offset)

    @property
    def size(self):
        return boxes_utils.get_size(self.tensor, self.offset)

    @property
    def centers(self):
        return boxes_utils.get_centers(self.tensor)

    @property
    def dims(self):
        return boxes_utils.get_dims(self.tensor, self.offset)

    def corners(self) -> torch.Tensor:
        """Return 4-corner representation with shape (N, 4, 2)."""
        return boxes_utils.boxes_to_corners(self.tensor, self.offset)

    def boxes(self) -> torch.Tensor:
        """Return a cloned tensor of boxes with shape (..., 4)."""
        return self.tensor.clone()

    def rotate(self, rot_mat_t: torch.Tensor | list | tuple) -> None:
        """Rotate boxes by applying the rotation to their corners.

        Args:
            rot_mat_t: (2,2) or homogeneous (3,2) rotation matrix (or compatible list/tuple).
        """
        if not isinstance(rot_mat_t, torch.Tensor):
            rot_mat_t = self.tensor.new_tensor(rot_mat_t)
        self.tensor = boxes_utils.rotate_boxes(self.tensor, rot_mat_t, self.offset)

    def flip(self, image_shape: tuple[int, int], direction: str = 'horizontal') -> None:
        """Flip boxes horizontally or vertically inside an image.

        Args:
            image_shape: (H, W) image size for boundary reference.
            direction: 'horizontal' or 'vertical'.
        """
        self.tensor = boxes_utils.flip_boxes(self.tensor, image_shape, direction, self.offset)

    def clip(self, image_shape: tuple[int, int]) -> None:
        """Clamp boxes to image boundaries.

        Args:
            image_shape: (H, W) image size.
        """
        self.tensor = boxes_utils.clip_boxes(self.tensor, image_shape, self.offset)

    def crop(self, crop_range: torch.Tensor | list | tuple, keep_outside_center: bool = False, keep_outside_boxes: bool = False) -> Any:
        """Crop boxes with a rectangle and optionally filter by center-inside
        policy.

        Args:
            crop_range: (x1, y1, x2, y2) crop rectangle.
            keep_outside_center: If True, do not filter by center-inside.
            keep_outside_boxes: If True, do not clamp boxes by crop edges before filtering.

        Returns:
            Indices kept (torch.Tensor or np.ndarray) depending on backend.
        """
        self.tensor, keep = boxes_utils.crop_boxes(crop_range, keep_outside_center, keep_outside_boxes)
        return keep

    def scale(self, scale_hw: tuple[float, float], image_shape: tuple[int, int]) -> None:
        """Scale boxes by (scale_h, scale_w) and clamp to image.

        Args:
            scale_hw: (scale_h, scale_w) factors.
            image_shape: (H, W) image size.
        """
        self.tensor = boxes_utils.scale_boxes(self.tensor, scale_hw, image_shape, self.offset)

    def overlaps(self, boxes: torch.Tensor | list | tuple, mode: str = 'iou', is_aligned: bool = False, offset: int = 0, eps: float = 1e-6) -> Any:
        """Compute pairwise IoU/IoA/GIoU between current boxes and input boxes.

        Args:
            boxes: Another set of boxes (..., 4).
            mode: 'iou' | 'ioa' | 'iob' | 'giou'.
            is_aligned: If True, compare element-wise (N,) else produce (N, M).
            offset: Inclusive width/height offset (e.g., 1 for pixel-inclusive).
            eps: Numerical stability epsilon.

        Returns:
            IoU-like metric tensor/array.
        """
        if not isinstance(boxes, torch.Tensor):
            boxes = self.tensor.new_tensor(boxes)
        return boxes_utils.overlaps(self.tensor, boxes, mode=mode, is_aligned=is_aligned, offset=offset, eps=eps)
