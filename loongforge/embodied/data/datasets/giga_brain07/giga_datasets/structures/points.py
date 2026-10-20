from typing import Any

import torch

from .base_structure import BaseStructure
from .utils import points_utils


class Points(BaseStructure):
    """2D points with geometry helpers (rotate/flip/clip/crop/scale)."""

    def __init__(self, tensor: torch.Tensor | list | tuple):
        """Create a 2D points wrapper.

        Args:
            tensor: (N, >=2) points array-like.
        """
        super(Points, self).__init__(tensor)
        assert self.tensor.ndim == 2 and self.tensor.shape[1] >= 2
        self.tensor = self.tensor.to(torch.float32)

    def points(self) -> torch.Tensor:
        """Return a cloned tensor of points of shape (N, D)."""
        return self.tensor.clone()

    def rotate(self, rot_mat_t: torch.Tensor | list | tuple) -> None:
        """Rotate 2D points by rotation matrix.

        Args:
            rot_mat_t: (2,2) or (3,2) homogeneous rotation.
        """
        if not isinstance(rot_mat_t, torch.Tensor):
            rot_mat_t = self.tensor.new_tensor(rot_mat_t)
        self.tensor = points_utils.rotate_points(self.tensor, rot_mat_t)

    def flip(self, image_shape: tuple[int, int], direction: str = 'horizontal') -> None:
        """Flip points inside an image.

        Args:
            image_shape: (H, W) image size.
            direction: 'horizontal' or 'vertical'.
        """
        self.tensor = points_utils.flip_points(self.tensor, image_shape, direction)

    def clip(self, image_shape: tuple[int, int]) -> None:
        """Clamp points onto image boundaries.

        Args:
            image_shape: (H, W) image size.
        """
        self.tensor = points_utils.clip_points(self.tensor, image_shape)

    def crop(self, crop_range: torch.Tensor | list | tuple, mode: str = 'filter') -> Any:
        """Crop points by a rectangle with filter/clip policies.

        Args:
            crop_range: (x1, y1, x2, y2) crop rectangle.
            mode: 'filter' to drop outside points, 'clip' to clamp.

        Returns:
            Kept indices after crop.
        """
        self.tensor, keep = points_utils.crop_points(self.tensor, crop_range, mode)
        return keep

    def scale(self, scale_hw: tuple[float, float], image_shape: tuple[int, int]) -> None:
        """Scale points by (scale_h, scale_w) and optionally clamp.

        Args:
            scale_hw: (scale_h, scale_w) factors.
            image_shape: (H, W) image size for optional clamping.
        """
        self.tensor = points_utils.scale_points(self.tensor, scale_hw, image_shape)
