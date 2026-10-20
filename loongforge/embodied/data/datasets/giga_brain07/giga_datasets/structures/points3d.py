from typing import Any

import torch

from .base_structure import BaseStructure
from .mode3d import Mode3D
from .utils import boxes3d_utils, points3d_utils


class Points3D(BaseStructure):
    """3D points with conversion and geometry helpers."""

    def __init__(self, tensor: torch.Tensor | list | tuple, mode: Any = None):
        """Create a 3D points wrapper.

        Args:
            tensor: (N, >=3) array-like representing xyz[+...] coordinates.
            mode: Coordinate mode (e.g., 'lidar'|'camera'|'depth' or Mode3D constant)
                used to infer rotation and BEV axes.
        """
        super(Points3D, self).__init__(tensor, mode=mode)
        assert self.tensor.ndim == 2 and self.tensor.shape[1] >= 3
        self.tensor = self.tensor.to(torch.float32)
        params = Mode3D.get_params(mode)
        self.mode = mode
        self.rot_axis = params['rot_axis']
        self.flip_axis = params['flip_axis']
        self.bev_axis = params['bev_axis']

    def convert_to(self, dst_mode: Any, rot_mat: torch.Tensor | list | tuple | None = None):
        """Convert points to another coordinate mode.

        Args:
            dst_mode: Destination mode ('lidar'|'camera'|'depth' or Mode3D constant).
            rot_mat: Optional (3x3 or 3x4) transform; if None, uses a mode-pair default.

        Returns:
            Converted points (keeps wrapper when possible).
        """
        return points3d_utils.convert_points3d(self, src_mode=self.mode, dst_mode=dst_mode, rot_mat=rot_mat)

    def points(self) -> torch.Tensor:
        """Return a cloned tensor of points of shape (N, D)."""
        return self.tensor.clone()

    def points3d(self) -> torch.Tensor:
        """Alias of ``points`` for clarity in 3D contexts."""
        return self.tensor.clone()

    def points2d(self, cam_intrinsic: torch.Tensor | list | tuple) -> torch.Tensor:
        """Project xyz points to pixel coordinates via camera intrinsics.

        Args:
            cam_intrinsic: (3x4) or (3x3) camera intrinsic matrix.

        Returns:
            (N, 2) pixel coordinates.
        """
        if not isinstance(cam_intrinsic, torch.Tensor):
            cam_intrinsic = self.tensor.new_tensor(cam_intrinsic)
        return points3d_utils.points3d_to_points2d(self.tensor, cam_intrinsic)

    def in_range(self, point_range: torch.Tensor | list | tuple) -> torch.Tensor:
        """Mask indicating whether points are within an axis-aligned 3D cuboid.

        Args:
            point_range: Iterable of length 6 [x_min, y_min, z_min, x_max, y_max, z_max].

        Returns:
            Bool mask of length N.
        """
        return points3d_utils.points3d_in_range(self.tensor, point_range)

    def in_range_bev(self, point_range: torch.Tensor | list | tuple) -> torch.Tensor:
        """Mask indicating whether points are within a 2D BEV rectangle.

        Args:
            point_range: Iterable of length 6 defining 3D range; only BEV axes are used.

        Returns:
            Bool mask of length N.
        """
        return points3d_utils.points3d_in_range_bev(self.tensor, point_range, self.bev_axis)

    def get_rot_mat_t(self, angle: torch.Tensor | list | tuple) -> torch.Tensor:
        """Build a (3x3) rotation matrix (transposed) for the configured axis.

        Args:
            angle: Angle(s) in radians; accepts scalar or array-like.

        Returns:
            Rotation matrix/matrices with shape (3, 3) or (N, 3, 3).
        """
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)
        return points3d_utils.get_rot_mat_t(angle, self.rot_axis)

    def rotate(self, angle: torch.Tensor | list | tuple | None = None, rot_mat_t: torch.Tensor | list | tuple | None = None) -> None:
        """Rotate xyz coordinates in-place.

        Args:
            angle: Angle(s) in radians; used only if ``rot_mat_t`` not provided.
            rot_mat_t: Precomputed rotation matrix/matrices (3x3), transposed form.

        Notes:
            Only the first 3 dimensions (xyz) are rotated; extra channels remain unchanged.
        """
        if rot_mat_t is None:
            rot_mat_t = self.get_rot_mat_t(angle)
        if not isinstance(rot_mat_t, torch.Tensor):
            rot_mat_t = self.tensor.new_tensor(rot_mat_t)
        self.tensor[:, :3] = points3d_utils.rotate_points3d(self.tensor[:, :3], rot_mat_t)

    def flip(self, bev_direction: str = 'horizontal') -> None:
        """Flip points across a BEV axis.

        Args:
            bev_direction: 'horizontal' flips along BEV x/y depending on mode; 'vertical' flips along the other.
        """
        self.tensor = points3d_utils.flip_points3d(self.tensor, self.flip_axis, bev_direction)

    def translate(self, trans_vector: torch.Tensor | list | tuple) -> None:
        """Translate xyz coordinates.

        Args:
            trans_vector: (3,) translation vector added to (x, y, z).
        """
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        self.tensor = points3d_utils.translate_points3d(self.tensor, trans_vector)

    def scale(self, scale_factor: float | int) -> None:
        """Uniformly scale xyz coordinates.

        Args:
            scale_factor: Scalar factor applied to (x, y, z).
        """
        self.tensor = points3d_utils.scale_points3d(self.tensor, scale_factor)

    def points3d_in_surfaces3d(self, surfaces3d: torch.Tensor | list | tuple) -> torch.Tensor:
        """Test whether points lie inside polyhedron surfaces.

        Args:
            surfaces3d: (M, 6, 4, 3) surfaces of oriented 3D boxes.

        Returns:
            (N, M) indicator matrix.
        """
        points = self.tensor[:, :3]
        return points3d_utils.points3d_in_surfaces3d(points, surfaces3d)

    def points3d_in_boxes3d(self, boxes3d: torch.Tensor | list | tuple) -> torch.Tensor:
        """Test whether points lie inside oriented 3D boxes.

        Args:
            boxes3d: (B, 9) oriented boxes (cx,cy,cz,w,h,d,angles).

        Returns:
            (N, B) indicator matrix.
        """
        points3d = self.tensor[:, :3]
        if not isinstance(boxes3d, torch.Tensor):
            boxes3d = self.tensor.new_tensor(boxes3d)
        return boxes3d_utils.points3d_in_boxes3d(points3d, boxes3d, self.rot_axis)
