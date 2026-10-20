from typing import Any

import numpy as np
import torch

from .base_structure import BaseStructure
from .mode3d import Mode3D
from .utils import boxes3d_utils, points3d_utils


class Boxes3D(BaseStructure):
    """3D oriented bounding boxes with conversion and geometry utilities."""

    def __init__(self, tensor: torch.Tensor | list | tuple, mode: Any = None):
        super(Boxes3D, self).__init__(tensor, mode=mode)
        assert self.tensor.ndim == 2 and self.tensor.shape[1] == 9
        self.tensor = self.tensor.to(torch.float32)
        params = Mode3D.get_params(mode)
        self.mode = mode
        self.rot_axis = params['rot_axis']
        self.flip_axis = params['flip_axis']
        self.bev_axis = params['bev_axis']

    @property
    def volume(self):
        return boxes3d_utils.get_volume(self.tensor)

    @property
    def centers(self):
        return self.tensor[:, :3]

    @property
    def dims(self):
        return self.tensor[:, 3:6]

    @property
    def angles(self):
        return self.tensor[:, 6:9] if self.rot_axis == -1 else self.tensor[:, 6 + self.rot_axis, None]

    def convert_to(self, dst_mode: Any, rot_mat: torch.Tensor | list | tuple | None = None):
        return boxes3d_utils.convert_boxes3d(self, src_mode=self.mode, dst_mode=dst_mode, rot_mat=rot_mat)

    def corners(self) -> torch.Tensor:
        return self.corners3d()

    def corners3d(self) -> torch.Tensor:
        """torch.Tensor: a tensor with N x 8 x 3 matrix."""
        return boxes3d_utils.boxes3d_to_corners3d(self.tensor, self.rot_axis)

    def corners2d(self, cam_intrinsic: torch.Tensor | list | tuple | None = None, bev: bool = False) -> torch.Tensor:
        """torch.Tensor: a tensor with N x 8 x 2 matrix."""
        if bev:
            bev_boxes = boxes3d_utils.boxes3d_to_bev_boxes2d(self.tensor, self.bev_axis)
            return boxes3d_utils.bev_boxes2d_to_corners2d(bev_boxes)
        else:
            if not isinstance(cam_intrinsic, torch.Tensor):
                cam_intrinsic = self.tensor.new_tensor(cam_intrinsic)
            return boxes3d_utils.boxes3d_to_corners2d(self.tensor, self.rot_axis, cam_intrinsic)

    def boxes(self) -> torch.Tensor:
        return self.tensor.clone()

    def boxes3d(self) -> torch.Tensor:
        return self.tensor.clone()

    def boxes2d(self, cam_intrinsic: torch.Tensor | list | tuple) -> torch.Tensor:
        """torch.Tensor: a tensor with N x 4 matrix."""
        if not isinstance(cam_intrinsic, torch.Tensor):
            cam_intrinsic = self.tensor.new_tensor(cam_intrinsic)
        return boxes3d_utils.boxes3d_to_boxes2d(self.tensor, self.rot_axis, cam_intrinsic)

    def surfaces3d(self) -> torch.Tensor:
        """torch.Tensor: a tensor with N x 6 X 4 X 3 matrix."""
        return boxes3d_utils.boxes3d_to_surfaces3d(self.tensor, self.rot_axis)

    def in_range(self, box_range: torch.Tensor | list | tuple) -> torch.Tensor:
        return points3d_utils.points3d_in_range(self.tensor[:, :3], box_range)

    def in_range_bev(self, box_range: torch.Tensor | list | tuple) -> torch.Tensor:
        return points3d_utils.points3d_in_range_bev(self.tensor[:, :3], box_range, self.bev_axis)

    def limit_angles(self, offset: float = 0.5, period: float = np.pi) -> None:
        self.tensor = boxes3d_utils.limit_angles(self.tensor, self.rot_axis, offset, period)

    def get_pose_mat(self) -> torch.Tensor:
        return boxes3d_utils.get_pose_mat(self.tensor, self.rot_axis)

    def get_rot_mat_t(self, angle: torch.Tensor | list | tuple | None = None) -> torch.Tensor:
        if angle is None:
            angle = self.angles
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)
        return points3d_utils.get_rot_mat_t(angle, self.rot_axis)

    def rotate(self, angle: torch.Tensor | list | tuple, rot_mat_t: torch.Tensor | list | tuple | None = None) -> None:
        if rot_mat_t is None:
            rot_mat_t = self.get_rot_mat_t(angle)
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)
        if not isinstance(rot_mat_t, torch.Tensor):
            rot_mat_t = self.tensor.new_tensor(rot_mat_t)
        self.tensor = boxes3d_utils.rotate_boxes3d(self.tensor, rot_mat_t, angle, self.rot_axis)

    def flip(self, bev_direction: str = 'horizontal') -> None:
        self.tensor = boxes3d_utils.flip_boxes3d(self.tensor, self.flip_axis, self.rot_axis, bev_direction)

    def translate(self, trans_vector: torch.Tensor | list | tuple) -> None:
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        self.tensor = boxes3d_utils.translate_boxes3d(self.tensor, trans_vector)

    def scale(self, scale_factor: float | int) -> None:
        self.tensor = boxes3d_utils.scale_boxes3d(self.tensor, scale_factor)

    def points3d_in_boxes3d(self, points3d: torch.Tensor | list | tuple) -> torch.Tensor:
        points3d = points3d[:, :3]
        if not isinstance(points3d, torch.Tensor):
            points3d = self.tensor.new_tensor(points3d)
        boxes3d = self.tensor
        return boxes3d_utils.points3d_in_boxes3d(points3d, boxes3d, self.rot_axis)

    def overlaps(self, boxes3d: torch.Tensor | list | tuple, method: str, mode: str = 'iou', offset: float = 0, eps: float = 1e-6) -> Any:
        if not isinstance(boxes3d, torch.Tensor):
            boxes3d = self.tensor.new_tensor(boxes3d)
        return boxes3d_utils.overlaps(
            self.tensor,
            boxes3d,
            method=method,
            mode=mode,
            offset=offset,
            eps=eps,
            rot_axis=self.rot_axis,
            bev_axis=self.bev_axis,
        )


class LidarBoxes3D(Boxes3D):
    def __init__(self, tensor):
        super(LidarBoxes3D, self).__init__(tensor, mode=Mode3D.LIDAR)


class CameraBoxes3D(Boxes3D):
    def __init__(self, tensor):
        super(CameraBoxes3D, self).__init__(tensor, mode=Mode3D.CAMERA)


class DepthBoxes3D(Boxes3D):
    def __init__(self, tensor):
        super(DepthBoxes3D, self).__init__(tensor, mode=Mode3D.DEPTH)
