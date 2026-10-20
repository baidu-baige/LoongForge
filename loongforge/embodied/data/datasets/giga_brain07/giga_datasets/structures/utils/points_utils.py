from typing import Any

import numpy as np
import torch


def _get_rot_mat_t(rot_sin: Any, rot_cos: Any) -> Any:
    """Assemble 2x2 rotation matrix from sine and cosine (transposed form)."""
    rot_mat_t = [rot_cos, rot_sin, -rot_sin, rot_cos]
    return rot_mat_t


def get_rot_mat_t(angle: Any) -> Any:
    """2D rotation matrices for input angles.

    Args:
        angle: Scalar or array-like in radians.

    Returns:
        Array/Tensor of shape (N, 2, 2).
    """
    if isinstance(angle, torch.Tensor):
        angle = angle.view(-1)
        rot_mat_t = _get_rot_mat_t(torch.sin(angle), torch.cos(angle))
        rot_mat_t = torch.cat(rot_mat_t, dim=0).reshape((2, 2, -1)).permute(2, 0, 1).contiguous()
    else:
        angle = np.array(angle)
        angle = angle.reshape((-1,))
        rot_mat_t = _get_rot_mat_t(np.sin(angle), np.cos(angle))
        rot_mat_t = np.concatenate(rot_mat_t, axis=0).reshape((2, 2, -1)).transpose(2, 0, 1)
    return rot_mat_t  # (N, 2, 2)


def rotate_points(points: Any, rot_mat_t: Any) -> Any:
    """Rotate 2D points (optionally homogeneous) by rotation matrix.

    Args:
        points: (..., 2) or (..., 3) if homogeneous.
        rot_mat_t: (2,2) or (3,2) or batched variants.

    Returns:
        Rotated points with same leading shape.
    """
    is_tensor = isinstance(points, torch.Tensor)
    point_shape = points.shape
    rot_mat_t_shape = rot_mat_t.shape
    if point_shape[-1] == 2 and rot_mat_t_shape[-2] == 3:
        ones_shape = list(point_shape)[:-1] + [1]
        if is_tensor:
            points = torch.cat([points, points.new_ones(ones_shape)], dim=-1)
        else:
            points = np.concatenate([points, np.ones(ones_shape)], axis=-1)
    if len(point_shape) == 2:
        if len(rot_mat_t_shape) == 3:
            assert rot_mat_t_shape[0] == 1
            rot_mat_t = rot_mat_t[0]
        if is_tensor:
            return torch.einsum('ij,jk->ik', (points, rot_mat_t))
        else:
            return np.einsum('ij,jk->ik', points, rot_mat_t)
    elif len(point_shape) == 3:
        if is_tensor:
            return torch.einsum('aij,ajk->aik', points, rot_mat_t)
        else:
            return np.einsum('aij,ajk->aik', points, rot_mat_t)
    else:
        assert False


def flip_points(points: Any, image_shape: tuple[int, int], direction: str = 'horizontal') -> Any:
    """Flip points inside an image along given axis.

    Args:
        points: (N, 2) array/tensor.
        image_shape: (H, W).
        direction: 'horizontal' or 'vertical'.

    Returns:
        Flipped points (same type as input).
    """
    image_h, image_w = image_shape[:2]
    if direction == 'horizontal':
        points[:, 0] = image_w - 1 - points[:, 0]
    elif direction == 'vertical':
        points[:, 1] = image_h - 1 - points[:, 1]
    else:
        assert False
    return points


def clip_points(points: Any, image_shape: tuple[int, int]) -> Any:
    """Clamp points onto image boundaries.

    Args:
        points: (N, 2) array/tensor.
        image_shape: (H, W).

    Returns:
        Clamped points.
    """
    image_h, image_w = image_shape[:2]
    if isinstance(points, torch.Tensor):
        points[:, 0] = torch.clamp(points[:, 0], min=0, max=image_w - 1)
        points[:, 1] = torch.clamp(points[:, 1], min=0, max=image_h - 1)
    else:
        points[:, 0] = np.clip(points[:, 0], a_min=0, a_max=image_w - 1)
        points[:, 1] = np.clip(points[:, 1], a_min=0, a_max=image_h - 1)
    return points


def crop_points(points: Any, crop_range: Any, mode: str = 'filter') -> tuple[Any, Any]:
    """Crop points by a rectangle with filter/clip policies.

    Args:
        points: (N, 2) input points.
        crop_range: (x1, y1, x2, y2) rectangle.
        mode: 'filter' to drop outside points, 'clip' to clamp.

    Returns:
        Tuple of (points_after, keep_indices or index array).
    """
    if isinstance(points, torch.Tensor):
        if mode == 'filter':
            keep = torch.logical_and(crop_range[:2] <= points, points <= crop_range[2:]).all(axis=1)
            points = points - crop_range[:2]
            keep = torch.nonzero(keep, as_tuple=False).view(-1)
            return points[keep], keep
        elif mode == 'clip':
            crop_range = crop_range.type_as(points)
            points = torch.min(torch.max(points, crop_range[0:2]), crop_range[2:4])
            points = points - crop_range[:2]
            return points, torch.arange(points.shape[0])
        else:
            assert False
    else:
        if mode == 'filter':
            keep = np.logical_and(crop_range[:2] <= points, points <= crop_range[2:]).all(axis=1)
            points = points - crop_range[:2]
            keep = np.where(keep == 1)[0]
            return points[keep], keep
        elif mode == 'clip':
            points = np.minimum(np.maximum(points, crop_range[0:2]), crop_range[2:4])
            points = points - crop_range[:2]
            return points, np.arange(points.shape[0])
        else:
            assert False


def scale_points(points: Any, scale_hw: tuple[float, float], image_shape: tuple[int, int] | None = None) -> Any:
    """Scale points by (scale_h, scale_w) and optionally clamp to image.

    Args:
        points: (N, 2) points.
        scale_hw: (scale_h, scale_w).
        image_shape: Optional (H, W) for clamping.

    Returns:
        Scaled (and possibly clamped) points.
    """
    scale_h, scale_w = scale_hw
    points[:, 0] *= scale_w
    points[:, 1] *= scale_h
    if image_shape is not None:
        points = clip_points(points, image_shape)
    return points
