from typing import Any

import numpy as np
import torch


def _get_rot_mat_t(rot_sin: Any, rot_cos: Any, ones: Any, zeros: Any, rot_axis: int):
    """Assemble 3x3 rotation matrix (transposed) for a given axis.

    Args:
        rot_sin: sin(theta) values.
        rot_cos: cos(theta) values.
        ones/zeros: Helpers with same shape as angles.
        rot_axis (int): 0 (x), 1 (y), 2 (z).

    Returns:
        Rotation matrix entries concatenated in column-major then reshaped outside.
    """
    if rot_axis == 0:
        rot_mat_t = [
            ones,
            zeros,
            zeros,
            zeros,
            rot_cos,
            rot_sin,
            zeros,
            -rot_sin,
            rot_cos,
        ]
    elif rot_axis == 1:
        rot_mat_t = [
            rot_cos,
            zeros,
            -rot_sin,
            zeros,
            ones,
            zeros,
            rot_sin,
            zeros,
            rot_cos,
        ]
    elif rot_axis == 2:
        rot_mat_t = [
            rot_cos,
            rot_sin,
            zeros,
            -rot_sin,
            rot_cos,
            zeros,
            zeros,
            zeros,
            ones,
        ]
    else:
        assert False
    return rot_mat_t


def get_rot_mat_t(angle3d: Any, rot_axis: int):
    """3D rotation matrices for given axis or Euler triplet.

    Args:
        angle3d: (N,) when single axis or (N,3) for Euler angles.
        rot_axis: 0/1/2 for x/y/z or -1 for ZYX composition.

    Returns:
        Array/Tensor of shape (N, 3, 3).
    """
    if isinstance(angle3d, torch.Tensor):
        if rot_axis in (0, 1, 2):
            if angle3d.ndim == 2 and angle3d.shape[1] == 3:
                angle3d = angle3d[:, rot_axis]
            angle3d = angle3d.view(-1)
            ones = torch.ones_like(angle3d)
            zeros = torch.zeros_like(angle3d)
            rot_mat_t = _get_rot_mat_t(torch.sin(angle3d), torch.cos(angle3d), ones, zeros, rot_axis)
            rot_mat_t = torch.cat(rot_mat_t, dim=0).reshape((3, 3, -1)).permute(2, 0, 1).contiguous()
        elif rot_axis == -1:
            angle3d = angle3d.view(-1, 3)
            ones = torch.ones_like(angle3d[:, 0])
            zeros = torch.zeros_like(angle3d[:, 0])
            rot_mat_t_x = _get_rot_mat_t(torch.sin(angle3d[:, 0]), torch.cos(angle3d[:, 0]), ones, zeros, 0)
            rot_mat_t_y = _get_rot_mat_t(torch.sin(angle3d[:, 1]), torch.cos(angle3d[:, 1]), ones, zeros, 1)
            rot_mat_t_z = _get_rot_mat_t(torch.sin(angle3d[:, 2]), torch.cos(angle3d[:, 2]), ones, zeros, 2)
            rot_mat_t_x = torch.cat(rot_mat_t_x, dim=0).reshape((3, 3, -1))
            rot_mat_t_y = torch.cat(rot_mat_t_y, dim=0).reshape((3, 3, -1))
            rot_mat_t_z = torch.cat(rot_mat_t_z, dim=0).reshape((3, 3, -1))
            rot_mat_t = torch.einsum('ija,jka,kha->aih', rot_mat_t_z, rot_mat_t_y, rot_mat_t_x)
        else:
            assert False
    else:
        angle3d = np.array(angle3d)
        if rot_axis in (0, 1, 2):
            if angle3d.ndim == 2 and angle3d.shape[1] == 3:
                angle3d = angle3d[:, rot_axis]
            angle3d = angle3d.reshape((-1,))
            ones = np.ones_like(angle3d)
            zeros = np.zeros_like(angle3d)
            rot_mat_t = _get_rot_mat_t(np.sin(angle3d), np.cos(angle3d), ones, zeros, rot_axis)
            rot_mat_t = np.concatenate(rot_mat_t, axis=0).reshape((3, 3, -1)).transpose(2, 0, 1)
        elif rot_axis == -1:
            angle3d = angle3d.reshape((-1, 3))
            ones = np.ones_like(angle3d[:, 0])
            zeros = np.zeros_like(angle3d[:, 0])
            rot_mat_t_x = _get_rot_mat_t(np.sin(angle3d[:, 0]), np.cos(angle3d[:, 0]), ones, zeros, 0)
            rot_mat_t_y = _get_rot_mat_t(np.sin(angle3d[:, 1]), np.cos(angle3d[:, 1]), ones, zeros, 1)
            rot_mat_t_z = _get_rot_mat_t(np.sin(angle3d[:, 2]), np.cos(angle3d[:, 2]), ones, zeros, 2)
            rot_mat_t_x = np.concatenate(rot_mat_t_x, axis=0).reshape((3, 3, -1))
            rot_mat_t_y = np.concatenate(rot_mat_t_y, axis=0).reshape((3, 3, -1))
            rot_mat_t_z = np.concatenate(rot_mat_t_z, axis=0).reshape((3, 3, -1))
            rot_mat_t = np.einsum('ija,jka,kha->aih', rot_mat_t_z, rot_mat_t_y, rot_mat_t_x)
        else:
            assert False
    return rot_mat_t  # (N, 3, 3)


def convert_points3d(points3d: Any, src_mode: Any, dst_mode: Any, rot_mat: Any | None = None):
    """Convert 3D points between LIDAR/CAMERA/DEPTH modes.

    Args:
        points3d: (N, C) where first 3 are xyz.
        src_mode: Source mode enum/name.
        dst_mode: Destination mode enum/name.
        rot_mat: Optional (3x3 or 3x4) rotation/transform.

    Returns:
        Converted points in same container type.
    """
    from ..mode3d import Mode3D
    from ..points3d import Points3D

    src_mode = Mode3D.get_mode(src_mode)
    dst_mode = Mode3D.get_mode(dst_mode)
    if src_mode == dst_mode:
        return points3d

    is_numpy = isinstance(points3d, np.ndarray)
    is_class = hasattr(points3d, 'tensor')
    if is_numpy:
        data = torch.from_numpy(points3d).clone()
    elif is_class:
        data = points3d.tensor.clone()
    else:
        data = points3d.clone()

    if rot_mat is None:
        if src_mode == Mode3D.LIDAR and dst_mode == Mode3D.CAMERA:
            rot_mat = data.new_tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

        elif src_mode == Mode3D.CAMERA and dst_mode == Mode3D.LIDAR:
            rot_mat = data.new_tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

        elif src_mode == Mode3D.DEPTH and dst_mode == Mode3D.CAMERA:
            rot_mat = data.new_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        elif src_mode == Mode3D.CAMERA and dst_mode == Mode3D.DEPTH:
            rot_mat = data.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        elif src_mode == Mode3D.LIDAR and dst_mode == Mode3D.DEPTH:
            rot_mat = data.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        elif src_mode == Mode3D.DEPTH and dst_mode == Mode3D.LIDAR:
            rot_mat = data.new_tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

        else:
            assert False

    if not isinstance(rot_mat, torch.Tensor):
        rot_mat = data.new_tensor(rot_mat)
    if rot_mat.size(1) == 4:
        extended_xyz = torch.cat([data[:, :3], data.new_ones(data.size(0), 1)], dim=-1)
        xyz = torch.matmul(extended_xyz, rot_mat.t())
    else:
        xyz = torch.matmul(data[:, :3], rot_mat.t())

    points3d = torch.cat([xyz[:, :3], data[:, 3:]], dim=-1)

    if is_numpy:
        return points3d.numpy()
    elif is_class:
        return Points3D(points3d, mode=dst_mode)
    else:
        return points3d


def rotate_points3d(points3d: Any, rot_mat_t: Any):
    """Rotate 3D points by rotation matrices (batched or single).

    Args:
        points3d: (..., 3) points or batched.
        rot_mat_t: (3,3) or batched (N,3,3).

    Returns:
        Rotated points with same leading shape.
    """
    is_tensor = isinstance(points3d, torch.Tensor)
    point_shape = points3d.shape
    if len(point_shape) == 2:
        if len(rot_mat_t.shape) == 3:
            assert rot_mat_t.shape[0] == 1
            rot_mat_t = rot_mat_t[0]
        if is_tensor:
            return torch.einsum('ij,jk->ik', (points3d, rot_mat_t))
        else:
            return np.einsum('ij,jk->ik', points3d, rot_mat_t)
    elif len(point_shape) == 3:
        if is_tensor:
            return torch.einsum('aij,ajk->aik', points3d, rot_mat_t)
        else:
            return np.einsum('aij,ajk->aik', points3d, rot_mat_t)
    else:
        assert False


def flip_points3d(points3d: Any, flip_axis: tuple[int, int], bev_direction: str = 'horizontal'):
    """Flip 3D points in BEV along given direction.

    Args:
        points3d: (N, 3) points.
        flip_axis: Axis indices (h, v) in BEV.
        bev_direction: 'horizontal' or 'vertical'.

    Returns:
        Flipped points.
    """
    if bev_direction == 'horizontal':
        points3d[:, flip_axis[0]] = -points3d[:, flip_axis[0]]
    elif bev_direction == 'vertical':
        points3d[:, flip_axis[1]] = -points3d[:, flip_axis[1]]
    else:
        assert False
    return points3d


def translate_points3d(points3d: Any, trans_vector: Any):
    """Translate 3D points by a vector."""
    points3d[:, :3] += trans_vector
    return points3d


def scale_points3d(points3d: Any, scale_factor: float | int):
    """Scale 3D points by a scalar factor (xyz only)."""
    points3d[:, :3] *= scale_factor
    return points3d


def points3d_to_points2d(points3d: Any, cam_intrinsic: Any):
    """Project 3D points to pixels via camera intrinsics.

    Args:
        points3d: (N, 3) xyz.
        cam_intrinsic: (3,4) or (3,3).

    Returns:
        (N, 2) pixel coordinates.
    """
    if isinstance(points3d, torch.Tensor):
        if cam_intrinsic.shape[1] == 4:
            points3d = torch.cat([points3d, points3d.new_ones((points3d.shape[0], 1))], dim=-1)  # (N, 4)
        points3d = torch.matmul(points3d, cam_intrinsic.t())  # (N, 3)
    else:
        if cam_intrinsic.shape[1] == 4:
            points3d = np.concatenate([points3d, np.ones((points3d.shape[0], 1))], axis=-1)  # (N, 4)
        points3d = np.dot(points3d, cam_intrinsic.T)  # (N, 3)
    points = points3d[:, :2] / points3d[:, 2:3]  # (N, 2)
    return points


def depth_to_points3d(depth: np.ndarray, cam_k: np.ndarray | None = None, cam_scale: float = 1.0, method: int = 2):
    """Backproject depth map to 3D points.

    Args:
        depth: (H, W) depth map.
        cam_k: (3,3) intrinsic matrix, optional.
        cam_scale: Scale factor to convert depth to meters.
        method: 1 uses normalized coords; 2 uses inverse K.

    Returns:
        (H*W, 3) array of points.
    """
    xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / cam_scale
    if cam_k is None:
        points3d = np.stack([xmap, ymap, points_z], axis=-1).astype(np.float32)
    elif method == 1:
        fx, fy = cam_k[0, 0], cam_k[1, 1]
        cx, cy = cam_k[0, 2], cam_k[1, 2]
        points_x = (xmap - cx) / fx
        points_y = (ymap - cy) / fy
        points3d = np.stack([points_x * points_z, points_y * points_z, points_z], axis=-1).astype(np.float32)
    elif method == 2:
        cam_k_inv = cam_k.copy()
        cam_k_inv[0, 0] = 1.0 / cam_k[0, 0]
        cam_k_inv[1, 1] = 1.0 / cam_k[1, 1]
        cam_k_inv[0, 2] = -cam_k[0, 2] / cam_k[0, 0]
        cam_k_inv[1, 2] = -cam_k[1, 2] / cam_k[1, 1]
        points3d = np.stack([xmap * points_z, ymap * points_z, points_z], axis=-1).astype(np.float32)
        points3d = np.dot(points3d, cam_k_inv.T)
    else:
        assert False
    return points3d.reshape((-1, 3))


def calc_square_dist(point_feat_a: torch.Tensor, point_feat_b: torch.Tensor, norm: bool = True):
    """Calculating square distance between a and b.

    Args:
        point_feat_a (Tensor): (B, N, C) Feature vector of each point.
        point_feat_b (Tensor): (B, M, C) Feature vector of each point.
        norm (Bool): Whether to normalize the distance.
            Default: True.

    Returns:
        Tensor: (B, N, M) Distance between each pair points.
    """
    length_a = point_feat_a.shape[1]
    length_b = point_feat_b.shape[1]
    num_channel = point_feat_a.shape[-1]
    # [bs, n, 1]
    a_square = torch.sum(point_feat_a.unsqueeze(dim=2).pow(2), dim=-1)
    # [bs, 1, m]
    b_square = torch.sum(point_feat_b.unsqueeze(dim=1).pow(2), dim=-1)
    a_square = a_square.repeat((1, 1, length_b))  # [bs, n, m]
    b_square = b_square.repeat((1, length_a, 1))  # [bs, n, m]

    coor = torch.matmul(point_feat_a, point_feat_b.transpose(1, 2))

    dist = a_square + b_square - 2 * coor
    if norm:
        dist = torch.sqrt(dist) / num_channel
    return dist


def points3d_in_range(points3d: Any, point_range: Any):
    """BEV cuboid range filter on 3D points."""
    # points3d: (N, 3)
    in_range_flags = (
        (points3d[:, 0] > point_range[0])
        & (points3d[:, 1] > point_range[1])
        & (points3d[:, 2] > point_range[2])
        & (points3d[:, 0] < point_range[3])
        & (points3d[:, 1] < point_range[4])
        & (points3d[:, 2] < point_range[5])
    )
    return in_range_flags


def points3d_in_range_bev(points3d: Any, point_range: Any, bev_axis: tuple[int, int, int, int, int]):
    """2D BEV range filter using selected axes."""
    # points3d: (N, 3)
    in_range_flags = (
        (points3d[:, bev_axis[0]] > point_range[bev_axis[0]])
        & (points3d[:, bev_axis[1]] > point_range[bev_axis[1]])
        & (points3d[:, bev_axis[0]] < point_range[bev_axis[0] + 3])
        & (points3d[:, bev_axis[1]] < point_range[bev_axis[1] + 3])
    )
    return in_range_flags


def points3d_in_surfaces3d(points3d: Any, surfaces3d: Any):
    """Check inclusion of points inside 3D polyhedron surfaces.

    Args:
        points3d: (N, 3) xyz points.
        surfaces3d: (M, 6, 4, 3) surfaces of boxes.

    Returns:
        (N, M) indicator matrix.
    """
    if isinstance(points3d, torch.Tensor):
        surface_vec = surfaces3d[:, :, :2, :] - surfaces3d[:, :, 1:3, :]  # (M, 6, 2, 3)
        # normal_vec: (a, b, c, d) with ax+by+cz+d=0
        normal_vec = torch.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :], dim=-1)  # (M, 6, 3)
        d = -torch.einsum('aij, aij->ai', normal_vec, surfaces3d[:, :, 0, :])  # (M, 6)
        normal_vec = torch.cat([normal_vec, d[:, :, None]], dim=-1)  # (M, 6, 4)
        points3d = torch.cat([points3d, points3d.new_ones((points3d.shape[0], 1))], dim=-1)  # (N, 4)
        indices = torch.einsum('aj, bij->abi', points3d, normal_vec)  # (N, M, 6)
        indices = torch.sum(indices >= 0, axis=2)  # (N, M)
        indices = indices == 0  # (N, M)
        indices = indices.to(torch.int32)
    else:
        surface_vec = surfaces3d[:, :, :2, :] - surfaces3d[:, :, 1:3, :]  # (M, 6, 2, 3)
        normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])  # (M, 6, 3)
        d = -np.einsum('aij, aij->ai', normal_vec, surfaces3d[:, :, 0, :])  # (M, 6)
        normal_vec = np.concatenate([normal_vec, d[:, :, None]], axis=-1)  # (M, 6, 4)
        points3d = np.concatenate([points3d, np.ones((points3d.shape[0], 1))], axis=-1)  # (N, 4)
        indices = np.einsum('aj, bij->abi', points3d, normal_vec)  # (N, M, 6)
        indices = np.sum(indices >= 0, axis=2)  # (N, M)
        indices = indices == 0  # (N, M)
        indices = indices.astype(np.int32)
    return indices


def get_rot_mat_and_translation(src_points: Any, dst_points: Any):
    """Calculates the least-squares best-fit transform that maps corresponding
    src_points to dst_points in m spatial dimensions.

    Args:
        src_points: (N, m) points.
        dst_points: (N, m) points.

    Returns:
        R: (m, m) rotation matrix.
        t: (m,) translation vector.
    """
    assert src_points.shape == dst_points.shape
    if isinstance(src_points, torch.Tensor):
        # translate points to their centroids
        src_centroid = torch.mean(src_points, dim=0)
        dst_centroid = torch.mean(dst_points, dim=0)
        src_points = src_points - src_centroid
        dst_points = dst_points - dst_centroid
        # rotation matirx
        H = torch.mm(src_points.t(), dst_points)
        U, S, V = torch.svd(H)
        R = torch.mm(V, U.t())
        # special reflection case
        if torch.det(R) < 0:
            V[:, -1] *= -1
            R = torch.mm(V, U.t())
        # translation
        t = dst_centroid - torch.mm(R, src_centroid.view(3, 1))[:, 0]
    else:
        # translate points to their centroids
        src_centroid = np.mean(src_points, axis=0)
        dst_centroid = np.mean(dst_points, axis=0)
        src_points = src_points - src_centroid
        dst_points = dst_points - dst_centroid
        # rotation matirx
        H = np.dot(src_points.T, dst_points)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        # translation
        t = dst_centroid - np.dot(R, src_centroid)
    return R, t
