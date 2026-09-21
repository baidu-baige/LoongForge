# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Human-hand target construction and morphology TCP conversion."""

import mujoco
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


# Fixed permutation from the Eq.2 EEF frame to the common parallel-gripper TCP frame.
R_EQ2_TO_HAND = np.array([[0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])

# MANO keypoint indices (matching step2_path_b.py).
WRIST_IDX, THUMB_TIP_IDX, INDEX_TIP_IDX, MIDDLE_TIP_IDX = 0, 4, 8, 12

# ============ Eq.1/2 retargeting (world frame) ============

def _camera_poses(head_pose, count):
    """Return camera-to-world rotations and camera origins."""
    rotations = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    origins = np.zeros((count, 3), dtype=float)
    if head_pose is None:
        return rotations, origins
    pose = np.asarray(head_pose, dtype=float).reshape(count, 7)
    origins[:] = pose[:, :3]
    for i, quat in enumerate(pose[:, 3:7]):
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        rotations[i] = mat.reshape(3, 3)
    return rotations, origins


def align_tcp_opening_to_2d(positions, orientations, keypoints_2d,
                            hand_sign, head_pose=None, focal_xy=None):
    """Make the TCP opening axis project onto the observed 2D opening line.

    The TCP y axis is the parallel gripper's opening axis. Merely copying the
    image-plane direction into its camera x/y components is not sufficient:
    perspective projection also depends on the TCP position and axis depth.
    This solves that projection constraint while retaining the 3D axis depth
    component whenever a positive projected solution exists.
    """
    p_world = np.asarray(positions, dtype=float).reshape(-1, 3)
    result = np.asarray(orientations, dtype=float).reshape(-1, 3, 3).copy()
    count = len(result)
    kp2d = np.asarray(keypoints_2d, dtype=float).reshape(count, 21, 2)
    camera_R, camera_p = _camera_poses(head_pose, count)
    p_camera = np.einsum(
        "nji,nj->ni", camera_R, p_world - camera_p)
    opening_camera = np.einsum(
        "nji,nj->ni", camera_R, result[:, :, 1])

    thumb2d = kp2d[:, THUMB_TIP_IDX]
    virtual2d = (0.7 * kp2d[:, INDEX_TIP_IDX] +
                 0.3 * kp2d[:, MIDDLE_TIP_IDX])
    opening2d = float(hand_sign) * (thumb2d - virtual2d)
    focal = (np.ones(2, dtype=float) if focal_xy is None else
             np.asarray(focal_xy, dtype=float).reshape(2))
    if not np.isfinite(focal).all() or np.any(focal <= 0.0):
        raise ValueError("focal_xy must contain two finite positive values")
    opening_normalized = opening2d / focal
    opening_norm = np.linalg.norm(opening_normalized, axis=1)
    relevant = kp2d[:, [THUMB_TIP_IDX, INDEX_TIP_IDX, MIDDLE_TIP_IDX]]
    valid = (np.isfinite(relevant).all(axis=(1, 2)) &
             (opening_norm > 1e-8) & np.isfinite(p_camera).all(axis=1) &
             (p_camera[:, 2] > 1e-6))

    for i in np.flatnonzero(valid):
        direction = opening_normalized[i] / opening_norm[i]
        position = p_camera[i]
        old_axis = opening_camera[i]
        depth = float(position[2])
        axis_z = float(np.clip(old_axis[2], -1.0, 1.0))

        # A projected infinitesimal axis obeys
        #   delta_uv = (axis_xy * p_z - p_xy * axis_z) / p_z^2.
        # Set delta_uv=lambda*direction and solve ||axis||=1 for lambda.
        a = 1.0 / (depth * depth)
        b = (2.0 * axis_z * np.dot(direction, position[:2]) /
             (depth * depth))
        c = axis_z * axis_z * (
            1.0 + np.dot(position[:2], position[:2]) / (depth * depth)) - 1.0
        discriminant = b * b - 4.0 * a * c
        candidates = []
        if discriminant >= 0.0:
            root = np.sqrt(discriminant)
            for scale in ((-b + root) / (2.0 * a),
                          (-b - root) / (2.0 * a)):
                if scale <= 1e-10:
                    continue
                axis = np.array([
                    (scale * direction[0] + position[0] * axis_z) / depth,
                    (scale * direction[1] + position[1] * axis_z) / depth,
                    axis_z,
                ])
                axis /= max(np.linalg.norm(axis), 1e-12)
                candidates.append(axis)
        # Some near-view-axis poses cannot retain the original depth component
        # and project in the requested direction. The camera-plane solution is
        # exact and avoids an arbitrary 90/180 degree orientation jump.
        if not candidates:
            candidates.append(np.array([direction[0], direction[1], 0.0]))
        corrected_camera = max(candidates, key=lambda x: np.dot(x, old_axis))
        corrected_world = camera_R[i] @ corrected_camera
        corrected_world /= max(np.linalg.norm(corrected_world), 1e-12)

        # Retain the smoothed approach direction modulo the now-constrained
        # opening axis, then rebuild a proper right-handed TCP rotation.
        approach = result[i, :, 2]
        approach -= corrected_world * np.dot(approach, corrected_world)
        if np.linalg.norm(approach) < 1e-8:
            approach = result[i, :, 0]
            approach -= corrected_world * np.dot(approach, corrected_world)
        approach /= max(np.linalg.norm(approach), 1e-12)
        lateral = np.cross(corrected_world, approach)
        lateral /= max(np.linalg.norm(lateral), 1e-12)
        result[i] = np.stack([lateral, corrected_world, approach], axis=1)
    return result


def opening_projection_plane_normals(positions, keypoints_2d, hand_sign,
                                     head_pose=None, focal_xy=None):
    """Return world normals of planes producing the observed opening line.

    For a TCP at camera point ``(x, y, z)``, every 3D opening axis whose
    perspective projection follows the observed 2D direction lies in one
    plane through the camera ray. Requiring the robot axis to be orthogonal to
    this plane normal preserves the visible orientation without inventing the
    monocularly unobservable axis-depth component.
    """
    p_world = np.asarray(positions, dtype=float).reshape(-1, 3)
    count = len(p_world)
    kp2d = np.asarray(keypoints_2d, dtype=float).reshape(count, 21, 2)
    camera_R, camera_p = _camera_poses(head_pose, count)
    p_camera = np.einsum("nji,nj->ni", camera_R, p_world - camera_p)
    thumb2d = kp2d[:, THUMB_TIP_IDX]
    virtual2d = (0.7 * kp2d[:, INDEX_TIP_IDX] +
                 0.3 * kp2d[:, MIDDLE_TIP_IDX])
    opening2d = float(hand_sign) * (thumb2d - virtual2d)
    focal = (np.ones(2, dtype=float) if focal_xy is None else
             np.asarray(focal_xy, dtype=float).reshape(2))
    if not np.isfinite(focal).all() or np.any(focal <= 0.0):
        raise ValueError("focal_xy must contain two finite positive values")
    direction = opening2d / focal
    direction_norm = np.linalg.norm(direction, axis=1)
    valid = (np.isfinite(direction).all(axis=1) &
             (direction_norm > 1e-8) & np.isfinite(p_camera).all(axis=1) &
             (p_camera[:, 2] > 1e-6))
    normals_world = np.full((count, 3), np.nan, dtype=float)
    if not np.any(valid):
        return normals_world
    direction[valid] /= direction_norm[valid, None]
    perpendicular = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
    normals_camera = np.stack([
        perpendicular[:, 0] * p_camera[:, 2],
        perpendicular[:, 1] * p_camera[:, 2],
        -np.einsum("ij,ij->i", perpendicular, p_camera[:, :2]),
    ], axis=1)
    normal_norm = np.linalg.norm(normals_camera, axis=1)
    valid &= normal_norm > 1e-8
    normals_camera[valid] /= normal_norm[valid, None]
    normals_world[valid] = np.einsum(
        "nij,nj->ni", camera_R[valid], normals_camera[valid])
    return normals_world


def target_axis_projection_plane_normals(positions, orientations, axis,
                                         head_pose=None):
    """Return camera-ray plane normals for a target-frame local axis.

    Unlike the observed opening line above, the second visible gripper axis
    comes from the perspective-corrected Eq.2 target.  Constraining the robot
    axis to the plane spanned by the camera ray and that target axis preserves
    its image-space direction while leaving monocular depth unconstrained.
    """
    p_world = np.asarray(positions, dtype=float).reshape(-1, 3)
    R_world = np.asarray(orientations, dtype=float).reshape(-1, 3, 3)
    local_axis = np.asarray(axis, dtype=float).reshape(3).copy()
    axis_norm = np.linalg.norm(local_axis)
    if not np.isfinite(local_axis).all() or axis_norm < 1e-8:
        raise ValueError("axis must be finite and non-zero")
    local_axis /= axis_norm
    camera_R, camera_p = _camera_poses(head_pose, len(p_world))
    ray_camera = np.einsum(
        "nji,nj->ni", camera_R, p_world - camera_p)
    axis_world = np.einsum("nij,j->ni", R_world, local_axis)
    axis_camera = np.einsum("nji,nj->ni", camera_R, axis_world)
    normals_camera = np.cross(ray_camera, axis_camera)
    normal_norm = np.linalg.norm(normals_camera, axis=1)
    valid = (np.isfinite(normals_camera).all(axis=1) &
             np.isfinite(ray_camera).all(axis=1) &
             (ray_camera[:, 2] > 1e-6) & (normal_norm > 1e-8))
    normals_world = np.full((len(p_world), 3), np.nan, dtype=float)
    normals_camera[valid] /= normal_norm[valid, None]
    normals_world[valid] = np.einsum(
        "nij,nj->ni", camera_R[valid], normals_camera[valid])
    return normals_world


def projection_plane_directions(positions, plane_normals, head_pose=None):
    """Return the positive in-plane direction for an image-axis constraint.

    A plane normal alone represents an unoriented image line.  ``normal x
    camera_ray`` is the tangent direction whose perspective projection has
    the positive sign used to construct that normal, distinguishing an axis
    from its 180-degree reversal.
    """
    p_world = np.asarray(positions, dtype=float).reshape(-1, 3)
    normals_world = np.asarray(plane_normals, dtype=float).reshape(-1, 3)
    _, camera_p = _camera_poses(head_pose, len(p_world))
    ray_world = p_world - camera_p
    directions_world = np.cross(normals_world, ray_world)
    direction_norm = np.linalg.norm(directions_world, axis=1)
    valid = (np.isfinite(directions_world).all(axis=1) &
             (direction_norm > 1e-8))
    result = np.full_like(directions_world, np.nan)
    result[valid] = directions_world[valid] / direction_norm[valid, None]
    return result

def eq12_pose_world(kp_seq, hand_sign, keypoints_2d=None, head_pose=None):
    """MANO 21 keypoints -> common parallel-gripper TCP pose in world frame.

    hand_sign: +1 for the right hand, -1 for the left hand.
    Degeneracy guard: when z x d is near zero, fall back to [0,0,1] or [0,1,0]
    (as in step2_path_b.py)."""
    N = kp_seq.shape[0]
    kp = kp_seq.reshape(N, 21, 3)
    wrist = kp[:, WRIST_IDX]
    thumb = kp[:, THUMB_TIP_IDX]
    index = kp[:, INDEX_TIP_IDX]
    middle = kp[:, MIDDLE_TIP_IDX]

    k_vf = 0.7 * index + 0.3 * middle
    p = 0.5 * (thumb + k_vf)
    w = np.linalg.norm(thumb - k_vf, axis=1)
    w_safe = np.maximum(w, 1e-8)

    d = k_vf - wrist
    z = hand_sign * (thumb - k_vf) / w_safe[:, None]
    zxd = np.cross(z, d)
    zxd_norm = np.linalg.norm(zxd, axis=1)
    degen = zxd_norm < 1e-6
    if np.any(degen):
        ref = np.tile([0.0, 0.0, 1.0], (N, 1))
        alt = np.abs(np.einsum("ij,ij->i", z, ref)) > 0.99
        ref[alt] = [0.0, 1.0, 0.0]
        fb = np.cross(z, ref)
        zxd[degen] = fb[degen]
        zxd_norm = np.linalg.norm(zxd, axis=1)
    y = zxd / np.maximum(zxd_norm, 1e-8)[:, None]
    x = np.cross(y, z)
    R_eq2 = np.stack([x, y, z], axis=2)          # (N,3,3)
    R_tcp = R_eq2 @ R_EQ2_TO_HAND
    if keypoints_2d is not None:
        R_tcp = align_tcp_opening_to_2d(
            p, R_tcp, keypoints_2d, hand_sign, head_pose=head_pose)
    return p, R_tcp, w, p


def _savgol_window(length, requested, polyorder):
    """Return a valid odd Savitzky-Golay window for a trajectory length."""
    if length < polyorder + 2:
        return None
    window = min(int(requested), length if length % 2 else length - 1)
    if window % 2 == 0:
        window -= 1
    return window if window >= polyorder + 2 else None


def _fill_invalid_keypoints(kp_seq):
    """Linearly fill missing MANO frames before temporal filtering.

    Invalid detections are common at the edge of a hand track. Letting one
    sentinel/NaN frame enter Eq.2 contaminates the cross-product orientation
    over the whole smoothing window, so interpolation is done per coordinate
    while the original validity mask is retained for quality reporting.
    """
    values = np.asarray(kp_seq, dtype=float).copy()
    if values.ndim != 2:
        values = values.reshape(len(values), -1)
    finite = np.isfinite(values) & (np.abs(values) < 1e8)
    if finite.all():
        return values
    frames = np.arange(len(values))
    for dim in range(values.shape[1]):
        good = finite[:, dim]
        if not np.any(good):
            values[:, dim] = 0.0
            continue
        if np.count_nonzero(good) == 1:
            values[:, dim] = values[good, dim][0]
        else:
            values[:, dim] = np.interp(frames, frames[good], values[good, dim])
    return values


def _smooth_rotations(rotations, sigma=2.0):
    """Gaussian quaternion mean with hemisphere alignment, as in EgoDex."""
    rotations = np.asarray(rotations, dtype=float)
    if len(rotations) < 2 or sigma <= 0.0:
        return rotations
    quats = Rotation.from_matrix(rotations).as_quat()
    half = max(1, int(3.0 * float(sigma)))
    smoothed = np.empty_like(quats)
    for t in range(len(quats)):
        lo, hi = max(0, t - half), min(len(quats), t + half + 1)
        idx = np.arange(lo, hi)
        weights = np.exp(-0.5 * ((idx - t) / float(sigma)) ** 2)
        weights /= max(weights.sum(), 1e-12)
        patch = quats[lo:hi].copy()
        patch[(patch @ quats[t]) < 0.0] *= -1.0
        q = (weights[:, None] * patch).sum(axis=0)
        norm = np.linalg.norm(q)
        smoothed[t] = q / max(norm, 1e-12)
    return Rotation.from_quat(smoothed).as_matrix()


def smooth_retarget_targets(positions, orientations, widths,
                            window=11, polyorder=3,
                            orientation_sigma=2.0, width_max=None,
                            keypoints_2d=None, hand_sign=None, head_pose=None):
    """Temporally smooth Eq.1/Eq.2 targets before base search and IK."""
    positions = _fill_invalid_keypoints(np.asarray(positions, dtype=float))
    orientations = np.asarray(orientations, dtype=float)
    widths = _fill_invalid_keypoints(
        np.asarray(widths, dtype=float).reshape(-1, 1)).reshape(-1)
    win = _savgol_window(len(positions), window, polyorder)
    if win is not None:
        positions = savgol_filter(positions, win, polyorder, axis=0)
        widths = savgol_filter(widths, win, polyorder, axis=0)
    widths = np.maximum(widths, 0.0)
    if width_max is not None:
        widths = np.minimum(widths, float(width_max))
    orientations = _smooth_rotations(orientations, orientation_sigma)
    if keypoints_2d is not None:
        if hand_sign is None:
            raise ValueError("hand_sign is required with keypoints_2d")
        orientations = align_tcp_opening_to_2d(
            positions, orientations, keypoints_2d, hand_sign,
            head_pose=head_pose)
    return positions, orientations, widths


def _quat_to_mat(quat):
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat, dtype=float))
    return out.reshape(3, 3)


def target_ref_pose(spec, p_tcp_world, R_tcp_world, tcp_rot_override=None,
                    opening_width=None):
    """Convert common TCP targets to the model's IK reference frame."""
    if spec.ee_site:
        # Some source XMLs place the named site at the gripper rail rather
        # than at the physical finger contact center.  Track the site at the
        # inverse of this calibrated offset so the visible pads meet the
        # human TCP.
        tcp_rot = _quat_to_mat(
            spec.tcp_rot_site if tcp_rot_override is None else tcp_rot_override)
        R_site = R_tcp_world @ tcp_rot.T
        offset = np.broadcast_to(
            np.asarray(spec.tcp_pos_site, dtype=float),
            np.asarray(p_tcp_world).shape,
        ).copy()
        width_gain = np.asarray(spec.tcp_pos_site_width_gain, dtype=float)
        if opening_width is not None and np.any(width_gain):
            width = np.clip(
                np.asarray(opening_width, dtype=float), 0.0,
                float(spec.gripper_max),
            )
            offset += width[..., None] * width_gain
        if np.any(offset):
            p_site = p_tcp_world - np.einsum(
                "nij,nj->ni", R_site, offset)
        else:
            p_site = p_tcp_world
        return p_site, R_site
    tcp_rot = _quat_to_mat(spec.tcp_rot_body)
    R_body = R_tcp_world @ tcp_rot.T
    p_body = p_tcp_world - np.einsum("nij,j->ni", R_body, np.asarray(spec.tcp_pos_body))
    return p_body, R_body
