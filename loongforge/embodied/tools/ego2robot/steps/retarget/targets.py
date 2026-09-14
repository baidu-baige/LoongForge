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

def eq12_pose_world(kp_seq, hand_sign):
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
                            orientation_sigma=2.0, width_max=None):
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
