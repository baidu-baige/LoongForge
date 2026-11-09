# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Similarity registration and projection of DVLT reconstructions.

The DVLT model predicts its own (arbitrarily scaled) camera poses and a world
point cloud.  The SLAM tracker provides metric camera poses for the same
trajectory, so an Umeyama similarity transform
``X_slam = s * R @ X_dvlt + t`` registers the reconstruction into the metric
SLAM frame; the resulting world points can then be projected for *any* frame
with the retarget-compatible pinhole model, yielding a metric scene depth map
aligned with the MuJoCo robot depth grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass
class Similarity:
    """Similarity transform ``dst = scale * R @ src + t`` (column vectors)."""

    scale: float
    R: np.ndarray        # (3, 3)
    t: np.ndarray        # (3,)

    def apply_pts(self, pts: np.ndarray) -> np.ndarray:
        """Apply to (M, 3) row-vector points."""
        return self.scale * (pts @ self.R.T) + self.t

    def apply_origins(self, origins: np.ndarray) -> np.ndarray:
        return self.apply_pts(np.asarray(origins, dtype=np.float64))


@dataclass
class AlignStats:
    scale: float = 1.0
    rot_err_deg: float = np.nan
    rmse: float = np.nan          # meters, after alignment
    reflected: bool = False
    n_pairs: int = 0
    origin_std: float = np.nan    # m; tiny values mean a degenerate (static) camera
    detail: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {
            "scale": float(self.scale),
            "rot_err_deg": float(self.rot_err_deg) if np.isfinite(self.rot_err_deg) else None,
            "rmse_m": float(self.rmse) if np.isfinite(self.rmse) else None,
            "reflected": bool(self.reflected),
            "n_pairs": int(self.n_pairs),
            "origin_std_m": float(self.origin_std) if np.isfinite(self.origin_std) else None,
            **{k: v for k, v in self.detail.items()},
        }


def umeyama(src: np.ndarray, dst: np.ndarray) -> Similarity:
    """Least-squares similarity transform ``dst ≈ scale * R @ src + t``.

    Both inputs are (M, 3) row-vector points.  Detects and corrects reflections.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    if n < 3:
        raise ValueError(f"umeyama needs >= 3 correspondences, got {n}")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    # Unnormalized scatter matrix: Hess/Umeyama scale is
    # s = tr(D S) / sum_i ||src_i - mu_src||^2, with D from the SVD of
    # (Y - mu_dst)^T (X - mu_src) -- keeping numerator and denominator at the
    # same (unscaled) magnitude avoids an n-fold scale error.
    cov = dst_c.T @ src_c
    U, d, Vt = np.linalg.svd(cov)
    sign = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0
    R = U @ np.diag([1.0, 1.0, sign]) @ Vt
    if np.linalg.norm(src_c) < 1e-12:
        scale = 1.0
    else:
        scale = float(np.sum(d * np.array([1.0, 1.0, sign])) /
                      np.sum(src_c ** 2))
    t = mu_dst - scale * R @ mu_src
    return Similarity(scale=scale, R=R, t=t)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> (3, 3) rotation (column-vector convention)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def align_pose_sequences(dvlt_poses7: np.ndarray,
                         slam_poses7: np.ndarray,
                         sample_indices: np.ndarray) -> Tuple[Similarity, AlignStats]:
    """Register DVLT camera poses to SLAM poses via camera-center Umeyama.

    Args:
        dvlt_poses7: (S, 7) [t xyz, q wxyz] camera-to-world from DVLT (arbitrary scale).
        slam_poses7: (S, 7) [t xyz, q wxyz] metric camera-to-world of DVLT's views.
        sample_indices: raw frame indices of the DVLT views (for stats only).

    Returns a :class:`Similarity` and alignment statistics.  The rotation
    consistency is checked on the camera axes: valid alignment should keep the
    mean per-pair rotation error well under a few degrees.
    """
    dvlt = np.asarray(dvlt_poses7, dtype=np.float64)
    slam = np.asarray(slam_poses7, dtype=np.float64)
    if len(dvlt) != len(slam):
        raise ValueError(f"pose count mismatch: {len(dvlt)} vs {len(slam)}")

    # Rotations of both families (R is c2w: X_w = R @ X_c + t).
    R_dvlt = np.stack([_quat_to_mat(q) for q in dvlt[:, 3:7]])
    R_slam = np.stack([_quat_to_mat(q) for q in slam[:, 3:7]])

    sim = umeyama(dvlt[:, :3], slam[:, :3])
    # Consistency: R_dvlt should satisfy s*R @ R_dvlt ≈ R_slam (up to scale on
    # the translation only, so compare the rotation parts directly).
    R_pred = sim.R @ R_dvlt
    angles = []
    for rp, rt in zip(R_pred, R_slam):
        rel = rt.T @ rp
        angles.append(float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))))
    rot_err = float(np.mean(angles)) if angles else np.nan

    resid = sim.apply_origins(dvlt[:, :3]) - slam[:, :3]
    origin_std = float(np.std(slam[:, :3], axis=0).mean())
    stats = AlignStats(
        scale=float(sim.scale),
        rot_err_deg=rot_err,
        rmse=float(np.sqrt(np.mean(resid ** 2))),
        reflected=bool(np.linalg.det(sim.R) < 0),
        n_pairs=len(dvlt),
        origin_std=origin_std,
        detail={
            "rot_err_max_deg": float(np.max(angles)) if angles else None,
            "sample_indices": np.asarray(sample_indices, dtype=np.int64).tolist()
            if len(sample_indices) < 4096 else "truncated",
            "n_samples": int(len(sample_indices)),
        },
    )
    return sim, stats


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def project_points_to_depth(points_w: np.ndarray, c2w_t: np.ndarray, K: np.ndarray,
                            height: int, width: int,
                            max_pts: int = 4_000_000) -> Tuple[np.ndarray, np.ndarray]:
    """Project world points into one camera and rasterize a depth map.

    Args:
        points_w: (M, 3) metric world points (already registered).
        c2w_t: (7,) [t xyz, q wxyz] camera-to-world of the target frame.
        K: (3, 3) pinhole intrinsics (retarget-compatible).
        height, width: output depth map size.

    Returns ``(depth, valid)``: depth (H, W) float32 in meters with NaN where
    no point projects (holes), and a boolean mask of rasterized pixels.
    """
    pts = np.asarray(points_w, dtype=np.float64)
    if len(pts) == 0:
        return (np.full((height, width), np.nan, np.float32),
                np.zeros((height, width), dtype=bool))
    if len(pts) > max_pts:
        sel = np.random.default_rng(0).choice(len(pts), size=max_pts, replace=False)
        pts = pts[sel]

    R = _quat_to_mat(c2w_t[3:7])
    t = c2w_t[:3]
    # X_cam = R^T (X_w - t)
    cam = (pts - t) @ R
    z = cam[:, 2]
    keep = z > 0.05  # meters; drop points behind / too close to the camera
    if not np.any(keep):
        return (np.full((height, width), np.nan, np.float32),
                np.zeros((height, width), dtype=bool))
    cam = cam[keep]
    z = z[keep]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = np.round(fx * cam[:, 0] / z + cx).astype(np.int64)
    v = np.round(fy * cam[:, 1] / z + cy).astype(np.int64)
    inb = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inb], v[inb], z[inb]
    depth = np.full((height, width), np.nan, np.float32)
    if len(z) == 0:
        return depth, np.zeros((height, width), dtype=bool)
    # Nearest surface wins: sort by z ascending, then the first occurrence of
    # every pixel is its minimum depth.
    order = np.argsort(z, kind="stable")
    u, v, z = u[order], v[order], z[order]
    flat = v * width + u
    _, first_idx = np.unique(flat, return_index=True)
    depth.ravel()[flat[first_idx]] = z[first_idx]
    valid = ~np.isnan(depth)
    return depth, valid


def unproject_slam_world(depth: np.ndarray, K: np.ndarray,
                         slam_poses7: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Unproject per-view camera depth into the metric SLAM world frame.

    DVLT's *own* camera poses carry an unobservable gauge (overall rotation
    drift for monocular video), so its ``world_points`` cannot be registered
    reliably.  The per-view **depth is pose-independent**, however, so we
    unproject it with the metric SLAM poses of the same frames:

        X_world = scale * R_slam @ (z_dvlt * K^{-1} [u, v, 1]) + t_slam

    ``scale`` converts DVLT's depth units into SLAM meters (a single scalar;
    see ``trajectory_scale``).  The rotation gauge never enters because the
    SLAM rotation is used directly.

    Args:
        depth: (S, h, w) camera-z depth in DVLT units.
        K: (S, 3, 3) per-view intrinsics (DVLT's predicted K, matching ``depth``).
        slam_poses7: (S, 7) [t xyz, q wxyz] metric camera-to-world.
        scale: DVLT-unit -> meter conversion factor.

    Returns (S, h, w, 3) world points in the SLAM metric frame.
    """
    depth = np.asarray(depth, dtype=np.float64) * float(scale)
    S, h, w = depth.shape
    K = np.asarray(K, dtype=np.float64)
    out = np.empty((S, h, w, 3), dtype=np.float64)
    u = np.arange(w, dtype=np.float64)
    v = np.arange(h, dtype=np.float64)
    for s in range(S):
        R = _quat_to_mat(slam_poses7[s, 3:7])
        t = slam_poses7[s, :3]
        fx, fy = K[s, 0, 0], K[s, 1, 1]
        cx, cy = K[s, 0, 2], K[s, 1, 2]
        z = depth[s]
        xc = (u[None, :] - cx) / fx * z
        yc = (v[:, None] - cy) / fy * z
        pts = np.stack([xc, yc, z], axis=-1)  # (h, w, 3)
        out[s] = pts @ R.T + t
    return out


def trajectory_scale(dvlt_poses7: np.ndarray, slam_poses7: np.ndarray) -> float:
    """Robust DVLT-unit -> meter scale from camera-trajectory length ratios.

    Unlike a full Umeyama registration, a length ratio is invariant to the
    monocular rotation gauge, so it stays well-conditioned even when the DVLT
    pose rotation drifts relative to SLAM.  Falls back to the Umeyama scale
    when the trajectory is degenerate (static camera).
    """
    dvlt = np.asarray(dvlt_poses7, dtype=np.float64)
    slam = np.asarray(slam_poses7, dtype=np.float64)
    d_dvlt = np.linalg.norm(np.diff(dvlt[:, :3], axis=0), axis=1).sum()
    d_slam = np.linalg.norm(np.diff(slam[:, :3], axis=0), axis=1).sum()
    if d_dvlt < 1e-9:
        return umeyama(dvlt[:, :3], slam[:, :3]).scale
    return float(d_slam / d_dvlt)


def estimate_scale_from_hand_keypoints(dvlt_depth: np.ndarray, dvlt_K: np.ndarray,
                                       slam_poses7: np.ndarray,
                                       kp_world: np.ndarray) -> Dict:
    """Fit ``z_slam_metric = scale * z_dvlt`` from hand keypoints (pure ratio).

    The zarr hand keypoints live in the SLAM world frame, so projecting them
    with the SLAM cameras gives per-view **metric** depths at known pixels of
    the DVLT depth grid.  Both quantities are camera-z distances to the same
    physical surface, so their ratio is a single scalar (no offset).  A
    linear-with-offset fit is degenerate here (the hand-region depth is nearly
    constant), so the median ratio with MAD trimming is used instead.  The
    pixel mapping uses DVLT's *own* predicted intrinsics, which are consistent
    with its depth grid.

    Args:
        dvlt_depth: (S, h, w) DVLT camera-z depth.
        dvlt_K: (S, 3, 3) DVLT intrinsics matching ``dvlt_depth``.
        slam_poses7: (S, 7) metric camera-to-world of the sampled frames.
        kp_world: (S, K, 3) world-frame keypoints (e.g. MANO 21 points).

    Returns dict with scale and fit stats (scale defaults to 1.0 if
    degenerate; ``n_pairs`` reports how many samples were used).
    """
    depth = np.asarray(dvlt_depth, dtype=np.float64)
    K = np.asarray(dvlt_K, dtype=np.float64)
    poses = np.asarray(slam_poses7, dtype=np.float64)
    kps = np.asarray(kp_world, dtype=np.float64)
    S, h, w = depth.shape
    ratios = []
    for s in range(min(S, len(poses))):
        R = _quat_to_mat(poses[s, 3:7])
        t = poses[s, :3]
        fx, fy = K[s, 0, 0], K[s, 1, 1]
        cx, cy = K[s, 0, 2], K[s, 1, 2]
        for kp in kps[s]:
            kp = np.asarray(kp, dtype=np.float64)
            if not np.isfinite(kp).all():
                continue
            xc, yc, zc = (kp - t) @ R  # X_cam = R^T (X_w - t)
            if zc < 0.005:
                continue
            u = int(round(fx * xc / zc + cx))
            v = int(round(fy * yc / zc + cy))
            if not (0 <= u < w and 0 <= v < h):
                continue
            d = depth[s, v, u]
            if not np.isfinite(d) or d < 1e-4:
                continue
            ratios.append(zc / d)
    n = len(ratios)
    if n < 8:
        return {"mode": "degenerate", "scale": 1.0, "n_pairs": n,
                "resid_ratio": None}
    ratios = np.asarray(ratios)
    med = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - med)))
    keep = np.abs(ratios - med) < 3 * (1.4826 * mad + 1e-6)
    scale = float(np.median(ratios[keep])) if keep.any() else med
    resid = float(np.median(np.abs(ratios[keep] / scale - 1.0))) if keep.any() else None
    if not (0.001 < abs(scale) < 1000.0):
        return {"mode": "degenerate", "scale": 1.0, "n_pairs": n,
                "resid_ratio": resid}
    return {"mode": "keypoints", "scale": scale, "n_pairs": n,
            "resid_ratio": resid}


def fill_depth_holes(depth: np.ndarray, iterations: int = 4) -> np.ndarray:
    """Fill NaN holes with the nearest valid pixel's depth (bounded EDT).

    A single distance-transform pass fills holes up to ``iterations`` pixels
    wide with the nearest valid depth (equivalent to the previous iterative
    4-neighbor growth, but ~30x faster).  Remaining holes (e.g. outside the
    reconstructed frustum) stay NaN, which the retarget compositor already
    treats as "no occluder" (alpha=1).
    """
    from scipy import ndimage
    d = np.asarray(depth, dtype=np.float32)
    valid = ~np.isnan(d)
    if valid.all() or not valid.any():
        return d
    iters = max(1, int(iterations))
    # Distance (in pixels) to the nearest valid pixel and its index.
    dist, (iy, ix) = ndimage.distance_transform_edt(
        ~valid, return_distances=True, return_indices=True)
    within = dist <= iters
    if not within.any():
        return d
    out = d.copy()
    out[within] = d[iy[within], ix[within]]
    return out