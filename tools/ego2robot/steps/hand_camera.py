# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Camera calibration and positive-depth rigid fitting for video hand tracks."""

from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

def validate_k(value):
    k = np.asarray(value, dtype=np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1])
            or abs(k[0, 1]) > 1e-8 or abs(k[1, 0]) > 1e-8):
        raise ValueError("intrinsics must be a finite, zero-skew 3x3 K with positive fx/fy")
    return k


def assumed_camera(width, height):
    return {"fl_x": float(max(width, height)), "fl_y": float(max(width, height)),
            "cx": width / 2., "cy": height / 2., "w": width, "h": height,
            "estimated": False, "source": "assumed:max(width,height)"}


def stabilize_camera_estimate(camera, prior_ratio=0.4, max_relative_deviation=None):
    """Explicit legacy camera-specific resolution prior, disabled by default.

    Absence of reference inputs does not make this camera-specific prior
    generalizable. Callers must explicitly opt in with a nonnegative gate.
    """
    if max_relative_deviation is None or max_relative_deviation < 0:
        return dict(camera)
    if not 0 < prior_ratio:
        raise ValueError("focal prior ratio must be positive and deviation nonnegative")
    out = dict(camera)
    width = float(camera["w"])
    focal = float(np.mean([camera["fl_x"], camera["fl_y"]]))
    prior = float(width * prior_ratio)
    relative_deviation = abs(focal - prior) / max(prior, 1e-8)
    out["focal_prior_ratio"] = float(prior_ratio)
    out["focal_prior_px"] = prior
    out["focal_relative_deviation"] = relative_deviation
    out["focal_prior_applied"] = False
    if relative_deviation > max_relative_deviation:
        out["raw_fl_x"] = float(camera["fl_x"])
        out["raw_fl_y"] = float(camera["fl_y"])
        out["fl_x"] = prior
        out["fl_y"] = prior
        out["source"] = f"{camera.get('source', 'video')}:resolution_prior"
        out["focal_prior_applied"] = True
    camera_matrix(out)
    return out


def estimate_camera(frames, repo=None, weights="pinhole", samples=8, device=None):
    """Estimate a shared pinhole camera from sampled RGB frames only.

    The GeoCalib implementation bundled by ViPE can be imported independently
    of its SLAM stack. No dataset, calibration JSON, Zarr or reference poses
    are accepted by this interface.
    """
    import sys
    import torch
    if samples < 1 or not frames:
        raise ValueError("camera estimation needs positive samples and at least one RGB frame")
    if repo:
        location = Path(repo).resolve()
        candidates = [location, location / "vipe/priors"]
        module_root = next((p for p in candidates if (p / "geocalib").is_dir()), None)
        if module_root is None:
            raise FileNotFoundError(f"GeoCalib module not found in {location}")
        sys.path.insert(0, str(module_root))
    try:
        from geocalib import GeoCalib
    except ImportError as exc:
        raise RuntimeError(
            "GeoCalib is unavailable: set --geocalib_repo to the ViPE checkout "
            "(or its vipe/priors directory) and install kornia") from exc
    model = GeoCalib(weights=weights).to(
        device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
    indices = np.unique(np.linspace(0, len(frames) - 1, min(samples, len(frames))).astype(int))
    if len(indices) == 0:
        raise ValueError("camera estimation needs at least one RGB frame")
    # Bound memory for the joint optimizer while retaining a shared focal
    # estimate across scene views, as in ViPE's pinhole initialization.
    rgb = torch.stack([torch.from_numpy(np.ascontiguousarray(frames[i][..., ::-1]))
                       .permute(2, 0, 1).float() / 255 for i in indices]).to(next(model.parameters()).device)
    with torch.no_grad():
        result = model.calibrate(rgb, shared_intrinsics=True)
    cam = result["camera"]
    focal = cam.f.detach().cpu().numpy()
    center = cam.c.detach().cpu().numpy()
    height, width = frames[0].shape[:2]
    camera = {"fl_x": float(np.median(focal[:, 0])), "fl_y": float(np.median(focal[:, 1])),
              "cx": float(np.median(center[:, 0])), "cy": float(np.median(center[:, 1])),
              "w": width, "h": height, "estimated": True,
              "source": "video:GeoCalib", "sample_frame_indices": indices.tolist(),
              "focal_uncertainty_px": float(result["focal_uncertainty"].median().item())}
    camera_matrix(camera)
    del model, rgb, result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return camera


def camera_matrix(camera):
    return validate_k([[camera["fl_x"], 0, camera["cx"]],
                       [0, camera["fl_y"], camera["cy"]], [0, 0, 1]])


def project(xyz, k):
    xyz = np.asarray(xyz, dtype=np.float64)
    return (xyz @ k.T)[..., :2] / xyz[..., 2:]


def fit_camera_pose(points, uv, k, method="pnp"):
    """Fit predicted shape to predicted 2D with fixed metric scale.

    Returns corrected joints, a wrist-centered rotation, wrist translation,
    and whether a PnP correction passed the positive-depth/reprojection gate.
    No reference 3D is an input.
    """
    k = validate_k(k)
    xyz = np.asarray(points, dtype=np.float64).reshape(21, 3)
    uv = np.asarray(uv, dtype=np.float64).reshape(21, 2)
    shape = xyz - xyz[:1]
    keep = np.isfinite(shape).all(1) & np.isfinite(uv).all(1)
    if keep.sum() < 6 or np.linalg.matrix_rank(shape[keep] - shape[keep].mean(0)) < 2:
        raise ValueError("camera fitting requires at least six non-collinear correspondences")
    p, obs = shape[keep], uv[keep]
    norm = (obs - k[:2, 2]) / [k[0, 0], k[1, 1]]
    a = np.zeros((2 * len(p), 3))
    a[0::2, 0], a[1::2, 1] = 1, 1
    a[0::2, 2], a[1::2, 2] = -norm[:, 0], -norm[:, 1]
    b = (norm * p[:, 2:] - p[:, :2]).reshape(-1)
    initial = np.linalg.lstsq(a, b, rcond=None)[0]
    lower = max(.02 - float(np.min(shape[:, 2])), .02)
    if initial[2] <= lower:
        depth = max(float(np.median(
            np.diag(k)[:2] * np.ptp(p[:, :2], axis=0)
            / np.maximum(np.ptp(obs, axis=0), 1))), lower + .05)
        xy = (obs.mean(0) - k[:2, 2]) * depth / np.diag(k)[:2] - p[:, :2].mean(0)
        initial = np.r_[xy, depth]

    def residual(t):
        return (project(p + t, k) - obs).reshape(-1)

    def jacobian(t):
        x = p + t
        z = x[:, 2]
        j = np.zeros((2 * len(p), 3))
        j[0::2, 0], j[1::2, 1] = k[0, 0] / z, k[1, 1] / z
        j[0::2, 2] = -k[0, 0] * x[:, 0] / z**2
        j[1::2, 2] = -k[1, 1] * x[:, 1] / z**2
        return j

    trans = least_squares(residual, initial, jac=jacobian,
                          bounds=([-np.inf, -np.inf, lower], [np.inf] * 3),
                          max_nfev=40).x
    corrected, rotation, accepted = shape + trans, np.eye(3), False
    if method in {"pnp", "pnp_ransac"}:
        try:
            evaluation_ids = np.arange(len(p))
            if method == "pnp_ransac":
                # A few unstable fingertip pixels should not rotate the whole
                # hand. RANSAC rejects such correspondences before LM refinement.
                ok, r, t, inliers = cv2.solvePnPRansac(
                    np.ascontiguousarray(p), np.ascontiguousarray(obs), k, None,
                    rvec=np.zeros((3, 1)), tvec=trans.reshape(3, 1).copy(),
                    useExtrinsicGuess=True, iterationsCount=100,
                    reprojectionError=4., confidence=.999,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                if ok and inliers is not None and len(inliers) >= 6:
                    ids = inliers.ravel()
                    evaluation_ids = ids
                    r, t = cv2.solvePnPRefineLM(
                        np.ascontiguousarray(p[ids]),
                        np.ascontiguousarray(obs[ids]), k, None, r, t)
                else:
                    ok, r, t = False, np.zeros((3, 1)), trans.reshape(3, 1)
            else:
                ok, r, t = cv2.solvePnP(
                    np.ascontiguousarray(p), np.ascontiguousarray(obs), k, None,
                    rvec=np.zeros((3, 1)), tvec=trans.reshape(3, 1).copy(),
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            candidate_rotation = cv2.Rodrigues(r)[0]
            candidate = shape @ candidate_rotation.T + t.reshape(3)
            before_residual = np.linalg.norm(
                project(corrected[keep], k)[evaluation_ids] - obs[evaluation_ids],
                axis=1)
            after_residual = np.linalg.norm(
                project(candidate[keep], k)[evaluation_ids] - obs[evaluation_ids],
                axis=1)
            before = np.median(before_residual ** 2)
            after = np.median(after_residual ** 2)
            if (ok and np.isfinite(candidate).all() and candidate[:, 2].min() > .02
                    and after <= before + 1e-4):
                corrected, rotation, trans = candidate, candidate_rotation, t.reshape(3)
                accepted = True
        except cv2.error:
            pass
    elif method != "translation":
        raise ValueError(f"unknown camera fit method: {method}")
    return corrected.astype(np.float32), rotation, trans, accepted


def calibrate_predictions(predictions, camera, method="pnp"):
    """Update joints AND MANO initialization under the same rigid correction."""
    k = camera_matrix(camera)
    stats = {"fitted": 0, "pnp_accepted": 0, "without_2d": 0, "invalid": 0}
    for item in predictions:
        if not np.isfinite(item["score"]) or item["score"] <= 0:
            item["score"] = 0.
            continue
        extra = item.get("_wilor_extra", {})
        obs = item.get("keypoints_2d", extra.get("keypoints_2d"))
        if obs is None:
            if camera["estimated"]:
                raise ValueError("calibrated reconstruction requires full-image keypoints_2d")
            stats["without_2d"] += 1
            continue
        obs = np.asarray(obs, dtype=np.float32)
        if obs.size == 42:
            obs = obs.reshape(21, 2)
        if obs.shape != (21, 2) or not np.isfinite(obs).all():
            item["score"] = 0.
            stats["invalid"] += 1
            continue
        old = np.asarray(item["keypoints"], np.float32).reshape(21, 3)
        try:
            fitted, correction, wrist, accepted = fit_camera_pose(old, obs, k, method)
        except ValueError:
            item["score"] = 0.0
            stats["invalid"] += 1
            continue
        if "global_orient" in extra and "cam_trans" in extra:
            root = np.asarray(extra["global_orient"]).squeeze()
            root = root if root.shape == (3, 3) else cv2.Rodrigues(root.reshape(3))[0]
            mirror = np.diag([1 if float(item["right"]) > .5 else -1, 1, 1])
            # MANO rotates around its shape-dependent wrist. Mirror the
            # correction into canonical right-hand space, keeping that pivot.
            extra["global_orient"] = (mirror @ correction @ mirror @ root).astype(np.float32)
            extra["cam_trans"] = (np.asarray(extra["cam_trans"]).reshape(3)
                                  + wrist - old[0]).astype(np.float32)
        item["keypoints"] = fitted
        item["keypoints_2d"] = np.asarray(obs, np.float32)
        stats["fitted"] += 1
        stats["pnp_accepted"] += int(accepted)
    return stats
