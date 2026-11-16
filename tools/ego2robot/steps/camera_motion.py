# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""RGB-only camera motion: fixed-K COLMAP, background PnP, metric depth scale.

GeoCalib supplies the initial K. An opt-in video self-calibration stage can
propose a shared focal length from masked feature geometry, with disjoint
temporal validation. Dynamic hand pixels are excluded using WiLoR observations.
A metric depth backend estimates shared scale without hand-size or GT fits.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .camera_geometry import complete_c2w, validate_trajectory
    from .hand_camera import camera_matrix
except ImportError:
    from camera_geometry import complete_c2w, validate_trajectory
    from hand_camera import camera_matrix


def background_mask(frame, predictions, t, margin=25, temporal_radius=2):
    """Mask hand hulls and their local temporal neighborhood, using RGB outputs."""
    mask = np.full(frame.shape[:2], 255, np.uint8)
    foreground = np.zeros_like(mask)
    for item in predictions:
        if abs(int(item["frame"]) - t) > temporal_radius or item["score"] <= 0:
            continue
        uv = item.get("keypoints_2d", item.get("_wilor_extra", {}).get("keypoints_2d"))
        if uv is None:
            continue
        uv = np.asarray(uv).reshape(-1, 2)
        if len(uv) < 3 or not np.isfinite(uv).all():
            continue
        hull = cv2.convexHull(np.clip(np.rint(uv), -10000, 10000).astype(np.int32))
        cv2.fillConvexPoly(foreground, hull, 255)
    if margin:
        foreground = cv2.dilate(foreground, np.ones((2 * margin + 1,) * 2, np.uint8))
    mask[foreground > 0] = 0
    return mask


def load_metric_depth(repo, model_path, device=None, backend="unidepth"):
    """Load one of the video-only metric depth backends.

    ``unidepth`` is loaded from the vendored ViPE source. ``da3_metric`` uses
    the official Depth Anything 3 package, optionally from ``repo`` when it is
    not installed in the active environment. Keeping the import here avoids
    making either depth package a mandatory import for static-camera mode.
    """
    import torch

    if backend not in {"unidepth", "da3_metric"}:
        raise ValueError(f"unknown metric depth backend: {backend}")
    if not model_path:
        flag = "--unidepth_model" if backend == "unidepth" else "--da3_metric_model"
        raise ValueError(f"COLMAP metric motion needs {flag}")
    if backend == "da3_metric":
        if repo:
            source = Path(repo)
            src = source / "src"
            if (src / "depth_anything_3").is_dir() and str(src) not in sys.path:
                sys.path.insert(0, str(src))
        try:
            module = importlib.import_module("depth_anything_3.api")
        except ImportError as exc:
            raise ImportError(
                "DA3Metric-Large requires the official depth-anything-3 package; "
                "install it with `pip install -e /path/to/depth-anything-3 "
                "--no-deps` or pass --da3_repo"
            ) from exc
        if not (Path(model_path) / "config.json").is_file():
            raise FileNotFoundError(f"local DA3Metric checkpoint not found: {model_path}")
        model = module.DepthAnything3.from_pretrained(str(model_path)).to(
            device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
        return {"backend": backend, "model": model}
    if not repo:
        raise ValueError("UniDepthV2 requires --unidepth_repo")
    root = Path(repo)
    candidates = [root, root / "vipe/priors/depth/unidepth"]
    location = next((p for p in candidates if (p / "models/unidepthv2/unidepthv2.py").is_file()), None)
    if location is None:
        raise FileNotFoundError(f"UniDepthV2 source not found: {root}")
    if not (Path(model_path) / "config.json").is_file():
        raise FileNotFoundError(f"local UniDepthV2 checkpoint not found: {model_path}")
    name = "_ego2robot_unidepth"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(location.resolve())]
        sys.modules[name] = package
    # The vendored encoder imports this pure-PyTorch attention wrapper by its
    # absolute name. Load only that file, avoiding ViPE's CUDA JIT __init__.
    attention_name = "vipe.ext.xformers"
    temporary_attention = attention_name not in sys.modules
    if temporary_attention:
        wrapper = location.resolve().parents[2] / "ext/xformers.py"
        if not wrapper.is_file():
            raise FileNotFoundError(f"vendored attention wrapper not found: {wrapper}")
        spec = importlib.util.spec_from_file_location(attention_name, wrapper)
        attention = importlib.util.module_from_spec(spec)
        sys.modules[attention_name] = attention
        spec.loader.exec_module(attention)
    try:
        module = importlib.import_module(name + ".models.unidepthv2.unidepthv2")
    finally:
        if temporary_attention:
            sys.modules.pop(attention_name, None)
    model = module.UniDepthV2.from_pretrained(str(model_path)).to(
        device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
    model.resolution_level = 0
    return {"backend": backend, "model": model}


def _resize_depth_output(values, shape):
    """Restore a model output to the RGB frame resolution."""
    values = np.asarray(values, dtype=np.float32)
    h, w = shape
    if values.shape == (h, w):
        return values
    if values.ndim != 2:
        raise ValueError(f"metric depth output must be 2-D, got {values.shape}")
    return cv2.resize(values, (w, h), interpolation=cv2.INTER_LINEAR)


def infer_metric_depth(model, frame, k, backend=None, process_res=504):
    import torch

    # Keep the old direct-model interface for unit tests and downstream users.
    if isinstance(model, dict):
        backend = model["backend"]
        model = model["model"]
    backend = backend or "unidepth"
    if backend == "da3_metric":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            prediction = model.inference(
                [rgb], process_res=int(process_res),
                process_res_method="upper_bound_resize",
                export_format="mini_npz")
        raw_depth = np.asarray(prediction.depth[0], dtype=np.float32)
        raw_conf = (np.ones_like(raw_depth) if prediction.conf is None else
                    np.asarray(prediction.conf[0], dtype=np.float32))
        # The DA3 metric checkpoint predicts canonical metric depth. Its
        # official conversion is focal_processed * net_output / 300. The
        # focal length must be scaled to the network's resized image before
        # the map is restored to the original video resolution.
        proc_h, proc_w = raw_depth.shape
        orig_h, orig_w = frame.shape[:2]
        focal_processed = .5 * (
            float(k[0, 0]) * proc_w / orig_w
            + float(k[1, 1]) * proc_h / orig_h
        )
        depth = raw_depth * (focal_processed / 300.0)
        return (_resize_depth_output(depth, (orig_h, orig_w)),
                _resize_depth_output(raw_conf, (orig_h, orig_w)))
    if backend != "unidepth":
        raise ValueError(f"unknown metric depth backend: {backend}")
    rgb = torch.from_numpy(np.ascontiguousarray(frame[..., ::-1])).permute(2, 0, 1)
    with torch.inference_mode():
        prediction = model.infer(rgb, torch.as_tensor(k, dtype=torch.float32)[None])
    depth = prediction["depth"].squeeze().float().cpu().numpy()
    confidence = prediction["confidence"].squeeze().float().cpu().numpy()
    if depth.shape != frame.shape[:2] or confidence.shape != depth.shape:
        raise ValueError("UniDepth must return original-resolution depth/confidence")
    return depth, confidence


def _sample_depth_patch(depth, confidence, uv, joint_ids, patch_size=7,
                        confidence_threshold=None):
    """Sample robust depth values around full-image hand joints."""
    depth = np.asarray(depth, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    h, w = depth.shape
    half = max(0, int(patch_size) // 2)
    values = np.full(len(joint_ids), np.nan, dtype=np.float64)
    for i, jid in enumerate(joint_ids):
        x, y = np.asarray(uv[jid], dtype=np.float64)
        if not np.isfinite([x, y]).all():
            continue
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue
        x0, x1 = max(0, xi - half), min(w, xi + half + 1)
        y0, y1 = max(0, yi - half), min(h, yi + half + 1)
        d = depth[y0:y1, x0:x1]
        c = confidence[y0:y1, x0:x1]
        valid = np.isfinite(d) & (d > .05) & np.isfinite(c)
        if confidence_threshold is not None:
            valid &= c >= confidence_threshold
        if valid.any():
            values[i] = float(np.median(d[valid]))
    return values


def validate_hand_depth_options(frame_stride, patch_size, max_mad, max_delta):
    if frame_stride < 1:
        raise ValueError("hand depth frame stride must be positive")
    if patch_size < 1 or patch_size % 2 != 1:
        raise ValueError("hand depth patch size must be positive and odd")
    if not (np.isfinite(max_mad) and max_mad > 0
            and np.isfinite(max_delta) and max_delta > 0):
        raise ValueError("hand depth MAD and correction limits must be finite and positive")


def align_hand_predictions_to_metric_depth(
        frames, predictions, depth_model, camera, metric_depth_backend,
        process_res=504, frame_stride=1, patch_size=7,
        max_translation_mad=.03, max_translation_delta=.15):
    """Correct hand absolute translation using video-only metric depth.

    WiLoR/Dyn-HaMR supplies the MANO-relative shape and full-image 2D joints.
    Metric depth supplies a 3D location for several thick joints (wrist and
    MCPs).  The translation is the robust median of
    ``backprojected_depth_joint - relative_joint``.  This is deliberately
    independent of ground-truth data and does not fit a camera trajectory.

    The input list is mutated in place so an optional Dyn-HaMR pass receives
    the corrected initialization.  A compact quality summary is returned.
    """
    validate_hand_depth_options(frame_stride, patch_size, max_translation_mad,
                                max_translation_delta)
    k = camera_matrix(camera)
    by_frame = {}
    for item in predictions:
        t = int(item.get("frame", -1))
        if 0 <= t < len(frames) and float(item.get("score", 0.)) > 0:
            by_frame.setdefault(t, []).append(item)
    all_frames = np.asarray(sorted(by_frame), dtype=np.int64)
    selected = all_frames[::int(frame_stride)]
    if len(selected) and selected[-1] != all_frames[-1]:
        selected = np.r_[selected, all_frames[-1]]

    reliable = (0, 5, 9, 13, 17)
    aligned = 0
    considered = 0
    aligned_frames = set()
    deltas, mads = [], []
    reasons = {}
    varying_confidence_frames = 0

    def reject(reason):
        reasons[reason] = reasons.get(reason, 0) + 1

    for t in selected.tolist():
        depth, confidence = infer_metric_depth(
            depth_model, frames[t], k,
            backend=metric_depth_backend, process_res=process_res)
        if depth.shape != frames[t].shape[:2] or confidence.shape != depth.shape:
            raise ValueError("hand metric depth/confidence must match the video resolution")
        finite_conf = confidence[np.isfinite(confidence)]
        if finite_conf.size and float(np.ptp(finite_conf)) > 1e-6:
            varying_confidence_frames += 1
        conf_threshold = (float(np.quantile(finite_conf, .4))
                          if finite_conf.size else None)
        for item in by_frame[t]:
            considered += 1
            points = np.asarray(item["keypoints"], dtype=np.float64).reshape(21, 3)
            uv = item.get("keypoints_2d")
            if uv is None:
                uv = item.get("_wilor_extra", {}).get("keypoints_2d")
            if uv is None:
                reject("missing_2d")
                continue
            uv = np.asarray(uv, dtype=np.float64).reshape(21, 2)
            if not (np.isfinite(points).all() and np.isfinite(uv).all()):
                reject("nonfinite_observation")
                continue
            relative = points - points[:1]
            joint_ids = list(reliable)
            sampled = _sample_depth_patch(
                depth, confidence, uv, joint_ids, patch_size, conf_threshold)
            valid = np.isfinite(sampled)
            if valid.sum() < 3:
                joint_ids = list(range(21))
                sampled = _sample_depth_patch(
                    depth, confidence, uv, joint_ids, patch_size, conf_threshold)
                valid = np.isfinite(sampled)
            if valid.sum() < 3:
                reject("insufficient_depth")
                continue
            ids = np.asarray(joint_ids, dtype=np.int64)[valid]
            d = sampled[valid]
            xyz = np.stack([
                (uv[ids, 0] - k[0, 2]) * d / k[0, 0],
                (uv[ids, 1] - k[1, 2]) * d / k[1, 1],
                d,
            ], axis=1)
            translations = xyz - relative[ids]
            center = np.median(translations, axis=0)
            residual = np.linalg.norm(translations - center, axis=1)
            # Median distance to the multivariate median. A constant but
            # large residual must not pass as a zero scalar residual MAD.
            mad = float(np.median(residual))
            delta = center - points[0]
            delta_norm = float(np.linalg.norm(delta))
            if mad > max_translation_mad:
                reject("joint_disagreement")
                continue
            if delta_norm > max_translation_delta:
                reject("large_correction")
                continue
            candidate = relative + center
            if candidate[:, 2].min() <= .05 or points[:, 2].min() <= .05:
                reject("nonpositive_depth")
                continue
            def reprojection_error(xyz):
                pixels = xyz @ k.T
                pixels = pixels[:, :2] / pixels[:, 2:]
                return float(np.median(np.linalg.norm(pixels - uv, axis=1)))
            before = reprojection_error(points)
            after = reprojection_error(candidate)
            if after > 12. or after > before + 3.:
                reject("reprojection")
                continue
            item["keypoints"] = candidate.astype(np.float32)
            extra = item.get("_wilor_extra")
            if isinstance(extra, dict) and extra.get("cam_trans") is not None:
                extra["cam_trans"] = (
                    np.asarray(extra["cam_trans"], dtype=np.float64).reshape(3)
                    + delta).astype(np.float32)
            aligned += 1
            aligned_frames.add(t)
            deltas.append(delta_norm * 1000.)
            mads.append(mad * 1000.)

    return dict(
        enabled=True, frames_considered=int(len(selected)),
        frames_aligned=int(len(aligned_frames)),
        hands_considered=int(considered), hands_aligned=int(aligned),
        median_translation_delta_mm=float(np.median(deltas)) if deltas else None,
        median_translation_mad_mm=float(np.median(mads)) if mads else None,
        max_translation_delta_mm=float(np.max(deltas)) if deltas else None,
        rejected=int(sum(reasons.values())), rejection_reasons=reasons,
        frame_stride=int(frame_stride),
        patch_size=int(patch_size),
        max_translation_mad_mm=float(max_translation_mad * 1000.),
        max_correction_mm=float(max_translation_delta * 1000.),
        max_reprojection_px=12., max_reprojection_increase_px=3.,
        varying_confidence_frames=int(varying_confidence_frames),
        confidence_informative=bool(varying_confidence_frames),
        metric_depth_backend=str(metric_depth_backend),
        reference_data_used=False)


def robust_metric_scale(frame_ratios, min_frames=3, source="UniDepthV2"):
    """Estimate scale from temporally balanced, robust frame consensus.

    Each valid frame first contributes one robust median. The sequence is then
    split into temporal blocks and each block contributes one median, preventing
    a long locally biased section from dominating the global scale.
    """
    medians = []
    counts = []
    for ratios in frame_ratios:
        r = np.asarray(ratios, dtype=np.float64)
        r = r[np.isfinite(r) & (r > 0)]
        if len(r) < 20:
            continue
        log_r = np.log(r)
        center = np.median(log_r)
        mad = np.median(np.abs(log_r - center))
        keep = np.abs(log_r - center) <= max(3 * 1.4826 * mad, .05)
        if keep.sum() >= 20:
            medians.append(float(np.median(r[keep])))
            counts.append(int(keep.sum()))
    if len(medians) < min_frames:
        raise RuntimeError("too few static background/depth correspondences for metric camera scale")
    values = np.asarray(medians, dtype=np.float64)
    block_count = min(3, len(values))
    blocks = np.array_split(values, block_count)
    block_scales = [float(np.median(block)) for block in blocks if len(block)]
    scale = float(np.median(block_scales))
    spread = float(np.median(np.abs(np.asarray(block_scales) - scale)) / scale)
    frame_spread = float(np.median(np.abs(values - scale)) / scale)
    if spread > .35:
        raise RuntimeError(
            f"inconsistent temporally blocked metric depth scale "
            f"(relative MAD={spread:.3f})")
    return scale, dict(
        source=f"video:{source}/static_COLMAP_points",
        meters_per_native_unit=scale, per_frame_scales=medians,
        inlier_counts=counts, relative_mad=spread,
        frame_relative_mad=frame_spread, temporal_block_scales=block_scales,
        temporal_block_count=len(block_scales),
        physically_calibrated=False, reference_data_used=False)


def image_map_points(reconstruction, image):
    uv, xyz = [], []
    for point in image.points2D:
        if not point.has_point3D():
            continue
        p = reconstruction.points3D[point.point3D_id]
        if p.error > 2.5 or p.track.length() < 3:
            continue
        uv.append(point.xy)
        xyz.append(p.xyz)
    return np.asarray(uv, np.float32).reshape(-1, 2), np.asarray(xyz, np.float64).reshape(-1, 3)


def depth_scale_observations(model, image, depth, confidence, mask):
    """Background scale samples keyed by video pixel, for fair K comparisons."""
    uv, xyz = image_map_points(model, image)
    w2c = image.cam_from_world().matrix()
    z = (xyz @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]
    h, w = depth.shape
    finite = np.isfinite(uv).all(1) & np.isfinite(z) & (z > 1e-6)
    ids = np.flatnonzero(finite)
    xy = np.rint(uv[ids]).astype(int)
    inside = ((xy[:, 0] >= 0) & (xy[:, 0] < w)
              & (xy[:, 1] >= 0) & (xy[:, 1] < h))
    ids, xy = ids[inside], xy[inside]
    x, y = xy.T
    finite_conf = confidence[np.isfinite(confidence)]
    threshold = np.quantile(finite_conf, .4) if len(finite_conf) else np.inf
    d, c = depth[y, x], confidence[y, x]
    keep = ((mask[y, x] > 0) & np.isfinite(d) & (d > .05)
            & np.isfinite(c) & (c >= threshold))
    # Quantization preserves subpixel SIFT identity across fresh databases.
    return {tuple(np.rint(p * 1000).astype(int)): float(r)
            for p, r in zip(uv[ids][keep], d[keep] / z[ids][keep])}


def compare_intrinsics_candidates(candidates, observations, width, height):
    """Choose a K using depth agreement on identical video observations.

    Reprojection is only a rejection gate: a wrong focal can still fit pixels.
    The independent depth maps are compared on common background SIFT pixels,
    with equal frame weights and a free *video-only* global metric scale.
    The original K wins ties and uncertain comparisons.
    """
    diagnostics = {}
    if len(candidates) != 2:
        raise ValueError("intrinsics comparison needs the original and one alternative")
    names = list(candidates)
    common_frames = sorted(set(observations[names[0]]) & set(observations[names[1]]))
    paired = {name: [] for name in names}
    counts, cells, comparison_frames = [], [], []
    for t in common_frames:
        common = sorted(set(observations[names[0]][t]) & set(observations[names[1]][t]))
        if len(common) < 20:
            continue
        uv = np.asarray(common, float) / 1000.
        grid = np.minimum((uv / [width, height] * [4, 3]).astype(int), [3, 2])
        cell_count = len(np.unique(grid, axis=0))
        if cell_count < 4:
            continue
        counts.append(len(common))
        cells.append(cell_count)
        comparison_frames.append(int(t))
        for name in names:
            paired[name].append(np.asarray([observations[name][t][p] for p in common]))
    for name in names:
        if len(counts) < 3:
            diagnostics[name] = dict(eligible=False, reason="insufficient_common_background")
            continue
        frame_centers = [float(np.median(np.log(r))) for r in paired[name]]
        center = float(np.median(frame_centers))
        # Includes both depth-shape error within views and temporal scale drift.
        score = float(np.mean([np.median(np.abs(np.log(r) - center))
                               for r in paired[name]]))
        try:
            _, scale_info = robust_metric_scale(paired[name])
        except RuntimeError as exc:
            diagnostics[name] = dict(eligible=False, reason=str(exc))
            continue
        diagnostics[name] = dict(
            eligible=True, log_depth_error=score,
            frame_scale_relative_mad=scale_info["relative_mad"],
            comparison_frames=len(counts), common_point_counts=counts,
            comparison_frame_indices=comparison_frames,
            spatial_grid_cells=cells,
            per_frame_log_depth_error=[
                float(np.median(np.abs(np.log(r) - center))) for r in paired[name]])
    original, alternative = names
    chosen, reason = original, "original_retained"
    if all(diagnostics[name]["eligible"] for name in names):
        base, alt = candidates[original], candidates[alternative]
        # Counts of 3D points change with triangulation fragmentation. Assess
        # enough *identical reliable observations* across space and time instead.
        span = ((comparison_frames[-1] - comparison_frames[0])
                / max(common_frames[-1] - common_frames[0], 1))
        improved_fraction = float(np.mean(
            np.asarray(diagnostics[alternative]["per_frame_log_depth_error"])
            < np.asarray(diagnostics[original]["per_frame_log_depth_error"])))
        support = dict(
            median_common_points=float(np.median(counts)),
            p10_common_points=float(np.quantile(counts, .1)),
            median_spatial_grid_cells=float(np.median(cells)),
            supported_frame_fraction=len(counts) / len(common_frames),
            supported_temporal_span_fraction=float(span),
            depth_improved_frame_fraction=improved_fraction,
            sparse_points_ratio=alt["sparse_points"] / max(base["sparse_points"], 1))
        diagnostics[alternative]["background_support"] = support
        gates = dict(
            coverage_preserved=alt["valid_fraction"] >= base["valid_fraction"] - .02,
            registration_preserved=alt["registered_fraction"] >= base["registered_fraction"] - .02,
            reliable_background_support=support["median_common_points"] >= 60
            and support["p10_common_points"] >= 30,
            spatial_support=support["median_spatial_grid_cells"] >= 6,
            temporal_support=support["supported_frame_fraction"] >= .75
            and support["supported_temporal_span_fraction"] >= .6,
            reprojection_acceptable=alt["reprojection_px"] <= max(2., 1.25 * base["reprojection_px"]),
            depth_improved=diagnostics[alternative]["log_depth_error"]
            < .9 * diagnostics[original]["log_depth_error"],
            depth_improved_across_views=improved_fraction >= .6)
        diagnostics[alternative]["gates"] = gates
        if all(gates.values()):
            chosen, reason = alternative, "common_background_depth_improved"
    return chosen, dict(policy="video_consistency_v2", selected=chosen, reason=reason,
                        candidates=diagnostics, reference_data_used=False)


def localize_frame(source_frame, target_frame, uv, xyz, k, initial_w2c, mask,
                   return_tracks=False):
    """Localize an unsampled/unregistered frame against static COLMAP points."""
    if len(uv) < 15:
        return None
    a = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    p = np.asarray(uv, np.float32).reshape(-1, 1, 2)
    q, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, winSize=(21, 21), maxLevel=3)
    if q is None:
        return None
    back, back_ok, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, winSize=(21, 21), maxLevel=3)
    if back is None:
        return None
    q = q.reshape(-1, 2)
    keep = ok.ravel().astype(bool) & back_ok.ravel().astype(bool)
    keep &= np.linalg.norm(back.reshape(-1, 2) - uv, axis=1) < 1.
    h, w = mask.shape
    keep &= np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    ids = np.flatnonzero(keep)
    ids = ids[mask[q[ids, 1].astype(int), q[ids, 0].astype(int)] > 0]
    if len(ids) < 15:
        return None
    rvec = cv2.Rodrigues(np.asarray(initial_w2c[:3, :3], np.float64))[0]
    tvec = np.asarray(initial_w2c[:3, 3], np.float64).reshape(3, 1).copy()
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        xyz[ids], q[ids].astype(np.float64), k, None, rvec, tvec, True,
        iterationsCount=200, reprojectionError=3., confidence=.999,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None or len(inliers) < 15:
        return None
    kept = ids[inliers.ravel()]
    rvec, tvec = cv2.solvePnPRefineLM(xyz[kept], q[kept].astype(np.float64), k, None, rvec, tvec)
    projected = cv2.projectPoints(xyz[kept], rvec, tvec, k, None)[0].reshape(-1, 2)
    error = float(np.median(np.linalg.norm(projected - q[kept], axis=1)))
    rot = cv2.Rodrigues(rvec)[0]
    if error > 2. or np.mean((xyz[kept] @ rot.T + tvec.ravel())[:, 2] > 0) < .95:
        return None
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rot, tvec.ravel()
    result = (np.linalg.inv(pose), len(kept), error)
    if return_tracks:
        return (*result, q[kept], xyz[kept])
    return result


def _pose_difference(first, second, scene_scale):
    """Return rotation degrees and normalized camera-center distance."""
    relative = first[:3, :3].T @ second[:3, :3]
    rotation_deg = float(np.rad2deg(Rotation.from_matrix(relative).magnitude()))
    translation = float(np.linalg.norm(first[:3, 3] - second[:3, 3])
                        / max(float(scene_scale), 1e-6))
    return rotation_deg, translation


def select_localization_candidate(candidates, scene_scale, target_frame=None,
                                  rotation_gate_deg=3.0,
                                  translation_gate=0.08):
    """Select a pose supported by independent map anchors.

    Each candidate is ``(c2w, inlier_count, reprojection_error, anchor,
    tracked_uv, tracked_xyz)``. A single valid anchor remains usable, while
    disagreeing multi-anchor estimates are rejected instead of silently
    choosing the first one.
    """
    if not candidates:
        return None, dict(candidate_count=0, consensus_count=0, rejected=False)
    support = []
    for i, candidate in enumerate(candidates):
        count = 0
        for j, other in enumerate(candidates):
            if i == j:
                continue
            rotation, translation = _pose_difference(
                candidate[0], other[0], scene_scale)
            if rotation <= rotation_gate_deg and translation <= translation_gate:
                count += 1
        support.append(count)
    best_support = max(support)
    eligible = [i for i, value in enumerate(support) if value == best_support]
    if target_frame is None:
        chosen = min(eligible, key=lambda i: (candidates[i][2], -candidates[i][1]))
    else:
        # Temporal locality is the primary tie breaker. A low reprojection
        # error from a distant anchor can be a locally consistent but globally
        # drifting solution; keep the nearest-anchor gauge continuous.
        chosen = min(eligible, key=lambda i: (
            abs(int(candidates[i][3]) - int(target_frame)),
            candidates[i][2], -candidates[i][1]))
    # With multiple independent anchors, no consensus means the pose is
    # ambiguous. A singleton is retained because there is no contradiction.
    rejected = len(candidates) > 1 and best_support == 0
    info = dict(
        candidate_count=len(candidates),
        consensus_count=int(best_support + 1),
        rejected=rejected,
        selected_anchor=int(candidates[chosen][3]),
        selected_inliers=int(candidates[chosen][1]),
        selected_reprojection_px=float(candidates[chosen][2]),
        candidate_anchors=[int(value[3]) for value in candidates],
        candidate_inliers=[int(value[1]) for value in candidates],
        candidate_reprojection_px=[float(value[2]) for value in candidates])
    return (None if rejected else candidates[chosen]), info


def map_with_parallax_recovery(pycolmap, database, images, root, options,
                              min_registered, enabled=False):
    """Retry weak initialization with smaller angles, keeping K and tracks fixed."""
    best, attempts, best_angle = None, [], None
    for angle in ((8., 4., 2.) if enabled else (8.,)):
        options.mapper.init_min_tri_angle = angle
        target = Path(root) / ("sparse" if angle == 8. else f"sparse_angle_{angle:g}")
        models = pycolmap.incremental_mapping(database, images, target, options=options)
        candidate = max(models.values(), key=lambda m: m.num_reg_images()) if models else None
        attempts.append(dict(init_min_tri_angle_deg=angle,
                             registered_keyframes=0 if candidate is None else candidate.num_reg_images()))
        if candidate is not None and (best is None or candidate.num_reg_images() > best.num_reg_images()):
            best, best_angle = candidate, angle
        if best is not None and best.num_reg_images() >= min_registered:
            break
    return best, dict(enabled=enabled, attempts=attempts,
                      selected_init_min_tri_angle_deg=best_angle)


def reconstruct(frames, camera, predictions, root, stride=3, max_keyframes=300,
                threads=4, mask_margin=25, min_registration_fraction=.7,
                low_parallax_recovery=False, self_calibrate=False):
    import pycolmap

    if stride < 1 or max_keyframes < 3 or len(frames) < 3:
        raise ValueError("camera reconstruction needs >=3 frames/keyframes and positive stride")
    root = Path(root)
    # Never resume a stale database belonging to a different video or K.
    if (root / "database.db").exists():
        raise FileExistsError(f"camera workspace already exists: {root}; use a new episode/output")
    images, masks = root / "images", root / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    n = len(frames)
    step = max(stride, int(np.ceil((n - 1) / (max_keyframes - 1))))
    indices = np.unique(np.r_[np.arange(0, n, step), n - 1])
    for t in indices:
        if not cv2.imwrite(str(images / f"{t:06d}.jpg"), frames[t],
                           [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"cannot write camera input frame {t}")
        cv2.imwrite(str(masks / f"{t:06d}.jpg.png"),
                    background_mask(frames[t], predictions, int(t), mask_margin))
    reader = pycolmap.ImageReaderOptions(
        camera_model="PINHOLE", mask_path=str(masks),
        camera_params=",".join(str(camera[k]) for k in ("fl_x", "fl_y", "cx", "cy")))
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = threads
    extraction.sift.max_num_features = 2048
    pycolmap.extract_features(root / "database.db", images,
        camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
        extraction_options=extraction, device=pycolmap.Device.cpu)
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = threads
    pycolmap.match_sequential(root / "database.db", matching_options=matching,
        pairing_options=pycolmap.SequentialPairingOptions(overlap=8, quadratic_overlap=False),
        device=pycolmap.Device.cpu)
    selfcal = dict(enabled=False, accepted=False, reference_data_used=False)
    if self_calibrate:
        try:
            from .camera_selfcal import apply_camera, propose_camera
        except ImportError:
            from camera_selfcal import apply_camera, propose_camera
        proposed, selfcal = propose_camera(root / "database.db", camera)
        if selfcal.get("accepted"):
            pairs_file = root / "selfcal_pairs.txt"
            apply_camera(root / "database.db", proposed, pairs_file)
            camera.update(proposed)
        else:
            selfcal["fallback"] = "initial_camera"
        (root / "selfcal_metadata.json").write_text(json.dumps(selfcal, indent=2))
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = threads
    options.random_seed = 42
    options.min_model_size = 10
    options.multiple_models = True
    options.max_num_models = 3
    options.extract_colors = False
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.ba_refine_sensor_from_rig = False
    options.mapper.abs_pose_refine_focal_length = False
    options.mapper.abs_pose_refine_extra_params = False
    options.mapper.init_min_tri_angle = 8.
    options.mapper.init_min_num_inliers = 60
    # The masked first views can have only a few dozen map observations.
    # Permit 15 verified correspondences, while tightening reprojection error.
    options.mapper.abs_pose_min_num_inliers = 15
    options.mapper.abs_pose_min_inlier_ratio = .15
    options.mapper.abs_pose_max_error = 6.
    model, initialization = map_with_parallax_recovery(
        pycolmap, root / "database.db", images, root, options,
        max(5, min_registration_fraction * len(indices)), low_parallax_recovery)
    (root / "initialization_metadata.json").write_text(json.dumps(initialization, indent=2))
    if model is None:
        raise RuntimeError("COLMAP could not initialize camera motion; inspect background texture/parallax")
    return localize_reconstruction(
        frames, camera, predictions, root, model, indices, step,
        mask_margin, min_registration_fraction, initialization, selfcal)


def localize_reconstruction(frames, camera, predictions, root, model, indices, step,
                            mask_margin=25, min_registration_fraction=.7,
                            initialization=None, self_calibration=None):
    """Track a video-only sparse map at fixed K, also supporting frozen-map replay."""
    import pycolmap

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    k = camera_matrix(camera)
    expected = np.asarray([camera[x] for x in ("fl_x", "fl_y", "cx", "cy")])
    for cam in model.cameras.values():
        if not np.allclose(cam.params, expected, atol=1e-6):
            raise RuntimeError("COLMAP changed the fixed video intrinsics")
    registered = {int(Path(image.name).stem): image for image in model.images.values()}
    if len(registered) < max(5, min_registration_fraction * len(indices)):
        raise RuntimeError(f"camera registration coverage too low: {len(registered)}/{len(indices)}")
    native = {t: np.linalg.inv(image.cam_from_world().matrix().tolist() + [[0, 0, 0, 1]])
              for t, image in registered.items()}
    map_points = {t: image_map_points(model, image) for t, image in registered.items()}
    tracking = {}
    for t in list(range(n)) + list(range(n - 1, -1, -1)):
        if t in native:
            continue
        mask = background_mask(frames[t], predictions, t, mask_margin)
        # Try nearby registered cameras; every localization uses an actual
        # background 2D/3D correspondence, including sequence endpoints.
        anchors = sorted(registered, key=lambda i: abs(i - t))[:4]
        # When the map's final views are occluded, propagate already verified
        # background 3D tracks from nearby localized frames. These remain
        # map-anchored PnP observations, never copied camera poses.
        # Rejected candidates also have diagnostic entries in ``tracking``.
        # Only accepted localizations carry a pose and map correspondences.
        localized = (i for i in tracking if i in native and i in map_points)
        anchors += sorted(localized, key=lambda i: abs(i - t))[:4]
        anchors = list(dict.fromkeys(anchors))
        candidates = []
        scene_scales = [
            np.linalg.norm(points[1], axis=1).mean()
            for points in map_points.values()
            if len(points[1])
        ]
        scene_scale = float(np.median(scene_scales)) if scene_scales else 1.
        for source in anchors:
            if abs(source - t) > max(12, 2 * step):
                continue
            uv, xyz = map_points[source]
            result = localize_frame(frames[source], frames[t], uv, xyz, k,
                                    np.linalg.inv(native[source]), mask, return_tracks=True)
            if result is not None:
                candidates.append((result[0], result[1], result[2], int(source),
                                   result[3], result[4]))
        selected, selection = select_localization_candidate(
            candidates, scene_scale, target_frame=t)
        if selected is not None:
            native[t], count, error, source, new_uv, new_xyz = selected
            map_points[t] = new_uv, new_xyz
            tracking[t] = dict(inliers=count, median_reprojection_px=error,
                               anchor_frame=source, **selection)
        elif candidates:
            tracking[t] = dict(**selection)
    pose_indices = np.asarray(sorted(native))
    poses = np.stack([native[t] for t in pose_indices])
    relative, pose_valid = complete_c2w(poses, pose_indices, n, max_gap=max(12, 2 * step))
    observed = np.zeros(n, bool)
    observed[pose_indices] = True
    stats = dict(
        source="video:COLMAP", pycolmap_version=pycolmap.__version__,
        initialization=initialization,
        self_calibration=self_calibration or {"enabled": False},
        fixed_intrinsics=True, sampled_frame_indices=indices.tolist(),
        registered_frame_indices=sorted(registered), registered_keyframes=len(registered),
        tracked_frames=len(tracking), interpolated_frames=int((~observed).sum()),
        localized_frame_indices=pose_indices.tolist(),
        unlocalized_frame_indices=np.flatnonzero(~observed).tolist(),
        sparse_points=model.num_points3D(), mean_reprojection_error_px=model.compute_mean_reprojection_error(),
        flow_pnp={str(t): value for t, value in tracking.items()},
        camera_reference_frame_index=int(pose_indices[0]),
        valid_camera_frames=int(pose_valid.sum()),
        invalid_camera_frame_indices=np.flatnonzero(~pose_valid).tolist(),
        reference_data_used=False)
    stats["interpolated_frames"] = int((pose_valid & ~observed).sum())
    (root / "reconstruction_metadata.json").write_text(json.dumps(stats, indent=2))
    exported = relative.copy()
    exported[~pose_valid] = np.nan
    np.savez_compressed(
        root / "native_trajectory.npz", c2w_unscaled=exported,
        pose_valid=pose_valid, pose_observed=observed, intrinsics=k)
    if pose_valid.mean() < min_registration_fraction:
        raise RuntimeError(f"camera trajectory coverage too low: {pose_valid.sum()}/{n}")
    return model, registered, relative, observed, pose_valid, stats


def estimate_motion(frames, camera, predictions, output_dir, unidepth_repo,
                    unidepth_model, device=None, stride=3, max_keyframes=300,
                    depth_samples=8, threads=4, mask_margin=25,
                    metric_depth_backend="unidepth", metric_depth_repo=None,
                    metric_depth_model=None, metric_depth_process_res=504,
                    min_registration_fraction=.7,
                    hand_depth_alignment=False, hand_camera=None,
                    hand_depth_stride=1, hand_depth_patch_size=7,
                    hand_depth_max_mad=.03, hand_depth_max_delta=.15,
                    intrinsics_policy="fixed", alternative_camera=None,
                    self_calibrate=False):
    if intrinsics_policy not in {"fixed", "video_consistency"}:
        raise ValueError(f"unknown motion intrinsics policy: {intrinsics_policy}")
    if self_calibrate and alternative_camera is not None:
        raise ValueError("video self-calibration cannot be combined with a resolution focal prior")
    metric_depth_repo = metric_depth_repo or unidepth_repo
    metric_depth_model = metric_depth_model or unidepth_model
    if not metric_depth_model or not (Path(metric_depth_model) / "config.json").is_file():
        name = "UniDepthV2" if metric_depth_backend == "unidepth" else "DA3Metric-Large"
        raise FileNotFoundError(f"set the local {name} checkpoint for camera motion")
    candidate_cameras = {"original": dict(camera)}
    if (intrinsics_policy == "video_consistency" and alternative_camera is not None
            and not np.allclose(camera_matrix(camera), camera_matrix(alternative_camera))):
        candidate_cameras["protected"] = alternative_camera
    reconstructions, failures = {}, {}
    for name, candidate in candidate_cameras.items():
        root = (Path(output_dir) / name if len(candidate_cameras) > 1
                else Path(output_dir))
        try:
            reconstruction = reconstruct(
                frames, candidate, predictions, root, stride, max_keyframes, threads,
                mask_margin, min_registration_fraction,
                intrinsics_policy == "video_consistency" and name == "protected",
                self_calibrate=self_calibrate)
            if self_calibrate and reconstruction[-1].get("selected_intrinsics"):
                candidate.update(reconstruction[-1]["selected_intrinsics"])
                if name == "original":
                    camera.update(candidate)
            reconstructions[name] = reconstruction
        except RuntimeError as exc:
            if len(candidate_cameras) == 1:
                raise
            failures[name] = str(exc)
    if not reconstructions:
        raise RuntimeError(f"all video camera candidates failed: {failures}")
    depth_model = load_metric_depth(metric_depth_repo, metric_depth_model, device,
                                    backend=metric_depth_backend)
    # Shared depth views and shared SIFT pixels prevent a candidate from
    # improving its score simply by dropping difficult frames or observations.
    common_registered = set.intersection(*(set(r[1]) for r in reconstructions.values()))
    selected = np.asarray(sorted(common_registered))
    selected = selected[np.unique(np.linspace(0, len(selected) - 1,
                                             min(depth_samples, len(selected))).astype(int))]
    if len(selected) < 3:
        raise ValueError("metric camera scale needs at least three depth views")
    candidate_depths = {name: [] for name in reconstructions}
    candidate_conf = {name: [] for name in reconstructions}
    observations = {name: {} for name in reconstructions}
    scale_frames = []
    original_k = camera_matrix(camera)
    for t in selected:
        depth, conf = infer_metric_depth(
            depth_model, frames[t], original_k, process_res=metric_depth_process_res)
        mask = background_mask(frames[t], predictions, int(t), mask_margin)
        for name, result in reconstructions.items():
            k = camera_matrix(candidate_cameras[name])
            if np.allclose(k, original_k):
                d, c = depth, conf
            elif (metric_depth_backend == "da3_metric"
                  and np.isclose(k[0, 0] / original_k[0, 0],
                                 k[1, 1] / original_k[1, 1])):
                # DA3Metric's network does not consume K; only its canonical
                # depth conversion changes. Reuse exactly the same network map.
                factor = k[0, 0] / original_k[0, 0]
                d, c = depth * factor, conf
            else:
                d, c = infer_metric_depth(
                    depth_model, frames[t], k, process_res=metric_depth_process_res)
            candidate_depths[name].append(d)
            candidate_conf[name].append(c)
            observations[name][int(t)] = depth_scale_observations(
                result[0], result[1][int(t)], d, c, mask)
        scale_frames.append(int(t))
    chosen = next(iter(reconstructions))
    selection_info = dict(policy=intrinsics_policy, selected=chosen,
                          reason="single_candidate", reference_data_used=False)
    if len(reconstructions) == 2:
        qualities = {}
        for name, result in reconstructions.items():
            s = result[-1]
            qualities[name] = dict(
                valid_fraction=float(result[4].mean()),
                registered_fraction=s["registered_keyframes"] / len(s["sampled_frame_indices"]),
                sparse_points=s["sparse_points"],
                reprojection_px=s["mean_reprojection_error_px"])
        chosen, selection_info = compare_intrinsics_candidates(
            qualities, observations, camera["w"], camera["h"])
        selection_info["reconstruction_quality"] = qualities
    selection_info["failures"] = failures
    selection_info["candidate_intrinsics"] = {
        name: {key: c[key] for key in ("fl_x", "fl_y", "cx", "cy", "source")}
        for name, c in candidate_cameras.items()}
    camera = candidate_cameras[chosen]
    k = camera_matrix(camera)
    if self_calibrate:
        try:
            from .hand_camera import calibrate_predictions
        except ImportError:
            from hand_camera import calibrate_predictions
        # Input predictions are raw WiLoR when self calibration is enabled.
        # Fit once using the same K as mapping and metric-depth conversion.
        hand_fit_stats = calibrate_predictions(predictions, camera)
        hand_camera = camera
    else:
        hand_fit_stats = None
    model, registered, relative, observed, pose_valid, stats = reconstructions[chosen]
    depths, confidences = candidate_depths[chosen], candidate_conf[chosen]
    ratios = [list(observations[chosen][int(t)].values()) for t in selected]
    backend_name = ("UniDepthV2" if metric_depth_backend == "unidepth"
                    else "DA3Metric-Large")
    scale, scale_info = robust_metric_scale(ratios, source=backend_name)
    hand_depth_stats = dict(enabled=False, reference_data_used=False)
    if hand_depth_alignment:
        # Invalid camera frames remain invalid and need no hand-depth pass.
        eligible_predictions = [
            item for item in predictions
            if 0 <= int(item["frame"]) < len(pose_valid)
            and pose_valid[int(item["frame"])]
        ]
        hand_depth_stats = align_hand_predictions_to_metric_depth(
            frames, eligible_predictions, depth_model,
            hand_camera or camera, metric_depth_backend,
            process_res=metric_depth_process_res,
            frame_stride=hand_depth_stride,
            patch_size=hand_depth_patch_size,
            max_translation_mad=hand_depth_max_mad,
            max_translation_delta=hand_depth_max_delta)
    c2w = relative.copy()
    c2w[:, :3, 3] *= scale
    validate_trajectory(c2w, len(frames))
    reference = stats["camera_reference_frame_index"]
    stats.update(translation_scale=scale_info,
                 self_calibration=stats.get("self_calibration", {}),
                 hand_camera_fit=hand_fit_stats,
                 intrinsics_selection=selection_info,
                 motion_intrinsics={key: camera[key] for key in
                                    ("fl_x", "fl_y", "cx", "cy", "w", "h", "source")},
                 coordinate_frame="first_camera_world" if reference == 0 else "first_valid_camera_world",
                 trajectory_units="meters",
                 metric_depth_backend=metric_depth_backend,
                 metric_depth_model=str(Path(metric_depth_model).resolve()),
                 metric_depth_process_res=int(metric_depth_process_res),
                 metric_depth_conditioned_on="video:GeoCalib_K" if camera["source"] == "video:GeoCalib"
                 else camera["source"], depth_frame_indices=scale_frames,
                 hand_depth_alignment=hand_depth_stats)
    public_poses = c2w.copy()
    public_poses[~pose_valid] = np.nan
    public_native = relative.copy()
    public_native[~pose_valid] = np.nan
    arrays = dict(c2w=public_poses, c2w_unscaled=public_native, pose_observed=observed,
                  pose_valid=pose_valid,
                  metric_depth=np.stack(depths), depth_confidence=np.stack(confidences),
                  depth_frame_indices=np.asarray(scale_frames), intrinsics=k,
                  meters_per_native_unit=np.asarray(scale), source=np.asarray("video:COLMAP"),
                  units=np.asarray("meters"), camera_reference_frame_index=np.asarray(reference),
                  coordinate_frame=np.asarray(stats["coordinate_frame"]))
    (Path(output_dir) / "camera_metadata.json").write_text(json.dumps(stats, indent=2))
    return c2w, arrays, stats
