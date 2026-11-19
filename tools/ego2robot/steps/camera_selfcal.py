# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Experimental shared-focal calibration from masked video feature matches.

COLMAP supplies the focal optimizer. This module adds disjoint temporal
validation and rejects weak or inconsistent evidence. It never accepts a
reference camera, depth, hand size, video name, or dataset identifier.
"""
from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path

import cv2
import numpy as np

try:
    from .hand_camera import camera_matrix
except ImportError:
    from hand_camera import camera_matrix


# Engineering gates, fixed before dataset evaluation; not learned confidence
# intervals. Keep this backend opt-in until tested on independent cameras.
MIN_PAIRS_PER_BLOCK = 5
MIN_PAIR_INLIERS = 30
MAX_FOCAL_DISAGREEMENT = .10
MIN_VALIDATION_IMPROVEMENT = .20
MIN_PARALLAX_DEG = 1.


def essential_error(fundamental, k):
    """Scale-invariant distance to the equal-singular-value essential constraint."""
    singular = np.linalg.svd(k.T @ fundamental @ k, compute_uv=False)
    return float(abs(singular[0] - singular[1]) / max(singular[0], 1e-15))


def _parallax(pair, k):
    """Cheirality and median angle between rays after removing camera rotation."""
    uv1, uv2 = pair["uv1"], pair["uv2"]
    e = k.T @ pair["F"] @ k
    try:
        count, r, _, mask = cv2.recoverPose(e, uv1, uv2, k)
    except cv2.error:
        return 0., 0.
    if count < MIN_PAIR_INLIERS:
        return 0., count / max(1, len(uv1))
    first = np.c_[uv1, np.ones(len(uv1))] @ np.linalg.inv(k).T
    second = np.c_[uv2, np.ones(len(uv2))] @ np.linalg.inv(k).T @ r
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second /= np.linalg.norm(second, axis=1, keepdims=True)
    angles = np.rad2deg(np.arccos(np.clip(np.sum(first * second, axis=1), -1, 1)))
    return float(np.median(angles[mask.ravel() != 0])), count / len(uv1)


def _fit(database, camera, pair_ids, multiplier, destination):
    import pycolmap

    shutil.copy2(database, destination)
    keep = set(pair_ids)
    with pycolmap.Database.open(destination) as db:
        for cam in db.read_all_cameras():
            cam.has_prior_focal_length = False
            cam.params = np.asarray([
                camera["fl_x"] * multiplier, camera["fl_y"] * multiplier,
                camera["cx"], camera["cy"]])
            db.update_camera(cam)
        ids, _ = db.read_two_view_geometries()
        for pair_id in ids:
            if pair_id not in keep:
                db.delete_two_view_geometry(*pycolmap.pair_id_to_image_pair(pair_id))
    options = pycolmap.ViewGraphCalibrationOptions(
        random_seed=42, cross_validate_prior_focal_lengths=False,
        reestimate_relative_pose=False)
    if not pycolmap.calibrate_view_graph(destination, options):
        raise RuntimeError("COLMAP view graph calibration failed")
    with pycolmap.Database.open(destination) as db:
        return float(db.read_all_cameras()[0].mean_focal_length())


def propose_camera(database, camera):
    """Return a video-only proposal and audit report; input database is read-only.

    Fit on the first two non-overlapping thirds of sampled frames. The last
    third supplies independent pairs for validation, never focal optimization.
    Principal point and square pixels remain assumptions, not estimated truth.
    """
    # pycolmap.Database.open can change SQLite headers even for reads. Snapshot
    # via SQLite's read-only API, including any committed WAL, before using it.
    with tempfile.TemporaryDirectory(prefix="path_b_selfcal_snapshot_") as tmp:
        snapshot = Path(tmp) / "database.db"
        source_uri = Path(database).resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(source_uri, uri=True) as source, sqlite3.connect(snapshot) as target:
            source.backup(target)
        return _propose_snapshot(snapshot, camera)


def _propose_snapshot(database, camera):
    import pycolmap

    camera_matrix(camera)
    report = dict(
        enabled=True, accepted=False, backend="COLMAP:view_graph_calibration",
        reference_data_used=False, experimental=True,
        protocol="two_temporal_training_blocks_one_disjoint_validation_block_v1",
        gates=dict(min_pairs_per_block=MIN_PAIRS_PER_BLOCK,
                   min_pair_inliers=MIN_PAIR_INLIERS,
                   max_focal_disagreement=MAX_FOCAL_DISAGREEMENT,
                   min_validation_improvement=MIN_VALIDATION_IMPROVEMENT,
                   min_parallax_deg=MIN_PARALLAX_DEG))

    def reject(reason):
        report["reason"] = reason
        return dict(camera), report

    if not np.isclose(camera["fl_x"], camera["fl_y"], rtol=.01):
        return reject("non_square_pixels_unsupported")
    if not hasattr(pycolmap, "calibrate_view_graph"):
        return reject("pycolmap_view_graph_calibration_unavailable")
    with pycolmap.Database.open(database) as db:
        cams = db.read_all_cameras()
        if len(cams) != 1 or cams[0].model_name != "PINHOLE":
            return reject("requires_one_shared_pinhole_camera")
        images = sorted(db.read_all_images(), key=lambda image: image.name)
        blocks = np.array_split([image.image_id for image in images], 3)
        block_index = {int(i): b for b, ids in enumerate(blocks) for i in ids}
        report["block_image_ids"] = [list(map(int, ids)) for ids in blocks]
        ids, geometries = db.read_two_view_geometries()
        pairs = [[] for _ in range(3)]
        for pair_id, geometry in zip(ids, geometries):
            i, j = pycolmap.pair_id_to_image_pair(pair_id)
            b = block_index[i]
            if b != block_index[j] or int(geometry.config) not in (2, 3):
                continue
            matches = np.asarray(geometry.inlier_matches)
            f = geometry.F
            if (len(matches) < MIN_PAIR_INLIERS or f is None
                    or not np.isfinite(f).all() or np.linalg.norm(f) < 1e-15):
                continue
            pairs[b].append(dict(
                id=pair_id, F=f, uv1=np.asarray(db.read_keypoints(i)[matches[:, 0], :2], float),
                uv2=np.asarray(db.read_keypoints(j)[matches[:, 1], :2], float)))
    report["eligible_pairs_per_block"] = [len(p) for p in pairs]
    if min(map(len, pairs)) < MIN_PAIRS_PER_BLOCK:
        return reject("insufficient_nonplanar_temporally_disjoint_pairs")
    with tempfile.TemporaryDirectory(prefix="path_b_selfcal_") as tmp:
        root = Path(tmp)
        specifications = [
            ([p["id"] for p in pairs[b]], 1.) for b in (0, 1)]
        train_ids = [p["id"] for block in pairs[:2] for p in block]
        specifications += [(train_ids, .5), (train_ids, 2.)]
        try:
            focal = [_fit(database, camera, ids, factor, root / f"{i}.db")
                     for i, (ids, factor) in enumerate(specifications)]
        except (RuntimeError, ValueError) as exc:
            report["error"] = str(exc)
            return reject("optimizer_failure")
    report["training_focals_px"] = focal
    focal = np.asarray(focal)
    if not np.isfinite(focal).all() or np.any(focal <= 0):
        return reject("invalid_focal")
    disagreement = float(np.ptp(focal) / np.median(focal))
    report["focal_disagreement"] = disagreement
    if disagreement > MAX_FOCAL_DISAGREEMENT:
        return reject("temporally_or_initialization_unstable")
    proposed_focal = float(np.median(focal[2:]))
    # Broad physical feasibility (10..170 degree horizontal FOV), not a
    # preferred focal length or a replacement value.
    hfov = float(np.rad2deg(2 * np.arctan(camera["w"] / (2 * proposed_focal))))
    report["horizontal_fov_deg"] = hfov
    if not 10 < hfov < 170:
        return reject("implausible_pinhole_field_of_view")
    proposed = dict(camera, fl_x=proposed_focal, fl_y=proposed_focal,
                    source=f"{camera['source']}:view_graph")
    old_k, new_k = camera_matrix(camera), camera_matrix(proposed)
    validation = pairs[2]
    old = np.asarray([essential_error(p["F"], old_k) for p in validation])
    new = np.asarray([essential_error(p["F"], new_k) for p in validation])
    # Every pair has equal weight, so an image with more SIFT points cannot
    # dominate the decision.
    improvement = float(1 - np.median(new) / max(np.median(old), 1e-15))
    angles = [_parallax(p, new_k) for p in validation]
    useful = sum(angle >= MIN_PARALLAX_DEG and fraction >= .75
                 for angle, fraction in angles)
    report["validation"] = dict(
        original_essential_error=float(np.median(old)),
        proposed_essential_error=float(np.median(new)),
        relative_improvement=improvement, improving_pair_fraction=float(np.mean(new < old)),
        pairs_with_parallax=useful, total_pairs=len(validation),
        median_parallax_deg=float(np.median([a for a, _ in angles])))
    if useful < MIN_PAIRS_PER_BLOCK or useful < len(validation) / 2:
        return reject("insufficient_validation_parallax_or_cheirality")
    if improvement < MIN_VALIDATION_IMPROVEMENT or np.mean(new < old) < .75:
        return reject("no_consistent_held_out_geometry_improvement")
    if abs(np.log(proposed_focal / camera["fl_x"])) < .01:
        return reject("negligible_focal_change")
    report.update(accepted=True, reason="stable_training_and_held_out_geometry",
                  selected_focal_px=proposed_focal)
    return proposed, report


def apply_camera(database, camera, pairs_path):
    """Reverify original matches under the accepted K before building a map."""
    import pycolmap

    with pycolmap.Database.open(database) as db:
        names = {image.image_id: image.name for image in db.read_all_images()}
        ids, _ = db.read_two_view_geometries()
        pairs = [pycolmap.pair_id_to_image_pair(i) for i in ids]
        for cam in db.read_all_cameras():
            cam.params = np.asarray([camera[x] for x in ("fl_x", "fl_y", "cx", "cy")])
            cam.has_prior_focal_length = True
            db.update_camera(cam)
        for i, j in pairs:
            db.delete_two_view_geometry(i, j)
    Path(pairs_path).write_text("".join(f"{names[i]} {names[j]}\n" for i, j in pairs))
    options = pycolmap.TwoViewGeometryOptions()
    options.ransac.random_seed = 42
    pycolmap.verify_matches(database, pairs_path, options)
