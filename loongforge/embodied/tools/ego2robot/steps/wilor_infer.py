# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Small batch adapter around the official WiLoR repository.

The upstream demo renders meshes but does not persist predictions.  This
adapter keeps its preprocessing/model path and writes one compact NPZ sequence
that ``steps.path_b`` can consume.
"""

from __future__ import annotations

import argparse
import inspect
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# WiLoR's MANO wrapper returns the 21 joints in its OpenPose-compatible order.
# Path B downstream expects the canonical MANO order used by EgoVerse and by
# ``action_alignment.py`` (wrist=0, thumb_tip=4, index_tip=8, middle_tip=12).
WILOR_TO_OPENPOSE = np.asarray(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)
OPENPOSE_TO_MANO = np.argsort(WILOR_TO_OPENPOSE)


def prepare_mano_assets(repo: Path, mano_dir: str | None = None):
    """Make the user-provided MANO files visible to WiLoR's fixed paths."""
    source_value = mano_dir or os.environ.get("EGO2ROBOT_MANO_DIR")
    if not source_value:
        raise ValueError(
            "MANO directory is not configured; pass --mano_dir or set "
            "EGO2ROBOT_MANO_DIR"
        )
    source = Path(source_value).expanduser().resolve()
    right = source / "MANO_RIGHT.pkl"
    mean = source / "mano_mean_params.npz"
    if not mean.is_file():
        mean = repo / "mano_data" / "mano_mean_params.npz"
    if not right.is_file():
        raise FileNotFoundError(
            f"MANO_RIGHT.pkl not found in {source}; pass --mano_dir or set "
            "EGO2ROBOT_MANO_DIR"
        )
    if not mean.is_file():
        raise FileNotFoundError(
            f"mano_mean_params.npz not found in {source} or {repo / 'mano_data'}"
        )
    target = repo / "mano_data"
    target.mkdir(parents=True, exist_ok=True)
    for src in (right, mean):
        dst = target / src.name
        # ``mean`` commonly falls back to WiLoR's own file, which is already
        # at the destination and must not be linked to itself.
        if src.absolute() == dst.absolute():
            continue
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and dst.resolve() == src.resolve():
                continue
            dst.unlink()
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def load_wilor_no_renderer(checkpoint_path: str, cfg_path: str):
    """Load WiLoR for keypoint inference without initializing pyrender/EGL."""
    from wilor.configs import get_config
    from wilor.models import WiLoR

    model_cfg = get_config(cfg_path, update_cachedir=True)
    if "vit" in model_cfg.MODEL.BACKBONE.TYPE and "BBOX_SHAPE" not in model_cfg.MODEL:
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, (
            f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        )
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        model_cfg.freeze()
    if "DATA_DIR" in model_cfg.MANO:
        model_cfg.defrost()
        model_cfg.MANO.DATA_DIR = "./mano_data/"
        model_cfg.MANO.MODEL_PATH = "./mano_data/"
        model_cfg.MANO.MEAN_PARAMS = "./mano_data/mano_mean_params.npz"
        model_cfg.freeze()
    model = WiLoR.load_from_checkpoint(
        checkpoint_path, strict=False, cfg=model_cfg, init_renderer=False
    )
    return model, model_cfg

def run(args):
    repo = Path(args.wilor_repo).resolve()
    if not (repo / "wilor").is_dir():
        raise FileNotFoundError(f"WiLoR package not found under {repo}")
    prepare_mano_assets(repo, args.mano_dir)
    # The released loader hard-codes relative ``./mano_data`` paths.
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    # MANO pickles depend on chumpy 0.70, whose Python 2/3.10-era import
    # still calls inspect.getargspec. Keep this compatibility shim local to
    # the adapter instead of modifying the installed third-party package.
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    # chumpy also imports NumPy 1.x scalar aliases removed in NumPy 2.
    for name, value in {
        "bool": bool, "int": int, "float": float, "complex": complex,
        "object": object, "unicode": str, "str": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)

    import torch
    from ultralytics import YOLO
    from wilor.utils import recursive_to
    from wilor.datasets.vitdet_dataset import ViTDetDataset

    cfg = args.config or str(repo / "pretrained_models" / "model_config.yaml")
    model, model_cfg = load_wilor_no_renderer(args.checkpoint, cfg)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()
    # Ultralytics 8.1.x predates PyTorch 2.6's ``weights_only`` default and
    # its detector checkpoint contains a serialized PoseModel class. This is
    # a trusted local file supplied by the user, so opt out only while loading
    # this checkpoint and restore torch.load immediately afterwards.
    torch_load = torch.load
    def _trusted_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return torch_load(*load_args, **load_kwargs)
    torch.load = _trusted_load
    try:
        detector = YOLO(args.detector).to(device)
    finally:
        torch.load = torch_load
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_kp, all_right, all_frame, all_score = [], [], [], []
    all_pose, all_orient, all_betas, all_trans, all_kp2d = [], [], [], [], []
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(Path(args.images).glob(ext))
    image_paths.sort()
    if not image_paths:
        raise RuntimeError(f"no image files found under {args.images}")

    for frame_idx, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        result = detector(image, conf=args.confidence, verbose=False)[0]
        boxes, rights, scores = [], [], []
        for box in result.boxes:
            row = box.data.detach().cpu().numpy().reshape(-1)
            if row.size < 6:
                continue
            boxes.append(row[:4])
            scores.append(float(row[4]))
            rights.append(float(box.cls.detach().cpu().item()))
        if not boxes:
            continue
        dataset = ViTDetDataset(model_cfg, image, np.asarray(boxes, np.float32),
                                np.asarray(rights, np.float32),
                                rescale_factor=args.rescale_factor,
                                fp16=args.fp16)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0)
        offset = 0
        for batch in loader:
            batch = recursive_to(batch, device)
            with torch.inference_mode():
                output = model(batch)
            kp = output["pred_keypoints_3d"].detach().cpu().numpy().astype(np.float32)
            kp = kp[:, OPENPOSE_TO_MANO, :]
            centers = batch["box_center"].detach().cpu().numpy().astype(np.float32)
            sizes = batch["box_size"].detach().cpu().numpy().astype(np.float32).reshape(-1)
            params = output["pred_mano_params"]
            pose = params["hand_pose"].detach().cpu().numpy().astype(np.float32)
            orient = params["global_orient"].detach().cpu().numpy().astype(np.float32)
            betas = params["betas"].detach().cpu().numpy().astype(np.float32)
            right = batch["right"].detach().cpu().numpy().astype(np.float32).reshape(-1)
            # WiLoR regresses camera translation in the 256px crop coordinate
            # system.  Convert it to the original image, exactly as the
            # official demo does, otherwise a hand of 10cm is placed several
            # metres away and renders as a few pixels.
            pred_cam = output["pred_cam"].detach().cpu().numpy().astype(np.float32)
            img_size = batch["img_size"].detach().cpu().numpy().astype(np.float32)
            scaled_focal = (
                float(model_cfg.EXTRA.FOCAL_LENGTH) /
                float(model_cfg.MODEL.IMAGE_SIZE) * np.max(img_size, axis=1)
            )
            cam_bbox = pred_cam.copy()
            cam_bbox[:, 1] *= 2.0 * right - 1.0
            centers_np = centers
            bs = sizes * cam_bbox[:, 0] + 1e-9
            full_trans = np.stack([
                2.0 * (centers_np[:, 0] - img_size[:, 0] / 2.0) / bs + cam_bbox[:, 1],
                2.0 * (centers_np[:, 1] - img_size[:, 1] / 2.0) / bs + cam_bbox[:, 2],
                2.0 * scaled_focal / bs,
            ], axis=1).astype(np.float32)
            for j in range(len(kp)):
                hand_sign = 2.0 * right[j] - 1.0
                # WiLoR's canonical output is root-relative MANO coordinates;
                # mirror x back to the common camera convention for left hands,
                # matching the upstream demo's postprocessing.
                kp_j = kp[j].copy()
                kp_j[:, 0] *= hand_sign
                trans_j = full_trans[j].copy()
                # ``pred_keypoints_3d`` is centered at the MANO wrist. First
                # recover the desired full-image 2D projection using WiLoR's
                # camera (this is also the signal Dyn-HaMR optimizes).
                pts_wilor = kp_j + trans_j[None, :]
                desired_2d = np.stack([
                    scaled_focal[j] * pts_wilor[:, 0] /
                    np.maximum(pts_wilor[:, 2], 1e-6) + img_size[j, 0] / 2.0,
                    scaled_focal[j] * pts_wilor[:, 1] /
                    np.maximum(pts_wilor[:, 2], 1e-6) + img_size[j, 1] / 2.0,
                ], axis=1).astype(np.float32)
                # Re-solve translation for the MuJoCo image focal length. A
                # uniform metric scale cannot correct a focal-length mismatch;
                # matching the 2D extent gives a stable, visible hand depth.
                render_focal = float(max(img_size[j, 0], img_size[j, 1]))
                extent3d = np.ptp(kp_j, axis=0)
                extent2d = np.ptp(desired_2d, axis=0)
                z_candidates = [
                    render_focal * float(extent3d[0]) /
                    max(float(extent2d[0]), 1.0),
                    render_focal * float(extent3d[1]) /
                    max(float(extent2d[1]), 1.0),
                ]
                z_target = float(np.clip(np.median(z_candidates), 0.25, 2.0))
                desired_center = desired_2d.mean(axis=0)
                kp_center = kp_j.mean(axis=0)
                trans_j = np.asarray([
                    (desired_center[0] - img_size[j, 0] / 2.0) * z_target /
                    render_focal - kp_center[0],
                    (desired_center[1] - img_size[j, 1] / 2.0) * z_target /
                    render_focal - kp_center[1],
                    z_target,
                ], dtype=np.float32)
                kp_j += trans_j[None, :]
                all_kp.append(kp_j)
                all_right.append(right[j])
                all_frame.append(frame_idx)
                all_score.append(scores[offset + j])
                all_pose.append(pose[j])
                all_orient.append(orient[j])
                all_betas.append(betas[j])
                all_trans.append(trans_j)
                all_kp2d.append(desired_2d)
            offset += len(kp)

    if not all_kp:
        raise RuntimeError("WiLoR detector found no hands in the input video")
    np.savez_compressed(out_dir / "wilor_predictions.npz",
                        pred_keypoints_3d=np.asarray(all_kp, np.float32),
                        right=np.asarray(all_right, np.float32),
                        frame=np.asarray(all_frame, np.int64),
                        score=np.asarray(all_score, np.float32),
                        hand_pose=np.asarray(all_pose, np.float32),
                        global_orient=np.asarray(all_orient, np.float32),
                        betas=np.asarray(all_betas, np.float32),
                        cam_trans=np.asarray(all_trans, np.float32),
                        keypoints_2d=np.asarray(all_kp2d, np.float32))
    print(f"WiLoR predictions: {len(all_kp)} hands -> {out_dir / 'wilor_predictions.npz'}")


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Export official WiLoR predictions as NPZ")
    ap.add_argument("--wilor_repo", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--mano_dir", default=None,
                    help="directory containing MANO_RIGHT.pkl; pass explicitly or set EGO2ROBOT_MANO_DIR")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--confidence", type=float, default=0.3)
    ap.add_argument("--rescale_factor", type=float, default=2.0)
    ap.add_argument("--fp16", action="store_true")
    return ap


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
