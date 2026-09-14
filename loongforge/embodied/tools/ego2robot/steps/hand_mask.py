# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Step 3: SAM3 Hand/Arm Mask

Input:
  - zarr episode (images.front_1 JPEG, left/right.obs_keypoints, obs_head_pose, intrinsics)
Output:
  - {output_dir}/{ep_name}/masks.npz  — (T, H, W) uint8 binary mask
  - {output_dir}/{ep_name}/mask_overlay.mp4 - visualization video

Strategy:
  1. Seed visible-person text prompts and hand/forearm geometric box prompts at the middle frame.
  2. Build arm coverage boxes by projecting hand keypoints and wrist extension points from world to image coordinates.
  3. Propagate the person and each hand/arm mask in separate video sessions, then merge the arm results.
  4. Split long videos into 400-frame chunks with a 50-frame overlap, propagate in both directions, and merge.
  5. Fill short temporal gaps, repair anomalous mask-area frames, and apply a 5x5 morphological close.
  6. Output separate masks and the merged human mask.

Usage:
  python cli.py mask --zarr_dir data_input/egoverse/narrow_tabletop --output_dir data_output/03_mask
"""

import argparse
import os
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import zarr
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import simplejpeg
    HAS_SIMPLEJPEG = True
except ImportError:
    HAS_SIMPLEJPEG = False

import torch

try:
    from . import config
except ImportError:
    import config

# ============================================================
# Projection (from vis_keypoints_video.py)
# ============================================================


def project_points_to_image(points_3d_world, head_pose, intrinsic_matrix):
    """3D world -> 2D pixel.

    Returns:
        pixels (M,2), valid (M,) bool
    """
    from scipy.spatial.transform import Rotation
    t_world = head_pose[:3]
    q_wxyz = head_pose[3:7]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    R_head_in_world = Rotation.from_quat(q_xyzw).as_matrix()
    R_inv = R_head_in_world.T
    t_inv = -R_inv @ t_world
    points_cam = (R_inv @ points_3d_world.T).T + t_inv
    K = intrinsic_matrix[:3, :3]
    z = points_cam[:, 2]
    valid = z > 0.01
    pixels = np.zeros((points_cam.shape[0], 2))
    if np.any(valid):
        proj = K @ points_cam[valid].T
        pixels[valid, 0] = proj[0] / proj[2]
        pixels[valid, 1] = proj[1] / proj[2]
    return pixels, valid


# ============================================================
# Generate point prompts for SAM3
# ============================================================

WRIST_IDX = 0
ELBOW_APPROX_OFFSET = np.array([0.0, 0.0, -0.25])  # Approximate the forearm 25 cm below the wrist.


def make_point_prompts(kp_world, head_pose, K, H, W):
    """
    Generate SAM3 point prompts from 21 MANO keypoints.
    Returns (points (N,2) np.float32, labels (N,) int: 1=foreground).
    """
    # The 21 hand keypoints.
    pixels, valid = project_points_to_image(kp_world, head_pose, K)

    # Extend forearm coverage with the wrist and 2-3 samples below it.
    wrist_3d = kp_world[WRIST_IDX]
    # Forearm direction: from the middle-finger base (index 9) toward the wrist.
    forearm_dir = wrist_3d - kp_world[9]
    forearm_dir_norm = np.linalg.norm(forearm_dir)
    if forearm_dir_norm > 1e-6:
        forearm_dir = forearm_dir / forearm_dir_norm
    else:
        forearm_dir = np.array([0, 0, -1.0])

    # Sample three points along the forearm direction, 5, 12, and 20 cm from the wrist.
    arm_pts_3d = np.array([
        wrist_3d + forearm_dir * 0.05,
        wrist_3d + forearm_dir * 0.12,
        wrist_3d + forearm_dir * 0.20,
    ])
    arm_pixels, arm_valid = project_points_to_image(arm_pts_3d, head_pose, K)

    # Merge all prompt points.
    all_pixels = []
    all_labels = []

    for i in range(21):
        if valid[i]:
            px, py = pixels[i]
            if 0 <= px < W and 0 <= py < H:
                all_pixels.append([px, py])
                all_labels.append(1)

    for i in range(3):
        if arm_valid[i]:
            px, py = arm_pixels[i]
            if 0 <= px < W and 0 <= py < H:
                all_pixels.append([px, py])
                all_labels.append(1)

    if len(all_pixels) == 0:
        return None, None

    return np.array(all_pixels, dtype=np.float32), np.array(all_labels, dtype=np.int32)


# ============================================================
# SAM3 wrapper
# ============================================================

class SAM3Segmenter:
    """SAM3 image predictor wrapper."""

    def __init__(self, checkpoint_path, device="cuda:0", confidence_threshold=0.5):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.device = device
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            device=device,
        )
        model = model.to(device=device, dtype=torch.float32)

        self._use_amp = device.startswith("cuda") and torch.cuda.is_available()
        if self._use_amp:
            self._amp_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            self._amp_dtype = torch.float32

        self.processor = Sam3Processor(
            model=model,
            device=device,
            confidence_threshold=confidence_threshold,
        )

    def _autocast(self):
        if not self._use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._amp_dtype)

    @staticmethod
    def _prompt_box(point_coords, width, height, padding_px=8):
        """Convert projected hand/arm points to one normalized XYWH box."""
        points = np.asarray(point_coords, dtype=np.float32)
        x0, y0 = points.min(axis=0) - padding_px
        x1, y1 = points.max(axis=0) + padding_px
        x0 = float(np.clip(x0, 0, width - 1))
        y0 = float(np.clip(y0, 0, height - 1))
        x1 = float(np.clip(x1, x0 + 1, width))
        y1 = float(np.clip(y1, y0 + 1, height))
        return [
            (x0 + x1) / (2.0 * width),
            (y0 + y1) / (2.0 * height),
            (x1 - x0) / width,
            (y1 - y0) / height,
        ]

    @staticmethod
    def _best_mask(state, height, width):
        masks = state["masks"]
        scores = state["scores"]
        if hasattr(masks, "detach"):
            masks = masks.detach().cpu().numpy()
        if hasattr(scores, "detach"):
            # NumPy does not support torch.bfloat16 arrays directly.
            scores = scores.detach().float().cpu().numpy()
        masks = np.asarray(masks)
        scores = np.asarray(scores)
        if masks.size == 0 or scores.size == 0:
            return np.zeros((height, width), dtype=np.uint8)
        if masks.ndim == 2:
            best = masks
        else:
            best = masks[int(np.argmax(scores))]
        best = np.squeeze(best)
        if best.shape != (height, width):
            raise ValueError(
                f"SAM3 returned mask shape {best.shape}, expected {(height, width)}"
            )
        return best.astype(bool).astype(np.uint8)

    @staticmethod
    def _combined_masks(state, height, width):
        """Union every mask retained by the processor confidence threshold."""
        masks = state["masks"]
        scores = state["scores"]
        if hasattr(masks, "detach"):
            masks = masks.detach().cpu().numpy()
        if hasattr(scores, "detach"):
            scores = scores.detach().float().cpu().numpy()
        masks = np.asarray(masks)
        scores = np.asarray(scores)
        if masks.size == 0 or scores.size == 0:
            return np.zeros((height, width), dtype=np.uint8)
        if masks.shape[-2:] != (height, width):
            raise ValueError(
                f"SAM3 returned mask shape {masks.shape}, expected (*, {height}, {width})"
            )
        return masks.reshape(-1, height, width).any(axis=0).astype(np.uint8)

    @staticmethod
    def _copy_state(state):
        copied = dict(state)
        if "backbone_out" in state:
            copied["backbone_out"] = dict(state["backbone_out"])
        return copied

    def segment_human(self, image_rgb, prompt_groups, body_prompt=None):
        """Segment visible body plus hand/arm groups with one image encoding.

        Each group keeps the old best-mask selection behavior, while all
        groups reuse the same image backbone features.
        """
        height, width = image_rgb.shape[:2]
        with torch.inference_mode(), self._autocast():
            state = self.processor.set_image(Image.fromarray(image_rgb, mode="RGB"))
            combined = np.zeros((height, width), dtype=np.uint8)

            if body_prompt:
                body_state = self.processor.set_text_prompt(
                    prompt=body_prompt,
                    state=self._copy_state(state),
                )
                combined |= self._combined_masks(body_state, height, width)

            # Geometric-only prompts need the processor's dummy "visual"
            # text features. Compute them once for this frame.
            hand_state = self._copy_state(state)
            if prompt_groups and "language_features" not in hand_state["backbone_out"]:
                hand_state["backbone_out"].update(
                    self.processor.model.backbone.forward_text(
                        ["visual"], device=self.device
                    )
                )

            for point_coords, _point_labels in prompt_groups:
                # Use the official SAM3 API with one enclosing hand/arm box.
                # A separate shallow state keeps left/right prompts isolated
                # while reusing the same image backbone features.
                group_state = self._copy_state(hand_state)
                group_state = self.processor.add_geometric_prompt(
                    box=self._prompt_box(point_coords, width, height),
                    label=True,
                    state=group_state,
                )
                combined |= self._best_mask(group_state, height, width)
            return combined

    def segment_groups(self, image_rgb, prompt_groups):
        """Segment hand/arm groups without a body text prompt."""
        return self.segment_human(image_rgb, prompt_groups, body_prompt=None)

    def segment(self, image_rgb, point_coords, point_labels):
        """
        image_rgb: (H, W, 3) uint8 RGB
        point_coords: (N, 2) float32 [x, y]
        point_labels: (N,) int (1=fg, 0=bg)
        Returns: mask (H, W) uint8 binary
        """
        return self.segment_groups(image_rgb, [(point_coords, point_labels)])


class SAM3VideoSegmenter:
    """SAM3 video predictor with temporal memory and mask propagation."""

    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 50

    def __init__(self, checkpoint_path, device="cuda:0"):
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("SAM3 video tracking requires a CUDA device")

        from sam3.model_builder import build_sam3_video_predictor

        gpu_index = int(device.split(":", 1)[1]) if ":" in device else 0
        torch.cuda.set_device(gpu_index)
        self.predictor = build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            gpus_to_use=[gpu_index],
            async_loading_frames=True,
        )

    @staticmethod
    def _center_to_xywh(box):
        cx, cy, width, height = box
        return [
            max(0.0, cx - width / 2.0),
            max(0.0, cy - height / 2.0),
            min(width, 1.0 - max(0.0, cx - width / 2.0)),
            min(height, 1.0 - max(0.0, cy - height / 2.0)),
        ]

    def _propagate_prompt(
        self,
        frame_dir,
        num_frames,
        frame_width,
        frame_height,
        prompt_request,
        anchor_frame,
        frame_start,
        frame_end,
    ):
        """Run one prompt over a bounded chunk in both temporal directions."""
        all_masks = np.zeros(
            (num_frames, frame_height, frame_width), dtype=np.uint8
        )
        session_id = None
        try:
            session = self.predictor.handle_request({
                "type": "start_session",
                "resource_path": str(frame_dir),
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": True,
            })
            session_id = session["session_id"]
            self.predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": anchor_frame,
                "output_prob_thresh": 0.5,
                **prompt_request,
            })

            def collect(direction, max_frames):
                if max_frames < 0:
                    return
                for result in self.predictor.handle_stream_request({
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": direction,
                    "start_frame_index": anchor_frame,
                    "max_frame_num_to_track": max_frames,
                    "output_prob_thresh": 0.5,
                }):
                    frame_idx = int(result["frame_index"])
                    if not frame_start <= frame_idx < frame_end:
                        continue
                    outputs = result.get("outputs") or {}
                    masks = np.asarray(outputs.get("out_binary_masks", []))
                    if masks.size == 0:
                        continue
                    if masks.shape[-2:] != (frame_height, frame_width):
                        raise ValueError(
                            f"SAM3 video returned mask shape {masks.shape}, "
                            f"expected (*, {frame_height}, {frame_width})"
                        )
                    all_masks[frame_idx] = (
                        masks.astype(bool).any(axis=0).astype(np.uint8)
                    )

            # Forward includes the anchor; backward starts at anchor - 1.
            collect("forward", frame_end - 1 - anchor_frame)
            collect("backward", anchor_frame - frame_start)
            return all_masks
        finally:
            if session_id is not None:
                self.predictor.handle_request({
                    "type": "close_session",
                    "session_id": session_id,
                })

    def segment_episode(
        self,
        frame_dir,
        num_frames,
        frame_width,
        frame_height,
        body_prompt,
        prompt_groups,
        mask_mode="both",
        prompt_groups_by_frame=None,
    ):
        """Track person and hand/arm masks independently, then return both."""
        empty = np.zeros(
            (num_frames, frame_height, frame_width), dtype=np.uint8
        )
        person_masks = empty.copy()
        arm_masks = empty.copy()

        chunk_step = self.CHUNK_SIZE - self.CHUNK_OVERLAP
        chunk_ranges = [
            (start, min(start + self.CHUNK_SIZE, num_frames))
            for start in range(0, num_frames, chunk_step)
        ]

        def groups_for(anchor):
            if prompt_groups_by_frame is None:
                return prompt_groups
            return prompt_groups_by_frame.get(anchor, prompt_groups)

        if mask_mode in ("both", "person"):
            print(
                f"  Temporal track: text prompt {body_prompt!r}, "
                f"anchor=middle, chunks={len(chunk_ranges)}"
            )
            for frame_start, frame_end in chunk_ranges:
                anchor = (frame_start + frame_end - 1) // 2
                person_masks |= self._propagate_prompt(
                    frame_dir,
                    num_frames,
                    frame_width,
                    frame_height,
                    {"text": body_prompt},
                    anchor,
                    frame_start,
                    frame_end,
                )

        if mask_mode in ("both", "arms"):
            print(
                f"  Temporal track: hand/arm geometric boxes, "
                f"chunks={len(chunk_ranges)}"
            )
            for frame_start, frame_end in chunk_ranges:
                anchor = (frame_start + frame_end - 1) // 2
                groups = groups_for(anchor)
                boxes = [
                    self._center_to_xywh(
                        SAM3Segmenter._prompt_box(
                            points, frame_width, frame_height
                        )
                    )
                    for points, _labels in groups
                ]
                # SAM3 requires exactly one visual box for an initial prompt.
                # Track each hand/arm independently, then union the propagated
                # masks so left and right arms remain separate in tracker memory.
                for box in boxes:
                    arm_masks |= self._propagate_prompt(
                        frame_dir,
                        num_frames,
                        frame_width,
                        frame_height,
                        {
                            "text": "visual",
                            "bounding_boxes": [box],
                            "bounding_box_labels": [1],
                        },
                        anchor,
                        frame_start,
                        frame_end,
                    )

        return person_masks, arm_masks

    def close(self):
        """Release resources held by the video predictor."""
        self.predictor.shutdown()


# ============================================================
# Main processing
# ============================================================

def _unwrap_img_bytes(val):
    while isinstance(val, np.ndarray) and val.ndim == 0:
        val = val.item()
    if isinstance(val, np.ndarray) and val.dtype == object:
        val = _unwrap_img_bytes(val.flat[0])
    return val


def _fill_short_temporal_gaps(masks, max_gap=3):
    """Fill short all-zero runs between valid mask frames."""
    masks = masks.astype(bool, copy=True)
    active = masks.any(axis=(1, 2))
    t = 0
    while t < len(active):
        if active[t]:
            t += 1
            continue
        start = t
        while t < len(active) and not active[t]:
            t += 1
        end = t
        if start > 0 and end < len(active) and end - start <= max_gap:
            bridge = masks[start - 1] | masks[end]
            masks[start:end] = bridge
            active[start:end] = True
    return masks.astype(np.uint8)


def _replace_small_area_frames(masks, window=11, min_ratio=0.5):
    """Replace frames whose mask area is anomalously small locally."""
    masks = masks.astype(np.uint8, copy=True)
    areas = masks.reshape(len(masks), -1).sum(axis=1).astype(np.float64)
    original = masks.copy()
    half = window // 2
    for t, area in enumerate(areas):
        lo = max(0, t - half)
        hi = min(len(areas), t + half + 1)
        local_median = float(np.median(areas[lo:hi]))
        if local_median <= 0 or area >= min_ratio * local_median:
            continue
        candidates = np.flatnonzero(areas[lo:hi] >= min_ratio * local_median) + lo
        if len(candidates) == 0:
            continue
        nearest = int(candidates[np.argmin(np.abs(candidates - t))])
        masks[t] = original[nearest]
    return masks


def _paper_mask_postprocess(masks):
    """Apply Ego2Robot A.4 temporal cleanup and 5x5 morphological close."""
    masks = _fill_short_temporal_gaps(masks, max_gap=3)
    masks = _replace_small_area_frames(masks, window=11, min_ratio=0.5)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = np.zeros_like(masks, dtype=np.uint8)
    for t in range(len(masks)):
        closed[t] = (
            cv2.morphologyEx(masks[t] * 255, cv2.MORPH_CLOSE, close_kernel) > 0
        ).astype(np.uint8)
    return closed


def process_episode(
    zarr_path,
    output_dir,
    ep_name,
    segmenter,
    fps=30.0,
    dilation_px=0,
    max_frames=None,
    body_prompt="person",
    mask_mode="both",
):
    """Generate SAM3 human masks for a single episode."""
    print(f"\n[{ep_name}] Loading zarr...")
    store = zarr.open_group(zarr_path, mode="r")
    total_frames = int(store.attrs["total_frames"])
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    left_kp = np.array(store["left.obs_keypoints"][:total_frames]).reshape(total_frames, 21, 3)
    right_kp = np.array(store["right.obs_keypoints"][:total_frames]).reshape(total_frames, 21, 3)
    head_pose = np.array(store["obs_head_pose"][:total_frames])

    left_invalid = np.any(np.abs(left_kp) > 1e8, axis=(1, 2))
    right_invalid = np.any(np.abs(right_kp) > 1e8, axis=(1, 2))
    head_invalid = np.any(np.abs(head_pose[:, :3]) > 1e8, axis=1)

    imgs = store["images.front_1"]
    first_bytes = _unwrap_img_bytes(imgs[0])
    first_img = simplejpeg.decode_jpeg(first_bytes, colorspace="RGB")
    H, W = first_img.shape[:2]

    # Resolve both legacy per-camera 3x4 intrinsics and the newer flattened
    # {fl_x, fl_y, cx, cy, w, h} form used by some Zarr exports.
    from steps.config import dataset_intrinsics_k, fallback_episode_attrs
    ep_attrs, _borrow_src = fallback_episode_attrs(zarr_path)
    K = (dataset_intrinsics_k(ep_attrs, camera="front_1", img_shape=(H, W))
         if ep_attrs is not None else None)
    if K is None:
        print("  ⚠️ No usable intrinsics for front_1, skipping")
        return

    ep_out = os.path.join(output_dir, ep_name)
    os.makedirs(ep_out, exist_ok=True)

    stats = {
        "total": total_frames,
        "masked": 0,
        "mask_mode": mask_mode,
        "person_prompted": int(mask_mode in ("both", "person")),
        "arm_prompted": 0,
        "person_masked": 0,
        "arm_masked": 0,
        "left_prompted": 0,
        "right_prompted": 0,
    }

    print(f"  Processing {total_frames} frames, resolution {W}x{H}...")
    # Build hand/arm geometric prompts independently at each chunk anchor.
    # This preserves the requested geometric-box prompt while following the
    # paper's middle-anchor strategy instead of relying on frame 0.
    chunk_step = SAM3VideoSegmenter.CHUNK_SIZE - SAM3VideoSegmenter.CHUNK_OVERLAP
    prompt_groups_by_frame = {}
    for frame_start in range(0, total_frames, chunk_step):
        frame_end = min(frame_start + SAM3VideoSegmenter.CHUNK_SIZE, total_frames)
        anchor = (frame_start + frame_end - 1) // 2
        groups = []
        if not head_invalid[anchor]:
            if not left_invalid[anchor]:
                pts, labels = make_point_prompts(
                    left_kp[anchor], head_pose[anchor], K, H, W
                )
                if pts is not None and len(pts) >= 3:
                    groups.append((pts, labels))
                    if anchor == (total_frames - 1) // 2:
                        stats["left_prompted"] = 1
            if not right_invalid[anchor]:
                pts, labels = make_point_prompts(
                    right_kp[anchor], head_pose[anchor], K, H, W
                )
                if pts is not None and len(pts) >= 3:
                    groups.append((pts, labels))
                    if anchor == (total_frames - 1) // 2:
                        stats["right_prompted"] = 1
        prompt_groups_by_frame[anchor] = groups
    middle_anchor = (total_frames - 1) // 2
    first_prompt_groups = prompt_groups_by_frame.get(middle_anchor, [])
    stats["arm_prompted"] = int(
        mask_mode in ("both", "arms") and bool(first_prompt_groups)
    )

    # SAM3 video inference needs a numeric frame directory. Reuse the original
    # JPEG bytes so staging does not decode/re-encode the episode.
    with tempfile.TemporaryDirectory(
        prefix=f".sam3_video_{ep_name}_", dir=output_dir
    ) as stage_root:
        frame_dir = Path(stage_root) / "frames"
        frame_dir.mkdir()
        for t in range(total_frames):
            img_bytes = _unwrap_img_bytes(imgs[t])
            if not isinstance(img_bytes, (bytes, bytearray, memoryview)):
                img_bytes = bytes(img_bytes)
            (frame_dir / f"{t:06d}.jpg").write_bytes(img_bytes)

        person_masks, arm_masks = segmenter.segment_episode(
            frame_dir=frame_dir,
            num_frames=total_frames,
            frame_width=W,
            frame_height=H,
            body_prompt=body_prompt,
            prompt_groups=first_prompt_groups,
            mask_mode=mask_mode,
            prompt_groups_by_frame=prompt_groups_by_frame,
        )

    expected_shape = (total_frames, H, W)
    if person_masks.shape != expected_shape or arm_masks.shape != expected_shape:
        raise ValueError(
            f"SAM3 video returned person={person_masks.shape}, arm={arm_masks.shape}, "
            f"expected {expected_shape}"
        )

    all_masks = np.zeros(expected_shape, dtype=np.uint8)
    person_masks = _paper_mask_postprocess(person_masks)
    arm_masks = _paper_mask_postprocess(arm_masks)
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_px * 2 + 1, dilation_px * 2 + 1),
        )
        for mask_seq in (person_masks, arm_masks):
            for frame_idx in range(total_frames):
                mask_seq[frame_idx] = (
                    cv2.dilate(mask_seq[frame_idx] * 255, kernel) > 0
                ).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(ep_out, "mask_overlay.mp4")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (W * 2, H))

    for t in range(total_frames):
        img_bytes = _unwrap_img_bytes(imgs[t])
        rgb = simplejpeg.decode_jpeg(img_bytes, colorspace="RGB")

        all_masks[t] = person_masks[t] | arm_masks[t]

        if np.any(person_masks[t]):
            stats["person_masked"] += 1
        if np.any(arm_masks[t]):
            stats["arm_masked"] += 1
        if np.any(all_masks[t]):
            stats["masked"] += 1

        # Visualization: original frame on the left, mask overlay on the right.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = bgr.copy()
        person_only = (person_masks[t] > 0) & (arm_masks[t] == 0)
        arm_only = (arm_masks[t] > 0) & (person_masks[t] == 0)
        both = (person_masks[t] > 0) & (arm_masks[t] > 0)
        overlay[person_only] = (
            overlay[person_only] * 0.4 + np.array([0, 0, 200]) * 0.6
        ).astype(np.uint8)
        overlay[arm_only] = (
            overlay[arm_only] * 0.4 + np.array([0, 200, 0]) * 0.6
        ).astype(np.uint8)
        overlay[both] = (
            overlay[both] * 0.4 + np.array([200, 0, 200]) * 0.6
        ).astype(np.uint8)
        # Draw the mask boundaries.
        contours, _ = cv2.findContours(all_masks[t], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        combined = np.hstack([bgr, overlay])
        cv2.putText(combined, f"frame {t}/{total_frames}", (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(combined)

        if t % 20 == 0:
            print(f"    frame {t}/{total_frames} mask_area={all_masks[t].sum()/(H*W)*100:.1f}%")

    writer.release()
    try:
        from .config import reencode_h264
    except ImportError:
        from config import reencode_h264
    reencode_h264(video_path)

    # Save mask arrays as NPZ files.
    mask_path = os.path.join(ep_out, "masks.npz")
    np.savez_compressed(
        os.path.join(ep_out, "person_masks.npz"), masks=person_masks
    )
    np.savez_compressed(
        os.path.join(ep_out, "hand_arm_masks.npz"), masks=arm_masks
    )
    np.savez_compressed(mask_path, masks=all_masks)

    print(f"  → {mask_path} ({all_masks.nbytes/1e6:.1f}MB raw)")
    print(f"  → {video_path}")
    print(f"  Stats: {stats}")
    return stats


def build_arg_parser():
    """Build the argument parser for the mask subcommand."""
    parser = argparse.ArgumentParser(description="Step 3: SAM3 Human Mask")
    parser.add_argument("--zarr_dir", required=True, help="EgoVerse episode directory")
    parser.add_argument("--output_dir", default="data_output/03_mask")
    parser.add_argument(
        "--sam3_checkpoint",
        default=None,
        help="Path to the SAM3 checkpoint (or set EGO2ROBOT_SAM3_CKPT)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--dilation_px",
        type=int,
        default=0,
        help="Optional extra dilation after Ego2Robot mask cleanup (default: 0)",
    )
    parser.add_argument("--body_prompt", default="person", help="SAM3 text prompt for visible human body")
    parser.add_argument(
        "--mask_mode",
        choices=("both", "person", "arms"),
        default="both",
        help="both=person+hand/arm tracks; person=text only; arms=geometric boxes only",
    )
    parser.add_argument(
        "--no_body",
        action="store_true",
        help="Deprecated alias for --mask_mode arms",
    )
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument("--max_frames", type=int, default=None, help="cap frames per episode (smoke test)")
    return parser


def run(args):
    """Load the SAM3 video predictor and propagate masks for all episodes."""
    if not HAS_CV2:
        print("ERROR: opencv-python required")
        sys.exit(1)
    if not HAS_SIMPLEJPEG:
        print("ERROR: simplejpeg required")
        sys.exit(1)

    sam3_checkpoint = config.resolve_sam3_checkpoint(args.sam3_checkpoint)
    if sam3_checkpoint is None:
        print("ERROR: Cannot find SAM3 checkpoint. Provide --sam3_checkpoint or set EGO2ROBOT_SAM3_CKPT")
        sys.exit(1)
    print(f"SAM3 checkpoint: {sam3_checkpoint}")

    # Initialize the SAM3 video predictor. Prompts are seeded once per episode and
    # propagated with the tracker's temporal memory.
    print("Loading SAM3 video model...")
    segmenter = SAM3VideoSegmenter(sam3_checkpoint, device=args.device)
    print("SAM3 video model ready.")

    os.makedirs(args.output_dir, exist_ok=True)

    zarr_root = Path(args.zarr_dir)
    all_eps = [p for p in sorted(zarr_root.iterdir()) if p.is_dir() and p.name != "videos"]

    if args.episodes:
        all_eps = [p for p in all_eps if p.name in args.episodes]

    print(f"Processing {len(all_eps)} episodes")
    mask_mode = getattr(args, "mask_mode", "both")
    if getattr(args, "no_body", False):
        mask_mode = "arms"
    if mask_mode in ("both", "person") and not getattr(args, "body_prompt", None):
        raise ValueError("--body_prompt must be non-empty for person tracking")

    all_stats = {}
    try:
        for ep_dir in all_eps:
            try:
                stats = process_episode(
                    str(ep_dir), args.output_dir, ep_dir.name,
                    segmenter, fps=args.fps, dilation_px=args.dilation_px,
                    max_frames=args.max_frames,
                    body_prompt=getattr(args, "body_prompt", "person"),
                    mask_mode=mask_mode,
                )
                if stats:
                    all_stats[ep_dir.name] = stats
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
    finally:
        segmenter.close()

    # Summary
    print("\n" + "=" * 60)
    print("Step 3 Summary")
    print("=" * 60)
    for ep, st in all_stats.items():
        print(f"  {ep[:30]:32s} masked={st['masked']}/{st['total']} "
              f"({st['masked']/st['total']*100:.0f}%) "
              f"person={st['person_masked']} arm={st['arm_masked']} "
              f"mode={st['mask_mode']}")
    print("Done.")


def main():
    """Entry point for the mask command."""
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
