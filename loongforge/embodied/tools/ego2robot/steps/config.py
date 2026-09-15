# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Global ego2robot configuration. Server-specific absolute paths are centralized
here and can all be overridden with environment variables.

These paths were module-level constants hard-coded in individual scripts under
`ego-to-robot-data-mvp/src/`. The release keeps them here so migration to a new
machine or container requires changing one place (or only setting environment
variables).
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Location of the standalone ProPainter repository (inpaint.py invokes
# inference_propainter.py there via subprocess).
PROPAINTER_DIR = os.environ.get(
    "EGO2ROBOT_PROPAINTER_DIR",
    str(PROJECT_ROOT / "third_party" / "ProPainter"),
)

# Staging root for ProPainter's intermediate PNG sequences (tmpfs is recommended
# to avoid filling the disk).
PROPAINTER_STAGE_ROOT = os.environ.get(
    "EGO2ROBOT_PROPAINTER_STAGE_ROOT",
    "/dev/shm/pp_stage",
)

# mujoco_menagerie cache directory (the robot_descriptions package downloads
# assets here at runtime; offline environments can point to a preloaded cache).
MENAGERIE_DIR = os.environ.get(
    "EGO2ROBOT_MENAGERIE_DIR",
    "/root/.cache/robot_descriptions/mujoco_menagerie",
)

# SAM3 checkpoint. The checkpoint is intentionally explicit because SAM3
# weights are not interchangeable across package revisions.
SAM3_CHECKPOINT_OVERRIDE = os.environ.get("EGO2ROBOT_SAM3_CKPT", None)

# Local Depth Anything 3 weights (official DA3-BASE format with config.json and
# model.safetensors). Point the depth step here for offline use instead of
# passing --model_id every time.
DA3_MODEL_DIR = os.environ.get("EGO2ROBOT_DA3_MODEL_DIR", None)


def resolve_da3_model_dir(cli_arg: str | None) -> str | None:
    """Resolve the DA3 weight source in priority order: CLI --model_id, then env.

    Return None when neither source is usable so the caller can fall back to the
    default Hugging Face model identifier.
    """
    if cli_arg:
        return cli_arg
    return DA3_MODEL_DIR


def resolve_sam3_checkpoint(cli_arg: str | None) -> str | None:
    """Resolve SAM3 checkpoint path: CLI argument takes precedence over env."""
    return cli_arg or SAM3_CHECKPOINT_OVERRIDE


def dataset_intrinsics_k(attrs, camera: str = "front_1", img_shape=None):
    """Extract camera intrinsics K (3, 3) from zarr episode attrs.

    The matrix is scaled to the target image resolution. Two storage formats are
    supported:
      - legacy: attrs["intrinsics"][camera] = (3, 4)/(3, 3) matrix already at
        image resolution
      - new: attrs["intrinsics"] = flat dict {fl_x, fl_y, cx, cy, w, h, ...}
        at the native resolution (for example 1920x1080), requiring scaling for
        a resized image such as 640x360
    Return None when no usable intrinsics are available.
    """
    import numpy as np
    raw = attrs.get("intrinsics", {})
    if not isinstance(raw, dict) or not raw:
        return None
    if camera in raw:
        K = np.asarray(raw[camera], dtype=np.float64)
        # Legacy format is (3, 4) or (3, 3); take the upper-left 3x3 block.
        if K.ndim == 2 and K.shape[0] >= 3 and K.shape[1] >= 3:
            return K[:3, :3]
        return None
    if "fl_x" not in raw:
        return None
    fl_x, fl_y = float(raw["fl_x"]), float(raw["fl_y"])
    cx = float(raw.get("cx", 0.0))
    cy = float(raw.get("cy", 0.0))
    nw = float(raw.get("w", 0.0))
    nh = float(raw.get("h", 0.0))
    if img_shape is not None:
        ih, iw = float(img_shape[0]), float(img_shape[1])
        sx = (iw / nw) if nw > 0 else 1.0
        sy = (ih / nh) if nh > 0 else 1.0
        fl_x, cx = fl_x * sx, cx * sx
        fl_y, cy = fl_y * sy, cy * sy
    return np.array([[fl_x, 0.0, cx], [0.0, fl_y, cy], [0.0, 0.0, 1.0]],
                    dtype=np.float64)


def dataset_fy_for_height(attrs, img_height: float) -> float | None:
    """Return vertical focal length fy for image height ``img_height``.

    This is used for retarget rendering.
    - Legacy matrix format: use K[1, 1] directly.
    - Flat dict format: scale ``fl_y`` by ``img_height / native_h``.
    Return None when no usable intrinsics are available; callers decide whether
    to fall back or raise an error.
    """
    import numpy as np
    raw = attrs.get("intrinsics", {})
    if not isinstance(raw, dict) or not raw:
        return None
    if "front_1" in raw:
        K = np.asarray(raw["front_1"], dtype=np.float64)
        if K.ndim >= 2 and K.shape[:2] == (3, 3):
            return float(K[1, 1])
        return None
    if "fl_y" in raw and float(raw.get("h", 0.0)) > 0:
        return float(raw["fl_y"]) * float(img_height) / float(raw["h"])
    return None


def _has_usable_intrinsics(raw) -> bool:
    """Return whether intrinsics can be used for projection."""
    if not isinstance(raw, dict) or not raw:
        return False
    return "front_1" in raw or "fl_x" in raw


def fallback_episode_attrs(episode_dir: str):
    """Return ``(episode_attrs, source_episode)``.

    If ``episode_dir`` lacks valid intrinsics, borrow them from another episode
    in the same parent directory (camera settings are consistent within a
    batch), print a notice, and return them. Return ``(None, None)`` if all are
    missing.
    """
    import glob
    import os
    import zarr as _z

    base = os.path.basename(episode_dir)
    own = None
    try:
        own = dict(_z.open_group(episode_dir, mode="r").attrs)
    except Exception:
        pass
    if own is not None and _has_usable_intrinsics(own.get("intrinsics")):
        return own, base

    zarr_dir = os.path.dirname(episode_dir)
    for cand in sorted(glob.glob(os.path.join(zarr_dir, "*"))):
        if cand == episode_dir or not os.path.isdir(cand):
            continue
        if not os.path.exists(os.path.join(cand, "zarr.json")):
            continue
        try:
            cand_attrs = dict(_z.open_group(cand, mode="r").attrs)
        except Exception:
            continue
        if _has_usable_intrinsics(cand_attrs.get("intrinsics")):
            print(f"  ⚠️ {base}: intrinsics missing, borrowing from "
                  f"{os.path.basename(cand)}")
            return cand_attrs, os.path.basename(cand)
    return None, None


def reencode_h264(src: str, crf: int = 20):
    """Re-encode a video as H.264 (yuv420p + faststart) for browser previews.

    OpenCV's mp4v output is MPEG-4 Part 2, which Chromium does not recognize.
    Steps write mp4v first and call this function at the end to replace it with
    H.264 through an atomic temporary-file replacement. On failure, keep the
    original file and print a warning without interrupting the pipeline.
    """
    import subprocess
    tmp = f"{src}.h264tmp.mp4"
    try:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
               "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", "-an", tmp]
        subprocess.run(cmd, check=True, timeout=1800)
        os.replace(tmp, src)
        return True
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        print(f"  ⚠️ H264 re-encode failed for {src}: {e}")
        return False

# ============================================================
# Downstream robot morphology declarations (for state/action dimension slicing).
# ============================================================
# Presets: (left_arm, left_grip, right_arm, right_grip, description)
ROBOT_PRESETS = {
    "panda_dual_g2": (7, 2, 7, 2, "Panda dual arm + 2-DOF gripper (full menagerie, default)"),
    "panda_dual_g1": (7, 1, 7, 1, "Panda dual arm + 1-DOF gripper (left finger, standard Franka convention)"),
    "panda_left_g2":  (7, 2, 0, 0, "Left arm only + 2-DOF gripper"),
    "panda_left_g1":  (7, 1, 0, 0, "Left arm only + 1-DOF gripper"),
}


def parse_robot_spec(spec: str | None):
    """Parse a robot morphology declaration.

    Return ``(left_arm, left_grip, right_arm, right_grip)``.
    - Preset names such as ``panda_dual_g2`` are supported.
    - Custom ``7g2r7g1`` syntax is supported:
      ``<left-arm-DOF>g<left-gripper-DOF>r<right-arm-DOF>g<right-gripper-DOF>``.
      Omitting ``r`` means no right arm. Arm DOF must be 7 or 0 and gripper DOF
      must be 0, 1, or 2. See the documentation for mapping to the source 18D
      data layout.
    """
    import re
    if not spec:
        return (7, 2, 7, 2)
    if spec in ROBOT_PRESETS:
        la, lg, ra, rg, _ = ROBOT_PRESETS[spec]
        return (la, lg, ra, rg)
    m = re.fullmatch(r"(\d+)g(\d+)r(\d+)g(\d+)", spec)
    if m:
        la, lg, ra, rg = (int(x) for x in m.groups())
    else:
        m = re.fullmatch(r"(\d+)g(\d+)", spec)
        if not m:
            raise ValueError(f"Unable to parse robot spec: {spec}")
        la, lg = (int(x) for x in m.groups())
        ra, rg = 0, 0
    if la not in (7, 0) or ra not in (7, 0) or lg not in (0, 1, 2) or rg not in (0, 1, 2):
        raise ValueError(
            "Supported values: arm DOF must be 7 or 0 (matching source IK); "
            "gripper DOF must be 0, 1, or 2"
        )
    return (la, lg, ra, rg)


def robot_layout(spec):
    """Compute output indices and names from a morphology declaration.

    Source layout (18D):
    [left_joint1..7, left_f1, left_f2, right_joint1..7, right_f1, right_f2]
    Return ``(indices: list[int], names: list[str])``.
    """
    la, lg, ra, rg = spec[0], spec[1], spec[2], spec[3]
    idx, names = [], []
    if la:
        idx += list(range(7))
        names += [f"left_joint{i}" for i in range(1, 8)]
        if lg == 1:
            idx += [7]
            names += ["left_finger_joint1"]
        elif lg == 2:
            idx += [7, 8]
            names += ["left_finger_joint1", "left_finger_joint2"]
    if ra:
        base = 9
        idx += [base + i for i in range(7)]
        names += [f"right_joint{i}" for i in range(1, 8)]
        if rg == 1:
            idx += [base + 7]
            names += ["right_finger_joint1"]
        elif rg == 2:
            idx += [base + 7, base + 8]
            names += ["right_finger_joint1", "right_finger_joint2"]
    return idx, names
