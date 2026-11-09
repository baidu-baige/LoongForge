# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Thin DVLT (Déjà View) inference wrapper.

All DVLT imports are contained here so the rest of the pipeline only depends
on this module's small, stable API.  The DVLT package (``pip install -e .``,
NVIDIA repo) must be importable in the running environment; the module is
loaded lazily so the gauss step can produce a clear error message when the
checkpoint or package is missing.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)
_ACCELERATOR = None  # shared accelerate state (created on first load)

# DVLT's default patch embedding is DINOv2 ViT-B/14 with register tokens.
DEFAULT_IMG_SIZE = 504
DEFAULT_PATCH_SIZE = 14
DEFAULT_CHECKPOINT = "nvidia/dvlt"  # local dir or HF hub id (offline: local dir)


@dataclass
class DvltResult:
    """Standardized DVLT outputs for one episode."""

    c2w: np.ndarray            # (S, 3, 4) camera-to-world, OpenCV convention
    K: np.ndarray              # (S, 3, 3) per-view intrinsics (image-size scale)
    depth: np.ndarray          # (S, h, w) float32, camera-z depth
    world_points: np.ndarray   # (S, h, w, 3) float32, DVLT world frame
    conf: np.ndarray           # (S, h, w) float32, world-point confidence [0, 1]
    img_size: int = DEFAULT_IMG_SIZE

    @property
    def height(self) -> int:
        return int(self.depth.shape[1])

    @property
    def width(self) -> int:
        return int(self.depth.shape[2])


@functools.lru_cache(maxsize=2)
def load_model(checkpoint: str,
               img_size: int = DEFAULT_IMG_SIZE,
               patch_size: int = DEFAULT_PATCH_SIZE,
               device: str = "cuda:0",
               dinov2_weights: Optional[str] = None):
    """Load a DVLT model (cached across episodes).

    ``checkpoint`` may be a local directory (offline) or an HF hub id.  The
    patch-embedding is *not* re-downloaded at construction (``load_patch_embed_weights
    =False``) because the checkpoint itself carries those weights; if it does
    not, ``dinov2_weights`` supplies the DINOv2 state dict offline.
    """
    from dvlt.model.dvlt.model import DVLT
    import torch
    # DVLT's load_pretrained logs through accelerate; initialize its state
    # first (no-op on a plain single-process run).
    from accelerate import Accelerator
    global _ACCELERATOR
    _ACCELERATOR = Accelerator(mixed_precision="bf16")

    ckpt = str(checkpoint or DEFAULT_CHECKPOINT)
    model = DVLT(img_size=img_size, patch_size=patch_size,
                 load_patch_embed_weights=False)
    try:
        missing, unexpected = model.load_pretrained(ckpt, strict=True)
    except Exception as e:
        raise RuntimeError(
            f"DVLT.load_pretrained({ckpt!r}) failed: {e}\n"
            f"Provide the nvidia/dvlt weights as a local directory "
            f"(--checkpoint <dir> with config.json + model.safetensors).") from e
    pe_missing = [k for k in (missing or []) if k.startswith("patch_embed")]
    if pe_missing:
        if not dinov2_weights:
            raise RuntimeError(
                f"checkpoint is missing {len(pe_missing)} patch_embed weights "
                f"(e.g. {pe_missing[0]}) and no DINOv2 weights were supplied: "
                f"pass --dinov2_weights <DINOv2-vitb14-reg4 .pth>")
        from dvlt.model_components import dinov2
        patch = model.model.patch_embed_encoder
        sd = torch.load(dinov2_weights, map_location="cpu")
        sd = sd.get("state_dict", sd)
        patch.load_state_dict(sd, strict=False)
        log.info("loaded DINOv2 patch-embed weights from %s", dinov2_weights)
    # setup_test unwraps, moves to the accelerator device, and switches the
    # wrapped torch module to eval mode (the DVLT facade is not an nn.Module).
    model.setup_test(_ACCELERATOR if _ACCELERATOR is not None else Accelerator(mixed_precision="bf16"))
    return model


def run_dvlt(model, frames: List, device: str = "cuda:0",
             img_size: int = DEFAULT_IMG_SIZE,
             patch_size: int = DEFAULT_PATCH_SIZE,
             inference_steps: Optional[int] = None) -> DvltResult:
    """Run feed-forward reconstruction over ``frames`` (PIL images)."""
    import torch
    from dvlt.common.constants import PredictionField
    from dvlt.util.preprocess import preprocess_images
    from accelerate import Accelerator

    if inference_steps is not None:
        model.inference_steps = inference_steps
    acc = Accelerator(mixed_precision="bf16")
    batch = preprocess_images(frames, img_size=img_size, patch_size=patch_size,
                              device=torch.device(device))
    with torch.no_grad(), acc.autocast():
        preds = model.predict(batch, acc)

    cameras = preds[PredictionField.CAMERAS][0]
    c2w = np.asarray(cameras.camera_to_worlds.detach().float().cpu().numpy(),
                     dtype=np.float64)          # (S, 3, 4)
    K = np.asarray(cameras.get_intrinsics_matrices().detach().float().cpu().numpy(),
                   dtype=np.float64)            # (S, 3, 3)
    depth = np.asarray(preds[PredictionField.DEPTHS][0].float().cpu().numpy(),
                       dtype=np.float32)        # (S, h, w)
    conf = np.asarray(preds[PredictionField.WORLD_POINTS_DIRECT_CONF][0]
                      .float().cpu().numpy(), dtype=np.float32)
    wp = np.asarray(preds[PredictionField.WORLD_POINTS][0].float().cpu().numpy(),
                    dtype=np.float32)           # (S, h, w, 3)
    if np.isnan(wp).any():
        log.warning("DVLT world points contain NaN; replacing with depth unprojection")
    return DvltResult(c2w=c2w, K=K, depth=depth, world_points=wp, conf=conf,
                      img_size=img_size)
