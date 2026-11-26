# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Builders for the **frozen** towers that sit outside the core.

A tower turns a raw modality into the tensor a stream or a condition reads:
the Wan video VAE produces the video stream's latents, the Wan CLIP image
encoder produces the ``clip_image`` condition's source. None of them is
trained, none of them holds a parameter the core's state dict knows about,
and all of them are selected by the spec rather than by a YAML string.

Why a module of its own rather than private helpers in ``factory/``: the spec
layer declares *which* tower feeds *which* key (``StreamSpec.source_encoder``
and ``ConditionSpec.source_encoder`` hold a ``Spec`` wrapping one of these
functions), while the shell layer in ``factory/`` is what actually builds it,
injecting the weight path its own config carries. A provider importing from
``factory/`` would invert the dependency; both layers importing from here does
not.

The heavy ``dreamzero`` imports stay inside the function bodies on purpose:
importing a spec module must not pull a VAE implementation into the process.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
from torch import nn

if TYPE_CHECKING:
    from .spec.world import WorldModelSpec

logger = logging.getLogger(__name__)

__all__ = ["build_wan_clip", "build_wan_vae"]

# The VAE generation is selected by latent width rather than by a YAML string.
# The two generations differ in channel count *and* spatial stride, and
# ``GeometrySpec`` already carries both numbers, so a separate preset name in
# the config would be a second place for the same fact to live.
_VAE_PRESETS: Dict[int, Tuple[str, int]] = {
    16: ("Wan2.1_VAE.pth", 8),      # Wan2.1: 16 latent channels, /8 spatially
    48: ("Wan2.2_VAE.pth", 16),     # Wan2.2: 48 latent channels, /16
}

_CLIP_FILENAME = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"


def _optional_file(root: str, filename: str) -> Optional[str]:
    """Resolve ``root`` to a concrete weight file, or ``None`` if absent.

    A weight path may be a HuggingFace repo id, a local Wan checkpoint
    directory or a direct path to one weight file. Only the last two can be
    loaded from disk; a repo id that has not been downloaded yet leaves the
    tower randomly initialised, which is what ``--dry_run`` and unit tests
    need.

    Args:
        root: Directory or file path from the config.
        filename: File looked for when ``root`` is a directory.

    Returns:
        An existing file path, or ``None`` when nothing was found.
    """
    if not root:
        return None
    if os.path.isfile(root):
        return root
    candidate = os.path.join(root, filename)
    return candidate if os.path.isfile(candidate) else None


def _load_into(module: nn.Module, weights: Optional[str], what: str) -> None:
    """Load a plain state dict into a frozen tower, logging the key counts.

    ``strict=False`` on purpose: the published Wan files carry the trunk only,
    while the wrappers add buffers of their own. Counting the leftovers and
    logging them is more informative than a hard failure, because a genuine
    mismatch shows up as "nearly every key missing".

    Args:
        module: Destination module (the wrapper's inner trunk).
        weights: Path to a ``torch.save``d state dict, or ``None`` to skip.
        what: Component name used in the log line.
    """
    if weights is None:
        return
    state = torch.load(weights, map_location="cpu")
    for key in ("state_dict", "model_state"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    missing, unexpected = module.load_state_dict(state, strict=False)
    logger.info("[arch] %s loaded from %s; missing=%d unexpected=%d",
                what, weights, len(missing), len(unexpected))


def build_wan_vae(*, spec: WorldModelSpec, path: str = "") -> nn.Module:
    """Build the frozen video VAE matching the spec's latent geometry.

    Args:
        spec: Built world-model spec; its geometry fixes both the latent
            channel count and the spatial stride the VAE must have.
        path: Directory (or file) holding the Wan encoder weights. Empty
            leaves the tower randomly initialised.

    Returns:
        A frozen, eval-mode VAE module.

    Raises:
        ValueError: The geometry asks for a latent width no Wan VAE produces,
            or the declared spatial stride disagrees with that generation's.
    """
    from loongforge.embodied.model.dreamzero.modules.wan_video_vae import (
        WanVideoVAE, WanVideoVAE38)

    channels = spec.geometry.latent_channels
    if channels not in _VAE_PRESETS:
        raise ValueError(
            f"geometry.latent_channels={channels} matches no Wan VAE; "
            f"known generations produce {sorted(_VAE_PRESETS)} channels")
    filename, stride = _VAE_PRESETS[channels]
    if spec.geometry.vae_spatial_stride != stride:
        raise ValueError(
            f"geometry says vae_spatial_stride="
            f"{spec.geometry.vae_spatial_stride}, but a {channels}-channel "
            f"Wan VAE compresses by {stride}. The latent grid would be off by "
            f"{stride / spec.geometry.vae_spatial_stride:g}x in each spatial "
            "axis, which shows up as a token-count mismatch, not as a "
            "shape error")
    weights = _optional_file(path, filename)
    factory = WanVideoVAE if channels == 16 else WanVideoVAE38
    vae = factory(z_dim=channels, vae_pretrained_path=weights)
    _load_into(getattr(vae, "model", vae), weights, "VAE")
    return vae.eval().requires_grad_(False)


def build_wan_clip(*, path: str = "") -> nn.Module:
    """Build the frozen CLIP image encoder feeding the ``clip_image`` slot.

    It takes no ``spec``: unlike the VAE, nothing about this tower varies with
    the geometry, and the 1280-wide output is already declared by the
    condition that reads it.

    Args:
        path: Directory (or file) holding the Wan encoder weights. Empty
            leaves the tower randomly initialised.

    Returns:
        A frozen, eval-mode ``WanImageEncoder``.
    """
    from loongforge.embodied.model.dreamzero.modules.wan_video_image_encoder \
        import WanImageEncoder

    weights = _optional_file(path, _CLIP_FILENAME)
    encoder = WanImageEncoder(image_encoder_pretrained_path=weights)
    _load_into(getattr(encoder, "model", encoder), weights, "CLIP")
    return encoder.eval().requires_grad_(False)
