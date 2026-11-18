# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Algorithm implementations for the L2 stream modules (Lane B).

Division of labour inside this directory: ``spec.py`` / ``state.py`` define
shapes and protocols, ``engine.py`` defines the call order, and this package
defines how a single layer is actually computed.

Classes here are ``nn.Module`` subclasses and own all parameters
(INV-3: the engine holds no parameters). This is a subpackage rather than a
single file because the WAN family and later non-WAN topologies share the
``StreamModule`` protocol but not the operators; keeping both operator sets
in one file would make them reference each other.
"""

from __future__ import annotations

from .wan_dit import (
    FP32LayerNorm,
    RMSNorm,
    SelfAttentionIO,
    SlottedCrossAttention,
    WanDiTBlock,
    assemble_freqs_3d,
    modulate,
    rope_apply,
    rope_freqs_1d,
    rope_freqs_3d,
    sinusoidal_embedding,
)
from .wan_streams import (
    ActionStream,
    WanDiTStream,
    WanVideoStream,
    build_context_projector,
)

__all__ = [
    "ActionStream", "FP32LayerNorm", "RMSNorm", "SelfAttentionIO",
    "SlottedCrossAttention", "WanDiTBlock", "WanDiTStream", "WanVideoStream",
    "assemble_freqs_3d", "build_context_projector", "modulate", "rope_apply",
    "rope_freqs_1d", "rope_freqs_3d", "sinusoidal_embedding",
]
