# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge compatibility exports for the installed Wall-OSS-0.5 ops."""

try:
    from wall_oss_05_op import (
        get_rope_index,
        get_window_index,
        log_backend_inventory,
        m_rope,
        permute,
        rmsnorm,
        rot_pos_emb,
        swiglu,
        unpermute,
    )
except ImportError as exc:
    raise ImportError(
        "The Wall-OSS-0.5 CUDA operators are unavailable. Install "
        "ops/cuda_source/wall_oss_05_op in the runtime Python environment."
    ) from exc


__all__ = [
    "get_rope_index",
    "get_window_index",
    "log_backend_inventory",
    "m_rope",
    "permute",
    "rmsnorm",
    "rot_pos_emb",
    "swiglu",
    "unpermute",
]
