# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Wall-X under the Apache-2.0 License.

"""wall_oss_05_op: standalone CUDA operator package.

Each operator proxy resolves lazily to a CUDA inline kernel when the compiled
extension is importable, and otherwise falls back to pure PyTorch.

Install:
    pip install --no-build-isolation -e .

Usage:
    from wall_oss_05_op import rope, m_rope, rot_pos_emb
    from wall_oss_05_op import rmsnorm, swiglu
    from wall_oss_05_op import permute, unpermute
    from wall_oss_05_op import get_rope_index, get_window_index
"""

from wall_oss_05_op.activation import swiglu
from wall_oss_05_op.base import backend_inventory, log_backend_inventory
from wall_oss_05_op.index import get_rope_index, get_window_index
from wall_oss_05_op.moe import permute, unpermute
from wall_oss_05_op.norm import rmsnorm
from wall_oss_05_op.rope import m_rope, rope, rot_pos_emb

__all__ = [
    "rmsnorm",
    "swiglu",
    "rope",
    "m_rope",
    "rot_pos_emb",
    "permute",
    "unpermute",
    "get_rope_index",
    "get_window_index",
    "backend_inventory",
    "log_backend_inventory",
]
