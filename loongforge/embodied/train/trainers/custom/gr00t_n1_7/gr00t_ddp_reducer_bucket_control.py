# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge compatibility exports for the installed DDP bucket AOT op."""

try:
    from gr00t_n1_7_op.gr00t_ddp_reducer_bucket_control import (
        get_buckets,
        initialize_buckets,
    )
except ImportError as exc:
    raise ImportError(
        "The GR00T-N1.7 AOT operators are unavailable. Install "
        "ops/cuda_source/gr00t_n1_7_op in the runtime Python environment."
    ) from exc


__all__ = ["get_buckets", "initialize_buckets"]
