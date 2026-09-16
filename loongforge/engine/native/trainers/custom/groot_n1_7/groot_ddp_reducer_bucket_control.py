# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge compatibility exports for the installed DDP bucket AOT op."""

try:
    from groot_n1_7_op.groot_ddp_reducer_bucket_control import (
        get_buckets,
        initialize_buckets,
    )
except ImportError as exc:
    raise ImportError(
        "The GR00T-N1.7 AOT operators are unavailable. Install "
        "DeepTraining/cuda_source/groot_n1_7_op in the runtime Python environment."
    ) from exc


__all__ = ["get_buckets", "initialize_buckets"]
