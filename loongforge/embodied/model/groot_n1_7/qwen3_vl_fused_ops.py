# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge compatibility exports for the installed GR00T-N1.7 AOT ops."""

try:
    from groot_n1_7_op.qwen3_vl_fused_ops import (
        qwen3_vl_fused_text_rope_forward,
        qwen3_vl_fused_text_rms_norm_forward,
        qwen3_vl_fused_text_silu_mul_forward,
        qwen3_vl_fused_vision_rope_forward,
    )
except ImportError as exc:
    raise ImportError(
        "The GR00T-N1.7 AOT operators are unavailable. Install "
        "DeepTraining/cuda_source/groot_n1_7_op in the runtime Python environment."
    ) from exc


__all__ = [
    "qwen3_vl_fused_text_rope_forward",
    "qwen3_vl_fused_text_rms_norm_forward",
    "qwen3_vl_fused_text_silu_mul_forward",
    "qwen3_vl_fused_vision_rope_forward",
]
