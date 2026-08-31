"""AOT package for the GR00T-N1.7 fused operators."""

try:
    from .groot_ddp_reducer_bucket_control import get_buckets, initialize_buckets
    from .groot_fused_adamw import (
        capturable_grad_scaled_step,
        capturable_step,
        eager_step,
    )
    from .qwen3_vl_fused_ops import (
        qwen3_vl_fused_text_rope_forward,
        qwen3_vl_fused_text_rms_norm_forward,
        qwen3_vl_fused_text_silu_mul_forward,
        qwen3_vl_fused_vision_rope_forward,
    )
except ImportError as exc:
    # The three compiled extensions are loaded eagerly, so any missing or
    # ABI-incompatible .so surfaces here. Python reports it as a suspected
    # circular import, which hides the real cause; restate it instead.
    raise ImportError(
        "groot_n1_7_op compiled extensions are unavailable. Build them from "
        "the package root:\n"
        "    pip install --no-build-isolation -e .\n"
        "If the build already succeeded, verify that the PyTorch version used "
        "to build matches the one importing this package.\n"
        f"Original error: {exc}"
    ) from exc

__all__ = [
    "capturable_grad_scaled_step",
    "capturable_step",
    "eager_step",
    "get_buckets",
    "initialize_buckets",
    "qwen3_vl_fused_text_rope_forward",
    "qwen3_vl_fused_text_rms_norm_forward",
    "qwen3_vl_fused_text_silu_mul_forward",
    "qwen3_vl_fused_vision_rope_forward",
]
