"""Python interface for Qwen3-VL fused inference operators."""

import torch

from . import _qwen3_vl_fused_ops


def _extension():
    """Return the AOT extension; kept as a function for API parity with JIT wrappers."""
    return _qwen3_vl_fused_ops


def qwen3_vl_fused_vision_rope_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3-VL vision RoPE with explicit FP32 operation rounding."""
    return _extension().qwen3_vl_fused_vision_rope(query, key, cos, sin)


def qwen3_vl_fused_text_rope_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3-VL text RoPE while preserving per-op dtype rounding."""
    return _extension().qwen3_vl_fused_text_rope(query, key, cos, sin)


def qwen3_vl_fused_text_rms_norm_forward(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Apply Qwen text RMSNorm using the native mean reduction."""
    extension = _extension()
    squared = extension.qwen3_vl_fused_text_rms_norm_square(hidden_states)
    variance = squared.mean(-1, keepdim=True)
    return extension.qwen3_vl_fused_text_rms_norm_finish(
        hidden_states, variance, weight, epsilon
    )


def qwen3_vl_fused_text_silu_mul_forward(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """Apply Qwen text SiLU and multiply with BF16/FP16 rounding."""
    return _extension().qwen3_vl_fused_text_silu_mul(gate, up)
