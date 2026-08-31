# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Wall-X (https://github.com/X-Square-Robot/wall-x)
# under the Apache-2.0 License.

"""RMS normalization operator."""

import logging

import torch

from wall_oss_05_op.base import OpsProxy

logger = logging.getLogger(__name__)


class RMSNormOp(OpsProxy):
    """RMS normalization: rmsnorm(x, weight, eps).

    Accepts ``rmsnorm(x, weight, eps)``.

    The CUDA backend is bit-identical to the PyTorch fallback in both forward and
    backward (the three reductions are ATen calls; only the pointwise work is
    fused), so switching backends does not change the training loss. bf16 and fp32
    are both supported -- the LLM norms (``input_layernorm`` /
    ``post_attention_layernorm`` / ``model.norm``) are deliberately kept in fp32 by
    ``convert_to_mix_precision``, so fp32 is the majority of the call sites.
    Anything else (fp16, mixed x/weight dtypes, non-contiguous last dim, tensor
    subclasses such as DTensor) falls back per call.
    """

    def __init__(self, name=None):
        """Initialize the RMSNorm operator proxy."""
        super().__init__(name)
        self._input_fallback_logged = False

    def _cuda_supported(self, hidden_states, weight):
        """Return whether the inputs satisfy the RMSNorm CUDA constraints."""
        return (
            type(hidden_states) is torch.Tensor
            and type(weight) in (torch.Tensor, torch.nn.Parameter)
            and hidden_states.is_cuda
            and hidden_states.dtype in (torch.bfloat16, torch.float32)
            and weight.dtype is hidden_states.dtype
            and hidden_states.stride(-1) == 1
            and weight.is_contiguous()
        )

    def _get_cuda_kernel(self):
        """Build and return the RMSNorm CUDA dispatch function when available."""
        try:
            from wall_oss_05_op._cuda_ext import is_exact_available
            from wall_oss_05_op._cuda_wrappers import (
                RMSNORM_EXACT_SYMBOLS,
                has_exact_symbols,
                rmsnorm_exact_kernel,
            )

            if not is_exact_available() or not has_exact_symbols(RMSNORM_EXACT_SYMBOLS):
                return None
        except ImportError:
            return None
        except Exception as e:
            logger.warning("RMSNormOp: CUDA kernel load failed: %s", e)
            return None

        def _dispatch(hidden_states, weight, eps=1e-6):
            """Dispatch one call to CUDA or the PyTorch fallback."""
            if self._cuda_supported(hidden_states, weight):
                return rmsnorm_exact_kernel(hidden_states, weight, eps)
            if not self._input_fallback_logged:
                self._input_fallback_logged = True
                logger.warning(
                    "RMSNormOp: CUDA kernel does not support these inputs, using "
                    "PyTorch per call (x=%s/%s, w=%s/%s)",
                    type(hidden_states).__name__,
                    hidden_states.dtype,
                    type(weight).__name__,
                    weight.dtype,
                )
            return self._pytorch_fallback(hidden_states, weight, eps)

        return _dispatch

    def _pytorch_fallback(self, hidden_states, weight, eps=1e-6):
        """Compute RMSNorm with eager PyTorch operators."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return (weight * hidden_states).to(input_dtype)


rmsnorm = RMSNormOp("rmsnorm")
