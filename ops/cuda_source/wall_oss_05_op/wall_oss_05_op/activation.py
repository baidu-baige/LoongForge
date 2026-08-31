# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Wall-X under the Apache-2.0 License.


"""Activation operators (SwiGLU)."""

import logging

import torch
import torch.nn.functional as F

from wall_oss_05_op.base import OpsProxy

logger = logging.getLogger(__name__)


class SwiGLUOp(OpsProxy):
    """Fused SwiGLU: swiglu(gate, up) -> silu(gate) * up.

    The CUDA backend is bit-identical to the PyTorch fallback in both forward and
    backward (eager's intermediate bf16 rounding points and ATen's association
    order for ``silu_backward`` are reproduced explicitly), so switching backends
    does not change the training loss. It also avoids keeping the intermediate
    activation in HBM.

    Inputs the kernel cannot consume -- anything but contiguous-last-dim bf16 on
    CUDA, e.g. fp32 diagnostic runs -- fall back per call. Non-contiguous
    ``split()`` views of a fused gate_up projection are supported directly.
    """

    def __init__(self, name=None):
        """Initialize the SwiGLU operator proxy."""
        super().__init__(name)
        self._input_fallback_logged = False

    def _cuda_supported(self, gate, up):
        """Return whether the inputs satisfy the CUDA kernel constraints."""
        return (
            type(gate) is torch.Tensor
            and type(up) is torch.Tensor
            and gate.is_cuda
            and gate.dtype is torch.bfloat16
            and up.dtype is torch.bfloat16
            and gate.shape == up.shape
            and gate.dim() >= 2
            and gate.stride(-1) == 1
            and up.stride(-1) == 1
        )

    def _get_cuda_kernel(self):
        """Build and return the CUDA dispatch function when available."""
        try:
            from wall_oss_05_op._cuda_ext import is_exact_available
            from wall_oss_05_op._cuda_wrappers import (
                SWIGLU_EXACT_SYMBOLS,
                has_exact_symbols,
                swiglu_exact_kernel,
            )

            if not is_exact_available() or not has_exact_symbols(SWIGLU_EXACT_SYMBOLS):
                return None
        except ImportError:
            return None
        except Exception as e:
            logger.warning("SwiGLUOp: CUDA kernel load failed: %s", e)
            return None

        def _dispatch(gate, up):
            """Dispatch one call to CUDA or the PyTorch fallback."""
            if self._cuda_supported(gate, up):
                return swiglu_exact_kernel(gate, up)
            if not self._input_fallback_logged:
                self._input_fallback_logged = True
                logger.warning(
                    "SwiGLUOp: CUDA kernel does not support these inputs, using "
                    "PyTorch per call (gate=%s/%s, up=%s/%s)",
                    type(gate).__name__,
                    gate.dtype,
                    type(up).__name__,
                    up.dtype,
                )
            return self._pytorch_fallback(gate, up)

        return _dispatch

    def _pytorch_fallback(self, gate, up):
        """Compute SwiGLU with eager PyTorch operators."""
        return F.silu(gate) * up


swiglu = SwiGLUOp("swiglu")
