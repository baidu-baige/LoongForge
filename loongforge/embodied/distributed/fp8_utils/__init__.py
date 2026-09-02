# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Framework-managed FP8 support for embodied models.

FP8 setup is split into backend-neutral policy and backend-specific execution:

* ``targets`` resolves model-owned defaults and CLI overrides into subtrees.
* ``backend`` dispatches conversion and forward context to TE or TorchAO.
* ``selection`` implements backend-neutral module selection.
* ``te_fp8`` implements ``te.Linear`` replacement and TE recipe/context.
* ``torchao_fp8`` implements native PyTorch ``Float8Linear`` replacement.

``apply_fp8_linear_conversion`` performs target resolution and replacement
before activation checkpointing and DDP/FSDP wrapping, including custom model
wrappers. The trainer later enters ``te.fp8_autocast`` around the complete model
forward. Replacing a module does not store its master parameters as FP8: weights
retain their original dtype, while TransformerEngine manages FP8 casts, scales,
and GEMMs during forward and backward.

TransformerEngine imports stay inside the functions that need them. A build
without TE can therefore import the embodied distributed stack and run normally
as long as ``--fp8`` remains disabled.
"""

from .backend import (
    FP8_BACKEND_CHOICES,
    apply_fp8_linear_conversion,
    convert_linear_for_fp8,
    resolve_fp8_forward_ctx,
)
from .selection import FP8_GEMM_ALIGNMENT
from .te_fp8 import (
    convert_linear_to_te,
    resolve_fp8_autocast_ctx,
    te_checkpoint_fn,
)
from .targets import get_default_fp8_targets, resolve_fp8_targets

__all__ = [
    "FP8_GEMM_ALIGNMENT",
    "FP8_BACKEND_CHOICES",
    "apply_fp8_linear_conversion",
    "convert_linear_for_fp8",
    "convert_linear_to_te",
    "get_default_fp8_targets",
    "resolve_fp8_autocast_ctx",
    "resolve_fp8_forward_ctx",
    "resolve_fp8_targets",
    "te_checkpoint_fn",
]
