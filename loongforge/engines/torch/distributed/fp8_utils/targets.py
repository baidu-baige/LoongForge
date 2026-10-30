# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Resolve model-owned FP8 policy into Linear-conversion target subtrees.

The framework owns how ``nn.Linear`` becomes ``te.Linear``; each model owns the
architectural decision of which repeated compute blocks are safe and useful to
convert. CLI options can override that policy for experiments and bisection.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


def get_default_fp8_targets(model: nn.Module) -> Optional[dict[str, Any]]:
    """Return the model-provided FP8 target specification, if available.

    Mirrors ``train.lora.get_default_lora_targets``: the framework owns the
    mechanism, the model owns the knowledge of which of its own layers must stay
    out of FP8. Defaults normally select large Transformer/DiT block subtrees
    and leave numerically sensitive input/output heads, explicitly fp32 modules,
    frozen backbones, and custom Linear implementations untouched.

    Expected keys:
        ``module_patterns``: Qualified subtree-root patterns. Every standard
            ``nn.Linear`` descendant becomes a conversion candidate.
        ``skip_modules``: Optional patterns excluded from the positive
            selection, typically used when a broad target contains fp32-critical
            projections.
    """
    provider = getattr(model, "default_fp8_targets", None)
    return provider() if callable(provider) else None


def resolve_fp8_targets(model: nn.Module, training_args) -> tuple[Any, Any]:
    """Resolve ``(module_patterns, skip_modules)`` for the FP8 conversion pass.

    Resolution order:

    1. Reject a model/configuration that reports ``fp8_unsupported_reason``.
       This is used for lifecycle incompatibilities such as post-FSDP
       materialization, not merely for the absence of defaults.
    2. Resolve module and skip patterns independently. A CLI value wins when
       explicitly set; otherwise that field falls back to
       ``default_fp8_targets()``.
    3. Require a positive target. ``--fp8`` must never degrade into a silent
       BF16 run because a model forgot to declare its defaults.

    The returned patterns are intentionally still raw here. ``linear`` parses
    them and validates that every positive and skip pattern matches a real
    standard Linear in the instantiated model.
    """
    # Check incompatibility before honoring CLI patterns: a different target
    # cannot fix a model lifecycle that makes every pre-wrap TE replacement
    # unsafe (for example, weights that remain on meta until after FSDP).
    backend = getattr(training_args, "fp8_backend", "te")
    unsupported = getattr(model, "fp8_unsupported_reason", None)
    if callable(unsupported):
        try:
            reason = unsupported(backend)
        except TypeError:
            # Keep compatibility with model providers that implement the
            # original zero-argument hook.
            reason = unsupported()
    else:
        reason = unsupported
    if reason:
        raise ValueError(
            f"--fp8 is not supported by {type(model).__name__}: {reason}"
        )

    defaults = dict(get_default_fp8_targets(model) or {})

    module_patterns = training_args.fp8_module_patterns
    if module_patterns is None:
        module_patterns = defaults.get("module_patterns")
    skip_modules = training_args.fp8_skip_modules
    if skip_modules is None:
        skip_modules = defaults.get("skip_modules")

    if not module_patterns:
        raise ValueError(
            "--fp8 is enabled but no FP8 module patterns were resolved. Define "
            f"{type(model).__name__}.default_fp8_targets() or pass "
            "--fp8-module-patterns."
        )
    return module_patterns, skip_modules
