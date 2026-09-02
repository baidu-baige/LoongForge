# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Select and replace standard Linear layers for TransformerEngine FP8.

The conversion is structural rather than a monkey-patch: each eligible
``nn.Linear`` is replaced in the model tree by a newly constructed
``te.Linear`` carrying the same dimensions, parameter values, dtype, bias
layout, and gradient flags. TransformerEngine performs the actual FP8 casting
only when the trainer enters ``te.fp8_autocast``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ..selection import FP8_GEMM_ALIGNMENT, resolve_linear_selection

logger = logging.getLogger(__name__)


def convert_linear_to_te(
    model: nn.Module,
    raw_module_patterns: str | list[str] | None,
    raw_skip_modules: str | list[str] | None,
    min_dim: int,
    device: torch.device,
) -> int:
    """Replace ``nn.Linear`` under the selected subtrees with ``te.Linear``.

    A module pattern names a *subtree root*. For example, ``model.blocks``
    selects standard Linear descendants such as
    ``model.blocks.7.mlp.up_proj``. This intentionally differs from activation
    checkpoint patterns, which name the exact modules to wrap: FP8 intent is
    normally expressed at Transformer/DiT-block granularity rather than by
    enumerating every projection.

    Only modules whose exact type is ``nn.Linear`` are considered. Custom
    Linear subclasses and domain/category-aware projections keep their own
    implementation. Selected modules are then filtered by ``min_dim`` and
    16-alignment: small GEMMs may lose performance to FP8 setup overhead, while
    unaligned GEMMs are unsupported. Both cases are counted and left unchanged.

    Replacements are created directly on the training ``device`` because this
    pass runs before DDP/FSDP moves or shards the original model and TE modules
    require CUDA. Meta parameters cannot be copied and are rejected explicitly.

    Returns:
        Number of ``nn.Linear`` modules replaced with ``te.Linear``.
    """
    selected_keys, skipped_keys = resolve_linear_selection(
        model,
        raw_module_patterns,
        raw_skip_modules,
    )
    if not selected_keys:
        return 0

    te_linear_cls = _import_te_linear()
    converted = 0
    converted_keys = []
    below_min_dim = 0
    unaligned = 0
    # te.Linear runs its own reset_parameters, which draws from the global RNG.
    # The drawn values are irrelevant (they are overwritten by the weight copy),
    # but the advanced RNG stream is not: models that sample noise per training
    # step -- flow matching, diffusion -- would then see a different noise
    # sequence than the same run without --fp8, and the resulting loss gap looks
    # like quantization error while being pure RNG misalignment. Forking keeps
    # the conversion invisible to the RNG stream so an FP8/bf16 A/B stays
    # comparable without a reseed workaround.
    fork_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        for module_key in selected_keys:
            if module_key in skipped_keys:
                continue
            linear = model.get_submodule(module_key)
            in_features, out_features = linear.in_features, linear.out_features
            if in_features % FP8_GEMM_ALIGNMENT or out_features % FP8_GEMM_ALIGNMENT:
                unaligned += 1
                continue
            if max(in_features, out_features) < min_dim:
                below_min_dim += 1
                continue
            # Replace at the qualified path so ordinary attributes, ModuleList,
            # and ModuleDict descendants all preserve the model's public shape.
            model.set_submodule(
                module_key, _build_te_linear(te_linear_cls, linear, device)
            )
            if type(model.get_submodule(module_key)) is nn.Linear:
                raise RuntimeError(
                    "FP8 conversion did not replace nn.Linear at "
                    f"{module_key} with te.Linear."
                )
            converted_keys.append(module_key)
            converted += 1

    logger.info(
        "Converted nn.Linear to te.Linear: converted=%d skipped_by_pattern=%d "
        "below_min_dim=%d unaligned=%d (min_dim=%d, alignment=%d)",
        converted,
        len(skipped_keys),
        below_min_dim,
        unaligned,
        min_dim,
        FP8_GEMM_ALIGNMENT,
    )
    if converted_keys:
        logger.info("TE FP8 converted modules: %s", ", ".join(converted_keys))
    else:
        logger.warning(
            "TE FP8 conversion converted no modules; selected=%d skipped=%d.",
            len(selected_keys),
            len(skipped_keys),
        )
    return converted


def _build_te_linear(
    te_linear_cls,
    linear: nn.Linear,
    device: torch.device,
) -> nn.Module:
    """Build the replacement while preserving the original parameter contract.

    The master weight remains in ``weight.dtype``; copying it into ``te.Linear``
    does not permanently quantize the checkpoint. TE creates and updates its FP8
    representations and scaling state when executing under ``fp8_autocast``.
    """
    weight = linear.weight
    if weight.device.type == "meta":
        raise RuntimeError(
            "FP8 conversion cannot copy weights off the meta device; "
            "--fp8 and --init-on-meta are incompatible."
        )

    te_linear = te_linear_cls(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        device=device,
        params_dtype=weight.dtype,
    )
    # te.Linear's random initialization is immediately overwritten. The caller
    # wraps construction in fork_rng so even the discarded initialization cannot
    # perturb diffusion/flow-matching noise sequences.
    with torch.no_grad():
        te_linear.weight.copy_(weight)
        if linear.bias is not None:
            te_linear.bias.copy_(linear.bias)
    # Preserve the freeze decisions made by _freeze_modules / the model itself,
    # which already ran by the time wrap_model gets here.
    te_linear.weight.requires_grad_(weight.requires_grad)
    if linear.bias is not None:
        te_linear.bias.requires_grad_(linear.bias.requires_grad)
    return te_linear


def _import_te_linear():
    """Import ``te.Linear``, turning a missing TE install into a clear error."""
    try:
        from transformer_engine.pytorch import Linear as TELinear
    except ImportError as exc:
        raise RuntimeError(
            "--fp8 requires the TransformerEngine PyTorch extension "
            "(transformer_engine.pytorch)."
        ) from exc
    return TELinear