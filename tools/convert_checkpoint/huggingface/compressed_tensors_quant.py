#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for writing official compressed-tensors pack-quantized HF weights."""

import json
import logging
import re
from pathlib import Path

import torch

from convert_checkpoint.huggingface.compressed_tensors_dequant import (
    DTYPE_MAP,
    build_quantization_scheme,
    iter_quantization_configs,
)
from convert_checkpoint.utils.utils import convert_fp8_to_bf16


LOGGER = logging.getLogger(__name__)

WEIGHT_SUFFIX = ".weight"
WEIGHT_PACKED_SUFFIX = ".weight_packed"
WEIGHT_SCALE_SUFFIX = ".weight_scale"
WEIGHT_SHAPE_SUFFIX = ".weight_shape"
WEIGHT_ZERO_POINT_SUFFIX = ".weight_zero_point"
WEIGHT_G_IDX_SUFFIX = ".weight_g_idx"

DEFAULT_KIMI_EXPERT_TARGET_RE = (
    r"^language_model\.model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(gate_proj|up_proj|down_proj)$"
)


def _load_config(config_file):
    config_path = Path(config_file)
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_pack_quantized_config(config):
    for _, quant_config in iter_quantization_configs(config):
        if quant_config.get("quant_method") != "compressed-tensors":
            continue
        if quant_config.get("format") != "pack-quantized":
            continue
        return quant_config
    return None


def _dtype_from_config(config):
    dtype_name = (
        config.get("text_config", {}).get("dtype")
        or config.get("text_config", {}).get("torch_dtype")
        or config.get("dtype")
        or config.get("torch_dtype")
        or "bfloat16"
    )
    return DTYPE_MAP.get(str(dtype_name).lower(), torch.bfloat16)


def _module_is_ignored(module_name, ignore_rules):
    for rule in ignore_rules or []:
        if rule.startswith("re:"):
            if re.match(rule[3:], module_name):
                return True
        elif module_name == rule or module_name.endswith(f".{rule}") or rule in module_name:
            return True
    return False


def _is_fp8_tensor(tensor):
    return tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def _materialize_weight(state_dict, weight_key, output_dtype):
    weight = state_dict[weight_key]
    scale_key = f"{weight_key[:-len(WEIGHT_SUFFIX)]}{WEIGHT_SCALE_SUFFIX}"
    scale = state_dict.get(scale_key)

    if _is_fp8_tensor(weight):
        if scale is None:
            raise KeyError(f"{weight_key} is FP8 but missing companion {scale_key}")
        return convert_fp8_to_bf16(weight, scale, dtype=output_dtype)

    return weight.to(output_dtype).contiguous()


def _calculate_int4_group_scale(weight, quantization_scheme, output_dtype):
    from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp

    scale, zero_point = compute_dynamic_scales_and_zp(
        weight.float(),
        quantization_scheme.weights,
        module=None,
    )
    scale = scale.to(output_dtype).contiguous()
    return scale, zero_point


def _replace_with_packed_weight(state_dict, weight_key, weight, scale, zero_point, quantization_scheme):
    from compressed_tensors import PackedQuantizationCompressor

    base_key = weight_key[:-len(WEIGHT_SUFFIX)]
    local_state = {"weight": weight, "weight_scale": scale}
    if not quantization_scheme.weights.symmetric:
        local_state["weight_zero_point"] = zero_point
    compressed = PackedQuantizationCompressor.compress(
        local_state,
        scheme=quantization_scheme,
    )

    for suffix in (
        WEIGHT_SUFFIX,
        WEIGHT_SCALE_SUFFIX,
        WEIGHT_SHAPE_SUFFIX,
        WEIGHT_ZERO_POINT_SUFFIX,
        WEIGHT_G_IDX_SUFFIX,
        WEIGHT_PACKED_SUFFIX,
    ):
        state_dict.pop(f"{base_key}{suffix}", None)

    state_dict[f"{base_key}{WEIGHT_PACKED_SUFFIX}"] = compressed["weight_packed"].contiguous()
    state_dict[f"{base_key}{WEIGHT_SCALE_SUFFIX}"] = compressed["weight_scale"].contiguous()
    state_dict[f"{base_key}{WEIGHT_SHAPE_SUFFIX}"] = compressed["weight_shape"].to(torch.int32).contiguous()
    if "weight_zero_point" in compressed:
        state_dict[f"{base_key}{WEIGHT_ZERO_POINT_SUFFIX}"] = compressed["weight_zero_point"].contiguous()
    if "weight_g_idx" in compressed:
        state_dict[f"{base_key}{WEIGHT_G_IDX_SUFFIX}"] = compressed["weight_g_idx"].contiguous()


def pack_state_dict_from_official_config(state_dict, config_file, target_regex=None):
    """Mutate a HF state_dict into the official compressed-tensors packed format.

    The official Kimi K2.5 config places the compressed-tensors quantization config
    under text_config. The conversion tool cannot reliably infer module classes
    from a plain state_dict, so callers should provide a target regex for the
    Linear weights that must be packed. If omitted, the Kimi routed expert
    projections are selected.
    """

    config = _load_config(config_file)
    quant_config = _find_pack_quantized_config(config)
    if quant_config is None:
        raise ValueError(f"No compressed-tensors pack-quantized config found in {config_file}")

    target_re = re.compile(target_regex or DEFAULT_KIMI_EXPERT_TARGET_RE)
    ignore_rules = quant_config.get("ignore", [])
    output_dtype = _dtype_from_config(config)
    quantization_scheme = build_quantization_scheme(
        load_path=str(Path(config_file).parent),
        config_file=config_file,
    )

    packed = 0
    normalized_fp8 = 0
    skipped = 0
    for weight_key in sorted(key for key in list(state_dict) if key.endswith(WEIGHT_SUFFIX)):
        weight = state_dict[weight_key]
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            skipped += 1
            continue

        module_name = weight_key[:-len(WEIGHT_SUFFIX)]
        scale_key = f"{module_name}{WEIGHT_SCALE_SUFFIX}"
        should_pack = target_re.match(module_name) is not None and not _module_is_ignored(module_name, ignore_rules)

        if should_pack:
            materialized = _materialize_weight(state_dict, weight_key, output_dtype)
            scale, zero_point = _calculate_int4_group_scale(materialized, quantization_scheme, output_dtype)
            _replace_with_packed_weight(
                state_dict,
                weight_key,
                materialized,
                scale,
                zero_point,
                quantization_scheme,
            )
            packed += 1
            continue

        if scale_key in state_dict:
            state_dict[weight_key] = _materialize_weight(state_dict, weight_key, output_dtype)
            state_dict.pop(scale_key, None)
            normalized_fp8 += 1
        elif _is_fp8_tensor(weight):
            raise KeyError(f"{weight_key} is FP8 but has no {scale_key}; cannot write official BF16 HF weight")

    for key in list(state_dict):
        if not key.endswith(WEIGHT_SCALE_SUFFIX):
            continue
        base_key = key[:-len(WEIGHT_SCALE_SUFFIX)]
        if f"{base_key}{WEIGHT_PACKED_SUFFIX}" not in state_dict:
            state_dict.pop(key, None)

    LOGGER.info(
        "Packed %d HF weight(s) with compressed-tensors from %s; normalized %d FP8/BF16 "
        "weight(s); skipped %d non-2D weight(s).",
        packed,
        config_file,
        normalized_fp8,
        skipped,
    )
    return packed
