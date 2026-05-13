#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for materializing compressed-tensors pack-quantized HF weights."""

import json
import logging
from pathlib import Path

import torch


LOGGER = logging.getLogger(__name__)

WEIGHT_PACKED_KEY = "weight_packed"
WEIGHT_SCALE_KEY = "weight_scale"
WEIGHT_SHAPE_KEY = "weight_shape"
WEIGHT_ZERO_POINT_KEY = "weight_zero_point"
WEIGHT_G_IDX_KEY = "weight_g_idx"

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

TORCH_DTYPE_NAME = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


def iter_quantization_configs(obj, path=()):
    if not isinstance(obj, dict):
        return

    quant_config = obj.get("quantization_config")
    if isinstance(quant_config, dict):
        yield path + ("quantization_config",), quant_config

    for key, value in obj.items():
        if isinstance(value, dict):
            yield from iter_quantization_configs(value, path + (key,))


def load_compressed_tensors_weight_config(load_path, config_file=None):
    config_path = Path(config_file).resolve() if config_file is not None else Path(load_path) / "config.json"
    if not config_path.exists():
        LOGGER.warning(
            "compressed-tensors config not found at %s; using fallback Kimi INT4 quantization args.",
            config_path,
        )
        return None, None

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    for path, quant_config in iter_quantization_configs(config):
        if quant_config.get("quant_method") != "compressed-tensors":
            continue
        if quant_config.get("format") != "pack-quantized":
            continue
        for group in quant_config.get("config_groups", {}).values():
            weights = group.get("weights") if isinstance(group, dict) else None
            if isinstance(weights, dict):
                return weights, f"{config_path}:{'.'.join(path)}"

    LOGGER.warning(
        "No pack-quantized compressed-tensors quantization_config found in %s; "
        "using fallback Kimi INT4 quantization args.",
        config_path,
    )
    return None, None


def build_quantization_scheme(
    load_path,
    config_file=None,
    ignore_config_quantization=False,
    num_bits=4,
    quant_strategy="group",
    group_size=32,
    symmetric=True,
    dynamic=False,
):
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    weight_config = None
    config_path = None
    if not ignore_config_quantization:
        weight_config, config_path = load_compressed_tensors_weight_config(load_path, config_file)

    if weight_config is None:
        weight_config = {
            "num_bits": num_bits,
            "type": "int",
            "strategy": quant_strategy,
            "group_size": group_size,
            "symmetric": symmetric,
            "dynamic": dynamic,
        }
    else:
        LOGGER.info("Using compressed-tensors weight config from %s.", config_path)

    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=weight_config.get("num_bits", num_bits),
            type=weight_config.get("type", "int"),
            strategy=weight_config.get("strategy", quant_strategy),
            group_size=weight_config.get("group_size", group_size),
            symmetric=weight_config.get("symmetric", symmetric),
            dynamic=weight_config.get("dynamic", dynamic),
        ),
    )


def get_packed_weight_keys(
    weight_key,
    packed_key=WEIGHT_PACKED_KEY,
    scale_key=WEIGHT_SCALE_KEY,
    shape_key=WEIGHT_SHAPE_KEY,
    zero_point_key=WEIGHT_ZERO_POINT_KEY,
    g_idx_key=WEIGHT_G_IDX_KEY,
):
    base = weight_key[: -len(".weight")] if weight_key.endswith(".weight") else weight_key
    return [
        f"{base}.{packed_key}",
        f"{base}.{scale_key}",
        f"{base}.{shape_key}",
        f"{base}.{zero_point_key}",
        f"{base}.{g_idx_key}",
    ]


def dequantize_int4_packed(
    weight_packed,
    weight_scale,
    weight_shape,
    quantization_scheme,
    tensor_name,
    weight_zero_point=None,
    weight_g_idx=None,
):
    from compressed_tensors import PackedQuantizationCompressor

    if weight_packed.dtype != torch.int32:
        raise TypeError(
            f"{tensor_name}.weight_packed must be torch.int32 for compressed-tensors "
            f"pack-quantized decompression, got {weight_packed.dtype}"
        )

    compressed_tensors = {
        "weight_packed": weight_packed,
        "weight_scale": weight_scale,
        "weight_shape": weight_shape,
    }
    if weight_zero_point is not None:
        compressed_tensors["weight_zero_point"] = weight_zero_point
    if weight_g_idx is not None:
        compressed_tensors["weight_g_idx"] = weight_g_idx

    decompressed = PackedQuantizationCompressor.decompress(
        compressed_tensors,
        scheme=quantization_scheme,
    )
    return decompressed["weight"]


def dequantize_state_dict(
    state_dict,
    load_path,
    output_dtype=torch.bfloat16,
    config_file=None,
    ignore_config_quantization=False,
    packed_key=WEIGHT_PACKED_KEY,
    scale_key=WEIGHT_SCALE_KEY,
    shape_key=WEIGHT_SHAPE_KEY,
    zero_point_key=WEIGHT_ZERO_POINT_KEY,
    g_idx_key=WEIGHT_G_IDX_KEY,
    num_bits=4,
    quant_strategy="group",
    group_size=32,
    symmetric=True,
    dynamic=False,
    target_weight_keys=None,
):
    packed_suffix = f".{packed_key}"
    if target_weight_keys is None:
        packed_keys = sorted(key for key in list(state_dict) if key.endswith(packed_suffix))
    else:
        packed_keys = []
        missing_packed_keys = []
        for weight_key in sorted(target_weight_keys):
            if not weight_key.endswith(".weight"):
                raise ValueError(f"Targeted dequant weight key must end with .weight, got: {weight_key}")
            key = f"{weight_key[:-len('.weight')]}.{packed_key}"
            if key in state_dict:
                packed_keys.append(key)
            else:
                missing_packed_keys.append(key)
        if missing_packed_keys:
            preview = ", ".join(missing_packed_keys[:5])
            suffix = "" if len(missing_packed_keys) <= 5 else f", ... ({len(missing_packed_keys)} total)"
            raise KeyError(
                "Targeted compressed-tensors dequant requested packed tensor(s) that were not loaded: "
                f"{preview}{suffix}"
            )
    if not packed_keys:
        return 0

    quantization_scheme = build_quantization_scheme(
        load_path,
        config_file=config_file,
        ignore_config_quantization=ignore_config_quantization,
        num_bits=num_bits,
        quant_strategy=quant_strategy,
        group_size=group_size,
        symmetric=symmetric,
        dynamic=dynamic,
    )

    converted = 0
    total = len(packed_keys)
    progress_interval = max(1, total // 20)
    LOGGER.info("Dequantizing %d target compressed-tensors packed INT4 weight(s).", total)
    for key in packed_keys:
        base = key[: -len(packed_suffix)]
        out_key = f"{base}.weight"
        if out_key in state_dict:
            raise KeyError(f"Refusing to overwrite existing tensor while dequantizing: {out_key}")

        scale_tensor_key = f"{base}.{scale_key}"
        shape_tensor_key = f"{base}.{shape_key}"
        missing = [name for name in (scale_tensor_key, shape_tensor_key) if name not in state_dict]
        if missing:
            raise KeyError(f"{key} is missing compressed-tensors companion tensor(s): {missing}")

        tensor = dequantize_int4_packed(
            state_dict[key],
            state_dict[scale_tensor_key],
            state_dict[shape_tensor_key],
            quantization_scheme,
            base,
            weight_zero_point=state_dict.get(f"{base}.{zero_point_key}"),
            weight_g_idx=state_dict.get(f"{base}.{g_idx_key}"),
        )
        state_dict[out_key] = tensor.to(output_dtype).contiguous()

        for remove_key in (
            key,
            scale_tensor_key,
            shape_tensor_key,
            f"{base}.{zero_point_key}",
            f"{base}.{g_idx_key}",
        ):
            state_dict.pop(remove_key, None)
        converted += 1
        if converted == total or converted % progress_interval == 0:
            LOGGER.info("Dequantized %d/%d compressed-tensors packed INT4 weight(s).", converted, total)

    return converted
