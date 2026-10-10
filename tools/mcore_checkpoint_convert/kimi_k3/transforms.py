# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""State-dict transforms that are specific to Kimi K3's HF contract."""

import torch


def normalize_kimi_k3_state_dict(state_dict: dict, c_config) -> None:
    """Normalize packed experts and remove K3 HF-only padding.

    K3 stores A_log at a fixed padded width while MCore stores only the
    effective heads. Its MXFP4 experts also use ``weight_packed`` and
    ``weight_scale`` names, while the shared dequant path consumes
    ``weight``/``scale`` pairs.
    """
    module = c_config.get("module", {}) if c_config is not None else {}
    target = module.get("_target_") or ""
    if module.get("model_type") != "kimi_k3" and not target.endswith("KimiK3Config"):
        return
    num_heads = int(
        module.get("kimi_linear_num_heads", module.get("num_attention_heads", 0))
    )
    num_experts = int(module.get("num_experts", 0))
    if num_heads <= 0:
        raise ValueError("Kimi K3 conversion requires kimi_linear_num_heads > 0")

    for key in [key for key in state_dict if key.endswith(".weight_packed")]:
        weight_key = key[: -len("_packed")]
        source_scale = f"{key[: -len('.weight_packed')]}.weight_scale"
        scale_key = f"{weight_key[: -len('.weight')]}.scale"
        if source_scale not in state_dict:
            raise KeyError(f"Missing MXFP4 scale for K3 expert weight {key}")
        if weight_key in state_dict or scale_key in state_dict:
            raise ValueError(f"K3 packed expert key collides with {weight_key}")
        state_dict[weight_key] = state_dict.pop(key)
        state_dict[scale_key] = state_dict.pop(source_scale)

    for key, value in list(state_dict.items()):
        if num_experts > 0 and key.endswith(".block_sparse_moe.gate.weight"):
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] < num_experts:
                shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
                raise ValueError(f"Invalid K3 router-weight shape for {key}: {shape}")
            if value.shape[0] > num_experts:
                state_dict[key] = value[:num_experts].contiguous()
            continue
        if num_experts > 0 and key.endswith(".block_sparse_moe.gate.e_score_correction_bias"):
            if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() < num_experts:
                raise ValueError(f"Invalid K3 expert-bias shape for {key}: {tuple(value.shape)}")
            if value.numel() > num_experts:
                state_dict[key] = value[:num_experts].contiguous()
            continue
        if not key.endswith(".A_log") or not isinstance(value, torch.Tensor):
            continue
        if value.ndim != 1 or value.numel() < num_heads:
            raise ValueError(f"Invalid K3 A_log shape for {key}: {tuple(value.shape)}")
        if value.numel() > num_heads:
            padding = value[num_heads:]
            if torch.count_nonzero(padding).item() != 0:
                raise ValueError(f"Non-zero K3 A_log padding in {key}")
            state_dict[key] = value[:num_heads].contiguous()
