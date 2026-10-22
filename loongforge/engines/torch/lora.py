# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model-independent LoRA support for embodied training."""

import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_PEFT_BASE_MODEL_PREFIX = "base_model.model."


def _require_peft():
    """Import PEFT lazily so full fine-tuning does not require it."""
    try:
        import peft
    except ImportError as exc:
        raise ImportError(
            "LoRA training requires the optional dependency 'peft'. Install "
            "LoongForge with the gpu extra or install 'peft>=0.18.0'."
        ) from exc
    return peft


def is_lora_enabled(training_args) -> bool:
    """Return whether generic LoRA fine-tuning is enabled."""
    return bool(training_args.use_lora)


def _split_csv(value: Optional[str]) -> Optional[list[str]]:
    """Parse a comma-separated CLI value into non-empty items."""
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _parse_init_lora_weights(value: str):
    """Convert boolean-like CLI values to PEFT initializer values."""
    normalized = (value or "true").strip().lower()
    if normalized in {"true", "t", "kaiming"}:
        return True
    if normalized in {"false", "f"}:
        return False
    return value


def get_default_lora_targets(model: nn.Module) -> Optional[dict[str, Any]]:
    """Return the model-provided LoRA target specification, if available."""
    provider = getattr(model, "default_lora_targets", None)
    return provider() if callable(provider) else None


def build_lora_config(training_args, default_targets: Optional[dict[str, Any]]):
    """Build a PEFT ``LoraConfig`` from model defaults and CLI overrides."""
    peft = _require_peft()
    defaults = dict(default_targets or {})

    cli_targets = _split_csv(training_args.lora_target_modules)
    target_modules = (
        cli_targets if cli_targets is not None else defaults.get("target_modules")
    )
    if not target_modules:
        raise ValueError(
            "LoRA is enabled but no target modules were resolved. Define "
            "model.default_lora_targets() or pass --lora-target-modules."
        )

    cli_modules_to_save = _split_csv(training_args.lora_modules_to_save)
    modules_to_save = (
        cli_modules_to_save
        if cli_modules_to_save is not None
        else defaults.get("modules_to_save")
    )

    return peft.LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        target_modules=(
            target_modules
            if isinstance(target_modules, str)
            else list(target_modules)
        ),
        modules_to_save=list(modules_to_save) if modules_to_save else None,
        init_lora_weights=_parse_init_lora_weights(training_args.lora_init),
    )


def _requires_pretrained_checkpoint(model: nn.Module) -> bool:
    """Return whether the model expects one consolidated base checkpoint."""
    provider = getattr(model, "lora_requires_pretrained_checkpoint", None)
    return bool(provider()) if callable(provider) else True


def apply_lora(
    model: nn.Module,
    training_args,
    *,
    require_base: bool = True,
    adapter_path: Optional[str] = None,
) -> nn.Module:
    """Freeze the base model and inject PEFT adapters before parallel wrapping."""
    if (
        require_base
        and _requires_pretrained_checkpoint(model)
        and not training_args.pretrained_checkpoint
    ):
        raise ValueError(
            "LoRA fine-tuning requires --pretrained-checkpoint for this model."
        )

    peft = _require_peft()
    if adapter_path is None:
        lora_config = build_lora_config(training_args, get_default_lora_targets(model))
    else:
        lora_config = peft.LoraConfig.from_pretrained(adapter_path)
        lora_config.inference_mode = False

    model.requires_grad_(False)
    model = peft.inject_adapter_in_model(lora_config, model)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    logger.info(
        "LoRA injected: r=%d alpha=%d targets=%s modules_to_save=%s; "
        "trainable=%d/%d (%.4f%%)",
        lora_config.r,
        lora_config.lora_alpha,
        lora_config.target_modules,
        lora_config.modules_to_save,
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )
    return model


def get_adapter_state_dict(
    state_dict: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Return a canonical PEFT adapter state dict from distributed model state."""
    from peft.utils.save_and_load import get_peft_model_state_dict

    normalized_state = {
        key.replace("_checkpoint_wrapped_module.", ""): value
        for key, value in state_dict.items()
    }
    adapter_state = get_peft_model_state_dict(model, state_dict=normalized_state)
    if not adapter_state:
        raise RuntimeError("PEFT produced an empty adapter state dict")
    return {
        key if key.startswith(_PEFT_BASE_MODEL_PREFIX) else f"{_PEFT_BASE_MODEL_PREFIX}{key}": value
        for key, value in adapter_state.items()
    }


def save_adapter_config(model: nn.Module, output_dir: str) -> None:
    """Write the active adapter's standard PEFT ``adapter_config.json``."""
    peft_config = getattr(model, "peft_config", {}).get("default")
    if peft_config is None:
        raise RuntimeError("Model has no default PEFT adapter configuration")

    inference_mode = peft_config.inference_mode
    peft_config.inference_mode = True
    try:
        peft_config.save_pretrained(output_dir)
    finally:
        peft_config.inference_mode = inference_mode


def load_adapter_into_model(
    model: nn.Module,
    adapter_path: str,
) -> nn.Module:
    """Load one standard PEFT adapter into an injected, unwrapped model."""
    from peft.utils.save_and_load import set_peft_model_state_dict
    from safetensors.torch import load_file

    if not os.path.isdir(adapter_path):
        raise ValueError(f"PEFT adapter path must be a directory: {adapter_path}")
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing PEFT adapter config: {config_path}")
    weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Missing PEFT adapter weights: {weights_path}")

    state_dict = load_file(weights_path)
    expected_state = get_adapter_state_dict(model.state_dict(), model)
    missing_keys = set(expected_state).difference(state_dict)
    unexpected_keys = set(state_dict).difference(expected_state)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "PEFT adapter tensors do not match the injected model: "
            f"missing={len(missing_keys)} unexpected={len(unexpected_keys)} "
            f"missing_first={sorted(missing_keys)[:5]} "
            f"unexpected_first={sorted(unexpected_keys)[:5]}"
        )

    mismatched_shapes = {
        key: (tuple(state_dict[key].shape), tuple(expected_state[key].shape))
        for key in state_dict
        if state_dict[key].shape != expected_state[key].shape
    }
    if mismatched_shapes:
        raise RuntimeError(
            "PEFT adapter tensor shapes do not match the injected model: "
            f"count={len(mismatched_shapes)} first={list(mismatched_shapes.items())[:5]}"
        )

    injected_state = {
        key.removeprefix(_PEFT_BASE_MODEL_PREFIX): value
        for key, value in state_dict.items()
    }
    load_result = set_peft_model_state_dict(model, injected_state)
    unexpected = getattr(load_result, "unexpected_keys", [])
    if unexpected:
        raise RuntimeError(f"Unexpected PEFT adapter keys: {unexpected[:5]}")
    logger.info(
        "Loaded PEFT adapter from %s: %d tensors applied",
        adapter_path,
        len(state_dict),
    )
    return model


def adapter_meta(training_args, model_cfg) -> dict[str, Any]:
    """Build LoongForge metadata accompanying a standard PEFT adapter."""
    return {
        "format": "peft",
        "format_version": 1,
        "base_checkpoint": training_args.pretrained_checkpoint,
        "model_name": training_args.model_name,
        "model_type": getattr(model_cfg, "model_type", None),
    }


__all__ = [
    "adapter_meta",
    "apply_lora",
    "build_lora_config",
    "get_adapter_state_dict",
    "get_default_lora_targets",
    "is_lora_enabled",
    "load_adapter_into_model",
    "save_adapter_config",
]
