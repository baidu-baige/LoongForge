# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import glob
import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .io import load_state_dict
from ..wan_video_clip import WanImageEncoderCLIP
from ..wan_video_dit import WanVideoDiT
from ..wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..wan_video_vae import WanVideoVAE
from ..logging_utils import get_logger

logger = get_logger(__name__)
SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"


@dataclass
class Wan21I2V14BLoadedComponents:
    dit: WanVideoDiT
    vae: WanVideoVAE
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    clip_encoder: WanImageEncoderCLIP | None
    dit_path: str
    vae_path: str
    text_encoder_path: str | None
    tokenizer_path: str | None
    clip_encoder_path: str | None


def _resolve_model_dir(model_id: str | os.PathLike[str]) -> Path:
    path = Path(model_id)
    if path.exists():
        return path
    raise FileNotFoundError(f"Local Wan model directory does not exist: {path}")


def _validate_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must be a dict, got {type(dit_config)}")
    validated = dict(dit_config)
    sig = inspect.signature(WanVideoDiT.__init__)
    allowed = {k for k in sig.parameters if k != "self"}
    unknown = sorted(set(validated) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown keys in `dit_config`: {unknown}. Allowed keys: {sorted(allowed)}"
        )
    return validated


def _load_json_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return {k: v for k, v in config.items() if not k.startswith("_")}


def _convert_vae_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    if any(k.startswith("model.") for k in state_dict):
        return state_dict
    return {f"model.{k}": v for k, v in state_dict.items()}


def _load_state_into_model(
    model: torch.nn.Module, path, torch_dtype: torch.dtype, strict: bool = True
):
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    if isinstance(state_dict, dict):
        for key in ("dit", "state_dict", "module", "model_state"):
            nested = state_dict.get(key)
            if isinstance(nested, dict):
                state_dict = nested
                break
    missing, unexpected = _load_state_dict_assign(model, state_dict, strict=strict)
    if missing or unexpected:
        logger.warning(
            "Loaded %s with missing=%d unexpected=%d", path, len(missing), len(unexpected)
        )
    return model


def _load_state_dict_assign(
    model: torch.nn.Module, state_dict: dict[str, torch.Tensor], strict: bool = True
):
    use_assign = os.environ.get("VPP2_LOAD_ASSIGN", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if use_assign and "assign" in inspect.signature(model.load_state_dict).parameters:
        return model.load_state_dict(state_dict, strict=strict, assign=True)
    return model.load_state_dict(state_dict, strict=strict)


def _maybe_stagger_model_load():
    delay_s = float(os.environ.get("VPP2_MODEL_LOAD_STAGGER_SECONDS", "0") or 0)
    if delay_s <= 0:
        return
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")) or 0)
    sleep_s = delay_s * rank
    if sleep_s > 0:
        logger.info("Staggering Wan2.1-I2V-14B load for rank=%d by %.1f seconds.", rank, sleep_s)
        time.sleep(sleep_s)


def load_wan21_i2v_14b_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "weights/Wan2.1-I2V-14B-480P",
    dit_checkpoint_path: str | os.PathLike[str] | None = None,
    tokenizer_model_id: str | None = None,
    tokenizer_max_len: int = 512,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
    load_clip_encoder: bool = True,
):
    logger.info("Loading Wan2.1-I2V-14B components...")
    _maybe_stagger_model_load()
    start = time.time()
    model_dir = _resolve_model_dir(model_id)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Wan2.1-I2V-14B model directory does not exist: {model_dir}")

    config = _load_json_config(model_dir)
    if dit_config is not None:
        config.update(dict(dit_config))
    validated_dit_config = _validate_dit_config(config)

    if dit_checkpoint_path:
        dit_checkpoint = Path(dit_checkpoint_path)
        if not dit_checkpoint.is_absolute():
            dit_checkpoint = model_dir / dit_checkpoint
        if not dit_checkpoint.is_file() and not skip_dit_load_from_pretrain:
            raise FileNotFoundError(f"DiT checkpoint does not exist: {dit_checkpoint}")
        dit_paths = [str(dit_checkpoint)]
    else:
        dit_paths = sorted(glob.glob(str(model_dir / "diffusion_pytorch_model-*.safetensors")))
        if not dit_paths:
            single = model_dir / "diffusion_pytorch_model.safetensors"
            if single.is_file():
                dit_paths = [str(single)]
        if not dit_paths and not skip_dit_load_from_pretrain:
            raise FileNotFoundError(f"No DiT checkpoint found under {model_dir}")

    vae_path = model_dir / "Wan2.1_VAE.pth"
    text_path = model_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    clip_path = model_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    tokenizer_path = (
        Path(tokenizer_model_id) if tokenizer_model_id else model_dir / "google" / "umt5-xxl"
    )

    if skip_dit_load_from_pretrain:
        logger.info("Skipping pretrained Wan2.1-14B DiT load.")
        dit = WanVideoDiT(**validated_dit_config).to(device=device, dtype=torch_dtype)
        dit_path_out = SKIPPED_PRETRAIN_SENTINEL
    else:
        dit = WanVideoDiT(**validated_dit_config)
        logger.info("Loading Wan2.1-14B DiT from %s.", ",".join(dit_paths))
        _load_state_into_model(dit, dit_paths, torch_dtype=torch_dtype, strict=True)
        dit = dit.to(device=device, dtype=torch_dtype)
        dit_path_out = ",".join(dit_paths)

    vae = WanVideoVAE(z_dim=16)
    vae_state = load_state_dict(str(vae_path), torch_dtype=torch.float32, device="cpu")
    _load_state_dict_assign(vae, _convert_vae_state_dict(vae_state), strict=True)
    vae = vae.eval().requires_grad_(False).to(device=device, dtype=torch.float32)

    text_encoder = None
    tokenizer = None
    text_encoder_path = None
    tokenizer_path_out = None
    if load_text_encoder:
        text_encoder = WanTextEncoder()
        _load_state_into_model(text_encoder, str(text_path), torch_dtype=torch_dtype, strict=True)
        text_encoder = text_encoder.to(device=device, dtype=torch_dtype)
        tokenizer = HuggingfaceTokenizer(
            name=str(tokenizer_path), seq_len=int(tokenizer_max_len), clean="whitespace"
        )
        text_encoder_path = str(text_path)
        tokenizer_path_out = str(tokenizer_path)

    clip_encoder = None
    clip_encoder_path = None
    if load_clip_encoder:
        clip_encoder = WanImageEncoderCLIP()
        clip_state = load_state_dict(str(clip_path), torch_dtype=torch.float16, device="cpu")
        clip_state = {
            k: v for k, v in clip_state.items() if k.startswith("visual.") or k == "log_scale"
        }
        missing, unexpected = _load_state_dict_assign(clip_encoder, clip_state, strict=True)
        if missing or unexpected:
            raise ValueError(
                f"Unexpected CLIP visual state mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        clip_encoder = (
            clip_encoder.eval().requires_grad_(False).to(device=device, dtype=torch.float16)
        )
        clip_encoder_path = str(clip_path)

    logger.info("Finished loading Wan2.1-I2V-14B components in %.2f seconds.", time.time() - start)
    return Wan21I2V14BLoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        clip_encoder=clip_encoder,
        dit_path=dit_path_out,
        vae_path=str(vae_path),
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path_out,
        clip_encoder_path=clip_encoder_path,
    )
