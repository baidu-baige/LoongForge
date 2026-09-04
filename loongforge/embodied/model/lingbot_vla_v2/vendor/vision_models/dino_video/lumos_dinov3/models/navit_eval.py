"""Vendor module implementation.

Build config-faithful NaViT video backbones for evaluation.
"""
from __future__ import annotations

import os
from typing import Any

import torch
from omegaconf import OmegaConf

from .navit_video_vision_transformer import (
    navit_video_vit_base,
    navit_video_vit_giant2,
    navit_video_vit_large,
    navit_video_vit_small,
)

_NAVIT_ARCH_DICT = {
    "vit_small": navit_video_vit_small,
    "vit_base": navit_video_vit_base,
    "vit_large": navit_video_vit_large,
    "vit_giant2": navit_video_vit_giant2,
}


def detect_checkpoint_format(ckpt_path: str):
    """Load checkpoint and return (state_dict, has_rope3d).

    Handles formats:
      - bare state_dict: {patch_embed.proj.weight, ...}
      - wrapped: {teacher: {backbone.xxx, ...}}
      - wrapped: {model: {backbone.xxx, ...}} or {state_dict: {...}}
    Strips a leading "backbone." prefix, and detects 3D RoPE via the presence of
    ``rope_embed.periods_t``.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "teacher" in raw:
        sd = raw["teacher"]
    elif "model" in raw:
        sd = raw["model"]
    elif "state_dict" in raw:
        sd = raw["state_dict"]
    else:
        sd = raw

    first_key = next(iter(sd.keys()))
    if first_key.startswith("backbone."):
        sd = {k.removeprefix("backbone."): v for k, v in sd.items()}

    has_rope3d = "rope_embed.periods_t" in sd
    return sd, has_rope3d


def default_config_path(ckpt_path: str) -> str:
    """<exp>/eval_checkpoints/teacher_step_*.pth -> <exp>/config.yaml."""
    return os.path.normpath(os.path.join(os.path.dirname(ckpt_path), "..", "config.yaml"))


def _student_cfg_to_kwargs(student_cfg, attention_mode: str, device: Any) -> dict:
    """Map a checkpoint config to eval-time model kwargs."""
    g = student_cfg.get
    return dict(
        patch_size=student_cfg.patch_size,
        pos_embed_rope_base=student_cfg.pos_embed_rope_base,
        pos_embed_rope_min_period=g("pos_embed_rope_min_period", None),
        pos_embed_rope_max_period=g("pos_embed_rope_max_period", None),
        pos_embed_rope_normalize_coords=student_cfg.pos_embed_rope_normalize_coords,
        pos_embed_rope_shift_coords=g("pos_embed_rope_shift_coords", None),
        pos_embed_rope_jitter_coords=g("pos_embed_rope_jitter_coords", None),
        pos_embed_rope_rescale_coords=g("pos_embed_rope_rescale_coords", None),
        pos_embed_rope_3d=g("pos_embed_rope_3d", False),
        pos_embed_rope_temporal_base=g("pos_embed_rope_temporal_base", 10000.0),
        pos_embed_rope_fps=g("pos_embed_rope_fps", None),
        pos_embed_rope_base_fps=g("pos_embed_rope_base_fps", 24.0),
        pos_embed_rope_dtype=g("pos_embed_rope_dtype", "bf16"),
        pos_embed_rope_prefix_temporal=g("pos_embed_rope_prefix_temporal", False),
        qkv_bias=student_cfg.qkv_bias,
        layerscale_init=student_cfg.layerscale,
        norm_layer=student_cfg.norm_layer,
        ffn_layer=student_cfg.ffn_layer,
        ffn_bias=student_cfg.ffn_bias,
        proj_bias=student_cfg.proj_bias,
        n_storage_tokens=student_cfg.n_storage_tokens,
        mask_k_bias=student_cfg.mask_k_bias,
        untie_cls_and_patch_norms=g("untie_cls_and_patch_norms", False),
        untie_global_and_local_cls_norm=g("untie_global_and_local_cls_norm", False),
        attention_mode=attention_mode,
        device=device,
    )


def build_navit_for_eval(
    ckpt_path: str,
    *,
    config_path: str | None = None,
    attention_mode: str | None = None,
    img_size: int = 224,
    device: str | torch.device = "cuda",
    verbose: bool = True,
):
    """Build an eval backbone and load checkpoint weights."""
    if config_path is None:
        config_path = default_config_path(ckpt_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"NaViT eval config not found at {config_path!r}. Pass config_path= "
            f"explicitly, or place the experiment config.yaml two levels above "
            f"the checkpoint (<exp>/config.yaml, ckpt under <exp>/eval_checkpoints/)."
        )

    cfg = OmegaConf.load(config_path)
    student_cfg = cfg.dinov3.student
    arch = student_cfg.arch
    if arch not in _NAVIT_ARCH_DICT:
        raise ValueError(f"unknown NaViT arch {arch!r}; expected one of {list(_NAVIT_ARCH_DICT)}")

    resolved_attn = attention_mode or cfg.get("attention_mode", "flash_attn3")
    kwargs = _student_cfg_to_kwargs(student_cfg, resolved_attn, device)

    model = _NAVIT_ARCH_DICT[arch](img_size=img_size, **kwargs)

    # Materialize buffers that may be omitted from teacher exports.
    model.init_weights()

    sd, has_rope3d = detect_checkpoint_format(ckpt_path)
    cfg_rope3d = bool(student_cfg.get("pos_embed_rope_3d", False))
    if has_rope3d != cfg_rope3d:
        print(
            f"  WARNING: rope3d mismatch — config says {cfg_rope3d} but checkpoint "
            f"{'has' if has_rope3d else 'lacks'} rope_embed.periods_t"
        )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose:
        real_missing = [k for k in missing if "periods_t" not in k]
        if real_missing:
            print(f"  WARNING: missing keys: {real_missing[:5]}...")
        real_unexpected = [k for k in unexpected if not k.startswith(("dino_head", "ibot_head"))]
        if real_unexpected:
            print(f"  WARNING: unexpected keys: {real_unexpected[:5]}...")

    model.eval().to(device)
    info = {
        "rope3d": cfg_rope3d,
        "attention_mode": resolved_attn,
        "embed_dim": model.embed_dim,
        "arch": arch,
        "config_path": config_path,
    }
    if verbose:
        print(f"  [navit_eval] arch={arch} rope3d={cfg_rope3d} "
              f"attention_mode={resolved_attn} embed_dim={model.embed_dim}")
    return model, info


def make_feature_model(model, inference_mode: str = "bidirectional", cls_pool: str = "mean"):
    """Return an object exposing get_intermediate_layers."""
    if inference_mode == "causal":
        from .inference import CausalEvalAdapter
        return CausalEvalAdapter(model, cls_pool=cls_pool)
    if inference_mode != "bidirectional":
        raise ValueError(f"Unknown inference_mode: {inference_mode}")
    return model
