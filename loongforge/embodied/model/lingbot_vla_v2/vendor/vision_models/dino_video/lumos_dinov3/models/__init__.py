"""Vendor module implementation."""
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging

import torch.nn as nn

from . import vision_transformer as vits

logger = logging.getLogger("dinov3")


def build_model(args, only_teacher=False, img_size=224, device=None):
    """Vendor interface."""
    assert "vit" in args.arch, f"Only ViT architectures are supported, got {args.arch}"

    vit_kwargs = dict(
        img_size=img_size,
        patch_size=args.patch_size,
        pos_embed_rope_base=args.pos_embed_rope_base,
        pos_embed_rope_min_period=args.pos_embed_rope_min_period,
        pos_embed_rope_max_period=args.pos_embed_rope_max_period,
        pos_embed_rope_normalize_coords=args.pos_embed_rope_normalize_coords,
        pos_embed_rope_shift_coords=args.pos_embed_rope_shift_coords,
        pos_embed_rope_jitter_coords=args.pos_embed_rope_jitter_coords,
        pos_embed_rope_rescale_coords=args.pos_embed_rope_rescale_coords,
        qkv_bias=args.qkv_bias,
        layerscale_init=args.layerscale,
        norm_layer=args.norm_layer,
        ffn_layer=args.ffn_layer,
        ffn_bias=args.ffn_bias,
        proj_bias=args.proj_bias,
        n_storage_tokens=args.n_storage_tokens,
        mask_k_bias=args.mask_k_bias,
        untie_cls_and_patch_norms=args.untie_cls_and_patch_norms,
        untie_global_and_local_cls_norm=args.untie_global_and_local_cls_norm,
        device=device,
    )
    teacher = vits.__dict__[args.arch](**vit_kwargs)
    if only_teacher:
        return teacher, teacher.embed_dim
    student = vits.__dict__[args.arch](
        **vit_kwargs,
        drop_path_rate=args.drop_path_rate,
    )
    embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher: bool = False):
    """Vendor interface."""
    outputs = build_model(
        cfg.student,
        only_teacher=only_teacher,
        img_size=cfg.crops.global_crops_size
        if isinstance(cfg.crops.global_crops_size, int)
        else max(cfg.crops.global_crops_size),
        device="meta",
    )
    if only_teacher:
        teacher, embed_dim = outputs
        return teacher, embed_dim
    else:
        student, teacher, embed_dim = outputs
        return student, teacher, embed_dim
