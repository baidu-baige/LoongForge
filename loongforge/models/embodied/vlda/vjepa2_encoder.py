# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""V-JEPA2 encoder used for online and EMA-teacher world targets."""

import logging
from functools import partial

import torch
from torch import nn

from .model_configuration_vlda import VJEPA2Config

logger = logging.getLogger(__name__)


class VJEPA2Encoder(nn.Module):
    """Construct the encoder without implicitly loading external weights."""

    def __init__(self, config: VJEPA2Config):
        super().__init__()
        from .vjepa2.vision_transformer import VisionTransformer

        self.config = config
        self.encoder = VisionTransformer(
            img_size=(config.img_size, config.img_size),
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_frames=config.num_frames,
            tubelet_size=config.tubelet_size,
            qkv_bias=config.qkv_bias,
            use_rope=config.use_rope,
            use_sdpa=config.use_sdpa,
            use_silu=config.use_silu,
            wide_silu=config.wide_silu,
            uniform_power=config.uniform_power,
            norm_layer=partial(nn.LayerNorm, eps=config.norm_layer_eps),
        )

    def load_encoder_state(self, state_dict: dict):
        """Load one encoder entry from an original V-JEPA2 checkpoint."""
        state_dict = {key.removeprefix("module.").removeprefix("backbone."): value for key, value in state_dict.items()}
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        # RoPE encoders do not use the fixed positional embedding of some exports.
        unexpected = [key for key in unexpected if key != "pos_embed"]
        if missing or unexpected:
            raise ValueError(f"V-JEPA2 checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        logger.info("Loaded V-JEPA2 encoder weights")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode normalized ``[B,3,2,256,256]`` images into patch tokens."""
        return self.encoder(images)
