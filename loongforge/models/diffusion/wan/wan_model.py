# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""wan model"""

import torch
import torch.nn as nn
from torch import Tensor

import math
from typing import Tuple, Dict, Literal, Optional
from einops import rearrange
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.spec_utils import ModuleSpec
from .wan_transformer_block import WanTransformerBlock

from loongforge.utils import get_args
from .communications import (
    split_forward_gather_backward,
    gather_forward_split_backward,
)
from megatron.core.parallel_state import (
    get_context_parallel_group,
)
from torch.amp import autocast


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float
    ):
        super().__init__()
        self.hidden_size = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x

from .wan_config import WanConfig
class WanModel(VisionModule):
    """Wan Transformer language model"""

    def __init__(
        self,
        config: WanConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        require_vae_embedding: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = True,
        position_embedding_type: Literal["learned_absolute", "rope"] = "rope",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
    ):
        super().__init__(config=config)
        self.require_clip_embedding = config.require_clip_embedding
        self.require_vae_embedding = require_vae_embedding
        self.args = get_args()
        self.pre_process = pre_process
        self.post_process = post_process
        self.freq_dim = config.freq_dim
        self.has_image_input = config.has_image_input
        self.patch_size = config.latent_patch_size
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

        self.hidden_size = config.hidden_size
        in_dim = config.in_dim
        out_dim = config.out_dim
        text_dim = config.text_dim
        patch_size = self.patch_size
        has_image_input = self.has_image_input
        eps = config.norm_epsilon
        self.has_image_pos_emb = config.has_image_pos_emb

        self.patch_embedding = nn.Conv3d(
            in_dim, config.hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, self.hidden_size), nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.head = Head(self.hidden_size, out_dim, patch_size, eps)
        head_dim = self.hidden_size // self.config.num_attention_heads
        f_freqs, h_freqs, w_freqs = precompute_freqs_cis_3d(head_dim)
        self.register_buffer('freqs_f', f_freqs, persistent=False)
        self.register_buffer('freqs_h', h_freqs, persistent=False)
        self.register_buffer('freqs_w', w_freqs, persistent=False)
        _f = (config.num_latent_frames - 1) // config.vae_temporal_compress + 1
        _h = config.max_latent_height // config.vae_spatial_compress // config.latent_patch_size[1]
        _w = config.max_latent_width // config.vae_spatial_compress // config.latent_patch_size[2]
        self._grid_f, self._grid_h, self._grid_w = _f, _h, _w
        _freqs = torch.cat([
            f_freqs[:_f].view(_f, 1, 1, -1).expand(_f, _h, _w, -1),
            h_freqs[:_h].view(1, _h, 1, -1).expand(_f, _h, _w, -1),
            w_freqs[:_w].view(1, 1, _w, -1).expand(_f, _h, _w, -1),
        ], dim=-1).reshape(_f * _h * _w, 1, -1)
        self.register_buffer('freqs_3d', _freqs, persistent=False)
        self.register_buffer('freqs_3d_cos', _freqs.real.squeeze(1).contiguous(), persistent=False)
        self.register_buffer('freqs_3d_sin', _freqs.imag.squeeze(1).contiguous(), persistent=False)

        if has_image_input:
            self.img_emb = MLP(
                1280, self.hidden_size, has_pos_emb=self.has_image_pos_emb
            )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
        )

        self.decoder = WanTransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
        )

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def pad_image(self, clip):
        seq = clip.shape[1]
        pad_num = (
            self.config.context_parallel_size - seq % self.config.context_parallel_size
        )
        pad_num = pad_num % self.config.context_parallel_size
        if pad_num != 0:
            pad = torch.zeros(clip.shape[0], pad_num, clip.shape[2], device=clip.device, dtype=clip.dtype)
            clip = torch.cat([clip, pad], dim=1)
        return pad_num, clip

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        f, h, w = self._grid_f, self._grid_h, self._grid_w
        freqs = self.freqs_3d
        rotary_pos_cos = self.freqs_3d_cos
        rotary_pos_sin = self.freqs_3d_sin

        if self.has_image_input:
            pad_num, clip_feature = self.pad_image(clip_feature)

        t_for_head = None
        timestep_mod = None
        if self.pre_process:
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_for_head = t

            timestep_mod = self.time_projection(t).unflatten(1, (6, self.hidden_size))
            context = self.text_embedding(context)
            if y is not None and self.require_vae_embedding:
                x = torch.cat([x, y], dim=1)
            if clip_feature is not None and self.require_clip_embedding:
                with autocast("cuda", dtype=torch.bfloat16):
                    clip_embdding = self.img_emb(clip_feature)
                    clip_embdding = rearrange(
                        clip_embdding, f"B S C ->S B C"
                    ).contiguous()

            x, (f, h, w) = self.patchify(x)

            x = rearrange(x, f"B S C ->S B C").contiguous()
            timestep_mod = rearrange(timestep_mod, f"B S C ->S B C").contiguous()
            context = rearrange(context, f"B S C ->S B C").contiguous()

            if self.has_image_input and clip_feature is not None and self.require_clip_embedding:
                context = torch.cat([clip_embdding, context], dim=0)

            cp = self.config.context_parallel_size
            if cp > 1:
                x = split_forward_gather_backward(
                    x, get_context_parallel_group(), dim=0, grad_scale="down"
                )
                context = split_forward_gather_backward(
                    context, get_context_parallel_group(), dim=0, grad_scale="down"
                )
                freqs = split_forward_gather_backward(
                    freqs, get_context_parallel_group(), dim=0, grad_scale="down"
                )
                rotary_pos_cos = split_forward_gather_backward(
                    rotary_pos_cos, get_context_parallel_group(), dim=0, grad_scale="down"
                )
                rotary_pos_sin = split_forward_gather_backward(
                    rotary_pos_sin, get_context_parallel_group(), dim=0, grad_scale="down"
                )

        else:
            x = None
            context = None
        extra_block_kwargs = {}
        x = self.decoder(
            hidden_states=x,
            attention_mask=None,
            context=context,
            context_mask=None,
            inference_params=None,
            packed_seq_params=None,
            rotary_pos_emb=freqs,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            timestep_mod=timestep_mod,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return x
        assert t_for_head is not None, (
            "WanModel post-process requires t_for_head, "
            "but pre_process=False. Pipeline parallel (pp>1) is not yet supported."
        )

        t = t_for_head.to(torch.bfloat16)

        x = rearrange(x, f"S B C ->B S C").contiguous()
        x = self.head(x, t)

        if self.config.context_parallel_size > 1:
            x = gather_forward_split_backward(
                x, get_context_parallel_group(), dim=1, grad_scale="up"
            )

        x = self.unpatchify(x, (f, h, w))
        return x

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1"
        self.decoder.set_input_tensor(input_tensor[0])
