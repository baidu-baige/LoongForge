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
from megatron.core.transformer.transformer_block import TransformerBlock

from loongforge.utils import get_args
from .communications import (
    split_forward_gather_backward,
    gather_forward_split_backward,
)
from megatron.core.parallel_state import (
    get_context_parallel_group,
)
from torch.cuda.amp import autocast as autocast


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    """
    Modulate input tensor x.

    Args:
        x (torch.Tensor): Input tensor.
        shift (torch.Tensor): Shift parameter.
        scale (torch.Tensor): Scale parameter.

    Returns:
        torch.Tensor: Modulated tensor.
    """
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    """
    Calculate one-dimensional sinusoidal embedding representation.

    Args:
        dim (int): Embedding dimension.
        position (torch.Tensor): Position index, size is (N,).

    Returns:
        torch.Tensor: One-dimensional sinusoidal embedding representation, size is (N, dim).
    """
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
    """
    3D rope calculation
    """
    # 3D rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    """
    1d rope precompute

    """

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    """
    Calculate rope

    """
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class MLP(torch.nn.Module):
    """
    MLP class definition
    """

    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        """
        Initialization method
        """
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
        """
        Forward function
        """
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    """
    Head class
    """

    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float
    ):
        """
        Initialization function
        """
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        """
        Forward function
        """
        shift, scale = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x

from .wan_config import WanConfig
class WanModel(VisionModule):
    """
    Wan Transformer language model
    """

    def __init__(
        self,
        #config: StditTransformerConfig,
        config: WanConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        ##
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        require_clip_embedding: bool = True,
        require_vae_embedding: bool = True,
        ##
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
        self.require_clip_embedding = require_clip_embedding
        self.require_vae_embedding = require_vae_embedding
        self.config.has_image_input = has_image_input
        self.args = get_args()
        self.pre_process = pre_process
        self.post_process = post_process

        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(
                1280, dim, has_pos_emb=has_image_pos_emb
            )  # clip_feature_dim = 1280
        self.has_image_pos_emb = has_image_pos_emb
        # timestep embedding
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
        )

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
        )

        # if self.post_process:
        #     self.head = Head(dim, out_dim, patch_size, eps)

    def patchify(self, x: torch.Tensor):
        """
        Perform patchify operation on input tensor.
        """
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        """
        Reassemble chunked tensors into complete tensor.
        """

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
        """image padding"""
        seq = clip.shape[1]
        pad_num = (
            self.config.context_parallel_size - seq % self.config.context_parallel_size
        )
        pad_num = pad_num % self.config.context_parallel_size
        if pad_num != 0:
            pad = torch.zeros(clip.shape[0], pad_num, clip.shape[2]).to(clip.device)
            clip = torch.cat([clip, pad], dim=1)
        return pad_num, clip

    def reorganize_x(self, hidden_state, context, timestep, t_s, clip_emb=None):
        """reorganize hidden_state, context, timestep, t_s for cp parallel"""
        cp = self.config.context_parallel_size
        assert context.shape[0] % cp == 0
        hidden_states = torch.chunk(hidden_state, cp, dim=0)
        contexts = torch.chunk(context, cp, dim=0)
        cated_x = []
        if clip_emb is not None:
            clip_embs = torch.chunk(clip_emb, cp, dim=0)
            for i in range(len(contexts)):
                cated_x.append(
                    torch.cat(
                        [hidden_states[i], clip_embs[i], contexts[i], timestep, t_s],
                        dim=0,
                    )
                )
        else:
            for i in range(len(contexts)):
                cated_x.append(
                    torch.cat([hidden_states[i], contexts[i], timestep, t_s], dim=0)
                )
        x = torch.cat(cated_x, dim=0)
        return x

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
        """
        Wan forward
        """
        if self.args.model_name == "wan2-2-i2v":
            f, h, w = (13, 30, 52)
        freqs = torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)

        if self.has_image_input:
            pad_num, clip_feature = self.pad_image(clip_feature)

        if self.pre_process:
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_s = t.unsqueeze(0)

            timestep = self.time_projection(t).unflatten(1, (6, self.dim))
            context = self.text_embedding(context)
            if y is not None and self.require_vae_embedding:
                x = torch.cat([x, y], dim=1) # (b, c_x + c_y, f, h, w)
            if clip_feature is not None and self.require_clip_embedding:
                with autocast(dtype=torch.bfloat16):
                    clip_embdding = self.img_emb(clip_feature)
                    clip_embdding = rearrange(
                        clip_embdding, f"B S C ->S B C"
                    ).contiguous()

            x, (f, h, w) = self.patchify(x)
            if self.args.model_name == "wan2-2-i2v":
                assert (f, h, w) == (13, 30, 52)

            x = rearrange(x, f"B S C ->S B C").contiguous()
            timestep = rearrange(timestep, f"B S C ->S B C").contiguous()
            context = rearrange(context, f"B S C ->S B C").contiguous().to(torch.float32)
            # Concatenate context
            if self.has_image_input:
                x = self.reorganize_x(x, context, timestep, t_s, clip_embdding)
            else:
                x = self.reorganize_x(x, context, timestep, t_s)
            ## Context parallel
            if self.config.context_parallel_size > 1:
                x = split_forward_gather_backward(
                    x, get_context_parallel_group(), dim=0, grad_scale="down"
                )
        else:
            x = None
        extra_block_kwargs = {}
        x = self.decoder(
            hidden_states=x,
            attention_mask=None,
            context=None,
            context_mask=None,
            inference_params=None,
            packed_seq_params=None,
            rotary_pos_emb=freqs,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return x
        if self.config.context_parallel_size > 1:
            x = gather_forward_split_backward(
                x, get_context_parallel_group(), dim=0, grad_scale="up"
            )
        t = x[-1:, :, :].squeeze(1).to(torch.bfloat16)
        vido_length = self.config.max_video_length

        ##cat latent
        chunk_size = x.shape[0] // self.config.context_parallel_size
        sp_latent_len = vido_length // self.config.context_parallel_size
        chunks = []
        for i in range(self.config.context_parallel_size):
            chunk = x[i * chunk_size : (i + 1) * chunk_size]
            front_x = chunk[:sp_latent_len]
            chunks.append(front_x)
        x = torch.cat(chunks, dim=0).to(torch.bfloat16)

        x = rearrange(x, f"S B C ->B S C").contiguous()
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1"
        self.decoder.set_input_tensor(input_tensor[0])
