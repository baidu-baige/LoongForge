# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from https://github.com/thu-ml/Motus under the Apache-2.0 License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wan2.2 DiT diffusion backbone: rotary embeddings, attention blocks, and the WanModel transformer."""
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention, flash_attention_dense

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    """Return sinusoidal position embeddings of dimension ``dim`` for the given ``position`` tensor."""
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    """Precompute complex-valued rotary position embedding frequencies for ``max_seq_len`` positions."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


from functools import lru_cache

# Cache of precomputed real-valued rotary tables (cos, sin), keyed by the grid
# shape and device.  Populating this touches the device (cat/expand), so it must
# happen during warmup; inside CUDA-Graph capture we only read from it.
_ROPE_REAL_CACHE: dict = {}


def _get_rope_cos_sin(fpart, hpart, wpart, f, h, w, device):
    """Return (cos, sin) real tensors of shape [seq_len, 1, C] for the fhw grid.

    Built once per (f, h, w, device) and cached, so the device-side
    cat/expand/reshape never runs during CUDA-Graph capture.
    """
    key = (f, h, w, device)
    cached = _ROPE_REAL_CACHE.get(key)
    if cached is not None:
        return cached
    seq_len = f * h * w
    fi = torch.cat([
        fpart[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        hpart[:h].view(1, h, 1, -1).expand(f, h, w, -1),
        wpart[:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(seq_len, 1, -1)  # complex [seq_len, 1, C]
    # Keep the rotation table in fp64 to match origin's complex128 rope math
    # (origin does the whole rotation in fp64 then downcasts). fp64 elementwise
    # ops are CUDA-graph-safe; only view_as_complex / in-place scatter were not.
    cos = fi.real.double().contiguous()
    sin = fi.imag.double().contiguous()
    _ROPE_REAL_CACHE[key] = (cos, sin)
    return cos, sin


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor,
               fhw: tuple = None) -> torch.Tensor:
    """Apply rotary position embedding.

    ``fhw`` is an optional (f, h, w) tuple of Python ints.  When provided the
    function skips all device-side tensor operations on ``grid_sizes`` and uses
    these constants directly, making the function fully CUDA-Graph compatible.
    When ``fhw`` is None the function falls back to reading ``grid_sizes`` from
    the tensor (safe outside of graph capture, e.g. inference with variable
    sequence lengths).
    """
    B, T, N, CC = x.shape
    assert CC % 2 == 0, "last dim must be 2C (real, imag)"
    C = CC // 2

    c_f = C - 2 * (C // 3)
    c_h = C // 3
    c_w = C // 3
    fpart, hpart, wpart = freqs.split([c_f, c_h, c_w], dim=1)

    if fhw is not None:
        # CUDA-Graph-safe path: pure real-valued rotary, no complex kernels and
        # no in-place scatter. The cos/sin table is precomputed & cached during
        # warmup (device cat/expand happens there, never inside capture).
        f, h, w = fhw
        seq_len = f * h * w

        cos, sin = _get_rope_cos_sin(fpart, hpart, wpart, f, h, w, x.device)
        # cos/sin: [seq_len, 1, C] -> broadcast over B, N: [1, seq_len, 1, C]
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        # Do the rotation in fp64 (matches origin's complex128 path), downcast
        # to fp32 only at the very end.
        xf = x.double().reshape(B, T, N, C, 2)
        x_re = xf[..., 0]
        x_im = xf[..., 1]

        # Rotate only the first seq_len positions; leave the tail untouched.
        xr = x_re[:, :seq_len]
        xi = x_im[:, :seq_len]
        out_re = xr * cos - xi * sin
        out_im = xr * sin + xi * cos

        rotated = torch.stack((out_re, out_im), dim=-1).reshape(B, seq_len, N, CC)
        if seq_len == T:
            # All positions rotated (video-only rope): no tail concat needed.
            return rotated.float()
        # Concatenate (not in-place scatter) with the unrotated tail.
        return torch.cat([rotated, x[:, seq_len:].double()], dim=1).float()
    else:
        x_c = torch.view_as_complex(x.to(torch.float64).reshape(B, T, N, -1, 2)).contiguous()
        y_c = x_c.clone()
        # Fallback: per-sample loop (inference / variable-length batches).
        gsz = grid_sizes.to(torch.long)
        for i, (f, h, w) in enumerate(gsz.tolist()):
            seq_len = f * h * w
            fi = torch.cat([
                fpart[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                hpart[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                wpart[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ], dim=-1).reshape(seq_len, 1, -1).contiguous()
            y_c[i, :seq_len] = x_c[i, :seq_len] * fi

        y = torch.view_as_real(y_c).reshape(B, T, N, -1).float()
        return y
    y = torch.view_as_real(y_c).reshape(B, T, N, -1).float()
    return y

@torch.amp.autocast('cuda', enabled=False)
def rope_apply_original(x, grid_sizes, freqs):
    """Apply 3D rotary position embedding to ``x`` per sample using a complex-valued reference loop."""
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):
    """Root-mean-square layer normalization with a learnable per-channel weight."""

    def __init__(self, dim, eps=1e-5):
        """Initialize the RMSNorm with feature dimension ``dim`` and numerical epsilon ``eps``."""
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        """Return ``x`` scaled by the reciprocal of its RMS over the last dimension."""
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    """Layer normalization that computes in float32 and casts the result back to the input dtype."""

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        """Initialize the layer norm over ``dim`` features with optional learnable affine parameters."""
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    """Multi-head self-attention with optional QK RMSNorm, rotary embeddings, and trimodal MoT support."""

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        """Initialize projections, optional QK norms, window size, and head configuration for self-attention."""
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs,
                action_q: torch.Tensor = None,
                action_k: torch.Tensor = None,
                action_v: torch.Tensor = None,
                und_q: torch.Tensor = None,
                und_k: torch.Tensor = None,
                und_v: torch.Tensor = None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            action_q/k/v(Tensor, optional): Action expert Q/K/V for trimodal MoT
            und_q/k/v(Tensor, optional): Understanding expert Q/K/V for trimodal MoT
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            """Project ``x`` into normalized query/key and value tensors reshaped to [B, S, num_heads, head_dim]."""
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # Trimodal MoT branch: WAN + Action + Understanding
        if action_q is not None or und_q is not None:
            L_x = q.size(1)
            
            # Apply RoPE only to video tokens (q, k)
            q_video_rope = rope_apply(q, grid_sizes, freqs, getattr(self, "graph_fhw", None))
            k_video_rope = rope_apply(k, grid_sizes, freqs, getattr(self, "graph_fhw", None))
            
            # Prepare parts for concatenation
            q_parts = [q_video_rope]
            k_parts = [k_video_rope] 
            v_parts = [v]
            
            # Add action tokens if provided
            if action_q is not None:
                q_parts.append(action_q)
                k_parts.append(action_k)
                v_parts.append(action_v)
                L_action = action_q.size(1)
            else:
                L_action = 0
                
            # Add understanding tokens if provided
            if und_q is not None:
                q_parts.append(und_q)
                k_parts.append(und_k)
                v_parts.append(und_v)
                L_und = und_q.size(1)
            else:
                L_und = 0
            
            # Concatenate all modalities
            q_cat = torch.cat(q_parts, dim=1)
            k_cat = torch.cat(k_parts, dim=1)
            v_cat = torch.cat(v_parts, dim=1)

            if getattr(self, "use_sdpa", False):
                attn_out = flash_attention_dense(
                    q_cat, k_cat, v_cat, window_size=self.window_size)
            else:
                attn_out = flash_attention(
                    q=q_cat,
                    k=k_cat,
                    v=v_cat,
                    k_lens=seq_lens,
                    window_size=self.window_size)

            # Split outputs back to respective modalities
            x_out = attn_out[:, :L_x, :, :]
            outputs = [x_out]
            
            start_idx = L_x
            if action_q is not None:
                action_out = attn_out[:, start_idx:start_idx + L_action, :, :]
                outputs.append(action_out)
                start_idx += L_action
            else:
                outputs.append(None)
                
            if und_q is not None:
                und_out = attn_out[:, start_idx:start_idx + L_und, :, :]
                outputs.append(und_out)
            else:
                outputs.append(None)

            # Project WAN branch; other branches returned in head shape for external projection
            x_out = x_out.flatten(2)
            x_out = self.o(x_out)
            outputs[0] = x_out
            
            return tuple(outputs)

        # Standard branch (no MoT)
        q = rope_apply(q, grid_sizes, freqs, getattr(self, "graph_fhw", None))
        k = rope_apply(k, grid_sizes, freqs, getattr(self, "graph_fhw", None))
        if getattr(self, "use_sdpa", False):
            x = flash_attention_dense(q, k, v, window_size=self.window_size)
        else:
            x = flash_attention(
                q=q,
                k=k,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    """Cross-attention where queries come from ``x`` and keys/values come from an external context sequence."""

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        if getattr(self, "use_sdpa", False):
            x = flash_attention_dense(q, k, v)
        else:
            x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    """Transformer block combining modulated self-attention, cross-attention, and a gated feed-forward network."""

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        """Initialize the norm layers, self/cross attention, feed-forward network, and modulation parameters."""
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)

        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            """Apply cross-attention against ``context`` followed by the modulated feed-forward network."""
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):
    """Output head that modulates, normalizes, and linearly projects features back to patchified pixels."""

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        """Initialize the norm, linear projection to ``prod(patch_size) * out_dim``, and modulation parameters."""
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        graph_fhw = getattr(self, "graph_fhw", None)
        if graph_fhw is not None:
            f, h, w = graph_fhw[0], graph_fhw[1], graph_fhw[2]
            seq_len = f * h * w
            u = x[:, :seq_len].view(x.size(0), f, h, w, *self.patch_size, c)
            u = torch.einsum('bfhwpqrc->bcfphqwr', u)
            return [v for v in u.reshape(x.size(0), c,
                                         f * self.patch_size[0],
                                         h * self.patch_size[1],
                                         w * self.patch_size[2])]

        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
