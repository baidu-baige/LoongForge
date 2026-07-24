# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.
#
# DreamZero WanVideoVAE / WanVideoVAE38 modules.
#
# Frozen by design: WanVideoVAE / WanVideoVAE38 internally call
# `.eval().requires_grad_(False)` on the inner VideoVAE / VideoVAE38 trunk.
# The VAE stays in its native PyTorch implementation because checkpoint
# compatibility is more important than rewriting this frozen component.
#
# Implementation notes:
# - `self.mean` / `self.std` are constructed with `device='cuda'` inside
#   `__init__`. Callers must instantiate on CUDA hosts.
# - No other behavioural changes.
"""DreamZero WanVideoVAE / WanVideoVAE38 modules and their supporting VAE building blocks."""

import logging

from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

CACHE_T = 2

logger = logging.getLogger(__name__)

_DREAMZERO_LOG_SEEN = set()


def _dreamzero_rank0():
    """Return True if running on rank 0, or if distributed training is not initialized."""
    try:
        import torch.distributed as _d

        return not (_d.is_available() and _d.is_initialized()) or _d.get_rank() == 0
    except Exception:
        return True


def _dreamzero_log_once(tag, msg):
    """Log `msg` at most once per `tag`, only from rank 0."""
    if tag in _DREAMZERO_LOG_SEEN:
        return
    _DREAMZERO_LOG_SEEN.add(tag)
    if _dreamzero_rank0():
        logger.info("%s", msg)


def check_is_instance(model, module_class):
    """Return True if `model` (or its wrapped `.module`) is an instance of `module_class`."""
    if isinstance(model, module_class):
        return True
    if hasattr(model, "module") and isinstance(model.module, module_class):
        return True
    return False


def block_causal_mask(x, block_size):
    """Build a block-wise causal attention mask for tensor `x` split into chunks of `block_size`."""
    # params
    b, n, s, _ = x.shape
    assert s % block_size == 0
    num_blocks = s // block_size

    # build mask
    mask = torch.zeros(b, n, s, s, dtype=torch.bool, device=x.device)
    for i in range(num_blocks):
        mask[:, :,
             i * block_size:(i + 1) * block_size, :(i + 1) * block_size] = 1
    return mask


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the causal 3d convolution and precompute its asymmetric padding."""
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        """Apply causal padding (optionally prepending cached frames) then run the 3d convolution."""
        padding = list(self._padding)
        if cache_x is not None and cache_x.shape[-2:] != x.shape[-2:]:
            cache_x = None
        if cache_x is not None and self._padding[4] > 0:
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMSNorm(nn.Module):
    """RMS normalization with a learnable scale (and optional bias)."""

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        """Set up the RMS norm's learnable gamma/bias parameters and broadcast shape."""
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        """Apply RMS normalization followed by the learnable scale and bias."""
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    """nn.Upsample wrapper that keeps nearest-neighbor interpolation numerically stable in bfloat16."""

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    """Spatial/temporal up- or down-sampling module used by the VAE encoder/decoder."""

    def __init__(self, dim, mode):
        """Build the resampling layers (and optional causal time conv) for the given mode."""
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(dim,
                                          dim * 2, (3, 1, 1),
                                          padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim,
                                          dim, (3, 1, 1),
                                          stride=(2, 1, 1),
                                          padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None):
        """Resample `x` spatially/temporally, using and updating `feat_cache` for causal chunks."""
        if feat_idx is None:
            feat_idx = [0]
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                            cache_x,
                        ], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x),
                            cache_x,
                        ], dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    prev_x = feat_cache[idx][:, :, -1:, :, :]
                    if prev_x.shape[-2:] == x.shape[-2:]:
                        x = torch.cat([prev_x, x], 2)
                    else:
                        x = torch.cat([x[:, :, :1, :, :], x], 2)
                    x = self.time_conv(x)
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        """Initialize `conv`'s weight as an identity mapping and zero its bias."""
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        conv_weight.data[:, :, 1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        """Initialize `conv`'s weight to duplicate an identity mapping across two halves and zero its bias."""
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)



def patchify(x, patch_size):
    """Rearrange spatial patches of size `patch_size` into the channel dimension."""
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(x,
                      "b c f (h q) (w r) -> b (c r q) f h w",
                      q=patch_size,
                      r=patch_size)
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")
    return x


def unpatchify(x, patch_size):
    """Inverse of `patchify`: expand channel-packed patches of size `patch_size` back into space."""
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(x,
                      "b (c r q) f h w -> b c f (h q) (w r)",
                      q=patch_size,
                      r=patch_size)
    return x


class Resample38(Resample):
    """Resample variant used by the 38-channel VAE variant (WanVideoVAE38)."""

    def __init__(self, dim, mode):
        """Build the resampling layers (and optional causal time conv) for the given mode."""
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        nn.Module.__init__(self)
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
        else:
            self.resample = nn.Identity()


class ResidualBlock(nn.Module):
    """Residual block: two causal conv + RMS-norm branches with a (possibly projected) shortcut."""

    def __init__(self, in_dim, out_dim, dropout=0.0):
        """Build the residual and shortcut paths for `in_dim` -> `out_dim` channels."""
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None):
        """Run the residual branch (with causal feature caching) and add the shortcut."""
        if feat_idx is None:
            feat_idx = [0]
        h = self.shortcut(x)
        for layer in self.residual:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    prev_x = feat_cache[idx][:, :, -1, :, :].unsqueeze(2)
                    if prev_x.shape[-2:] == cache_x.shape[-2:]:
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            prev_x,
                            cache_x,
                        ], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        x = x + h
        return x


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        """Build the QKV projection, output projection, and norm for single-head attention."""
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        """Apply single-head causal self-attention over the spatial dimensions of `x`."""
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(
            0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            #attn_mask=block_causal_mask(q, block_size=h * w)
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class AvgDown3D(nn.Module):
    """Average-pooling based down-sampler that reshapes and averages grouped channels."""
    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
    ):
        """Configure the temporal/spatial downsampling factors and channel grouping."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pad, reshape into groups, and average-pool `x` down by the configured factors."""
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):
    """Duplication-based up-sampler; the inverse counterpart of `AvgDown3D`."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
    ):
        """Configure the temporal/spatial upsampling factors and channel repeat count."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        """Repeat-interleave and reshape `x` to up-sample it by the configured factors."""
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class DownResidualBlock(nn.Module):
    """Down-sampling stage combining stacked residual blocks with an averaged shortcut."""

    def __init__(
        self, in_dim, out_dim, dropout, mult, temperal_downsample=False, down_flag=False
    ):
        """Build the residual main path and the AvgDown3D shortcut for this stage."""
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample38(out_dim, mode=mode))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=None):
        """Run the residual main path and add the averaged-downsample shortcut."""
        if feat_idx is None:
            feat_idx = [0]
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)

        shortcut = self.avg_shortcut(x_copy)
        x = x + shortcut
        return x


class UpResidualBlock(nn.Module):
    """Up-sampling stage combining stacked residual blocks with a duplicated shortcut."""

    def __init__(
        self, in_dim, out_dim, dropout, mult, temperal_upsample=False, up_flag=False
    ):
        """Build the residual main path and the optional DupUp3D shortcut for this stage."""
        super().__init__()
        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample38(out_dim, mode=mode))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        """Run the residual main path and add the optional duplicated-upsample shortcut."""
        if feat_idx is None:
            feat_idx = [0]
        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        else:
            return x_main


class Encoder3d(nn.Module):
    """3D convolutional VAE encoder producing latent mean/log-variance features."""

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=None,
                 num_res_blocks=2,
                 attn_scales=None,
                 temperal_downsample=None,
                 dropout=0.0):
        """Build the downsample/middle/head stages of the encoder."""
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_downsample is None:
            temperal_downsample = [True, True, False]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim, dropout),
                                    AttentionBlock(out_dim),
                                    ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(RMSNorm(out_dim, images=False), nn.SiLU(),
                                  CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=None):
        """Encode `x` through the downsample, middle, and head stages, with causal feature caching."""
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                    cache_x,
                ], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                        cache_x,
                    ], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Encoder3d38(nn.Module):
    """3D convolutional VAE encoder variant for the 38-channel VAE (WanVideoVAE38)."""

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=None,
                 num_res_blocks=2,
                 attn_scales=None,
                 temperal_downsample=None,
                 dropout=0.0):
        """Build the downsample/middle/head stages of the encoder."""
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_downsample is None:
            temperal_downsample = [False, True, True]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = (
                temperal_downsample[i] if i < len(temperal_downsample) else False
            )
            downsamples.append(
                DownResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # # output blocks
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )


    def forward(self, x, feat_cache=None, feat_idx=None):
        """Encode `x` through the downsample, middle, and head stages, with causal feature caching."""
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class Decoder3d(nn.Module):
    """3D convolutional VAE decoder that reconstructs video frames from latent codes."""

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=None,
                 num_res_blocks=2,
                 attn_scales=None,
                 temperal_upsample=None,
                 dropout=0.0):
        """Build the middle/upsample/head stages of the decoder."""
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_upsample is None:
            temperal_upsample = [False, True, True]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0], dropout),
                                    AttentionBlock(dims[0]),
                                    ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(RMSNorm(out_dim, images=False), nn.SiLU(),
                                  CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=None):
        """Decode latent `x` through the middle, upsample, and head stages, with causal feature caching."""
        if feat_idx is None:
            feat_idx = [0]
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                    cache_x,
                ], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                        cache_x,
                    ], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x



class Decoder3d38(nn.Module):
    """3D convolutional VAE decoder variant for the 38-channel VAE (WanVideoVAE38)."""

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=None,
                 num_res_blocks=2,
                 attn_scales=None,
                 temperal_upsample=None,
                 dropout=0.0):
        """Build the middle/upsample/head stages of the decoder."""
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_upsample is None:
            temperal_upsample = [False, True, True]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0], dropout),
                                    AttentionBlock(dims[0]),
                                    ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(
                UpResidualBlock(in_dim=in_dim,
                                out_dim=out_dim,
                                dropout=dropout,
                                mult=num_res_blocks + 1,
                                temperal_upsample=t_up_flag,
                                up_flag=i != len(dim_mult) - 1))
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(RMSNorm(out_dim, images=False), nn.SiLU(),
                                  CausalConv3d(out_dim, 12, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        """Decode latent `x` through the middle, upsample, and head stages, with causal feature caching."""
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    """Count the number of `CausalConv3d` layers contained in `model`."""
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class VideoVAE(nn.Module):
    """Core VAE trunk (encoder + decoder) shared by WanVideoVAE, without any I/O normalization."""

    def __init__(self,
                 dim=96,
                 z_dim=16,
                 dim_mult=None,
                 num_res_blocks=2,
                 attn_scales=None,
                 temperal_downsample=None,
                 dropout=0.0):
        """Build the encoder/decoder pair and the latent projection convs."""
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_downsample is None:
            temperal_downsample = [False, True, True]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

        self._enc_conv_num = count_conv3d(self.encoder)
        self._dec_conv_num = count_conv3d(self.decoder)

    def encode(self, x, scale):
        """Encode video `x` into a scaled latent mean, processing frames in causal chunks."""
        feat_map = [None] * self._enc_conv_num

        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4

        out = self.encoder(
            x[:, :, :1, :, :],
            feat_cache=feat_map,
            feat_idx=[0],
        )

        for i in range(1, iter_):
            out_ = self.encoder(
                x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                feat_cache=feat_map,
                feat_idx=[0],
            )
            out = torch.cat([out, out_], dim=2)
        mu, _ = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        return mu

    def decode(self, z, scale):
        """Decode a scaled latent `z` back into video frames, processing frames in causal chunks."""
        feat_map = [None] * self._dec_conv_num

        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=z.dtype, device=z.device)
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)

        out = self.decoder(
            x[:, :, 0:1, :, :],
            feat_cache=feat_map,
            feat_idx=[0],
        )

        for i in range(1, iter_):
            out_ = self.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=feat_map,
                feat_idx=[0],
            )
            out = torch.cat([out, out_], dim=2)
        return out

    def reparameterize(self, mu, log_var):
        """Sample a latent via the VAE reparameterization trick given `mu` and `log_var`."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu


class WanVideoVAE(nn.Module):
    """Wan video VAE wrapper: normalizes latents and adds tiled/batched encode-decode helpers."""

    def __init__(self, z_dim=16, vae_pretrained_path: str | None = None):
        """Set up latent normalization stats and construct the frozen `VideoVAE` trunk."""
        super().__init__()
        self.batch_encode = False

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, device='cuda')
        self.std = torch.tensor(std, device='cuda')
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = VideoVAE(z_dim=z_dim).eval().requires_grad_(False)
        self.upsampling_factor = 8
        self.z_dim = z_dim
        self.vae_pretrained_path = vae_pretrained_path


    def build_1d_mask(self, length, left_bound, right_bound, border_width, device):
        """Build a 1D blending mask that ramps up/down near unbounded edges over `border_width`."""
        x = torch.ones((length,), device=device)
        border = (torch.arange(border_width, device=device) + 1)
        if not left_bound:
            x[:border_width] = border / border_width
        if not right_bound:
            x[-border_width:] = torch.flip(border / border_width, dims=(0,))
        return x


    def build_mask(self, data, is_bound, border_width):
        """Build a 2D blending mask for `data` by combining the H and W 1D masks."""
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0], device=data.device)
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1], device=data.device)

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask


    def tiled_decode(self, hidden_states, tile_size, tile_stride):
        """Decode `hidden_states` tile-by-tile and blend overlapping regions with `build_mask`."""
        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        out_T = T * 4 - 3
        weight = torch.zeros(
            (1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        values = torch.zeros(
            (1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for h, h_, w, w_ in tqdm(tasks, desc="VAE decoding"):
            hidden_states_batch = hidden_states[:, :, :, h:h_, w:w_]
            hidden_states_batch = self.model.decode(hidden_states_batch, self.scale)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) * self.upsampling_factor,
                    (size_w - stride_w) * self.upsampling_factor,
                )
            ).to(dtype=hidden_states.dtype)

            target_h = h * self.upsampling_factor
            target_w = w * self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        values = values.clamp_(-1, 1)
        return values

    def tiled_encode(self, video, tile_size, tile_stride):
        """Encode `video` tile-by-tile and blend overlapping regions with `build_mask`."""
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        out_T = (T + 3) // 4
        weight = torch.zeros(
            (1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
            dtype=video.dtype,
            device=video.device,
        )
        values = torch.zeros(
            (1, self.z_dim, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
            dtype=video.dtype,
            device=video.device,
        )

        for h, h_, w, w_ in tqdm(tasks, desc="VAE encoding"):
            hidden_states_batch = video[:, :, :, h:h_, w:w_]
            hidden_states_batch = self.model.encode(hidden_states_batch, self.scale)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) // self.upsampling_factor,
                    (size_w - stride_w) // self.upsampling_factor,
                )
            ).to(dtype=video.dtype)

            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        return values

    def single_encode(self, video):
        """Encode a single (non-tiled) video clip into its latent representation."""
        x = self.model.encode(video, self.scale)
        # The outputs of torch compile always need to be cloned before being used.
        x = x.clone()
        return x

    def single_decode(self, hidden_state):
        """Decode a single (non-tiled) latent into a video clip, clamped to [-1, 1]."""
        video = self.model.decode(hidden_state, self.scale)
        return video.clamp_(-1, 1)

    def encode(self, videos, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        """Encode a batch of videos, using batched, per-video, or tiled encoding as configured."""
        if self.batch_encode and not tiled:
            _dreamzero_log_once(
                "batch_vae_encode",
                f"[dreamzero-vae] batch_vae_encode=true, "
                f"encode non-tiled video batch directly; batch_size={videos.shape[0]}",
            )
            return self.single_encode(videos)

        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                tile_size = (tile_size[0] * self.upsampling_factor, tile_size[1] * self.upsampling_factor)
                tile_stride = (tile_stride[0] * self.upsampling_factor, tile_stride[1] * self.upsampling_factor)
                hidden_state = self.tiled_encode(video, tile_size, tile_stride)
            else:
                hidden_state = self.single_encode(video)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)
        hidden_states = torch.stack(hidden_states)
        return hidden_states

    def decode(self, hidden_states, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        """Decode a batch of latents into videos, using tiled or single decoding as configured."""
        if tiled:
            video = self.tiled_decode(hidden_states, tile_size, tile_stride)
        else:
            video = self.single_decode(hidden_states)
        return video


    @staticmethod
    def state_dict_converter():
        """Return the state-dict converter used to load Civitai-format checkpoints."""
        return WanVideoVAEStateDictConverter()


class WanVideoVAEStateDictConverter:
    """Converts third-party (Civitai-style) VAE checkpoints into this module's state-dict format."""

    def __init__(self):
        """No state to initialize; present for interface consistency."""
        pass

    def from_civitai(self, state_dict):
        """Convert a Civitai-format `state_dict` into this module's `model.`-prefixed format."""
        state_dict_ = {}
        if 'model_state' in state_dict:
            state_dict = state_dict['model_state']
        for name in state_dict:
            state_dict_['model.' + name] = state_dict[name]
        return state_dict_


class VideoVAE38(VideoVAE):
    """VAE trunk variant for the 38-channel VAE, using the `_38` encoder/decoder and patchify."""

    def __init__(self,
                 dim=160,
                 z_dim=48,
                 dec_dim=256,
                 dim_mult=None,
                 num_res_blocks=2,
                 attn_scales=None,
                 temperal_downsample=None,
                 dropout=0.0):
        """Build the `_38` encoder/decoder pair and the latent projection convs."""
        nn.Module.__init__(self)
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_downsample is None:
            temperal_downsample = [False, True, True]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d38(dim, z_dim * 2, dim_mult, num_res_blocks,
                                   attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d38(dec_dim, z_dim, dim_mult, num_res_blocks,
                                   attn_scales, self.temperal_upsample, dropout)

        self._enc_conv_num = count_conv3d(self.encoder)
        self._dec_conv_num = count_conv3d(self.decoder)

    def encode(self, x, scale):
        """Patchify and encode video `x` into a scaled latent mean, processing causal chunks."""
        feat_map = [None] * self._enc_conv_num
        x = patchify(x, patch_size=2)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4

        out = self.encoder(
            x[:, :, :1, :, :],
            feat_cache=feat_map,
            feat_idx=[0],
        )

        for i in range(1, iter_):
            out_ = self.encoder(
                x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                feat_cache=feat_map,
                feat_idx=[0],
            )
            out = torch.cat([out, out_], dim=2)

        mu, _ = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        return mu

    def decode(self, z, scale):
        """Decode a scaled latent `z` back into video frames and unpatchify the result."""
        feat_map = [None] * self._dec_conv_num

        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=z.dtype, device=z.device)
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)

        out = self.decoder(
            x[:, :, 0:1, :, :],
            feat_cache=feat_map,
            feat_idx=[0],
            first_chunk=True,
        )

        for i in range(1, iter_):
            out_ = self.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=feat_map,
                feat_idx=[0],
            )
            out = torch.cat([out, out_], 2)

        out = unpatchify(out, patch_size=2)
        return out


class WanVideoVAE38(WanVideoVAE):
    """38-channel variant of `WanVideoVAE`, wrapping `VideoVAE38` with matching normalization."""

    def __init__(self, z_dim=48, dim=160, vae_pretrained_path: str | None = None):
        """Set up latent normalization stats and construct the frozen `VideoVAE38` trunk."""
        nn.Module.__init__(self)
        self.batch_encode = False

        mean = [
            -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
            -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
            -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
            -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
            -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
            0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667
        ]
        std = [
            0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
            0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
            0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
            0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
            0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
            0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744
        ]
        self.mean = torch.tensor(mean, device='cuda')
        self.std = torch.tensor(std, device='cuda')
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = VideoVAE38(z_dim=z_dim, dim=dim).eval().requires_grad_(False)
        self.upsampling_factor = 16
        self.z_dim = z_dim
        self.vae_pretrained_path = vae_pretrained_path
