# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image RoPE implementation"""

import functools
import math
from typing import List

import torch
import torch.nn as nn

# Fused Triton RoPE — reuse wan's interleaved kernel since qwen_image uses the
# same view_as_complex layout (consecutive pairs).
try:
    from ..wan.custom_ops import (
        apply_rotary_interleaved as _apply_rotary_interleaved,
        _rotary_interleaved_kernel,
    )
    import triton as _triton
    _TRITON_ROPE_AVAILABLE = True
except Exception:  # pragma: no cover - triton not built / import failure
    _apply_rotary_interleaved = None
    _rotary_interleaved_kernel = None
    _triton = None
    _TRITON_ROPE_AVAILABLE = False


def apply_rotary_emb_qwen(x: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply Qwen-style rotary embedding to ``x`` using complex ``freqs_cis``.

    Reference PyTorch implementation. ``x`` layout: [..., seq, hn].
    """
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


def _apply_rotary_interleaved_sbnd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Invoke the wan interleaved-RoPE kernel directly on an [s, b, np, hn]
    layout without permuting/contig-copying.

    The wan kernel is stride-aware; we simply remap its "batch" and "seqlen"
    axes so that ``pid_m`` iterates over the ``s`` dimension (which is what
    cos/sin are indexed by), and ``pid_batch`` iterates over ``b``.
    """
    seqlen, batch, nheads, headdim = x.shape
    seqlen_ro = cos.shape[0]
    rotary_dim = cos.shape[1] * 2

    output = torch.empty_like(x)
    if rotary_dim < headdim:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = _triton.next_power_of_2(rotary_dim)
    BLOCK_M = 4
    grid = (_triton.cdiv(seqlen, BLOCK_M), batch, nheads)

    with torch.cuda.device(x.device.index):
        _rotary_interleaved_kernel[grid](
            output, x, cos, sin,
            seqlen, rotary_dim, seqlen_ro,
            # output strides: (batch=b, seqlen=s, nheads, headdim)
            output.stride(1), output.stride(0), output.stride(2), output.stride(3),
            # x strides
            x.stride(1), x.stride(0), x.stride(2), x.stride(3),
            BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=1,
        )
    return output


class _FusedQwenRope(torch.autograd.Function):
    """Autograd wrapper for the fused interleaved-RoPE Triton kernel.

    The interleaved RoPE is a per-pair rotation ``y_even = x_even*c - x_odd*s``,
    ``y_odd = x_even*s + x_odd*c``. Its Jacobian w.r.t. ``x`` is a rotation by
    ``-theta``, which is the same forward kernel with ``sin`` negated. cos/sin
    are position tables (no gradient needed).
    """

    @staticmethod
    def forward(ctx, x, cos, sin):
        """Apply interleaved rotary embedding in forward pass."""
        ctx.save_for_backward(cos, sin)
        return _apply_rotary_interleaved_sbnd(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_out):
        """Compute gradient by applying rotary with negated sin."""
        cos, sin = ctx.saved_tensors
        grad_in = _apply_rotary_interleaved_sbnd(grad_out.contiguous(), cos, -sin)
        return grad_in, None, None


def apply_rotary_emb_qwen_fused_sbnd(x: torch.Tensor, freqs_cis: torch.Tensor):
    """Fused Triton RoPE for [s, b, np, hn] tensors.

    ``freqs_cis`` is a complex tensor of shape ``[seq_ro, hn/2]`` (produced by
    ``QwenEmbedRope`` via ``torch.polar``). We split its real/imag parts into
    cos/sin real tensors and invoke a stride-remapped launch of the wan
    interleaved kernel, avoiding the [s,b,n,d]<->[b,n,s,d] permutes entirely.

    Wrapped in a ``torch.autograd.Function`` so gradients flow through Q/K.

    Falls back to the reference implementation if triton is unavailable.
    """
    seqlen = x.shape[0]
    if not _TRITON_ROPE_AVAILABLE:
        y = x.permute(1, 2, 0, 3).contiguous()
        y = apply_rotary_emb_qwen(y, freqs_cis)
        return y.permute(2, 0, 1, 3).contiguous()

    cos = freqs_cis[:seqlen].real.contiguous()
    sin = freqs_cis[:seqlen].imag.contiguous()
    return _FusedQwenRope.apply(x.contiguous(), cos, sin)


class QwenEmbedRope(nn.Module):
    """RoPE frequency table for Qwen-Image (2-D positional axes)."""

    def __init__(self, theta: int, axes_dim: list[int], scale_rope: bool = False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = self._build_freqs(pos_index)
        self.neg_freqs = self._build_freqs(neg_index)
        self.rope_cache = {}
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """Return complex RoPE frequencies for the given index and axis dim."""
        if dim % 2 != 0:
            raise ValueError("RoPE axis dim must be even")
        freqs = torch.outer(
            index,
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)),
        )
        return torch.polar(torch.ones_like(freqs), freqs)

    def _build_freqs(self, index):
        return torch.cat(
            [self.rope_params(index, dim, self.theta) for dim in self.axes_dim], dim=1
        )

    def _expand_pos_freqs_if_needed(self, video_fhw, txt_seq_lens):
        if isinstance(video_fhw, list):
            video_fhw = tuple(max(i[j] for i in video_fhw) for j in range(3))
        _, height, width = video_fhw
        max_vid_index = max(height // 2, width // 2) if self.scale_rope else max(height, width)
        required_len = max_vid_index + max(txt_seq_lens)
        if required_len <= self.pos_freqs.shape[0]:
            return
        new_max_len = math.ceil(required_len / 512) * 512
        device = self.pos_freqs.device
        pos_index = torch.arange(new_max_len, device=device)
        neg_index = torch.arange(new_max_len, device=device).flip(0) * -1 - 1
        self.pos_freqs = self._build_freqs(pos_index).to(device)
        self.neg_freqs = self._build_freqs(neg_index).to(device)
        self.rope_cache.clear()

    def forward(self, video_fhw, txt_seq_lens, device):
        """Return the (video, text) RoPE frequency tensors for the requested shapes."""
        self._expand_pos_freqs_if_needed(video_fhw, txt_seq_lens)
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)
            self.rope_cache.clear()

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"
            if rope_key not in self.rope_cache:
                seq_lens = frame * height * width
                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                if self.scale_rope:
                    freqs_height = torch.cat(
                        [freqs_neg[1][-(height - height // 2):], freqs_pos[1][: height // 2]], dim=0
                    )
                    freqs_width = torch.cat(
                        [freqs_neg[2][-(width - width // 2):], freqs_pos[2][: width // 2]], dim=0
                    )
                    freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
                else:
                    freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)
                freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
                self.rope_cache[rope_key] = freqs.clone().contiguous()
            vid_freqs.append(self.rope_cache[rope_key])
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        return torch.cat(vid_freqs, dim=0), txt_freqs

    def forward_sampling(self, video_fhw, txt_seq_lens, device):
        """Convenience alias for ``forward``; provided for sampling code paths."""
        return self.forward(video_fhw, txt_seq_lens, device)


class QwenEmbedLayer3DRope(nn.Module):
    """Layer-aware 3-D RoPE variant used when ``use_layer3d_rope`` is enabled."""

    def __init__(self, theta: int, axes_dim: List[int], scale_rope: bool = False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = self._build_freqs(pos_index)
        self.neg_freqs = self._build_freqs(neg_index)
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """Return complex RoPE frequencies for the given index and axis dim."""
        if dim % 2 != 0:
            raise ValueError("RoPE axis dim must be even")
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    def _build_freqs(self, index):
        return torch.cat([self.rope_params(index, dim, self.theta) for dim in self.axes_dim], dim=1)

    def forward(self, video_fhw, txt_seq_lens, device):
        """Return the (video, text) 3-D RoPE frequency tensors for the requested shapes."""
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)
        video_fhw = [video_fhw] if not isinstance(video_fhw, list) else video_fhw
        if video_fhw and not isinstance(video_fhw[0], (list, tuple)):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        layer_num = len(video_fhw) - 1
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            if idx == layer_num:
                video_freq = self._compute_condition_freqs(frame, height, width)
            else:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            vid_freqs.append(video_freq.to(device))
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_vid_index = max(max_vid_index, layer_num)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max(txt_seq_lens), ...]
        return torch.cat(vid_freqs, dim=0), txt_freqs

    @functools.lru_cache(maxsize=None)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2):], freqs_pos[1][: height // 2]], dim=0
            )
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2):], freqs_pos[2][: width // 2]], dim=0
            )
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)
        return torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1).clone().contiguous()

    @functools.lru_cache(maxsize=None)
    def _compute_condition_freqs(self, frame, height, width):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_frame = freqs_neg[0][-1:].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2):], freqs_pos[1][: height // 2]], dim=0
            )
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2):], freqs_pos[2][: width // 2]], dim=0
            )
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)
        return torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1).clone().contiguous()
