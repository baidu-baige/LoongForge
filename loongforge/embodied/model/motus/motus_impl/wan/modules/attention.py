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

"""Attention backends for the Wan2.2 model: FlashAttention-2 varlen and scaled-dot-product fallbacks."""
import torch

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'sdpa_attention',
    'flash_attention_dense',
    'attention',
]


def flash_attention_dense(
    q,
    k,
    v,
    dropout_p=0.,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """Dense (non-varlen) flash attention for the CUDA-graph capture path.

    Inputs are [B, L, N, C] with uniform sequence lengths (training case), so
    the varlen packing origin uses (flash_attn_varlen_func) degenerates to this
    fixed-shape dense call. Unlike flash_attn_varlen_func it needs no dynamic
    cu_seqlens / python-int max_seqlen, so it is CUDA-graph capturable, while
    still dispatching to the SAME flash_attn 2.x kernels origin uses -> matches
    origin's attention numerics far better than PyTorch's bundled SDPA-FLASH.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    out_dtype = q.dtype

    if q.dtype not in half_dtypes:
        q = q.to(dtype)
    if k.dtype not in half_dtypes:
        k = k.to(dtype)
    if v.dtype not in half_dtypes:
        v = v.to(dtype)

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    out = flash_attn.flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
    )

    return out.type(out_dtype)


def sdpa_attention(
    q,
    k,
    v,
    dropout_p=0.,
    softmax_scale=None,
    causal=False,
    dtype=torch.bfloat16,
):
    """Compute attention via ``torch.nn.functional.scaled_dot_product_attention`` and restore the input dtype."""
    half_dtypes = (torch.float16, torch.bfloat16)
    out_dtype = q.dtype

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if q.dtype not in half_dtypes:
        q = q.to(dtype)
    if k.dtype not in half_dtypes:
        k = k.to(dtype)
    if v.dtype not in half_dtypes:
        v = v.to(dtype)

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )

    return out.transpose(1, 2).contiguous().type(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        """Cast ``x`` to the target half-precision dtype unless it is already float16/bfloat16."""
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    # apply attention
    assert FLASH_ATTN_2_AVAILABLE
    x = flash_attn.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
            0, dtype=torch.int32).to(q.device, non_blocking=True),
        cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
            0, dtype=torch.int32).to(q.device, non_blocking=True),
        max_seqlen_q=lq,
        max_seqlen_k=lk,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic).unflatten(0, (b, q.shape[0] // b))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """Dispatch to FlashAttention-2 when available, otherwise fall back to scaled-dot-product attention."""
    if FLASH_ATTN_2_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        return sdpa_attention(
            q=q,
            k=k,
            v=v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            dtype=dtype,
        )
