# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

"""RoPE primitives for DreamZero's 3D causal attention with action/state
register slicing.

These are pure tensor ops (no nn.Module / no TE). Kept in a dedicated file
to avoid a circular dependency on the attention block module.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def _reshape_register_freqs_like(freqs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Reshape 1D register freqs to match ``reference``'s rank (3D vs 2D)."""
    if reference.dim() == 3:
        return freqs.view(freqs.shape[0], 1, -1)
    return freqs.view(freqs.shape[0], -1)


if triton is not None:

    @triton.jit
    def _dreamzero_fused_rope_kernel(
        X,
        Y,
        FREQ_R,
        FREQ_I,
        ACTION_R,
        ACTION_I,
        STATE_R,
        STATE_I,
        TOTAL_PAIRS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HALF_DIM: tl.constexpr,
        BASE_SEQ_LEN: tl.constexpr,
        ACTION_LEN: tl.constexpr,
        STATE_LEN: tl.constexpr,
        ACTION_OFFSET: tl.constexpr,
        STATE_OFFSET: tl.constexpr,
        X_STRIDE_B: tl.constexpr,
        X_STRIDE_S: tl.constexpr,
        X_STRIDE_H: tl.constexpr,
        X_STRIDE_D: tl.constexpr,
        Y_STRIDE_B: tl.constexpr,
        Y_STRIDE_S: tl.constexpr,
        Y_STRIDE_H: tl.constexpr,
        Y_STRIDE_D: tl.constexpr,
        FREQ_STRIDE_S: tl.constexpr,
        FREQ_STRIDE_D: tl.constexpr,
        ACTION_STRIDE_S: tl.constexpr,
        ACTION_STRIDE_D: tl.constexpr,
        STATE_STRIDE_S: tl.constexpr,
        STATE_STRIDE_D: tl.constexpr,
        BACKWARD: tl.constexpr,
        USE_FP64: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < TOTAL_PAIRS

        pair_idx = offsets % HALF_DIM
        head_idx = (offsets // HALF_DIM) % NUM_HEADS
        seq_idx = (offsets // (HALF_DIM * NUM_HEADS)) % SEQ_LEN
        batch_idx = offsets // (HALF_DIM * NUM_HEADS * SEQ_LEN)

        even_idx = pair_idx * 2
        x_even_off = (
            batch_idx * X_STRIDE_B
            + seq_idx * X_STRIDE_S
            + head_idx * X_STRIDE_H
            + even_idx * X_STRIDE_D
        )
        x_odd_off = x_even_off + X_STRIDE_D
        y_even_off = (
            batch_idx * Y_STRIDE_B
            + seq_idx * Y_STRIDE_S
            + head_idx * Y_STRIDE_H
            + even_idx * Y_STRIDE_D
        )
        y_odd_off = y_even_off + Y_STRIDE_D

        is_base = seq_idx < BASE_SEQ_LEN
        is_action = (seq_idx >= BASE_SEQ_LEN) & (seq_idx < BASE_SEQ_LEN + ACTION_LEN)
        is_state = (
            (STATE_LEN > 0)
            & (seq_idx >= BASE_SEQ_LEN + ACTION_LEN)
            & (seq_idx < BASE_SEQ_LEN + ACTION_LEN + STATE_LEN)
        )
        action_seq = ACTION_OFFSET + seq_idx - BASE_SEQ_LEN
        state_seq = STATE_OFFSET + seq_idx - BASE_SEQ_LEN - ACTION_LEN

        base_freq_off = seq_idx * FREQ_STRIDE_S + pair_idx * FREQ_STRIDE_D
        action_freq_off = action_seq * ACTION_STRIDE_S + pair_idx * ACTION_STRIDE_D
        state_freq_off = state_seq * STATE_STRIDE_S + pair_idx * STATE_STRIDE_D

        if USE_FP64:
            freq_r = tl.load(FREQ_R + base_freq_off, mask=mask & is_base, other=1.0).to(tl.float64)
            freq_i = tl.load(FREQ_I + base_freq_off, mask=mask & is_base, other=0.0).to(tl.float64)
            action_r = tl.load(ACTION_R + action_freq_off, mask=mask & is_action, other=1.0).to(tl.float64)
            action_i = tl.load(ACTION_I + action_freq_off, mask=mask & is_action, other=0.0).to(tl.float64)
            state_r = tl.load(STATE_R + state_freq_off, mask=mask & is_state, other=1.0).to(tl.float64)
            state_i = tl.load(STATE_I + state_freq_off, mask=mask & is_state, other=0.0).to(tl.float64)
            x_even = tl.load(X + x_even_off, mask=mask, other=0.0).to(tl.float64)
            x_odd = tl.load(X + x_odd_off, mask=mask, other=0.0).to(tl.float64)
        else:
            freq_r = tl.load(FREQ_R + base_freq_off, mask=mask & is_base, other=1.0).to(tl.float32)
            freq_i = tl.load(FREQ_I + base_freq_off, mask=mask & is_base, other=0.0).to(tl.float32)
            action_r = tl.load(ACTION_R + action_freq_off, mask=mask & is_action, other=1.0).to(tl.float32)
            action_i = tl.load(ACTION_I + action_freq_off, mask=mask & is_action, other=0.0).to(tl.float32)
            state_r = tl.load(STATE_R + state_freq_off, mask=mask & is_state, other=1.0).to(tl.float32)
            state_i = tl.load(STATE_I + state_freq_off, mask=mask & is_state, other=0.0).to(tl.float32)
            x_even = tl.load(X + x_even_off, mask=mask, other=0.0).to(tl.float32)
            x_odd = tl.load(X + x_odd_off, mask=mask, other=0.0).to(tl.float32)
        freq_r = tl.where(is_action, action_r, tl.where(is_state, state_r, freq_r))
        freq_i = tl.where(is_action, action_i, tl.where(is_state, state_i, freq_i))

        if BACKWARD:
            y_even = x_even * freq_r + x_odd * freq_i
            y_odd = -x_even * freq_i + x_odd * freq_r
        else:
            y_even = x_even * freq_r - x_odd * freq_i
            y_odd = x_even * freq_i + x_odd * freq_r

        tl.store(Y + y_even_off, y_even, mask=mask)
        tl.store(Y + y_odd_off, y_odd, mask=mask)


class _DreamZeroFusedRoPE(torch.autograd.Function):
    """Autograd wrapper around the fused Triton RoPE kernel."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        freqs_real: torch.Tensor,
        freqs_imag: torch.Tensor,
        action_real: torch.Tensor,
        action_imag: torch.Tensor,
        state_real: torch.Tensor,
        state_imag: torch.Tensor,
        action_len: int,
        state_len: int,
        action_offset: int,
        state_offset: int,
        use_fp64: bool,
    ) -> torch.Tensor:
        """Apply the fused RoPE kernel and save inputs needed by backward."""
        b, seq_len, num_heads, dim = x.shape
        half_dim = dim // 2
        y = torch.empty_like(x)
        total_pairs = b * seq_len * num_heads * half_dim
        grid = (triton.cdiv(total_pairs, 256),)
        _dreamzero_fused_rope_kernel[grid](
            x,
            y,
            freqs_real,
            freqs_imag,
            action_real,
            action_imag,
            state_real,
            state_imag,
            total_pairs,
            seq_len,
            num_heads,
            half_dim,
            freqs_real.shape[0],
            action_len,
            state_len,
            action_offset,
            state_offset,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            y.stride(3),
            freqs_real.stride(0),
            freqs_real.stride(-1),
            action_real.stride(0),
            action_real.stride(-1),
            state_real.stride(0),
            state_real.stride(-1),
            False,
            use_fp64,
            BLOCK_SIZE=256,
        )
        ctx.save_for_backward(freqs_real, freqs_imag, action_real, action_imag, state_real, state_imag)
        ctx.shape = x.shape
        ctx.action_len = action_len
        ctx.state_len = state_len
        ctx.action_offset = action_offset
        ctx.state_offset = state_offset
        ctx.use_fp64 = use_fp64
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """Backward pass for the fused rotary embedding kernel."""
        freqs_real, freqs_imag, action_real, action_imag, state_real, state_imag = ctx.saved_tensors
        b, seq_len, num_heads, dim = ctx.shape
        half_dim = dim // 2
        grad_x = torch.empty_like(grad_out)
        total_pairs = b * seq_len * num_heads * half_dim
        grid = (triton.cdiv(total_pairs, 256),)
        _dreamzero_fused_rope_kernel[grid](
            grad_out,
            grad_x,
            freqs_real,
            freqs_imag,
            action_real,
            action_imag,
            state_real,
            state_imag,
            total_pairs,
            seq_len,
            num_heads,
            half_dim,
            freqs_real.shape[0],
            ctx.action_len,
            ctx.state_len,
            ctx.action_offset,
            ctx.state_offset,
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            grad_out.stride(3),
            grad_x.stride(0),
            grad_x.stride(1),
            grad_x.stride(2),
            grad_x.stride(3),
            freqs_real.stride(0),
            freqs_real.stride(-1),
            action_real.stride(0),
            action_real.stride(-1),
            state_real.stride(0),
            state_real.stride(-1),
            True,
            ctx.use_fp64,
            BLOCK_SIZE=256,
        )
        return grad_x, None, None, None, None, None, None, None, None, None, None, None


def _as_fused_freq_views(freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return (real, imag) views of a complex freqs tensor, or None if unsupported."""
    if triton is None or not freqs.is_complex():
        return None
    real = freqs.real
    imag = freqs.imag
    if real.dim() == 2:
        real = real.unsqueeze(1)
        imag = imag.unsqueeze(1)
    if real.dim() != 3:
        return None
    return real, imag


def _rope_apply_polar_fused(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor | None = None,
    freqs_state: torch.Tensor | None = None,
    action_len: int = 0,
    state_len: int = 0,
    action_offset: int = 0,
    state_offset: int = 0,
    enabled: bool = False,
    use_fp64: bool = False,
) -> torch.Tensor | None:
    """Attempt the fused Triton RoPE path; return None to fall back to the polar path."""
    if (
        triton is None
        or not enabled
        or not x.is_cuda
        or x.shape[-1] % 2 != 0
    ):
        return None
    base_views = _as_fused_freq_views(freqs)
    if base_views is None:
        return None
    freqs_real, freqs_imag = base_views
    action_views = _as_fused_freq_views(freqs_action) if freqs_action is not None else None
    state_views = _as_fused_freq_views(freqs_state) if freqs_state is not None else None
    if action_len > 0:
        if action_views is None or state_views is None:
            return None
        action_real, action_imag = action_views
        state_real, state_imag = state_views
    else:
        action_real, action_imag = freqs_real, freqs_imag
        state_real, state_imag = freqs_real, freqs_imag
    if x.shape[-1] // 2 > freqs_real.shape[-1]:
        return None
    return _DreamZeroFusedRoPE.apply(
        x,
        freqs_real,
        freqs_imag,
        action_real,
        action_imag,
        state_real,
        state_imag,
        int(action_len),
        int(state_len),
        int(action_offset),
        int(state_offset),
        bool(use_fp64),
    )


# ----------------------------------------------------------------------
# 1D sinusoidal embedding
# ----------------------------------------------------------------------
def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Compute a 1D sinusoidal positional embedding for ``position``."""
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position,
        torch.pow(
            10000,
            -torch.arange(half, dtype=position.dtype, device=position.device).div(half),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# ----------------------------------------------------------------------
# RoPE frequency tables.
# ----------------------------------------------------------------------
def rope_params(max_seq_len, dim, theta=10000):
    """Compute RoPE frequency table (alias of ``rope_params_polar``)."""
    return rope_params_polar(max_seq_len, dim, theta)


def rope_params_polar(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    """Compute the polar-form RoPE frequency table."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0
        / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# ----------------------------------------------------------------------
# Plain RoPE apply (1D, no action/state register)
# ----------------------------------------------------------------------
def rope_apply(
    x,
    grid_sizes,
    freqs,
    *,
    fused_rope: bool = False,
    fused_rope_fp64: bool = False,
):
    """Apply RoPE to ``x`` (fused Triton path when enabled, else polar fallback)."""
    fused = _rope_apply_polar_fused(
        x,
        freqs,
        enabled=fused_rope,
        use_fp64=fused_rope_fp64,
    )
    if fused is not None:
        return fused
    return rope_apply_polar(x, freqs)


def rope_apply_polar(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply plain RoPE (1D, no action/state register) via complex multiplication."""
    B, seq_len, n, _ = x.shape

    # precompute multipliers
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )

    # apply rotary embedding
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)
    return x


# ----------------------------------------------------------------------
# RoPE apply with action/state register tokens (1D, non-causal)
# ----------------------------------------------------------------------
def rope_action_apply(
    x,
    freqs,
    freqs_action,
    freqs_state,
    action_register_length,
    num_action_per_block=32,
    num_state_per_block=1,
    freqs_action_state: torch.Tensor | None = None,
    fused_rope: bool = False,
    fused_rope_fp64: bool = False,
):
    """Apply RoPE with action/state register tokens (1D, non-causal)."""
    if freqs_action_state is not None and action_register_length is not None:
        freqs = freqs_action_state
        action_register_length = None
    action_len = 0
    state_len = 0
    if action_register_length is not None:
        assert num_action_per_block is not None
        assert num_state_per_block is not None
        chunk_size = action_register_length // (
            num_action_per_block + num_state_per_block
        )
        action_len = chunk_size * num_action_per_block
        state_len = chunk_size * num_state_per_block
    fused = _rope_apply_polar_fused(
        x,
        freqs,
        freqs_action,
        freqs_state,
        action_len=action_len,
        state_len=state_len,
        enabled=fused_rope,
        use_fp64=fused_rope_fp64,
    )
    if fused is not None:
        return fused
    return rope_action_apply_polar(
        x,
        freqs,
        freqs_action,
        freqs_state,
        action_register_length,
        num_action_per_block,
        num_state_per_block,
    )


def rope_action_apply_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int | None = None,
    num_state_per_block: int | None = None,
) -> torch.Tensor:
    """Apply RoPE with action/state register tokens via complex multiplication."""
    B, seq_len, n, _ = x.shape

    # precompute multipliers
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )

    if action_register_length is not None:
        assert num_action_per_block is not None
        assert num_state_per_block is not None

        chunk_size = action_register_length // (
            num_action_per_block + num_state_per_block
        )

        freqs_1d_action = _reshape_register_freqs_like(
            freqs_action[: chunk_size * num_action_per_block],
            freqs,
        )
        freqs_1d_state = _reshape_register_freqs_like(
            freqs_state[: chunk_size * num_state_per_block],
            freqs,
        )
        freqs = torch.cat([freqs, freqs_1d_action, freqs_1d_state], dim=0)

    # apply rotary embedding
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)
    return x


# ----------------------------------------------------------------------
# Causal 3D RoPE apply (per-chunk action/state register slicing)
# Used inside CausalWanSelfAttention; differs from rope_action_apply_*
# in that it indexes a *single chunk* of action/state freqs per call,
# enabling chunked causal attention over interleaved video+action+state
# tokens.
# ----------------------------------------------------------------------
def causal_rope_action_apply(
    x,
    freqs,
    freqs_action,
    freqs_state,
    action_register_length,
    num_action_per_block,
    num_state_per_block,
    action_state_index,
    *,
    fused_rope: bool = False,
    fused_rope_fp64: bool = False,
):
    """Apply causal 3D RoPE, indexing a single chunk of action/state freqs."""
    action_len = 0
    state_len = 0
    action_offset = 0
    state_offset = 0
    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)
        action_len = num_action_per_block
        state_len = num_state_per_block
        action_offset = action_state_index * num_action_per_block
        state_offset = action_state_index * num_state_per_block
    fused = _rope_apply_polar_fused(
        x,
        freqs,
        freqs_action,
        freqs_state,
        action_len=action_len,
        state_len=state_len,
        action_offset=action_offset,
        state_offset=state_offset,
        enabled=fused_rope,
        use_fp64=fused_rope_fp64,
    )
    if fused is not None:
        return fused
    return causal_rope_action_apply_polar(
        x,
        freqs,
        freqs_action,
        freqs_state,
        action_register_length,
        num_action_per_block,
        num_state_per_block,
        action_state_index,
    )


def causal_rope_action_apply_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
):
    """Apply causal 3D RoPE via complex multiplication, indexing one chunk of registers."""
    B, seq_len, n, _ = x.shape

    # precompute multipliers
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )

    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)
        freqs_action = freqs_action[
            action_state_index
            * num_action_per_block : (action_state_index + 1)
            * num_action_per_block
        ]
        freqs_state = freqs_state[
            action_state_index
            * num_state_per_block : (action_state_index + 1)
            * num_state_per_block
        ]
        freqs_1d = torch.cat([freqs_action, freqs_state], dim=0).view(
            action_register_length, 1, -1
        )
        freqs = torch.cat([freqs, freqs_1d], dim=0)

    # apply rotary embedding
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)

    return x


__all__ = [
    "sinusoidal_embedding_1d",
    "rope_params",
    "rope_params_polar",
    "rope_apply",
    "rope_apply_polar",
    "rope_action_apply",
    "rope_action_apply_polar",
    "causal_rope_action_apply",
    "causal_rope_action_apply_polar",
    "_rope_apply_polar_fused",
]
