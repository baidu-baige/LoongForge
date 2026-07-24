# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# DreamZero Wan2.1 attention helpers.
# Backend selection is driven by ``DreamZeroConfig.attention_backend``.

"""DreamZero Wan2.1 attention helpers and backend dispatch utilities."""

import logging
import os
from typing import Iterable, Optional
import warnings

import torch

logger = logging.getLogger(__name__)

try:
    import flash_attn_interface

    def is_hopper_gpu():
        """Return ``True`` when the current CUDA device is a Hopper GPU."""
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name

    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from loongforge.embodied.model.dreamzero.modules.cudnn_attention import DotProductAttention

    TRANSFORMER_ENGINE_AVAILABLE = True
except ModuleNotFoundError:
    TRANSFORMER_ENGINE_AVAILABLE = False


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _allow_dense_flash_attention(
    lq: int,
    lk: int,
    *,
    enabled: bool = False,
    min_q: int = 0,
    policy: str = "legacy_min_q",
) -> bool:
    if not enabled:
        return False

    if lq < min_q:
        return False

    policy = policy.lower().replace("-", "_")
    if policy in {"legacy", "legacy_min_q", "min_q"}:
        return True
    if policy in {"equal", "equal_only", "strict_equal"}:
        return lq == lk
    if policy in {"shape_safe", "long_or_equal", "safe_long_or_equal"}:
        if lq == lk:
            return True
        safe_min_q = 512
        if lq < safe_min_q:
            return False
        max_kq_ratio = 32.0
        if max_kq_ratio > 0.0 and (lk / max(lq, 1)) > max_kq_ratio:
            return False
        return True

    warnings.warn(
        f"Unsupported flash_attention_dense_policy={policy!r}; "
        "disabling dense FlashAttention for this call."
    )
    return False


_FA_UNIFORM_LENS_CACHE: dict[tuple[str, int, int, int], torch.Tensor] = {}
_FA_UNIFORM_CU_SEQLENS_CACHE: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _device_cache_index(device: torch.device) -> int:
    if device.index is not None:
        return device.index
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.current_device()
    return -1


def _get_uniform_lens_and_cu_seqlens(
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache fixed-length varlen FlashAttention metadata.

    DreamZero's common FA2 path passes dense [B, L, H, D] tensors with
    q_lens/k_lens=None, then uses varlen FlashAttention. The sequence lengths
    are therefore uniform and immutable for a given (device, B, L). Reusing
    these int32 tensors avoids repeated tiny CUDA allocations/cumsum launches.

    This is intentionally controlled by an explicit performance option. On
    14B Full, reusing cu_seqlens objects can
    perturb backward/optimizer trajectories even when step-1 forward loss is
    identical, so do not enable it for strict loss alignment.
    """
    key = (device.type, _device_cache_index(device), batch_size, seq_len)
    cached = _FA_UNIFORM_CU_SEQLENS_CACHE.get(key)
    if cached is not None:
        return cached

    lens = torch.tensor([seq_len] * batch_size, dtype=torch.int32, device=device)
    zeros = torch.zeros([1], dtype=torch.int32, device=device)
    cu_seqlens = torch.cat([zeros, lens]).cumsum(0).to(torch.int32)
    _FA_UNIFORM_CU_SEQLENS_CACHE[key] = (lens, cu_seqlens)
    return lens, cu_seqlens


def _get_uniform_lens(
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Cache only the fixed q_lens/k_lens tensors.

    This is more conservative than caching cu_seqlens: FlashAttention still gets
    fresh cumulative-sequence tensors every call, while repeated small int32
    length tensors are reused for the common dense-as-varlen path.
    """
    key = (device.type, _device_cache_index(device), batch_size, seq_len)
    cached = _FA_UNIFORM_LENS_CACHE.get(key)
    if cached is not None:
        return cached

    lens = torch.tensor([seq_len] * batch_size, dtype=torch.int32, device=device)
    _FA_UNIFORM_LENS_CACHE[key] = lens
    return lens


def preseed_uniform_fa_lens_cache(
    batch_size: int,
    seq_lens: Iterable[int],
    device: torch.device,
    *,
    cache_fa_lens: bool = False,
    cache_fa_cu_seqlens: bool = False,
) -> None:
    """Pre-populate FA length metadata cache before torch.compile tracing.

    Dynamo guards on global dict size when a compiled helper indirectly reads
    the cache. Pre-seeding all expected lengths keeps the dict stable during
    later compiled attention calls without changing the tensor math path.
    """
    cache_fa_lens = cache_fa_cu_seqlens or cache_fa_lens
    if not cache_fa_lens:
        return

    seen: set[int] = set()
    for raw_seq_len in seq_lens:
        seq_len = int(raw_seq_len)
        if seq_len <= 0 or seq_len in seen:
            continue
        seen.add(seq_len)
        if cache_fa_cu_seqlens:
            _get_uniform_lens_and_cu_seqlens(batch_size, seq_len, device)
        else:
            _get_uniform_lens(batch_size, seq_len, device)


def _gpu_supports_flash_attention():
    """FlashAttention requires Ampere (compute capability 8.0) or newer."""
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return False
    try:
        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability()
        return cap[0] >= 8
    except Exception:
        return False


def _sdpa_attention_fallback(
    q, k, v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    dtype=torch.bfloat16,
):
    """PyTorch SDPA fallback for GPUs that don't support FlashAttention (e.g. pre-Ampere)."""
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            'Padding mask is disabled when using scaled_dot_product_attention on this GPU. '
            'It can have a slight impact on quality.'
        )
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    if q_scale is not None:
        q = q * q_scale
    if softmax_scale is not None:
        q = q * softmax_scale
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p
    )
    return out.transpose(1, 2).contiguous()


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor] = None,
    k_lens: Optional[torch.Tensor] = None,
    dropout_p: float = 0.,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Optional[tuple[int, int]] = None,
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: Optional[int] = None,
    flash_attention_dense: bool = False,
    flash_attention_dense_min_q: int = 0,
    flash_attention_dense_policy: str = "legacy_min_q",
    cache_fa_lens: bool = False,
    cache_fa_lens_clone: bool = False,
    cache_fa_cu_seqlens: bool = False,
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
    version:        int. 2 for flash attention 2, 3 for flash attention 3.

    Returns:
        x:              [B, Lq, Nq, C2].
    """
    if window_size is None:
        window_size = (-1, -1)
    if version is None:
        version = 3
    deterministic = deterministic or _env_truthy("FLASH_ATTENTION_DETERMINISTIC")

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # Use PyTorch SDPA on pre-Ampere GPUs (FlashAttention requires Ampere or newer)
    if not _gpu_supports_flash_attention():
        return _sdpa_attention_fallback(
            q, k, v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        )

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    cache_fa_lens = cache_fa_cu_seqlens or cache_fa_lens
    clone_cached_fa_lens = cache_fa_lens and cache_fa_lens_clone
    cu_seqlens_q = None
    cu_seqlens_k = None

    if (
        q_lens is None
        and k_lens is None
        and version == 2
        and FLASH_ATTN_2_AVAILABLE
        and _allow_dense_flash_attention(
            lq,
            lk,
            enabled=flash_attention_dense,
            min_q=flash_attention_dense_min_q,
            policy=flash_attention_dense_policy,
        )
    ):
        q = half(q)
        k = half(k)
        v = half(v)
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        if q_scale is not None:
            q = q * q_scale
        return flash_attn.flash_attn_func(
            q=q,
            k=k,
            v=v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).type(out_dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        if cache_fa_cu_seqlens:
            q_lens, cu_seqlens_q = _get_uniform_lens_and_cu_seqlens(b, lq, q.device)
        elif cache_fa_lens:
            q_lens = _get_uniform_lens(b, lq, q.device)
        else:
            q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
        if clone_cached_fa_lens:
            q_lens = q_lens.clone()
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        if cache_fa_cu_seqlens:
            k_lens, cu_seqlens_k = _get_uniform_lens_and_cu_seqlens(b, lk, k.device)
        elif cache_fa_lens:
            k_lens = _get_uniform_lens(b, lk, k.device)
        else:
            k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
        if clone_cached_fa_lens:
            k_lens = k_lens.clone()
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )
    if cu_seqlens_q is None:
        zeros = torch.zeros([1], dtype=torch.int32, device=q.device)
        cu_seqlens_q = torch.cat([zeros, q_lens]).cumsum(0).to(torch.int32)
    if cu_seqlens_k is None:
        zeros = torch.zeros([1], dtype=torch.int32, device=k.device)
        cu_seqlens_k = torch.cat([zeros, k_lens]).cumsum(0).to(torch.int32)

    # apply attention
    if version == 3 and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        raise ValueError(f"Invalid version: {version}")

    # output
    return x.type(out_dtype)


class AttentionModule(torch.nn.Module):
    """Attention module dispatching to the configured attention backend."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout_p: float = 0.,
        softmax_scale: Optional[float] = None,
        q_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Optional[tuple[int, int]] = None,
        deterministic: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        backend: Optional[str] = None,
        flash_attention_dense: bool = False,
        flash_attention_dense_min_q: int = 0,
        flash_attention_dense_policy: str = "legacy_min_q",
        cache_fa_lens: bool = False,
        cache_fa_lens_clone: bool = False,
    ):
        super().__init__()
        if backend is None:
            backend = "FA2"

        # Fall back to FA backend if TE is specified but not available
        if backend == "TE" and not TRANSFORMER_ENGINE_AVAILABLE:
            logger.warning("Transformer Engine is not available. Falling back to FA2 backend.")
            backend = "FA2"

        assert backend in ["torch", "FA2", "FA3", "TE", "torch_onnx"]
        self.backend = backend

        if backend == "torch":
            def _torch_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                out_dtype = q.dtype
                q = q.transpose(1, 2).to(dtype)
                k = k.transpose(1, 2).to(dtype)
                v = v.transpose(1, 2).to(dtype)

                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    is_causal=causal,
                    dropout_p=dropout_p,
                    scale=softmax_scale,
                )

                out = out.transpose(1, 2).contiguous()
                return out.to(out_dtype)
            self.attn_func = _torch_impl

        elif  backend == "torch_onnx":
            def _torch_onnx_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                out_dtype = q.dtype
                # use torch.nn.functional.scaled_dot_product_attention for tensorrt export

                # The input is (s, n, d), but sdpa needs (b, h, s, d).
                # We add a batch dimension and transpose.
                q = q.unsqueeze(0).transpose(1, 2).to(dtype)
                k = k.unsqueeze(0).transpose(1, 2).to(dtype)
                v = v.unsqueeze(0).transpose(1, 2).to(dtype)

                # Fix for ONNX export: repeat k and v to match q's batch size in cross-attention
                if q.shape[0] != k.shape[0] and k.shape[0] == 1:
                    k = k.repeat(q.shape[0], 1, 1, 1)
                    v = v.repeat(q.shape[0], 1, 1, 1)

                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    is_causal=causal,
                    dropout_p=dropout_p,
                    scale=softmax_scale,
                )

                # Transpose back to (b, s, n, d) format.
                out = out.transpose(1, 2).contiguous()
                return out.to(out_dtype)
            self.attn_func = _torch_onnx_impl

        elif backend == "TE" and TRANSFORMER_ENGINE_AVAILABLE:
            te_kwargs = {
                "num_attention_heads": num_heads,
                "kv_channels": head_dim,
                "qkv_format": "bshd",
                "attn_mask_type": "causal" if causal else "no_mask",
                "window_size": window_size,
                "attention_dropout": dropout_p,
                "deterministic": deterministic or _env_truthy("FLASH_ATTENTION_DETERMINISTIC"),
            }
            try:
                self.attn_backend = DotProductAttention(**te_kwargs)
            except TypeError as exc:
                if "deterministic" not in str(exc):
                    raise
                te_kwargs.pop("deterministic", None)
                self.attn_backend = DotProductAttention(**te_kwargs)

            def _te_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                out_dtype = q.dtype
                if q.shape[1] == 1 and k.shape[1] == 1:
                    return v.to(out_dtype)
                return self.attn_backend(
                    query_layer=q.to(dtype),
                    key_layer=k.to(dtype),
                    value_layer=v.to(dtype),
                ).to(out_dtype)
            self.attn_func = _te_impl

        elif backend == "FA2" or backend == "FA3":
            def _flash_attn_impl(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                q_lens: Optional[torch.Tensor], k_lens: Optional[torch.Tensor],
            ) -> torch.Tensor:
                return flash_attention(
                    q=q, k=k, v=v,
                    q_lens=q_lens, k_lens=k_lens,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    q_scale=q_scale,
                    causal=causal,
                    window_size=window_size,
                    deterministic=deterministic,
                    dtype=dtype,
                    version=3 if backend == "FA3" else 2,
                    flash_attention_dense=flash_attention_dense,
                    flash_attention_dense_min_q=flash_attention_dense_min_q,
                    flash_attention_dense_policy=flash_attention_dense_policy,
                    cache_fa_lens=cache_fa_lens,
                    cache_fa_lens_clone=cache_fa_lens_clone,
                )
            self.attn_func = _flash_attn_impl

        else:
            raise ValueError(f"Invalid backend: {backend}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens: Optional[torch.Tensor] = None,
        k_lens: Optional[torch.Tensor] = None,
    ):
        """Run attention with the selected backend over ``q``, ``k`` and ``v``."""
        if (
            self.backend == "torch" or
            self.backend == "torch_onnx" or
            (self.backend == "TE" and TRANSFORMER_ENGINE_AVAILABLE)
        ):
            if q_lens is not None or k_lens is not None:
                warnings.warn(
                    'Padding mask is disabled when using scaled_dot_product_attention. '
                    'It can have a significant impact on performance.'
                )
            return self.attn_func(q, k, v)  # type: ignore[call-arg]
        else:
            return self.attn_func(q, k, v, q_lens, k_lens)  # type: ignore[call-arg]
