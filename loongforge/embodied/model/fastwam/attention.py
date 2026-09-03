# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Attention backends and structured masks for FastWAM MoT attention."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ATTENTION_BACKENDS = ("sdpa", "fa2", "fa3", "fa4", "auto")
AUTO_ATTENTION_BACKENDS = ("fa4", "fa3", "fa2", "sdpa")
_FLASH_KERNELS: dict[str, Callable] = {}
_LOGGED_SELECTIONS: set[tuple[str, str]] = set()


def normalize_attention_backend(backend: str) -> str:
    value = str(backend).strip().lower()
    if value not in ATTENTION_BACKENDS:
        raise ValueError(
            f"Unsupported FastWAM attention backend: {backend!r}. "
            f"Expected one of {list(ATTENTION_BACKENDS)}."
        )
    return value


@dataclass(frozen=True)
class AttentionSegment:
    """A contiguous query range and the key ranges visible to its rows."""

    query_start: int
    query_end: int
    key_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class StructuredAttentionMask:
    """Dense reference mask plus a FlashAttention-compatible segmentation."""

    dense: torch.Tensor
    segments: tuple[AttentionSegment, ...]

    @property
    def ndim(self) -> int:
        return self.dense.ndim

    @property
    def shape(self) -> torch.Size:
        return self.dense.shape

    def to(self, *args, **kwargs) -> "StructuredAttentionMask":
        return StructuredAttentionMask(self.dense.to(*args, **kwargs), self.segments)

    def slice(
        self,
        query_start: int,
        query_end: int,
        key_start: int = 0,
        key_end: Optional[int] = None,
    ) -> "StructuredAttentionMask":
        if key_end is None:
            key_end = self.dense.shape[1]
        if not (0 <= query_start <= query_end <= self.dense.shape[0]):
            raise ValueError("Invalid structured attention query slice.")
        if not (0 <= key_start <= key_end <= self.dense.shape[1]):
            raise ValueError("Invalid structured attention key slice.")

        segments = []
        for segment in self.segments:
            start = max(segment.query_start, query_start)
            end = min(segment.query_end, query_end)
            if start >= end:
                continue
            key_ranges = []
            for range_start, range_end in segment.key_ranges:
                clipped_start = max(range_start, key_start)
                clipped_end = min(range_end, key_end)
                if clipped_start < clipped_end:
                    key_ranges.append((clipped_start - key_start, clipped_end - key_start))
            segments.append(
                AttentionSegment(
                    start - query_start,
                    end - query_start,
                    tuple(key_ranges),
                )
            )
        return StructuredAttentionMask(
            self.dense[query_start:query_end, key_start:key_end],
            tuple(segments),
        )


def build_structured_attention_mask(
    query_len: int,
    key_len: int,
    segments: Sequence[AttentionSegment],
    device: torch.device,
) -> StructuredAttentionMask:
    """Validate segments and materialize their dense reference mask."""
    if query_len <= 0 or key_len <= 0:
        raise ValueError(f"Attention lengths must be positive, got query={query_len}, key={key_len}.")

    normalized = tuple(segments)
    dense = torch.zeros((query_len, key_len), dtype=torch.bool, device=device)
    cursor = 0
    for segment in normalized:
        if segment.query_start != cursor or not (segment.query_start < segment.query_end <= query_len):
            raise ValueError("Attention segments must cover query rows once, contiguously, and in order.")
        previous_key_end = 0
        for key_start, key_end in segment.key_ranges:
            if not (previous_key_end <= key_start < key_end <= key_len):
                raise ValueError("Attention key ranges must be ordered, non-overlapping, and in bounds.")
            dense[segment.query_start : segment.query_end, key_start:key_end] = True
            previous_key_end = key_end
        cursor = segment.query_end
    if cursor != query_len:
        raise ValueError("Attention segments must cover every query row.")
    return StructuredAttentionMask(dense=dense, segments=normalized)


def video_attention_segments(
    mode: str,
    sequence_start: int,
    sequence_len: int,
    tokens_per_frame: int,
) -> list[AttentionSegment]:
    """Describe a Wan video mask as query/key ranges at a sequence offset."""
    sequence_end = sequence_start + sequence_len
    first_frame_end = sequence_start + min(tokens_per_frame, sequence_len)
    if mode == "bidirectional":
        return [AttentionSegment(sequence_start, sequence_end, ((sequence_start, sequence_end),))]
    if mode == "first_frame_causal":
        segments = [
            AttentionSegment(sequence_start, first_frame_end, ((sequence_start, first_frame_end),))
        ]
        if first_frame_end < sequence_end:
            segments.append(
                AttentionSegment(first_frame_end, sequence_end, ((sequence_start, sequence_end),))
            )
        return segments
    if mode == "per_frame_causal":
        if sequence_len % tokens_per_frame != 0:
            raise ValueError(
                "Video sequence length must be divisible by tokens_per_frame in per_frame_causal mode."
            )
        return [
            AttentionSegment(
                frame_start,
                frame_start + tokens_per_frame,
                ((sequence_start, frame_start + tokens_per_frame),),
            )
            for frame_start in range(sequence_start, sequence_end, tokens_per_frame)
        ]
    raise ValueError(f"Unsupported video attention mask mode: {mode!r}")


def _load_flash_kernel(backend: str) -> Callable:
    if backend in _FLASH_KERNELS:
        return _FLASH_KERNELS[backend]
    try:
        if backend == "fa2":
            kernel = import_module("flash_attn").flash_attn_func
        elif backend == "fa3":
            kernel = import_module("flash_attn_interface").flash_attn_func
        elif backend == "fa4":
            kernel = import_module("flash_attn.cute").flash_attn_func
        else:
            raise ValueError(f"No external FlashAttention kernel for backend: {backend!r}")
    except (ImportError, AttributeError) as exc:
        package = {"fa2": "flash-attn", "fa3": "flash_attn_interface", "fa4": "flash-attn-4"}[backend]
        raise ImportError(f"attention_backend={backend!r} requires {package}.") from exc
    _FLASH_KERNELS[backend] = kernel
    return kernel


def _resolve_backend(requested: str, q: torch.Tensor) -> str:
    requested = normalize_attention_backend(requested)
    eligible = (
        q.device.type == "cuda"
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] <= 256
    )
    if requested == "sdpa" or not eligible:
        return "sdpa"
    if requested != "auto":
        _load_flash_kernel(requested)
        return requested
    for candidate in AUTO_ATTENTION_BACKENDS[:-1]:
        try:
            _load_flash_kernel(candidate)
            return candidate
        except ImportError:
            continue
    return "sdpa"


def _call_external_flash(backend: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    kernel = _load_flash_kernel(backend)
    if backend == "fa2":
        output = kernel(q, k, v, dropout_p=0.0, causal=False)
    else:
        output = kernel(q, k, v, causal=False)
    return output[0] if isinstance(output, tuple) else output


def _segmented_flash_attention(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: StructuredAttentionMask,
) -> torch.Tensor:
    outputs = []
    for segment in mask.segments:
        q_segment = q[:, segment.query_start : segment.query_end]
        key_parts = [k[:, start:end] for start, end in segment.key_ranges]
        value_parts = [v[:, start:end] for start, end in segment.key_ranges]
        if not key_parts:
            outputs.append(torch.zeros_like(q_segment))
            continue
        k_segment = key_parts[0] if len(key_parts) == 1 else torch.cat(key_parts, dim=1)
        v_segment = value_parts[0] if len(value_parts) == 1 else torch.cat(value_parts, dim=1)
        outputs.append(_call_external_flash(backend, q_segment, k_segment, v_segment))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)


def run_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    attention_mask: Optional[torch.Tensor | StructuredAttentionMask] = None,
    backend: str = "sdpa",
) -> torch.Tensor:
    """Run SDPA or segmented external FlashAttention with identical mask semantics."""
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q/k/v must be [B, S, H*D] tensors.")
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError("q/k/v batch and hidden dimensions must match.")
    if q.shape[2] % num_heads != 0:
        raise ValueError(f"Attention width {q.shape[2]} is not divisible by num_heads={num_heads}.")

    head_dim = q.shape[2] // num_heads
    q_heads = q.reshape(q.shape[0], q.shape[1], num_heads, head_dim)
    k_heads = k.reshape(k.shape[0], k.shape[1], num_heads, head_dim)
    v_heads = v.reshape(v.shape[0], v.shape[1], num_heads, head_dim)
    requested = normalize_attention_backend(backend)
    selected = _resolve_backend(requested, q_heads)
    use_external = selected != "sdpa" and (
        attention_mask is None or isinstance(attention_mask, StructuredAttentionMask)
    )
    execution_backend = selected if use_external else "sdpa"
    log_key = (requested, execution_backend)
    if log_key not in _LOGGED_SELECTIONS:
        _LOGGED_SELECTIONS.add(log_key)
        logger.info("FastWAM attention backend: requested=%s selected=%s", requested, execution_backend)

    if use_external and attention_mask is None:
        output = _call_external_flash(selected, q_heads, k_heads, v_heads)
    elif use_external:
        output = _segmented_flash_attention(selected, q_heads, k_heads, v_heads, attention_mask)
    else:
        dense_mask = attention_mask.dense if isinstance(attention_mask, StructuredAttentionMask) else attention_mask
        if dense_mask is not None:
            dense_mask = dense_mask.to(device=q.device)
            if dense_mask.ndim == 2:
                dense_mask = dense_mask.unsqueeze(0).unsqueeze(0)
            elif dense_mask.ndim == 3:
                dense_mask = dense_mask.unsqueeze(1)
            elif dense_mask.ndim != 4:
                raise ValueError(f"Attention mask must be 2D/3D/4D, got {tuple(dense_mask.shape)}")
        output = F.scaled_dot_product_attention(
            q_heads.transpose(1, 2),
            k_heads.transpose(1, 2),
            v_heads.transpose(1, 2),
            attn_mask=dense_mask,
            dropout_p=0.0,
        ).transpose(1, 2).contiguous()
    return output.reshape(output.shape[0], output.shape[1], -1)
