# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-graph-safe padded FlashAttention helpers for GR00T-N1.7."""

from __future__ import annotations

from collections.abc import Callable

import torch


_FA2_PATCHES_INSTALLED = False
_FA2_GRAPH_BUFFERS: dict[tuple, dict[str, torch.Tensor]] = {}


def _is_cuda_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _is_graph_mode_active() -> bool:
    """Return whether this call should use the padded FA2 implementation.

    Warmup remains the ordinary eager FlashAttention path; only capture needs
    the padded implementation, and replay executes the recorded kernels.
    """
    return _is_cuda_capturing()


def _flash_attention_mask_no_sync(
    batch_size,
    cache_position,
    kv_length,
    kv_offset=0,
    mask_function=None,
    attention_mask=None,
    **kwargs,
):
    """Keep the 2D padding mask without reducing it to a Python boolean."""
    del batch_size, cache_position, kv_offset, mask_function, kwargs
    if attention_mask is not None:
        attention_mask = attention_mask[:, -kv_length:]
    return attention_mask


def _get_fa2_buffers(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    key = (
        batch_size,
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        dtype,
        device,
    )
    buffers = _FA2_GRAPH_BUFFERS.get(key)
    if buffers is not None:
        return buffers
    if _is_cuda_capturing():
        raise RuntimeError(
            "FlashAttention buffers for this sequence length were not initialized "
            f"before CUDA graph capture: requested={key}, "
            f"available={tuple(_FA2_GRAPH_BUFFERS)}."
        )

    total = batch_size * seq_len
    buffers = {
        "seq_ids": torch.arange(batch_size, device=device)
        .unsqueeze(1)
        .expand(-1, seq_len)
        .reshape(-1),
        "pos_in_seq": torch.arange(seq_len, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .reshape(-1),
        "sort_key": torch.zeros(total, dtype=torch.int64, device=device),
        # torch.tensor([total], device=...) performs a host-to-device copy,
        # which is illegal when a new sequence-length variant is first seen
        # during CUDA graph capture.  torch.full initializes it on-device.
        "total": torch.full((1,), total, dtype=torch.int32, device=device),
        "zero": torch.zeros(1, dtype=torch.int32, device=device),
        "zero_q": torch.zeros(1, num_q_heads, head_dim, dtype=dtype, device=device),
        "zero_kv": torch.zeros(1, num_kv_heads, head_dim, dtype=dtype, device=device),
    }
    _FA2_GRAPH_BUFFERS[key] = buffers
    return buffers


def prime_graph_safe_fa2_buffers(
    sequence_lengths: tuple[int, ...],
    device: torch.device,
) -> None:
    """Initialize every distributed sequence length before graph capture."""
    configurations = {
        (batch_size, num_q_heads, num_kv_heads, head_dim, dtype, cached_device)
        for (
            batch_size,
            _seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            dtype,
            cached_device,
        ) in tuple(_FA2_GRAPH_BUFFERS)
        if cached_device == device
    }
    for seq_len in set(sequence_lengths):
        for (
            batch_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            dtype,
            cached_device,
        ) in configurations:
            _get_fa2_buffers(
                batch_size,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                dtype,
                cached_device,
            )


def graph_safe_unpad_and_attend(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    flash_attn_varlen_func: Callable,
    flash_kwargs: dict,
) -> torch.Tensor:
    """Apply padded FA2 using only fixed-shape, graph-capturable operations."""
    batch_size, seq_len, num_kv_heads, head_dim = key_states.shape
    total = batch_size * seq_len
    num_q_heads = query_states.shape[2]
    buffers = _get_fa2_buffers(
        batch_size,
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        query_states.dtype,
        query_states.device,
    )

    flat_mask = attention_mask.reshape(-1).to(dtype=torch.int32)
    sort_key = buffers["sort_key"]
    sort_key.copy_(
        buffers["seq_ids"] * seq_len
        + buffers["pos_in_seq"]
        + (1 - flat_mask).to(dtype=torch.int64) * total
    )
    sorted_indices = torch.argsort(sort_key, stable=True)
    reverse_indices = torch.argsort(sorted_indices)

    sequence_lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
    cumulative_lengths = torch.cumsum(sequence_lengths, dim=0, dtype=torch.int32)

    # The sorted packed buffer contains all valid tokens followed by padding
    # tokens. Represent that padding as B fixed dummy sequences, each bounded
    # by ``seq_len``. A single dummy sequence could be B*S long and would force
    # FA2 to use a different kernel tile (and substantially enlarge graph-pool
    # memory). The extra sequences are never read by valid queries; their
    # outputs are removed by the original-position mask below.
    valid_total = cumulative_lengths[-1:]
    invalid_total = buffers["total"] - valid_total
    dummy_offsets = torch.arange(
        1,
        batch_size + 1,
        device=attention_mask.device,
        dtype=torch.int32,
    ) * seq_len
    remaining = invalid_total - (dummy_offsets - seq_len)
    dummy_lengths = torch.clamp(remaining, min=0, max=seq_len)
    dummy_cumulative = valid_total + torch.cumsum(dummy_lengths, dim=0, dtype=torch.int32)
    cu_seqlens = torch.cat([buffers["zero"], cumulative_lengths, dummy_cumulative])

    packed_query = torch.cat(
        [
            query_states.reshape(total, num_q_heads, head_dim)[sorted_indices],
            buffers["zero_q"],
        ],
        dim=0,
    )
    packed_key = torch.cat(
        [
            key_states.reshape(total, num_kv_heads, head_dim)[sorted_indices],
            buffers["zero_kv"],
        ],
        dim=0,
    )
    packed_value = torch.cat(
        [
            value_states.reshape(total, num_kv_heads, head_dim)[sorted_indices],
            buffers["zero_kv"],
        ],
        dim=0,
    )

    packed_output = flash_attn_varlen_func(
        packed_query,
        packed_key,
        packed_value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        **flash_kwargs,
    )
    if isinstance(packed_output, tuple):
        packed_output = packed_output[0]

    output = packed_output[:total][reverse_indices]
    output = output * flat_mask[:, None, None].to(dtype=output.dtype)
    return output.reshape(batch_size, seq_len, num_q_heads, head_dim)


def _install_graph_safe_fa2_forward() -> None:
    import transformers.integrations.flash_attention as flash_attention_integration
    import transformers.modeling_flash_attention_utils as flash_attention_utils

    original_forward = flash_attention_utils._flash_attention_forward
    if getattr(original_forward, "_loongforge_groot_n1_7_graph_safe", False):
        return

    def graph_safe_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        is_causal,
        dropout=0.0,
        position_ids=None,
        softmax_scale=None,
        sliding_window=None,
        use_top_left_mask=False,
        softcap=None,
        deterministic=None,
        cu_seq_lens_q=None,
        cu_seq_lens_k=None,
        max_length_q=None,
        max_length_k=None,
        target_dtype=None,
        attn_implementation=None,
        **kwargs,
    ):
        if (
            attention_mask is not None
            and query_length == key_states.shape[1]
            and cu_seq_lens_q is None
            and _is_graph_mode_active()
        ):
            (_, flash_varlen_func, _, _), process_flash_kwargs = (
                flash_attention_utils.lazy_import_flash_attention(attn_implementation)
            )
            query_states, key_states, value_states = (
                flash_attention_utils.fa_peft_integration_check(
                    query_states,
                    key_states,
                    value_states,
                    target_dtype,
                )
            )
            flash_kwargs = process_flash_kwargs(
                query_length=query_length,
                key_length=key_states.shape[1],
                is_causal=is_causal,
                dropout=dropout,
                softmax_scale=softmax_scale,
                sliding_window=sliding_window,
                use_top_left_mask=use_top_left_mask,
                softcap=softcap,
                deterministic=deterministic,
                **kwargs,
            )
            return graph_safe_unpad_and_attend(
                query_states,
                key_states,
                value_states,
                attention_mask,
                flash_varlen_func,
                flash_kwargs,
            )

        return original_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal,
            dropout=dropout,
            position_ids=position_ids,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            use_top_left_mask=use_top_left_mask,
            softcap=softcap,
            deterministic=deterministic,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            target_dtype=target_dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )

    graph_safe_forward._loongforge_groot_n1_7_graph_safe = True
    graph_safe_forward._loongforge_original = original_forward
    flash_attention_utils._flash_attention_forward = graph_safe_forward
    flash_attention_integration._flash_attention_forward = graph_safe_forward


def maybe_install_graph_safe_fa2_patches(*, force: bool = False) -> bool:
    """Install padded FA2 compatibility only for local CUDA graph runs."""
    global _FA2_PATCHES_INSTALLED
    if _FA2_PATCHES_INSTALLED or (not force and not _is_graph_mode_active()):
        return False

    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    ALL_MASK_ATTENTION_FUNCTIONS._global_mapping["flash_attention_2"] = (
        _flash_attention_mask_no_sync
    )
    _install_graph_safe_fa2_forward()
    _FA2_PATCHES_INSTALLED = True
    return True
