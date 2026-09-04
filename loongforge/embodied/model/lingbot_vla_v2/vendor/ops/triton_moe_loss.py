# Copyright 2025 Ant Group Co., Ltd. All Rights Reserved.
# Developer: xiancun
# Project： Lumos VIdeo Generation Foundation Model
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

"""
Triton-optimized MoE auxiliary loss functions.

Provides numerically equivalent replacements for the functions in loss.py,
with two key optimizations:
  1. Eliminate Python for-loops via vectorized segment-wise operations.
  2. Fuse topK + counting into Triton kernels to avoid huge intermediate tensors.

Usage:
    from loongforge.embodied.model.lingbot_vla_v2.vendor.ops.triton_moe_loss import (
        triton_load_balancing_loss_func,
        triton_sequence_wise_balance_loss,
    )
    # Drop-in replacement — same signature and return type as loss.py
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


# ═══════════════════════════════════════════════════════════════════════
# Section 1: Triton availability check + kernel definitions
# ═══════════════════════════════════════════════════════════════════════

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True

    # ── Kernel 1: per-segment topK counting (for sequence_wise_balance_loss) ──
    @triton.jit
    def _topk_segment_count_kernel(
        logits_ptr,         # [N_total, E]
        seg_starts_ptr,     # [S_total]
        seg_lengths_ptr,    # [S_total]
        f_out_ptr,          # [S_total, E]
        stride_logits_n,    # stride of logits along token dim
        E: tl.constexpr,    # actual num_experts
        K: tl.constexpr,    # top_k
        BLOCK_E: tl.constexpr,  # next power of 2 >= E
    ):
        """Each program computes f_i (expert counts) for one segment.

        Iterates over tokens in the segment, performs K rounds of argmax to
        find top-K experts, and accumulates per-expert hit counts.
        No gradient needed — f_i is always detached in the loss.
        """
        seg_id = tl.program_id(0)
        seg_start = tl.load(seg_starts_ptr + seg_id)
        seg_len = tl.load(seg_lengths_ptr + seg_id)

        expert_offs = tl.arange(0, BLOCK_E)  # [BLOCK_E]
        mask_e = expert_offs < E
        f_acc = tl.zeros((BLOCK_E,), dtype=tl.float32)

        for t in range(0, seg_len):
            row_ptr = logits_ptr + (seg_start + t) * stride_logits_n
            logits_row = tl.load(row_ptr + expert_offs, mask=mask_e, other=float('-inf'))

            # K rounds of argmax to find top-K indices
            row_copy = logits_row
            for _k in range(K):
                max_val = tl.max(row_copy, axis=0)
                is_max = (row_copy == max_val)
                # Distribute count evenly among ties (rare with float32)
                n_ties = tl.sum(is_max.to(tl.float32), axis=0)
                f_acc += tl.where(is_max, 1.0 / n_ties, 0.0)
                row_copy = tl.where(is_max, float('-inf'), row_copy)

        # Write f_count (unnormalized) — caller normalizes by (E / K) / seg_len
        out_ptr = f_out_ptr + seg_id * E
        tl.store(out_ptr + expert_offs, f_acc, mask=mask_e)

    # ── Kernel 2: blocked topK counting (for load_balancing_loss_func) ──
    @triton.jit
    def _topk_count_with_mask_kernel(
        routing_weights_ptr,  # [N, E] — softmax probabilities
        mask_ptr,             # [N] — 1.0 for valid, 0.0 for padding
        partial_f_ptr,        # [num_blocks, E] — partial expert counts
        partial_p_ptr,        # [num_blocks, E] — partial masked prob sums
        N,
        stride_rw_n,          # stride along token dim
        has_mask: tl.constexpr,
        E: tl.constexpr,
        K: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Each program accumulates topK counts + masked probs for a token block.

        Two-phase reduction: writes partial [E] results per block;
        caller sums across blocks in PyTorch.
        """
        pid = tl.program_id(0)
        n_start = pid * BLOCK_N

        expert_offs = tl.arange(0, BLOCK_E)
        mask_e = expert_offs < E
        f_local = tl.zeros((BLOCK_E,), dtype=tl.float32)
        p_local = tl.zeros((BLOCK_E,), dtype=tl.float32)

        for t_offset in range(BLOCK_N):
            t = n_start + t_offset
            # Guard: skip if t >= N (handles last block)
            if t < N:
                row_ptr = routing_weights_ptr + t * stride_rw_n
                rw_row = tl.load(row_ptr + expert_offs, mask=mask_e, other=0.0)

                if has_mask:
                    m = tl.load(mask_ptr + t)
                else:
                    m = 1.0

                # Accumulate masked probs
                p_local += rw_row * m

                # TopK counting
                row_copy = rw_row
                for _k in range(K):
                    max_val = tl.max(row_copy, axis=0)
                    is_max = (row_copy == max_val)
                    n_ties = tl.sum(is_max.to(tl.float32), axis=0)
                    f_local += tl.where(is_max, m / n_ties, 0.0)
                    row_copy = tl.where(is_max, float('-inf'), row_copy)

        # Write partial results (only E valid elements)
        f_ptr = partial_f_ptr + pid * E
        p_ptr = partial_p_ptr + pid * E
        tl.store(f_ptr + expert_offs, f_local, mask=mask_e)
        tl.store(p_ptr + expert_offs, p_local, mask=mask_e)

except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Vectorized PyTorch fallback (no Triton needed)
# ═══════════════════════════════════════════════════════════════════════

def _build_segment_info(
    seq_lengths_per_layer: List[int],
    num_layers: int,
    N_per_layer: int,
    device: torch.device,
):
    """Build segment IDs and metadata for all layers combined.

    Returns:
        segment_ids:    [num_layers * N_valid_per_layer] int64
        seg_starts:     [total_segments] int64
        seg_lengths:    [total_segments] int64
        total_segments: int
    """
    S = len(seq_lengths_per_layer)
    total_segments = num_layers * S
    seg_lengths_t = torch.tensor(seq_lengths_per_layer, dtype=torch.int64, device=device)

    # Repeat for all layers
    all_seg_lengths = seg_lengths_t.repeat(num_layers)  # [total_segments]

    # Per-layer segment starts: cumsum of seq_lengths
    per_layer_starts = torch.cumsum(seg_lengths_t, 0) - seg_lengths_t  # [S]
    # Layer offsets in the concatenated tensor
    layer_offsets = torch.arange(num_layers, device=device, dtype=torch.int64) * N_per_layer
    # [L, S] → [L*S]
    all_seg_starts = (per_layer_starts.unsqueeze(0) + layer_offsets.unsqueeze(1)).reshape(-1)

    # Segment IDs: [num_layers * N_valid]
    segment_ids = torch.repeat_interleave(
        torch.arange(total_segments, device=device), all_seg_lengths,
    )

    return segment_ids, all_seg_starts, all_seg_lengths, total_segments


def _vectorized_segment_f_i(
    logits: torch.Tensor,         # [N_total, E]
    seg_starts: torch.Tensor,     # [S_total]
    seg_lengths: torch.Tensor,    # [S_total]
    top_k: int,
) -> torch.Tensor:
    """Compute f_i per segment without Triton, using vectorized PyTorch ops.

    Returns: f_i [S_total, E]
    """
    N, E = logits.shape
    S_total = seg_starts.shape[0]

    # TopK over all tokens at once
    _, topk_idx = torch.topk(logits, k=top_k, dim=-1)  # [N, K]

    # Build one-hot mask efficiently: scatter into [N, E]
    mask = torch.zeros(N, E, device=logits.device, dtype=torch.float32)
    mask.scatter_(1, topk_idx, 1.0)

    # Segment-wise sum using scatter_add
    segment_ids = torch.repeat_interleave(
        torch.arange(S_total, device=logits.device), seg_lengths,
    )
    seg_ids_exp = segment_ids.unsqueeze(1).expand(-1, E)  # [N, E]

    f_sum = torch.zeros(S_total, E, device=logits.device, dtype=torch.float32)
    f_sum.scatter_add_(0, seg_ids_exp, mask)

    # Normalize: f_i = (E / K) * f_sum / T_s
    inv_lens = (float(E) / top_k) / seg_lengths.unsqueeze(1).float().clamp(min=1)
    f_i = f_sum * inv_lens

    return f_i


def _vectorized_topk_count(
    routing_weights: torch.Tensor,  # [N, E]
    top_k: int,
    flat_mask: Optional[torch.Tensor] = None,  # [N] float
) -> torch.Tensor:
    """Count per-expert topK selections, optionally masked. Returns [E]."""
    N, E = routing_weights.shape
    _, topk_idx = torch.topk(routing_weights, k=top_k, dim=-1)  # [N, K]

    tokens_per_expert = torch.zeros(E, device=routing_weights.device, dtype=torch.float32)
    weight = flat_mask if flat_mask is not None else torch.ones(N, device=routing_weights.device, dtype=torch.float32)

    # K rounds of scatter_add — K is small (typically 2-8), no Python overhead concern
    for k in range(top_k):
        tokens_per_expert.scatter_add_(0, topk_idx[:, k], weight)

    return tokens_per_expert


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Triton-accelerated wrappers
# ═══════════════════════════════════════════════════════════════════════

def _triton_segment_f_i(
    logits: torch.Tensor,         # [N_total, E]
    seg_starts: torch.Tensor,     # [S_total]
    seg_lengths: torch.Tensor,    # [S_total]
    top_k: int,
) -> torch.Tensor:
    """Compute f_i per segment using the Triton kernel. Returns [S_total, E]."""
    N, E = logits.shape
    S_total = seg_starts.shape[0]
    BLOCK_E = _next_power_of_2(E)

    f_counts = torch.zeros(S_total, E, device=logits.device, dtype=torch.float32)

    _topk_segment_count_kernel[(S_total,)](
        logits,
        seg_starts,
        seg_lengths,
        f_counts,
        logits.stride(0),
        E=E,
        K=top_k,
        BLOCK_E=BLOCK_E,
    )

    # Normalize: f_i = (E / K) * counts / T_s
    inv_lens = (float(E) / top_k) / seg_lengths.unsqueeze(1).float().clamp(min=1)
    return f_counts * inv_lens


def _triton_topk_count(
    routing_weights: torch.Tensor,  # [N, E]
    top_k: int,
    flat_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute tokens_per_expert [E] using the Triton kernel."""
    N, E = routing_weights.shape
    BLOCK_E = _next_power_of_2(E)
    BLOCK_N = 256
    num_blocks = (N + BLOCK_N - 1) // BLOCK_N

    partial_f = torch.zeros(num_blocks, E, device=routing_weights.device, dtype=torch.float32)
    partial_p = torch.zeros(num_blocks, E, device=routing_weights.device, dtype=torch.float32)

    has_mask = flat_mask is not None
    _topk_count_with_mask_kernel[(num_blocks,)](
        routing_weights,
        flat_mask if has_mask else routing_weights,  # dummy ptr when no mask
        partial_f,
        partial_p,
        N,
        routing_weights.stride(0),
        has_mask=has_mask,
        E=E,
        K=top_k,
        BLOCK_E=BLOCK_E,
        BLOCK_N=BLOCK_N,
    )

    # Phase 2: reduce across blocks
    tokens_per_expert = partial_f.sum(dim=0)  # [E]

    if flat_mask is not None:
        n_valid = flat_mask.sum().clamp(min=1)
    else:
        n_valid = float(N)
    tokens_per_expert = tokens_per_expert / n_valid

    return tokens_per_expert


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Main API — drop-in replacements for loss.py
# ═══════════════════════════════════════════════════════════════════════

def triton_sequence_wise_balance_loss(
    router_logits_list: tuple,
    top_k: int,
    seq_lengths: Optional[List[int]] = None,
    padding_len: int = 0,
    score_func: str = "softmax",
) -> List[torch.Tensor]:
    """Triton-optimized DeepSeek-V3 sequence-wise balance loss.

    Numerically equivalent to sequence_wise_balance_loss() in loss.py,
    but eliminates all Python for-loops by:
      - Processing all layers simultaneously via concatenation
      - Using segment-wise parallel reduction (scatter_add) instead of per-sequence loops
      - Fusing topK + counting in a Triton kernel (with PyTorch vectorized fallback)

    Args / Returns: same as sequence_wise_balance_loss in loss.py.
    """
    if router_logits_list is None or not isinstance(router_logits_list, (tuple, list)):
        return []

    valid_logits = [rl for rl in router_logits_list if rl is not None]
    if len(valid_logits) == 0:
        return []

    num_layers = len(valid_logits)
    device = valid_logits[0].device
    E = valid_logits[0].shape[1]

    # ── Step 1: Concatenate all layers, remove padding ──
    all_logits_list = []
    N_per_layer = None
    for logits in valid_logits:
        logits_f32 = logits.to(dtype=torch.float32)
        N = logits_f32.shape[0]
        if padding_len > 0:
            logits_f32 = logits_f32[:N - padding_len]
        all_logits_list.append(logits_f32)
        if N_per_layer is None:
            N_per_layer = logits_f32.shape[0]

    # Check if all layers have the same valid length (common case)
    same_length = all(l.shape[0] == N_per_layer for l in all_logits_list)

    if not same_length:
        # Rare: different MoE layers have different token counts
        return _fallback_per_layer(valid_logits, top_k, seq_lengths, padding_len, score_func)

    if seq_lengths is None or len(seq_lengths) == 0:
        seq_lengths_effective = [N_per_layer]
    else:
        seq_lengths_effective = seq_lengths

    S = len(seq_lengths_effective)
    all_logits = torch.cat(all_logits_list, dim=0)  # [L * N_valid, E]

    # ── Step 2: Build segment metadata ──
    segment_ids, seg_starts, seg_lengths_t, total_segments = _build_segment_info(
        seq_lengths_effective, num_layers, N_per_layer, device
    )

    # ── Step 3: P_i via PyTorch (gradient path) ──
    if score_func == "sigmoid":
        all_scores = all_logits.sigmoid()
        all_probs = all_scores / all_scores.sum(dim=-1, keepdim=True)
    else:
        all_probs = F.softmax(all_logits, dim=-1)  # [L * N_valid, E]
    seg_ids_exp = segment_ids.unsqueeze(1).expand(-1, E)  # [L * N_valid, E]

    P_sum = torch.zeros(total_segments, E, device=device, dtype=torch.float32)
    P_sum.scatter_add_(0, seg_ids_exp, all_probs)
    P_i = P_sum / seg_lengths_t.unsqueeze(1).float().clamp(min=1)  # [total_segments, E]

    # ── Step 4: f_i (no gradient needed) ──
    with torch.no_grad():
        if _HAS_TRITON and all_logits.is_cuda:
            f_i = _triton_segment_f_i(all_logits, seg_starts, seg_lengths_t, top_k)
        else:
            f_i = _vectorized_segment_f_i(all_logits, seg_starts, seg_lengths_t, top_k)

    # ── Step 5: Per-segment loss → per-layer mean ──
    loss_per_seg = (f_i * P_i).sum(dim=-1)  # [total_segments]
    loss_per_seg = loss_per_seg.reshape(num_layers, S)
    layer_losses = loss_per_seg.mean(dim=1)  # [L]

    return list(layer_losses.unbind(0))


def triton_load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    """Triton-optimized Switch Transformer load balancing loss.

    Numerically equivalent to load_balancing_loss_func() in loss.py,
    but avoids the huge [N, K, E] one_hot intermediate tensor by
    directly counting expert assignments via Triton or scatter_add.

    Memory reduction: O(N*K*E) → O(N*E + num_blocks*E)

    Args / Returns: same as load_balancing_loss_func in loss.py.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    gate_logits = tuple(g for g in gate_logits if g is not None)
    if len(gate_logits) == 0:
        return 0

    compute_device = gate_logits[0].device
    concatenated = torch.cat(
        [g.to(device=compute_device, dtype=torch.float32) for g in gate_logits],
        dim=0,
    )  # [L*N, E]

    # ── Step 1: softmax (gradient path) ──
    routing_weights = F.softmax(concatenated, dim=-1)  # [L*N, E]

    # ── Step 2: Build flat mask ──
    N_total = routing_weights.shape[0]
    if attention_mask is not None:
        batch_size, seq_len = attention_mask.shape
        num_layers = N_total // (batch_size * seq_len)
        flat_mask = (
            attention_mask
            .unsqueeze(0)
            .expand(num_layers, -1, -1)
            .reshape(-1)
            .to(device=compute_device, dtype=torch.float32)
        )
    else:
        flat_mask = None

    # ── Step 3: tokens_per_expert (no gradient) ──
    with torch.no_grad():
        if _HAS_TRITON and routing_weights.is_cuda:
            tokens_per_expert = _triton_topk_count(routing_weights, top_k, flat_mask)
        else:
            tokens_per_expert = _vectorized_topk_count(routing_weights, top_k, flat_mask)
            if flat_mask is not None:
                tokens_per_expert = tokens_per_expert / flat_mask.sum().clamp(min=1)
            else:
                tokens_per_expert = tokens_per_expert / float(N_total)

    # ── Step 4: router_prob_per_expert (gradient path) ──
    if flat_mask is not None:
        n_valid = flat_mask.sum().clamp(min=1)
        router_prob_per_expert = (routing_weights * flat_mask.unsqueeze(1)).sum(0) / n_valid
    else:
        router_prob_per_expert = routing_weights.mean(dim=0)

    # ── Step 5: loss ──
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert)
    return overall_loss * num_experts


# ═══════════════════════════════════════════════════════════════════════
# Section 5: Fallback for edge cases
# ═══════════════════════════════════════════════════════════════════════

def _fallback_per_layer(
    valid_logits: List[torch.Tensor],
    top_k: int,
    seq_lengths: Optional[List[int]],
    padding_len: int,
    score_func: str = "softmax",
) -> List[torch.Tensor]:
    """Fallback when layers have different valid token counts.

    Still vectorized within each layer (no per-sequence for-loop).
    """
    layer_loss_list = []
    for logits in valid_logits:
        logits = logits.to(dtype=torch.float32)
        N, E = logits.shape
        if padding_len > 0:
            logits = logits[:N - padding_len]
        if logits.shape[0] == 0:
            continue

        if seq_lengths is not None and len(seq_lengths) > 0:
            S = len(seq_lengths)
            device = logits.device
            seg_lengths_t = torch.tensor(seq_lengths, dtype=torch.int64, device=device)
            seg_starts = torch.cumsum(seg_lengths_t, 0) - seg_lengths_t

            # P_i (gradient path)
            if score_func == "sigmoid":
                scores = logits.sigmoid()
                probs = scores / scores.sum(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
            segment_ids = torch.repeat_interleave(torch.arange(S, device=device), seg_lengths_t)
            seg_ids_exp = segment_ids.unsqueeze(1).expand(-1, E)
            P_sum = torch.zeros(S, E, device=device, dtype=torch.float32)
            P_sum.scatter_add_(0, seg_ids_exp, probs)
            P_i = P_sum / seg_lengths_t.unsqueeze(1).float().clamp(min=1)

            # f_i (no gradient)
            with torch.no_grad():
                if _HAS_TRITON and logits.is_cuda:
                    f_i = _triton_segment_f_i(logits, seg_starts, seg_lengths_t, top_k)
                else:
                    f_i = _vectorized_segment_f_i(logits, seg_starts, seg_lengths_t, top_k)

            loss_per_seq = (f_i * P_i).sum(dim=-1)
            layer_loss_list.append(loss_per_seq.mean())
        else:
            if score_func == "sigmoid":
                scores = logits.sigmoid()
                probs = scores / scores.sum(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
            P_i = probs.mean(dim=0)

            with torch.no_grad():
                _, topk_idx = torch.topk(logits, k=top_k, dim=-1)
                mask = torch.zeros_like(logits)
                mask.scatter_(1, topk_idx, 1.0)
                f_i = (E / top_k) * mask.mean(dim=0)

            layer_loss_list.append(torch.sum(f_i * P_i))

    return layer_loss_list


# ═══════════════════════════════════════════════════════════════════════
# Section 6: Numerical alignment test
# ═══════════════════════════════════════════════════════════════════════

def _test_alignment(
    num_tokens: int = 2048,
    num_experts: int = 128,
    top_k: int = 8,
    num_layers: int = 4,
    num_seqs: int = 8,
    device: str = "cuda",
):
    """Verify that triton_loss functions match the original loss.py numerically.

    Run:
        python -c (
            "from loongforge.embodied.model.lingbot_vla_v2.vendor.ops."
            "triton_moe_loss import _test_alignment; _test_alignment()"
        )
    """
    from loongforge.embodied.model.lingbot_vla_v2.vendor.ops.moe_loss import (
        load_balancing_loss_func,
        sequence_wise_balance_loss,
    )

    torch.manual_seed(42)

    # Generate random per-sequence lengths
    tokens_per_seq = num_tokens // num_seqs
    seq_lengths = [tokens_per_seq] * (num_seqs - 1)
    seq_lengths.append(num_tokens - sum(seq_lengths))

    padding_len = 64

    router_logits_list = tuple(
        torch.randn(num_tokens + padding_len, num_experts, device=device, dtype=torch.float32)
        for _ in range(num_layers)
    )

    # ── Test 1: sequence_wise_balance_loss ──
    print("Testing sequence_wise_balance_loss alignment...")

    orig_losses = sequence_wise_balance_loss(
        router_logits_list, top_k, seq_lengths=seq_lengths, padding_len=padding_len,
    )
    triton_losses = triton_sequence_wise_balance_loss(
        router_logits_list, top_k, seq_lengths=seq_lengths, padding_len=padding_len,
    )

    assert len(orig_losses) == len(triton_losses), (
        f"Length mismatch: {len(orig_losses)} vs {len(triton_losses)}"
    )
    max_diff = 0.0
    for i, (o, t) in enumerate(zip(orig_losses, triton_losses)):
        diff = (o - t).abs().item()
        max_diff = max(max_diff, diff)
        print(f"  Layer {i}: orig={o.item():.6f}, triton={t.item():.6f}, diff={diff:.2e}")
    assert max_diff < 1e-4, f"sequence_wise max diff {max_diff:.2e} exceeds tolerance"
    print("  PASSED ✓\n")

    # ── Test 2: load_balancing_loss_func (no mask) ──
    print("Testing load_balancing_loss_func alignment...")

    orig_lb = load_balancing_loss_func(
        router_logits_list, num_experts, top_k, attention_mask=None,
    )
    triton_lb = triton_load_balancing_loss_func(
        router_logits_list, num_experts, top_k, attention_mask=None,
    )
    diff = (orig_lb - triton_lb).abs().item()
    print(f"  No mask:   orig={orig_lb.item():.6f}, triton={triton_lb.item():.6f}, diff={diff:.2e}")
    assert diff < 1e-4, f"No-mask mismatch: {diff}"

    # ── Test 3: load_balancing_loss_func (with mask) ──
    attn_mask = torch.ones(1, num_tokens + padding_len, device=device)
    attn_mask[0, num_tokens:] = 0.0
    orig_lb_m = load_balancing_loss_func(
        router_logits_list, num_experts, top_k, attention_mask=attn_mask,
    )
    triton_lb_m = triton_load_balancing_loss_func(
        router_logits_list, num_experts, top_k, attention_mask=attn_mask,
    )
    diff = (orig_lb_m - triton_lb_m).abs().item()
    print(f"  With mask: orig={orig_lb_m.item():.6f}, triton={triton_lb_m.item():.6f}, diff={diff:.2e}")
    assert diff < 1e-4, f"Masked mismatch: {diff}"

    print("  PASSED ✓\n")
    print("All alignment tests passed!")


if __name__ == "__main__":
    _test_alignment()
