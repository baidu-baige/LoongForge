# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Wall-X (https://github.com/X-Square-Robot/wall-x)
# under the Apache-2.0 License.

"""CUDA kernel wrappers with autograd support."""

import torch
from torch.autograd import Function


def _m():
    """Get the compiled CUDA extension module."""
    from wall_oss_05_op._cuda_ext import load
    return load()


class _RopeFunction(Function):
    """Autograd adapter for the out-of-place RoPE CUDA kernel."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        cos,
        sin,
        interleave,
        fwd_kernel=None,
        bwd_kernel=None,
    ):
        """Run the RoPE forward kernel and save tensors for backward."""
        q_embed = q
        k_embed = k
        ctx.save_for_backward(q, k, cos, sin)
        ctx.interleave = interleave
        ctx.bwd_kernel = bwd_kernel
        fwd_kernel(q, k, q_embed, k_embed, cos, sin, interleave)
        return q_embed, k_embed

    @staticmethod
    def backward(ctx, grad_q_embed, grad_k_embed):
        """Run the RoPE backward kernel."""
        q, k, cos, sin = ctx.saved_tensors
        interleave = ctx.interleave
        bwd_kernel = ctx.bwd_kernel
        grad_q = grad_q_embed
        grad_k = grad_k_embed

        bwd_kernel(
            grad_q_embed,
            grad_k_embed,
            grad_q,  # output
            grad_k,  # output
            cos,
            sin,
            interleave,
        )

        return grad_q, grad_k, None, None, None, None, None


class _RopePackFunction(Function):
    """Autograd adapter for the packed RoPE CUDA kernel."""

    @staticmethod
    def forward(
        ctx,
        qkv,
        q_num_heads,
        kv_num_heads,
        cos,
        sin,
        interleave,
        fwd_kernel=None,
        bwd_kernel=None,
    ):
        """Run the packed RoPE forward kernel."""
        ctx.save_for_backward(cos, sin)
        ctx.q_num_heads = q_num_heads
        ctx.kv_num_heads = kv_num_heads
        ctx.interleave = interleave
        ctx.bwd_kernel = bwd_kernel

        fwd_kernel(qkv, cos, sin, q_num_heads, kv_num_heads, interleave)
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        """Run the packed RoPE backward kernel."""
        if dqkv is None:
            return None, None, None, None, None, None, None, None

        cos, sin = ctx.saved_tensors
        if dqkv.stride(-1) != 1:
            dqkv = dqkv.contiguous()

        ctx.bwd_kernel(
            dqkv,
            cos,
            sin,
            ctx.q_num_heads,
            ctx.kv_num_heads,
            ctx.interleave,
        )
        return dqkv, None, None, None, None, None, None, None


class Rope:
    """Wrapper exposing standard and packed RoPE CUDA operations."""

    def __init__(self):
        """Load the standard RoPE CUDA entry points."""
        self.fwd_kernel = _m().rope
        self.inplace_kernel = _m().rope_inplace
        self.pack_kernel = _m().rope_inplace_pack
        self.pack_bwd_kernel = _m().rope_inplace_pack_bwd
        self.bwd_kernel = _m().rope_bwd

    def pack(
        self,
        qkv,
        q_num_heads,
        kv_num_heads,
        cos,
        sin,
        interleave=False,
        inference=False,
    ):
        """Packed qkv rope, splits/offsets computed inside the kernel.

        Args:
            qkv:          [seq_len, q_dim + 2*kv_dim]
            q_num_heads:  number of query heads
            kv_num_heads: number of key/value heads
            cos, sin:     rotary position encoding [1, seq_len, head_dim/2]
            interleave:   whether to use interleaved RoPE
            inference:    run without autograd bookkeeping
        """
        if inference:
            self.pack_kernel(qkv, cos, sin, q_num_heads, kv_num_heads, interleave)
            return qkv

        return _RopePackFunction.apply(
            qkv,
            q_num_heads,
            kv_num_heads,
            cos,
            sin,
            interleave,
            self.pack_kernel,
            self.pack_bwd_kernel,
        )

    def pack_backward(
        self, dqkv, q_num_heads, kv_num_heads, cos, sin, interleave=False
    ):
        """Packed dqkv backward for the packed rope kernel."""
        self.pack_bwd_kernel(dqkv, cos, sin, q_num_heads, kv_num_heads, interleave)
        return dqkv

    def __call__(self, q, k, cos, sin, interleave=False, inference=False):
        """Apply RoPE to query and key tensors."""
        if inference:
            self.inplace_kernel(q, k, cos, sin, interleave)
            return q, k

        return _RopeFunction.apply(
            q,
            k,
            cos,
            sin,
            interleave,
            self.fwd_kernel,
            self.bwd_kernel,
        )


class _MRopeFunction(Function):
    """Autograd adapter for the out-of-place M-RoPE CUDA kernel."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        cos,
        sin,
        mrope_section,
        fwd_kernel=None,
        bwd_kernel=None,
    ):
        """Run the M-RoPE forward kernel."""
        first = mrope_section[0]
        second = mrope_section[1]
        ctx.save_for_backward(q, k, cos, sin)
        ctx.mrope_section = mrope_section
        ctx.bwd_kernel = bwd_kernel
        q_embed = q
        k_embed = k
        fwd_kernel(q, k, q_embed, k_embed, cos, sin, first, second)
        return q_embed, k_embed

    @staticmethod
    def backward(ctx, grad_q_embed, grad_k_embed):
        """Run the M-RoPE backward kernel."""
        if grad_q_embed.stride(-1) != 1:
            grad_q_embed = grad_q_embed.contiguous()
        if grad_k_embed.stride(-1) != 1:
            grad_k_embed = grad_k_embed.contiguous()
        q, k, cos, sin = ctx.saved_tensors
        mrope_section = ctx.mrope_section
        bwd_kernel = ctx.bwd_kernel
        grad_q = grad_q_embed
        grad_k = grad_k_embed
        first = mrope_section[0]
        second = mrope_section[1]

        bwd_kernel(
            grad_q_embed,
            grad_k_embed,
            grad_q,  # output
            grad_k,  # output
            cos,
            sin,
            first,
            second,
        )

        return grad_q, grad_k, None, None, None, None, None


class _MRopePackFunction(Function):
    """Autograd adapter for the packed M-RoPE CUDA kernel."""

    @staticmethod
    def forward(
        ctx,
        qkv,
        q_num_heads,
        kv_num_heads,
        cos,
        sin,
        mrope_section,
        fwd_kernel=None,
        bwd_kernel=None,
    ):
        """Run the packed M-RoPE forward kernel."""
        first = mrope_section[0]
        second = mrope_section[1]
        ctx.save_for_backward(cos, sin)
        ctx.q_num_heads = q_num_heads
        ctx.kv_num_heads = kv_num_heads
        ctx.mrope_section = mrope_section
        ctx.bwd_kernel = bwd_kernel

        fwd_kernel(qkv, cos, sin, q_num_heads, kv_num_heads, first, second)
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        """Run the packed M-RoPE backward kernel."""
        if dqkv is None:
            return None, None, None, None, None, None, None, None

        cos, sin = ctx.saved_tensors
        if dqkv.stride(-1) != 1:
            dqkv = dqkv.contiguous()
        first = ctx.mrope_section[0]
        second = ctx.mrope_section[1]

        ctx.bwd_kernel(
            dqkv,
            cos,
            sin,
            ctx.q_num_heads,
            ctx.kv_num_heads,
            first,
            second,
        )
        return dqkv, None, None, None, None, None, None, None


class MRope:
    """Wrapper exposing standard and packed M-RoPE CUDA operations."""

    def __init__(self):
        """Load the M-RoPE CUDA entry points."""
        self.fwd_kernel = _m().m_rope
        self.inplace_kernel = _m().m_rope_inplace
        self.pack_kernel = _m().m_rope_inplace_pack
        self.pack_bwd_kernel = _m().m_rope_inplace_pack_bwd
        self.bwd_kernel = _m().m_rope_bwd

    def pack(
        self,
        qkv,
        q_num_heads,
        kv_num_heads,
        cos,
        sin,
        mrope_section,
        inference=False,
    ):
        """Packed qkv m-rope. qkv must be 3D: [bz, seq_len, q_dim+2*kv_dim].

        Args:
            qkv:          [bz, seq_len, q_dim + 2*kv_dim]
            q_num_heads:  number of query heads
            kv_num_heads: number of key/value heads
            cos, sin:     rotary position encoding [3, bz, seq_len, head_dim/2]
            mrope_section: (first, second) M-RoPE section parameters
            inference:    run without autograd bookkeeping
        """
        if inference:
            self.pack_kernel(
                qkv,
                cos,
                sin,
                q_num_heads,
                kv_num_heads,
                mrope_section[0],
                mrope_section[1],
            )
            return qkv

        return _MRopePackFunction.apply(
            qkv,
            q_num_heads,
            kv_num_heads,
            cos,
            sin,
            mrope_section,
            self.pack_kernel,
            self.pack_bwd_kernel,
        )

    def pack_backward(self, dqkv, q_num_heads, kv_num_heads, cos, sin, mrope_section):
        """Packed dqkv backward for the m-rope kernel."""
        self.pack_bwd_kernel(
            dqkv,
            cos,
            sin,
            q_num_heads,
            kv_num_heads,
            mrope_section[0],
            mrope_section[1],
        )
        return dqkv

    def __call__(self, q, k, cos, sin, mrope_section, inference=False):
        """Apply M-RoPE to query and key tensors."""
        if inference:
            first = mrope_section[0]
            second = mrope_section[1]
            self.inplace_kernel(q, k, cos, sin, first, second)
            return q, k

        return _MRopeFunction.apply(
            q,
            k,
            cos,
            sin,
            mrope_section,
            self.fwd_kernel,
            self.bwd_kernel,
        )


class RotPos:
    """Wrapper for fused vision rotary-position embedding kernels."""

    def __call__(self, inv_freq, grid_thw, spatial_merge_size, metadata=None):
        """Compute vision rotary-position embeddings."""
        assert (
            inv_freq.dtype == torch.float32
        ), f"Expected float32, got {inv_freq.dtype}"
        get_token_counts_kernel = _m().get_token_counts
        rot_pos_kernel = _m().rot_pos

        num_grids = grid_thw.size(0)
        token_counts = torch.zeros(
            (num_grids), dtype=grid_thw.dtype, device=grid_thw.device
        )

        get_token_counts_kernel(grid_thw, token_counts, spatial_merge_size)
        cumsum_tokens = torch.cat(
            [
                torch.zeros(1, dtype=token_counts.dtype, device=token_counts.device),
                token_counts.cumsum(dim=0),
            ],
            dim=0,
        ).to(grid_thw.dtype)

        total_tokens = (
            metadata.total_tokens if metadata is not None else cumsum_tokens[-1].item()
        )
        output = torch.empty(
            (total_tokens, inv_freq.size(0) * 2),
            dtype=torch.float,
            device=inv_freq.device,
        )
        rot_pos_kernel(inv_freq, grid_thw, output, cumsum_tokens, spatial_merge_size)

        return output


class GetRopeIndex:
    """Wrapper for the CUDA 3D RoPE-index kernel."""

    def __call__(
        self,
        input_ids,
        image_grid_thw,
        video_grid_thw,
        second_per_grid_ts,
        attention_mask,
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        tokens_per_second,
    ):
        """Compute 3D position IDs and M-RoPE position deltas."""
        get_workspace = _m().get_rope_index_getworkspace
        get_rope_index_kernel = _m().get_rope_index
        work_space_size = get_workspace(input_ids, image_grid_thw, video_grid_thw)

        workspace = torch.empty(
            work_space_size, dtype=torch.uint8, device=input_ids.device
        )
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        position_ids = torch.empty(
            (3, batch_size, seq_len), dtype=torch.int64, device=input_ids.device
        )
        mrope_deltas = torch.empty(
            (batch_size, 1), dtype=torch.int64, device=input_ids.device
        )
        get_rope_index_kernel(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
            attention_mask,
            position_ids,
            mrope_deltas,
            workspace,
            spatial_merge_size,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            tokens_per_second,
        )

        return position_ids, mrope_deltas


def get_window_index_cuda(
    grid_thw,
    window_size,
    spatial_merge_size,
    patch_size,
    spatial_merge_unit=1,
    metadata=None,
):
    """Compute window indices and cumulative window sequence lengths."""
    get_window_index_kernel = _m().get_window_index
    get_totals_kernel = _m().get_totals

    if grid_thw.size(0) == 0:
        return (
            torch.empty(0, dtype=grid_thw.dtype, device=grid_thw.device),
            torch.zeros(1, dtype=grid_thw.dtype, device=grid_thw.device),
        )

    vit_merger_window_size = window_size // spatial_merge_size // patch_size

    grid_info_tensor = torch.empty(
        (grid_thw.size(0), 6), dtype=grid_thw.dtype, device=grid_thw.device
    )
    global_totals_tensor = torch.zeros(
        (2), dtype=grid_thw.dtype, device=grid_thw.device
    )
    get_totals_kernel(
        grid_thw,
        grid_info_tensor,
        global_totals_tensor,
        spatial_merge_size,
        vit_merger_window_size,
    )
    if metadata is None:
        total_elements = global_totals_tensor[0].item()
        total_windows = global_totals_tensor[1].item()
    else:
        total_elements = metadata.total_elements
        total_windows = metadata.total_windows
    if total_elements == 0 or total_windows == 0:
        return (
            torch.empty(0, dtype=grid_thw.dtype, device=grid_thw.device),
            torch.zeros(1, dtype=grid_thw.dtype, device=grid_thw.device),
        )

    window_indices = torch.empty(
        total_elements, dtype=grid_thw.dtype, device=grid_thw.device
    )
    cu_window_seqlens = torch.empty(
        (total_windows + 1), dtype=grid_thw.dtype, device=grid_thw.device
    )
    window_counts_tensor = torch.empty(
        (total_windows), dtype=grid_thw.dtype, device=grid_thw.device
    )

    max_grid_t = (
        metadata.max_grid_t if metadata is not None else grid_thw[:, 0].max().item()
    )
    get_window_index_kernel(
        grid_thw,
        grid_info_tensor,
        window_indices,
        cu_window_seqlens,
        window_counts_tensor,
        max_grid_t,
        spatial_merge_size,
        vit_merger_window_size,
        patch_size,
        spatial_merge_unit,
    )

    return window_indices, cu_window_seqlens


################################################################################################
##
## PermuteMoE topK
##
################################################################################################


class PermuteMoETopK(torch.autograd.Function):
    """Autograd function for CUDA top-k MoE token permutation."""

    max_expanded_token_num = 0

    @staticmethod
    def forward(
        ctx,
        input_act: torch.Tensor,
        indices: torch.Tensor,
        num_out_tokens: int,
        max_token_num: int,
    ):
        """
        indices: for topK=1, indices in a 1-d tensor of shape [num_tokens],
                 otherwise, it's a 2-d tensor of shape [num_tokens, topK]
        """
        if not input_act.numel():
            return input_act, None

        # For top1 case, view the indices as 2D tensor to unify the shape for topk>=2 cases.
        if indices.dim() == 1:
            indices = indices.view(-1, 1)

        # Device check
        if input_act.is_cpu:
            raise RuntimeError(
                "[Error] The input `input_act` of permute_topK op is on the device: CPU!"
            )

        # Data type check
        if indices.dtype != torch.int32:
            indices = indices.to(torch.int32)

        # Contiguous check
        if not input_act.is_contiguous():
            input_act = input_act.contiguous()
        if not indices.is_contiguous():
            indices = indices.contiguous()

        num_topK = indices.size(1)

        input_max_expanded_token_num = max(max_token_num, input_act.size(0)) * num_topK
        if PermuteMoETopK.max_expanded_token_num < input_max_expanded_token_num:
            PermuteMoETopK.max_expanded_token_num = input_max_expanded_token_num

        permute_kernel = _m().permute
        sorted_indices = torch.empty(
            PermuteMoETopK.max_expanded_token_num,
            dtype=torch.int32,
            device=input_act.device,
        )
        row_id = torch.arange(
            PermuteMoETopK.max_expanded_token_num,
            dtype=torch.int32,
            device=input_act.device,
        )
        sorted_row_id = torch.empty(
            PermuteMoETopK.max_expanded_token_num,
            dtype=torch.int32,
            device=input_act.device,
        )
        get_storage_bytes_kernel = _m().cub_sort_pair_get_storage_bytes
        temp_storage_bytes = get_storage_bytes_kernel(
            PermuteMoETopK.max_expanded_token_num
        )
        temp_storage = torch.empty(
            temp_storage_bytes, dtype=torch.int8, device=input_act.device
        )
        num_out = (
            num_out_tokens if (num_out_tokens > 0) else (indices.size(0) * num_topK)
        )
        permuted_output = torch.empty(
            (num_out, input_act.size(1)), dtype=input_act.dtype, device=input_act.device
        )
        row_id_map = torch.empty(
            (indices.size(0) * num_topK), dtype=torch.int32, device=input_act.device
        )
        permute_kernel(
            input_act,
            indices,
            sorted_indices,
            row_id,
            sorted_row_id,
            temp_storage,
            permuted_output,
            row_id_map,
            num_out_tokens,
            PermuteMoETopK.max_expanded_token_num,
        )

        ctx.row_id_map = row_id_map
        ctx.num_tokens = indices.size(0)
        ctx.num_topK = num_topK

        return permuted_output, row_id_map

    @staticmethod
    def backward(ctx, permuted_act_grad, _):
        """Restore gradients to the original token order."""
        if not permuted_act_grad.numel():
            return permuted_act_grad, None, None, None

        unpermute_kernel = _m().unpermute
        if not permuted_act_grad.is_contiguous():
            permuted_act_grad = permuted_act_grad.contiguous()

        row_id_map = ctx.row_id_map
        num_tokens = ctx.num_tokens
        num_topK = ctx.num_topK
        num_cols = permuted_act_grad.size(1)
        unpermuted_output = torch.empty(
            (num_tokens, num_cols),
            dtype=permuted_act_grad.dtype,
            device=permuted_act_grad.device,
        )
        unpermute_kernel(
            permuted_act_grad, row_id_map, None, unpermuted_output, num_tokens, num_topK
        )

        return unpermuted_output, None, None, None


################################################################################################
##
## UnpermuteMoE topK
##
################################################################################################


class UnpermuteMoETopK(torch.autograd.Function):
    """Autograd function for CUDA top-k MoE token unpermutation."""

    @staticmethod
    def forward(
        ctx,
        input_act: torch.Tensor,
        row_id_map: torch.Tensor,
        probs: torch.Tensor = None,
    ):
        """Merge permuted tokens back into their original order."""

        if not input_act.numel():
            ctx.probs = probs
            return input_act

        # Device check
        if input_act.is_cpu:
            raise RuntimeError(
                "[Error] The input `input_act` of unpermute_topK op is on the device: CPU!"
            )
        if row_id_map.is_cpu:
            row_id_map = row_id_map.cuda()
        if probs is not None and probs.is_cpu:
            probs = probs.cuda()

        # Shape check
        if probs is not None and row_id_map.size(0) != probs.size(0) * probs.size(1):
            raise RuntimeError(
                f"[Error] unpermute_topK op input `probs` shape mismatch! "
                f"Expect {row_id_map.size(0)}, but got {probs.size(0) * probs.size(1)}."
            )

        # Data type check
        if row_id_map.dtype != torch.int32:
            row_id_map = row_id_map.to(torch.int32)
        if probs is not None and probs.dtype != torch.float32:
            probs = probs.to(torch.float32)

        # Contiguous check
        if not input_act.is_contiguous():
            input_act = input_act.contiguous()
        if not row_id_map.is_contiguous():
            row_id_map = row_id_map.contiguous()
        if probs is not None and not probs.is_contiguous():
            probs = probs.contiguous()

        num_tokens = probs.size(0) if probs is not None else input_act.size(0)
        num_topK = probs.size(1) if probs is not None else 1
        unpermute_kernel = _m().unpermute
        num_cols = input_act.size(1)
        unpermuted_output = torch.empty(
            (num_tokens, num_cols), dtype=input_act.dtype, device=input_act.device
        )
        unpermute_kernel(
            input_act, row_id_map, probs, unpermuted_output, num_tokens, num_topK
        )

        ctx.save_for_backward(input_act, row_id_map, probs)

        return unpermuted_output

    @staticmethod
    def backward(ctx, unpermuted_act_grad):
        """Propagate gradients through the unpermutation operation."""

        if not unpermuted_act_grad.numel():
            return unpermuted_act_grad, None, ctx.probs

        if not unpermuted_act_grad.is_contiguous():
            unpermuted_act_grad = unpermuted_act_grad.contiguous()

        input_act, row_id_map, probs = ctx.saved_tensors

        act_grad = None
        unpermute_bwd_kernel = _m().unpermute_bwd
        if ctx.needs_input_grad[0]:
            num_cols = unpermuted_act_grad.size(1)
            act_grad = torch.empty(
                (input_act.size(0), num_cols),
                dtype=unpermuted_act_grad.dtype,
                device=unpermuted_act_grad.device,
            )
            prob_grad = torch.empty(
                (probs.size(0), probs.size(1)),
                dtype=torch.float32,
                device=unpermuted_act_grad.device,
            )
            unpermute_bwd_kernel(
                unpermuted_act_grad, input_act, row_id_map, probs, act_grad, prob_grad
            )

        if not ctx.needs_input_grad[2]:
            prob_grad = None

        return act_grad, None, prob_grad


def permute_kernel(input_act, indices, num_out_tokens=None, max_token_num=0):
    """Permute tokens using the CUDA top-k MoE implementation."""
    num_out_tokens = 0 if num_out_tokens is None else num_out_tokens
    return PermuteMoETopK.apply(input_act, indices, num_out_tokens, max_token_num)


def unpermute_kernel(input_act, row_id_map, probs=None):
    """Unpermute tokens using the CUDA top-k MoE implementation."""
    return UnpermuteMoETopK.apply(input_act, row_id_map, probs)


# ----------------------------------------------------------------------------
# Bitwise-exact fused SwiGLU / RMSNorm
#
# Both reproduce eager PyTorch's rounding sequence exactly (forward and
# backward), so enabling them does not change the training loss at all. They
# live in a separate extension module built without --use_fast_math, so
# ``has_exact_symbols`` must be used before selecting them.
# ----------------------------------------------------------------------------

SWIGLU_EXACT_SYMBOLS = ("swiglu_exact_fwd", "swiglu_exact_bwd")
RMSNORM_EXACT_SYMBOLS = (
    "rmsnorm_exact_square",
    "rmsnorm_exact_fwd_out",
    "rmsnorm_exact_bwd_prod",
    "rmsnorm_exact_bwd_dx",
)


def _m_exact():
    """Get the compiled bitwise-exact CUDA extension module."""
    from wall_oss_05_op._cuda_ext import load_exact
    return load_exact()


def has_exact_symbols(names):
    """Return True when the exact extension exposes every given entry point."""
    try:
        module = _m_exact()
    except Exception:
        return False
    return all(hasattr(module, name) for name in names)


class _SwigluExactFunction(Function):
    """silu(gate) * up, bit-identical to eager in forward and backward."""

    @staticmethod
    def forward(ctx, gate, up):
        """Run the exact SwiGLU CUDA forward kernel."""
        out = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
        _m_exact().swiglu_exact_fwd(gate, up, out)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Run the exact SwiGLU CUDA backward kernel."""
        gate, up = ctx.saved_tensors
        if grad_out.stride(-1) != 1:
            grad_out = grad_out.contiguous()
        dgate = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
        dup = torch.empty(up.shape, dtype=up.dtype, device=up.device)
        _m_exact().swiglu_exact_bwd(grad_out, gate, up, dgate, dup)
        return dgate, dup


def swiglu_exact_kernel(gate, up):
    """Apply exact CUDA SwiGLU with autograd support."""
    return _SwigluExactFunction.apply(gate, up)


class _RmsNormExactFunction(Function):
    """RMSNorm, bit-identical to eager in forward and backward.

    The three reductions (``mean`` over the last dim, ``sum`` for dw, ``sum`` for
    d_inv) are ATen calls on purpose: their accumulation order is what makes the
    result bitwise-reproducible, and it cannot be replicated in a hand-written
    kernel portably. Only the pointwise work is fused.

    Saves bf16 ``x`` + fp32 ``inv`` instead of eager's fp32 intermediates, which
    is where the activation-memory saving comes from.
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, eps):
        """Run the exact RMSNorm CUDA forward path."""
        sq = torch.empty(hidden_states.shape, dtype=torch.float32, device=hidden_states.device)
        module = _m_exact()
        module.rmsnorm_exact_square(hidden_states, sq)
        inv = torch.rsqrt(sq.mean(-1, keepdim=True) + eps)
        del sq
        out = torch.empty(
            hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        module.rmsnorm_exact_fwd_out(hidden_states, inv.reshape(-1), weight, out)
        ctx.save_for_backward(hidden_states, weight, inv)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Run the exact RMSNorm CUDA backward path."""
        hidden_states, weight, inv = ctx.saved_tensors
        if grad_out.stride(-1) != 1:
            grad_out = grad_out.contiguous()
        module = _m_exact()
        inv_flat = inv.reshape(-1)
        opts = dict(dtype=torch.float32, device=hidden_states.device)
        p_dw = torch.empty(hidden_states.shape, **opts)
        p_inv = torch.empty(hidden_states.shape, **opts)
        module.rmsnorm_exact_bwd_prod(grad_out, hidden_states, inv_flat, weight, p_dw, p_inv)

        dw = p_dw.sum(dim=tuple(range(hidden_states.dim() - 1))).to(weight.dtype)
        g_inv = p_inv.sum(-1, keepdim=True)
        del p_dw, p_inv

        n = hidden_states.shape[-1]
        g_var = (-0.5 * g_inv) * inv.pow(3)
        g_sq2 = ((g_var / n) * 2.0).reshape(-1).contiguous()
        dx = torch.empty(
            hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        module.rmsnorm_exact_bwd_dx(grad_out, hidden_states, inv_flat, g_sq2, weight, dx)
        return dx, dw, None


def rmsnorm_exact_kernel(hidden_states, weight, eps=1e-6):
    """Apply exact CUDA RMSNorm with autograd support."""
    return _RmsNormExactFunction.apply(hidden_states, weight, eps)
