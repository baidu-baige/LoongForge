"""
Kernels for Deepseek Sparse Attention.
"""

import dataclasses
from typing import Optional, Tuple, Any
from packaging.version import Version as PkgVersion

import torch
from torch import Tensor

_DSA_FUSED_DEPS_HINT = (
    "dsa_fused requires optional dependencies. "
    "Install them with: pip install -r requirements_dsa_fused.txt"
)

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise ImportError(_DSA_FUSED_DEPS_HINT) from exc

try:
    import deep_gemm
    import flashinfer
except ImportError as exc:
    raise ImportError(_DSA_FUSED_DEPS_HINT) from exc

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.utils import get_te_version, is_te_min_version

try:
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
except ImportError as exc:
    raise ImportError(_DSA_FUSED_DEPS_HINT) from exc

try:
    from flash_mla import flash_mla_sparse_fwd
except ImportError as exc:
    raise ImportError(_DSA_FUSED_DEPS_HINT) from exc
from .sparse_mla_bwd import sparse_mla_bwd_interface


@triton.jit
def triton_attn_dist_kernel(
    p_out_ptr,
    output_ptr,
    sm_scale,
    s_q, topk,
    stride_p_s: tl.int64, stride_p_h: tl.int64, stride_p_k: tl.int64,
    stride_o_s: tl.int64, stride_o_k: tl.int64,
    H_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Triton kernel for computing attention distribution over heads.

    This kernel computes the attention distribution by:
    1) Iterating over each query position and head
    2) Applying softmax over the topk dimension (after scaling by sm_scale)
    3) Averaging the resulting probabilities across heads

    Args:
        p_out_ptr: Pointer to the attention logits tensor of shape [s_q, h_q, topk]
        output_ptr: Pointer to the output tensor of shape [s_q, topk]
        sm_scale: Scaling factor for logits before softmax
        s_q: Number of query positions, Not used.
        topk: Number of top elements to consider, Not used.
        stride_p_s: Stride for p_out_ptr along sequence dimension
        stride_p_h: Stride for p_out_ptr along head dimension
        stride_p_k: Stride for p_out_ptr along topk dimension
        stride_o_s: Stride for output_ptr along sequence dimension
        stride_o_k: Stride for output_ptr along topk dimension
        H_Q: Number of heads (constexpr)
        BLOCK_K: Block size for topk dimension (constexpr, must equal topk)
    """
    s_idx = tl.program_id(0)
    k_offs = tl.arange(0, BLOCK_K)    
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    for h_idx in range(H_Q):
        p_ptrs = p_out_ptr + s_idx * stride_p_s + h_idx * stride_p_h + k_offs * stride_p_k

        p = tl.load(p_ptrs)
        attn_score = p * sm_scale

        max_val = tl.max(attn_score, axis=0)
        exp_val = tl.exp(attn_score - max_val)
        sum_exp = tl.sum(exp_val, axis=0)
        attn_prob = exp_val / sum_exp  # [BLOCK_K]

        acc += attn_prob

    o_ptrs = output_ptr + s_idx * stride_o_s + k_offs * stride_o_k
    tl.store(o_ptrs, acc / H_Q)


def triton_attn_dist(p_out: torch.Tensor, sm_scale) -> torch.Tensor:
    """
    Compute an averaged attention distribution over heads for a *top-k* attention tensor.

    This is a convenience wrapper around a Triton kernel that:
      1) For each query position `s`, iterates over all heads `h`.
      2) Applies a per-head softmax over the `topk` dimension to convert logits to probabilities:
           attn_prob[s, h, :] = softmax(p_out[s, h, :] * sm_scale)
      3) Averages the resulting probability distributions across heads:
           output[s, :] = mean_h(attn_prob[s, h, :])

    Args:
        p_out (torch.Tensor):
            Attention logits of shape (s_q, h_q, topk). Typically `float16`/`bfloat16`.
            The softmax is computed independently for each (s, h) over the last dimension.
        sm_scale (float | torch.Tensor):
            Scaling factor applied to logits before softmax (commonly 1/sqrt(d_k)).
            Must be broadcastable to a scalar in the Triton kernel.

    Returns:
        torch.Tensor:
            Averaged attention distribution of shape (s_q, topk), same dtype/device as `p_out`.

    Shape:
        Input:  (s_q, h_q, topk)
        Output: (s_q, topk)

    Notes:
        - The kernel computes softmax in fp32 for improved numerical stability.
        - `topk` must be a power of 2, enforced by:
              assert topk == triton.next_power_of_2(topk)
          If your `topk` is not a power of 2, pad `p_out` along the last dimension first.
        - The kernel assumes `BLOCK_K == topk` and processes the full last dimension in one program.
        - This function returns the *mean* probability across heads (not sum).

    Example:
        >>> p_out = torch.randn(1024, 16, 128, device="cuda", dtype=torch.float16)
        >>> sm_scale = (128 ** -0.5)
        >>> dist = triton_attn_dist(p_out, sm_scale)
        >>> dist.shape
        torch.Size([1024, 128])
    """
    s_q, h_q, topk = p_out.shape
    output = torch.empty((s_q, topk), device=p_out.device, dtype=p_out.dtype)

    assert topk == triton.next_power_of_2(topk)

    grid = (s_q,)

    triton_attn_dist_kernel[grid](
        p_out, output, sm_scale,
        s_q, topk,
        p_out.stride(0), p_out.stride(1), p_out.stride(2),
        output.stride(0), output.stride(1),
        H_Q=h_q,
        BLOCK_K=topk,
    )
    return output


def padded_flashinfer_topk(logits, topk, sk, *, sorted=True):
    """
    Compute top-k values and indices from logits tensor, with padding support.

    This function computes the top-k values and indices from the last dimension of the logits tensor.
    If the requested topk is larger than the last dimension, it first computes full top-d and then pads
    the results with -infinity for values and sk (sequence length) for indices.

    Args:
        logits (torch.Tensor): Input tensor of shape [..., d] containing logits
        topk (int): Number of top elements to return
        sk (int): Sequence length, used for padding indices when topk > d
        sorted (bool, optional): Whether to sort the returned values and indices. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - vals: Top-k values tensor of shape [..., topk]
            - idx: Top-k indices tensor of shape [..., topk]

    Examples:
        >>> logits = torch.randn(10, 20)  # [batch, seq_len]
        >>> vals, idx = padded_flashinfer_topk(logits, 5, 20)
        >>> vals.shape, idx.shape
        (torch.Size([10, 5]), torch.Size([10, 5]))

        >>> # Case where topk > seq_len
        >>> vals, idx = padded_flashinfer_topk(logits, 25, 20)
        >>> vals.shape, idx.shape
        (torch.Size([10, 25]), torch.Size([10, 25]))
    """
    d = logits.size(-1)
    topk = int(topk)
    if topk <= d:
        return flashinfer.top_k(logits, topk, sorted=sorted)

    # compute full top-d, then pad
    vals, idx = flashinfer.top_k(logits, d, sorted=sorted)
    pad = topk - d
    vals = torch.cat([vals, vals.new_full((*vals.shape[:-1], pad), float("-inf"))], dim=-1)
    idx  = torch.cat([idx, idx.new_full((*idx.shape[:-1], pad), sk)], dim=-1)  # use key length to fill
    return vals, idx


class DSADotProductAttentionFunction(torch.autograd.Function):
    """
    Tilelang's sparse_mla interface for BF16.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        indices,
        chunk_offset=0,
        sm_scale=None,
        d_v=512,
        return_p_out=False,
        packed_seq_params=None
    ):
        """
        Flash MLA sparse_mla_forward.
        """
        # q_flash [sq, n, d]
        # kv_flash [skv, 1, d]
        # indices_flash [sq, 1, topk]
        q_flash = q
        kv_flash = kv
        indices_flash = indices

        sq = q.size(0)
        log2e = 1.44269504
        
        out, _, lse, *p_out = flash_mla_sparse_fwd(
            q_flash,  # q: [s_q, h_q, d_qk], bfloat16
            kv_flash,  # kv: [s_kv, h_kv, d_qk], bfloat16
            indices_flash,  # [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
            sm_scale,
            d_v,
            q_start_index_s=chunk_offset,
            write_p_out=return_p_out
        )

        ctx.save_for_backward(q_flash, kv_flash, indices_flash, out, lse / log2e)
        ctx.sm_scale = sm_scale
        ctx.chunk_offset = chunk_offset
        ctx.sq = sq

        out = out.unsqueeze(0)

        if return_p_out:
            return out, p_out[0]
        else:
            return out

    @staticmethod
    def backward(ctx, grad_out, grad_p_out=None):
        """
        TileLang's sparse_mla_backward.
        """

        q, kv, indices, out, lse = ctx.saved_tensors

        offsets = torch.tensor([0, ctx.sq], dtype=torch.int32, device="cuda")
        grad_out = grad_out.squeeze(0).contiguous()

        grad_q, grad_kv = sparse_mla_bwd_interface(
            q,
            kv,
            out,
            grad_out,
            indices,
            lse,
            offsets,
            chunk_offset=ctx.chunk_offset,
            sm_scale=ctx.sm_scale,
            return_kernel=False,
            delta=None
        )
        
        return grad_q, grad_kv, None, None, None, None, None, None


class DSADotProductAttention(MegatronModule):
    """
    Wrapper for TileLang's `sparse_mla_interface` that supports DeepSeek sparse attention.

    This class provides an interface for computing sparse attention using the TileLang
    sparse attention operations optimized for DeepSeek models.
    """

    cp_stream: torch.cuda.Stream = None

    def __init__(
        self,
        config: TransformerConfig,
        softmax_scale: Optional[float] = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.qkv_format: str = 'sbhd'
        # layer_number is not used.
        # attention_type is not used.
        # attention_dropout is not used.
        self.softmax_scale = softmax_scale
        # k_channels is not used.
        # v_channels is not used.
        # cp_comm_type is not used.

        self.kept_packed_seq_params = set(field.name for field in dataclasses.fields(PackedSeqParams))
        if get_te_version() < PkgVersion("1.3.0"):
            # TE 1.3.0 introduces precomputing max_seqlen to remove unnecessary kernels and D2H
            # copies (#555)
            # These two arguments did not exist prior to 1.3.0
            self.kept_packed_seq_params.discard("max_seqlen_q")
            self.kept_packed_seq_params.discard("max_seqlen_kv")

        if get_te_version() < PkgVersion("1.10.0"):
            # TE 1.8.0 introduces cu_seqlens_padded which is the cu_seqlens with paddings counted
            # in each individual sequence in THD format dataset
            # These two arguments did not exist prior to 1.8.0. Full support added in 1.10.0 (#1012)
            self.kept_packed_seq_params.discard("cu_seqlens_q_padded")
            self.kept_packed_seq_params.discard("cu_seqlens_kv_padded")

    def forward(
        self,
        query: Tensor,
        kv: Tensor,
        indices: Tensor,
        chunk_offset: int,
        attn_mask_type: AttnMaskType,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        return_p_out: bool = False,
    ):
        """Forward."""
        if attn_mask_type is None:
            attn_mask_type = AttnMaskType.causal

        assert (attn_mask_type == AttnMaskType.causal or attn_mask_type == AttnMaskType.padding_causal), (
            "DSADotProductAttention only support causal attention."
            "Please use TEDotProductAttention instead."
        )
        assert attention_bias is None, "Attention bias is not supported for DSADotProductAttention."

        assert self.config.qk_pos_emb_head_dim == 64, (
            f"DSADotProductAttention only support qk_pos_emb_head_dim 64, but got {self.config.qk_pos_emb_head_dim}."
        )

        packed_seq_kwargs = (
            {key: getattr(packed_seq_params, key) for key in self.kept_packed_seq_params}
            if packed_seq_params is not None
            else {}
        )
        # overwrite self.qkv_format depending on self.config.apply_rope_fusion, which can be set
        # after init
        if self.config.apply_rope_fusion and is_te_min_version("0.13.0", check_equality=False):
            self.qkv_format = 'bshd'

        qkv_format = packed_seq_kwargs.get('qkv_format', self.qkv_format)

        # if qkv_format == 'bshd'
        #     query: [b, sq, n, kv_lora_rank + qk_pos_emb_head_dim]
        #     kv: [b, skv, kv_lora_rank + qk_pos_emb_head_dim]
        if self.config.apply_rope_fusion and qkv_format == 'bshd':
            query = query.transpose(0, 1).contiguous()
            kv = kv.unsqueeze(2).transpose(0, 1).contiguous()

        # Case: sequence packing
        #   query: [sq*b, n, kv_lora_rank + qk_pos_emb_head_dim]
        #   kv: [skv*b, kv_lora_rank + qk_pos_emb_head_dim]
        # FlashMLA accept query [s, h, d] and kv [s, 1, d], so no need to add extra batch dim
        elif qkv_format == 'thd':
            kv = kv.unsqueeze(1)  # add dummy head dim for kv

        # if qkv_format == 'sbhd':
        #     query: [sq, b, n, kv_lora_rank + qk_pos_emb_head_dim]
        #     kv: [skv, b, kv_lora_rank + qk_pos_emb_head_dim]
        elif qkv_format == 'sbhd':
            query = query.transpose(0, 1).contiguous()
            kv = kv.unsqueeze(2).transpose(0, 1).contiguous()

        # indices: [b, sq, topk] -> [b, sq, 1, topk]
        assert indices is not None, "DSADotProductAttention need topk_indices."
        indices = indices.unsqueeze(2)

        # FlashMLA does not support batched input
        if query.ndim == 4:
            query = query.squeeze(0)  # [b, sq, h, d] -> [sq, h, d]
        if kv.ndim == 4:
            kv = kv.squeeze(0)  # [b, skv, h, d] -> [skv, h, d]
        if indices.ndim == 4:
            indices = indices.squeeze(0)  # [b, sq, 1, topk] -> [sq, 1, topk]

        args = (
            query,
            kv,
            indices,
            chunk_offset,
            self.softmax_scale,
            self.config.kv_lora_rank,
            return_p_out,
            packed_seq_params,
        )
        # core_attn_out [b, s/TP, h, d_v], p_out: list of [s/TP, h, topk]
        core_attn_out, *p_out = DSADotProductAttentionFunction.apply(*args)
        
        if qkv_format == 'sbhd':
            core_attn_out = core_attn_out.contiguous()
        elif qkv_format == 'thd':
            core_attn_out = core_attn_out.squeeze(1).contiguous()
        else:
            core_attn_out = core_attn_out.contiguous()

        if return_p_out:
            return core_attn_out, p_out[0]
        else:
            return core_attn_out


class DSAIndexerKernelFunction(torch.autograd.Function):
    """
    Autograd function for DeepGEMM FP8Indexer kernel.

    This function implements the forward and backward passes for computing
    sparse attention indices using quantized query and key tensors with
    Float8BlockQuantizer.

    The forward pass:
    1. Quantizes input query and key tensors using Float8BlockQuantizer
    2. Computes scaled dot product between query and key in FP8
    3. Computes top-k indices from the resulting score matrix
    4. Handles sequence packing and causal masking

    The backward pass:
    1. Computes gradients for query, key and weights
    2. Handles the scaling factors properly during gradient computation

    Attributes:
        quantizer (Float8BlockQuantizer): FP8 quantizer with blockwise quantization
    """
    quantizer = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        amax_epsilon=1e-12,
        force_pow_2_scales=True,
        block_scaling_dim=1
    )

    @staticmethod
    def forward(
        ctx,
        index_q: torch.Tensor,  # [s, h, d]
        index_k: torch.Tensor,  # [s, d]
        weights: torch.Tensor,  # [s, h]
        index_topk: int,
        chunk_offset: int,
        packed_seq_params: Optional[PackedSeqParams],
    ):
        """
        DeepGEMM FP8Indexer forward.

        Quantizer: Float8BlockQuantizer with blockwise = 128.
        """
        assert index_q.ndim == 3 and index_k.ndim == 2 and weights.ndim == 2
        seq_q, head, dim = index_q.size()
        seq_k, _dim = index_k.size()
        assert dim == _dim, "Query and Key have diff dim."
        assert dim == 128, "Only support dim with size 128."
        device = index_q.device
        
        softmax_scale = (dim ** -0.5)

        quantized_q = DSAIndexerKernelFunction.quantizer.quantize(index_q)
        quantized_k = DSAIndexerKernelFunction.quantizer.quantize(index_k)
        q_fp8 = quantized_q.get_data_tensors(rowwise_data=True, columnwise_data=False).view(torch.float8_e4m3fn)
        k_fp8 = quantized_k.get_data_tensors(rowwise_data=True, columnwise_data=False).view(torch.float8_e4m3fn)
        q_scale = quantized_q._rowwise_scale_inv.reshape(index_q.shape[:-1])  # [seq_q, head, 1] -> [seq_q, head]
        k_scale = quantized_k._rowwise_scale_inv.reshape(index_k.shape[:-1])  # [seq_k, head, 1] -> [seq_k, head]

        if packed_seq_params is None:
            k_start = torch.zeros(seq_q, dtype=torch.int, device=device)
        else:
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            seqlens = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
            full_seq_ids = torch.repeat_interleave(torch.arange(len(seqlens), device=device, dtype=torch.int), seqlens)
            local_seq_ids = full_seq_ids[chunk_offset:chunk_offset + seq_q]
            k_start = cu_seqlens_kv[local_seq_ids]

        k_end = torch.arange(seq_q, dtype=torch.int, device=device) + chunk_offset + 1

        weight_scaled = weights * q_scale * softmax_scale  # absorb the `sf_q` and `softmax_scale` into weights

        if packed_seq_params is None:
            # index_score [sq, sk]
            index_score = deep_gemm.fp8_mqa_logits(q_fp8, (k_fp8, k_scale), weight_scaled, k_start, k_end)
        else:
            # index_score [sq, max_seqlen_k]
            max_seqlen_k = 0 if packed_seq_params is None else packed_seq_params.max_seqlen_kv
            index_score = deep_gemm.fp8_mqa_logits(
                q_fp8,
                (k_fp8, k_scale),
                weight_scaled,
                k_start, k_end,
                clean_logits=False,
                max_seqlen_k=max_seqlen_k
            )
            # Post-process to clean logits, apply causal mask, k_start is all zeros so omit here
            mask = torch.arange(max_seqlen_k, device='cuda')[None, :] < (k_end - k_start)[:, None]
            index_score = index_score.masked_fill(~mask, float('-inf'))

        index_score_topk, topk_indices = padded_flashinfer_topk(index_score.contiguous(), index_topk, seq_k)

        # In sft packing case, index_score is of shape [sq, max_seqlen_kv],
        # the topk_indices is relative indices within its sequence, convert to global indices by adding k_start
        if packed_seq_params is not None:
            topk_indices = topk_indices + k_start.unsqueeze(1)  # index may exceed sk, sparse_attn will handle that

        ctx.softmax_scale = softmax_scale
        ctx.index_topk = index_topk
        ctx.save_for_backward(q_fp8, k_fp8, q_scale, k_scale, weight_scaled, topk_indices, k_start, k_end)

        return index_score_topk, topk_indices

    @staticmethod
    def backward(ctx, grad_score, grad_topk):
        """
        DeepGEMM FP8Indexer backward.

        Computes gradients for:
        1. Quantized query tensor (d_q)
        2. Quantized key tensor (d_k)
        3. Weights tensor (d_weights)

        Args:
            ctx: Autograd context containing saved tensors from forward pass
            grad_score: Gradient of loss wrt index_score_topk (output from forward pass)
            grad_topk: Gradient of loss wrt topk_indices (output from forward pass)

        Returns:
            Tuple containing gradients for:
            - d_q: Gradient wrt quantized query tensor
            - d_k: Gradient wrt quantized key tensor
            - d_weights: Gradient wrt weights tensor
            - None: Placeholder for index_topk gradient (not differentiable)
            - None: Placeholder for chunk_offset gradient (not differentiable)
            - None: Placeholder for packed_seq_params gradient (not differentiable)
        """
        q_fp8, k_fp8, q_scale, k_scale, weight_scaled, topk_indices, ks, ke = ctx.saved_tensors

        d_q, d_k, d_weights = deep_gemm.fp8_mqa_logits_bwd(
            grad_score.contiguous(),
            q_fp8,
            (k_fp8, k_scale),
            weight_scaled,
            ks,
            ke,
            topk_indices=topk_indices.int(),
            topk=ctx.index_topk
        )

        d_weights = d_weights * q_scale * ctx.softmax_scale
        d_q = d_q / q_scale.unsqueeze(-1)
        d_k = d_k / k_scale.unsqueeze(-1)

        return d_q, d_k, d_weights, None, None, None


class DSAIndexerKernel(torch.nn.Module):
    """
    Wrapper for DeepGEMM FP8Indexer kernel that computes sparse attention indices.

    This class provides an interface for computing quantized query-key dot products
    and extracting top-k indices using FP8 quantization for efficiency.

    The computation involves:
    1. Quantizing query and key tensors using FP8 block quantization
    2. Computing scaled dot product in FP8 precision
    3. Extracting top-k indices with optional sequence packing support
    4. Handling causal masking for autoregressive models

    Attributes:
        quantizer (Float8BlockQuantizer): FP8 quantizer with blockwise quantization
    """

    def forward(self, index_q, index_k, weights, index_topk, chunk_offset, packed_seq_params):
        """
        Call to the autograd function.
        """
        args = (
            index_q,
            index_k,
            weights,
            index_topk,
            chunk_offset,
            packed_seq_params,
        )
        return DSAIndexerKernelFunction.apply(*args)
