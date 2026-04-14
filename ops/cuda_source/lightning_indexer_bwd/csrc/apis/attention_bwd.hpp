#pragma once

#include "../../vendor/deep_gemm_csrc/utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/smxx_fp8_mqa_logits_bwd.hpp"
#endif

#include "../../vendor/deep_gemm_csrc/apis/layout.hpp"

namespace deep_gemm::attention_bwd
{

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fp8_mqa_logits_bwd(
    const torch::Tensor& grad_logits,
    const torch::Tensor& q,
    const std::pair<torch::Tensor, torch::Tensor>& kv,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end,
    const torch::Tensor& topk_indices,
    const bool& clean_logits,
    const int& max_seqlen_k,
    const int& topk
)
{
    // DG_HOST_ASSERT(clean_logits); // TODO

    // const auto& [seq_len_logits, seq_len_kv_logits] = get_shape<2>(grad_logits);
    const auto& [seq_len, num_heads, head_dim] = get_shape<3>(q);
    const auto& [seq_len_kv, head_dim_] = get_shape<2>(kv.first);
    const auto& [seq_len_, num_heads_] = get_shape<2>(weights);
    const auto& [seq_len_kv_] = get_shape<1>(kv.second);

    // DG_HOST_ASSERT(seq_len == seq_len_ and seq_len == seq_len_logits);
    DG_HOST_ASSERT(seq_len == seq_len_);
    DG_HOST_ASSERT(num_heads == num_heads_ and head_dim == head_dim_);
    // DG_HOST_ASSERT(seq_len_kv == seq_len_kv_ and seq_len_kv == seq_len_kv_logits);
    DG_HOST_ASSERT(seq_len_kv == seq_len_kv_);
    DG_HOST_ASSERT(cu_seq_len_k_start.size(0) == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_end.size(0) == seq_len);
    DG_HOST_ASSERT(topk_indices.size(1) == topk);

    DG_HOST_ASSERT(grad_logits.is_contiguous());
    DG_HOST_ASSERT(q.is_contiguous() and kv.first.is_contiguous());
    DG_HOST_ASSERT(kv.second.is_contiguous());
    DG_HOST_ASSERT(weights.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_end.is_contiguous());

    DG_HOST_ASSERT(grad_logits.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(q.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(kv.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(kv.second.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(cu_seq_len_k_end.scalar_type() == torch::kInt);

    constexpr int alignment = 4;
    const auto aligned_seq_len = align(seq_len, alignment);
    const auto aligned_seq_len_kv = align(seq_len_kv, alignment);
    const auto aligned_head_dim = align(head_dim, alignment);
    const auto aligned_num_heads = align(num_heads, alignment);

    torch::Tensor grad_q;
    grad_q = torch::zeros({aligned_seq_len, aligned_num_heads, aligned_head_dim}, q.options().dtype(torch::kFloat));
    grad_q = grad_q.index({torch::indexing::Slice(0, seq_len), torch::indexing::Slice(0, num_heads), torch::indexing::Slice(0, head_dim)});

    torch::Tensor grad_kv;
    grad_kv = torch::zeros({aligned_seq_len_kv, aligned_head_dim}, q.options().dtype(torch::kFloat));
    grad_kv = grad_kv.index({torch::indexing::Slice(0, seq_len_kv), torch::indexing::Slice(0, head_dim)});

    torch::Tensor grad_weights;
    grad_weights = torch::zeros({aligned_seq_len, aligned_num_heads}, q.options().dtype(torch::kFloat));
    grad_weights = grad_weights.index({torch::indexing::Slice(0, seq_len), torch::indexing::Slice(0, num_heads)});

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if ( arch_major == 9 or arch_major == 10 )
    {
        // Only support 128 aligned
        DG_HOST_ASSERT(seq_len_kv % 128 == 0);

        int stride_logits = align(seq_len_kv, 4);
        smxx_fp8_mqa_logits_bwd(
            grad_logits, q, kv.first, kv.second, weights, cu_seq_len_k_start, cu_seq_len_k_end,
            topk_indices, grad_q, grad_kv, grad_weights,
            seq_len, seq_len_kv, max_seqlen_k, stride_logits, num_heads, head_dim, topk, alignment
        );
    }
    else
    {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> grad_tuple(grad_q, grad_kv, grad_weights);
    return grad_tuple;
}
#endif 

static void register_apis( pybind11::module_& m )
{
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    m.def(
        "fp8_mqa_logits_bwd", &fp8_mqa_logits_bwd,
        py::arg("grad_logits"),
        py::arg("q"),
        py::arg("kv"),
        py::arg("weights"),
        py::arg("cu_seq_len_k_start"),
        py::arg("cu_seq_len_k_end"),
        py::arg("topk_indices"),
        py::arg("clean_logits") = true,
        py::arg("max_seqlen_k") = 0,
        py::arg("topk") = 0
    );
#endif
}

} // namespace deep_gemm::attention_bwd
