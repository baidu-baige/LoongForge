// Adapted from FlashMLA/csrc/api/sparse_fwd.h
// Simplified for standalone sparse MLA forward module (SM100 only)

#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/fast_math.h>

#include "params.h"
#include "smxxx/sparse_mla_fwd.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == \
    torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

inline int int64_stride_to_int(int64_t orig_stride) {
    if (orig_stride > std::numeric_limits<int>::max()) {
        TORCH_CHECK(false, "[Sparse MLA FWD] Stride exceeds int32 limit: ", orig_stride);
    }
    return static_cast<int>(orig_stride);
}

std::vector<at::Tensor> sparse_prefill_fwd(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    float sm_scale,
    int d_v,
    int q_start_index_s,
    bool write_p_out,
    const std::optional<at::Tensor> &topk_length
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm100 = dprops->major == 10;
    TORCH_CHECK(is_sm100, "Sparse Attention Forward Kernel is only supported on SM100 architecture");

    CHECK_DEVICE(q);
    CHECK_DEVICE(kv);
    CHECK_DEVICE(indices);
    if (topk_length.has_value()) {
        CHECK_DEVICE(topk_length.value());
    }

    TORCH_CHECK(q.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv.dtype() == torch::kBFloat16);
    TORCH_CHECK(indices.dtype() == torch::kInt32);
    if (topk_length.has_value()) {
        TORCH_CHECK(topk_length.value().dtype() == torch::kInt32);
    }

    int s_q = q.size(0);
    int s_kv = kv.size(0);
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int d_qk = q.size(2);
    int topk = indices.size(2);
    bool have_topk_length = topk_length.has_value();

    TORCH_CHECK(d_qk == 576 || d_qk == 512, "Invalid d_qk: ", d_qk, ". Expected 512 or 576.");
    TORCH_CHECK(d_v == 512, "Invalid d_v: ", d_v);
    TORCH_CHECK(q_start_index_s >= 0, "q_start_index_s must be >= 0");
    TORCH_CHECK(h_q == 128 || h_q == 64, "Sparse attention forward requires h_q=128 or h_q=64. Got h_q=", h_q);
    TORCH_CHECK(h_kv == 1, "Sparse attention forward currently only supports h_kv=1. Got h_kv=", h_kv);
    if (h_q == 128) {
        TORCH_CHECK(topk > 0 && topk % 128 == 0, "Sparse attention forward (head128) requires topk to be a positive multiple of 128. Got topk=", topk);
    } else {
        TORCH_CHECK(topk > 0 && topk % 64 == 0, "Sparse attention forward (head64) requires topk to be a positive multiple of 64. Got topk=", topk);
    }

    CHECK_SHAPE(q, s_q, h_q, d_qk);
    CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    CHECK_SHAPE(indices, s_q, h_kv, topk);
    if (have_topk_length) {
        CHECK_SHAPE(topk_length.value(), s_q);
    }

    TORCH_CHECK(q.stride(-1) == 1);
    TORCH_CHECK(kv.stride(-1) == 1);
    TORCH_CHECK(indices.stride(-1) == 1);

    // Allocate output tensors
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();

    at::Tensor out = torch::empty({s_q, h_q, d_v}, opts);
    at::Tensor max_logits = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat32));
    at::Tensor lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat32));
    at::Tensor p_out;
    if (write_p_out) {
        p_out = torch::empty({s_q, h_q, topk}, opts.dtype(torch::kFloat32));
    }

    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(max_logits);
    CHECK_CONTIGUOUS(lse);
    if (write_p_out) {
        CHECK_CONTIGUOUS(p_out);
    }

    SparseAttnFwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        q_start_index_s,
        sm_scale, sm_scale * 1.44269504f,

        // Input tensors
        (cutlass::bfloat16_t*)q.data_ptr(),
        (cutlass::bfloat16_t*)kv.data_ptr(),
        (int*)indices.data_ptr(),
        nullptr,  // attn_sink not supported in this interface
        have_topk_length ? (int*)topk_length.value().data_ptr() : nullptr,

        // Strides
        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        // Output tensors
        (cutlass::bfloat16_t*)out.data_ptr(),
        (float*)max_logits.data_ptr(),
        (float*)lse.data_ptr(),
        write_p_out ? (float*)p_out.data_ptr() : nullptr,

        dprops->multiProcessorCount,
        at::cuda::getCurrentCUDAStream().stream()
    };
    if (h_q == 128) {
        if (d_qk == 576) {
            sm100::fwd::head128::run_fwd_phase1_kernel<576>(params);
        } else {
            sm100::fwd::head128::run_fwd_phase1_kernel<512>(params);
        }
    } else {
        if (d_qk == 576) {
            sm100::fwd::head64::run_fwd_phase1_kernel<576>(params);
        } else {
            sm100::fwd::head64::run_fwd_phase1_kernel<512>(params);
        }
    }
    return {out, max_logits, lse, p_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLAFWD";
    m.def("sparse_prefill_fwd", &sparse_prefill_fwd);
}
