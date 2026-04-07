// Adapted from FlashMLA/csrc/api/sparse_bwd.h
// Simplified for standalone sparse MLA backward module

#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/fast_math.h>

#include "params.h"
#include "sm100/sparse_mla_bwd.h"
#include "sm100/head128_2kernels/phase.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == \
    torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

inline int int64_stride_to_int(int64_t orig_stride) {
    if (orig_stride > std::numeric_limits<int>::max()) {
        TORCH_CHECK(false, "[Sparse MLA BWD] Stride exceeds int32 limit: ", orig_stride);
    }
    return static_cast<int>(orig_stride);
}

std::vector<at::Tensor> sparse_prefill_bwd(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &o,
    const at::Tensor &dO,
    const at::Tensor &indices,
    const at::Tensor &lse,
    float sm_scale,
    int d_v,
    const std::optional<at::Tensor> &topk_length,
    int q_start_index_s,
    bool fast_mode
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm100 = dprops->major == 10;
    TORCH_CHECK(is_sm100, "Sparse Attention Backward Kernel is only supported on SM100 architecture");

    CHECK_DEVICE(q);
    CHECK_DEVICE(kv);
    CHECK_DEVICE(o);
    CHECK_DEVICE(dO);
    CHECK_DEVICE(indices);
    CHECK_DEVICE(lse);
    if (topk_length.has_value()) {
        CHECK_DEVICE(topk_length.value());
    }

    TORCH_CHECK(q.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv.dtype() == torch::kBFloat16);
    TORCH_CHECK(o.dtype() == torch::kBFloat16);
    TORCH_CHECK(dO.dtype() == torch::kBFloat16);
    TORCH_CHECK(indices.dtype() == torch::kInt32);
    TORCH_CHECK(lse.dtype() == torch::kFloat32);
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

    TORCH_CHECK(d_qk == 576, "Invalid d_qk: ", d_qk);
    TORCH_CHECK(d_v == 512, "Invalid d_v: ", d_v);
    TORCH_CHECK(q_start_index_s >= 0, "q_start_index_s must be >= 0");
    TORCH_CHECK(h_kv == 1, "Sparse attention backward currently only supports h_kv=1. Got h_kv=", h_kv);
    TORCH_CHECK(topk > 0 && topk % 64 == 0, "Sparse attention backward requires topk to be a positive multiple of 64. Got topk=", topk);

    CHECK_SHAPE(q, s_q, h_q, d_qk);
    CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    CHECK_SHAPE(o, s_q, h_q, d_v);
    CHECK_SHAPE(dO, s_q, h_q, d_v);
    CHECK_SHAPE(indices, s_q, h_kv, topk);
    CHECK_SHAPE(lse, s_q, h_q);
    if (have_topk_length) {
        CHECK_SHAPE(topk_length.value(), s_q);
    }

    TORCH_CHECK(q.stride(-1) == 1);
    TORCH_CHECK(kv.stride(-1) == 1);
    TORCH_CHECK(o.stride(-1) == 1);
    TORCH_CHECK(dO.stride(-1) == 1);
    TORCH_CHECK(indices.stride(-1) == 1);

    // Allocate output and intermediate tensors
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();

    at::Tensor dQ = torch::empty({s_q, h_q, d_qk}, opts);
    at::Tensor dKV = torch::zeros({s_kv, h_kv, d_qk}, opts.dtype(torch::kFloat32));
    at::Tensor delta = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat32));
    at::Tensor s = torch::empty({s_q, h_q, topk}, opts);
    at::Tensor ds = torch::empty({s_q, h_q, topk}, opts);

    CHECK_CONTIGUOUS(dQ);
    CHECK_CONTIGUOUS(dKV);
    CHECK_CONTIGUOUS(delta);
    CHECK_CONTIGUOUS(s);
    CHECK_CONTIGUOUS(ds);

    SparseAttnBwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        q_start_index_s,
        sm_scale, sm_scale * 1.44269504f,

        // Input tensors
        (cutlass::bfloat16_t*)q.data_ptr(),
        (cutlass::bfloat16_t*)kv.data_ptr(),
        (cutlass::bfloat16_t*)o.data_ptr(),
        (cutlass::bfloat16_t*)dO.data_ptr(),
        (int*)indices.data_ptr(),
        (float*)lse.data_ptr(),
        have_topk_length ? (int*)topk_length.value().data_ptr() : nullptr,

        // Strides
        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(o.stride(0)), int64_stride_to_int(o.stride(1)),
        int64_stride_to_int(dO.stride(0)), int64_stride_to_int(dO.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        // Output tensors
        (cutlass::bfloat16_t*)dQ.data_ptr(),
        (float*)dKV.data_ptr(),
        (float*)delta.data_ptr(),
        int64_stride_to_int(dQ.stride(0)), int64_stride_to_int(dQ.stride(1)),
        int64_stride_to_int(dKV.stride(0)), int64_stride_to_int(dKV.stride(1)),
        int64_stride_to_int(delta.stride(0)), int64_stride_to_int(delta.stride(1)),

        // Intermediate tensors (for fused mode)
        (cutlass::bfloat16_t*)s.data_ptr(),
        (cutlass::bfloat16_t*)ds.data_ptr(),
        int64_stride_to_int(s.stride(0)), int64_stride_to_int(s.stride(1)),
        int64_stride_to_int(ds.stride(0)), int64_stride_to_int(ds.stride(1)),

        dprops->multiProcessorCount,
        at::cuda::getCurrentCUDAStream().stream()
    };

    if (h_q == 128) {
        if (fast_mode) {
            sm100::bwd::head128_2kernels::fused::run_bwd_fused_phase_kernel<576>(params);
        } else {
            sm100::bwd::head128::run_bwd_phase1_kernel<576>(params);
        }
    } else if (h_q == 64) {
        sm100::bwd::head64::run_bwd_phase1_kernel<576>(params);
    } else {
        TORCH_CHECK(false, "Sparse attention backward currently only supports h_q=128 or 64. Got h_q=", h_q);
    }
    // Convert dKV from float32 to bfloat16
    at::Tensor dKV_bf16 = dKV.to(torch::kBFloat16);

    return {dQ, dKV_bf16};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLABWD";
    m.def("sparse_prefill_bwd", &sparse_prefill_bwd, py::arg("q"), py::arg("kv"), py::arg("o"),
          py::arg("dO"), py::arg("indices"), py::arg("lse"), py::arg("sm_scale"),
          py::arg("d_v"), py::arg("topk_length") = py::none(),
          py::arg("q_start_index_s") = 0, py::arg("fast_mode") = false);
}