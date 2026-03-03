// Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/fast_math.h>

#include "params1.h"
#include "params.h"
#include "sm90/sparse_mla_fwd.h"
#include "sm100/sparse_mla_fwd.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == \ 
torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

struct Arch {
    int major;
    int minor;

    bool is_sm90() const {
        return major == 9 && minor == 0;
    }

    bool is_sm100() const {
        return major == 10;
    }

    void assert_is_supported() const {
        TORCH_CHECK(is_sm90() || is_sm100(), "Only SM90 and SM100 are supported");
    }
};

inline int int64_stride_to_int(int64_t orig_stride) {
    if (orig_stride > std::numeric_limits<int>::max()) {
        TORCH_CHECK(false, "[Sparse TopK Attention] Stride exceeds int32 limit: ", orig_stride);
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
    bool write_p_out
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm90 = dprops->major == 9;
    bool is_sm100 = dprops->major == 10;
    TORCH_CHECK(is_sm90 || is_sm100, "Sparse Attention Forward Kernel (sparse_prefill_fwd) is only supported on SM90 or SM100 architectures");

    CHECK_DEVICE(q);
    CHECK_DEVICE(kv);
    CHECK_DEVICE(indices);

    TORCH_CHECK(q.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv.dtype() == torch::kBFloat16);
    TORCH_CHECK(indices.dtype() == torch::kInt32);
    TORCH_CHECK(q_start_index_s >= 0, "q_start_index_s must be >= 0");

    int s_q = q.size(0);
    int s_kv = kv.size(0);
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int d_qk = q.size(2);
    int topk = indices.size(2);

    CHECK_SHAPE(q, s_q, h_q, d_qk);
    CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    CHECK_SHAPE(indices, s_q, h_kv, topk);

    TORCH_CHECK(q.stride(-1) == 1);
    TORCH_CHECK(kv.stride(-1) == 1);
    TORCH_CHECK(indices.stride(-1) == 1);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();
    at::Tensor out = torch::empty({s_q, h_q, d_v}, opts);
    CHECK_CONTIGUOUS(out);
    
    at::Tensor buf_attn_score, max_logits, lse, p_out;
    max_logits = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    if (write_p_out) {
        p_out = torch::empty({s_q, h_q, topk}, opts.dtype(torch::kFloat));
    }
    CHECK_CONTIGUOUS(max_logits);
    CHECK_CONTIGUOUS(lse);
    if (write_p_out) CHECK_CONTIGUOUS(p_out);

    SparsePrefillParams1 params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        q_start_index_s,
        sm_scale, sm_scale * 1.44269504f,

        (cutlass::bfloat16_t*)q.data_ptr(),
        (cutlass::bfloat16_t*)kv.data_ptr(),
        (int*)indices.data_ptr(),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        (cutlass::bfloat16_t*)out.data_ptr(),
        (float*)max_logits.data_ptr(),
        (float*)lse.data_ptr(),
        write_p_out ? (float*)p_out.data_ptr() : nullptr,

        at::cuda::getCurrentCUDAStream().stream()
    };
    
    if (is_sm90) {
        sm90::run_fwd_kernel1(params);
    } else if (is_sm100) {
        sm100::run_fwd_kernel1(params);
    } else {
        TORCH_CHECK(false, "Unknown architecture");
    }

    return {out, max_logits, lse, p_out};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLAFWD";
    m.def("sparse_prefill_fwd", &sparse_prefill_fwd);
}
