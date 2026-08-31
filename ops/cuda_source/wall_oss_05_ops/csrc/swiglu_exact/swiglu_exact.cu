// Copyright 2026 The LoongForge Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Bitwise-exact fused SwiGLU (silu(gate) * up) for bf16.
//
// Reproduces eager PyTorch's exact rounding sequence so that forward AND backward
// are bit-identical to `F.silu(gate) * up`, while staying a single fused kernel
// (no HBM round-trip for the intermediate activation).
//
// Two things must be preserved:
//   1. Intermediate bf16 rounding points. Eager rounds silu(gate) to bf16 before
//      multiplying by up; every backward intermediate is a separate kernel and is
//      therefore bf16-rounded too.
//   2. fp32 association order with NO floating-point contraction. ATen's
//      silu_backward is ((dy*s) * (1 + x*(1-s))). All fp32 arithmetic here uses
//      __fmul_rn/__fadd_rn so the result does not depend on the build using
//      -fmad=false.
//
// Tensors are treated as (rows, cols) with cols contiguous (stride == 1) and an
// explicit row stride, so non-contiguous `split()` views of a fused gate_up
// projection are consumed in place without a copy.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../common/cuda_utils.h"

namespace wallx_cuda_swiglu_exact {

namespace {

__device__ __forceinline__ float sigmoidf_rn(float x) {
    // matches torch.sigmoid on fp32 CUDA bitwise, and ATen's opmath sigmoid
    // inside silu / silu_backward. The fast variant __expf must NOT be used.
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x)));
}

struct Shape2D {
    long rows;
    long cols;
    long row_stride;
};

Shape2D collapse(const at::Tensor &t) {
    TORCH_CHECK(t.dim() >= 2, "swiglu_exact: expected >=2D tensor");
    TORCH_CHECK(t.stride(-1) == 1, "swiglu_exact: last dim must be contiguous");
    long cols = t.size(-1);
    long rows = t.numel() / cols;
    long rs = t.stride(-2);
    for (int d = t.dim() - 3; d >= 0; --d) {
        TORCH_CHECK(t.stride(d) == t.size(d + 1) * t.stride(d + 1),
                    "swiglu_exact: leading dims must nest over the row stride");
    }
    return {rows, cols, rs};
}

void check_bf16(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, "swiglu_exact: ", name, " must be bfloat16");
    TORCH_CHECK(t.is_cuda(), "swiglu_exact: ", name, " must be a CUDA tensor");
}

constexpr int kThreads = 256;

long grid_for(long total) { return (total + kThreads - 1) / kThreads; }

}  // namespace

__global__ void swiglu_exact_fwd_kernel(const __nv_bfloat16 *__restrict__ g, long g_rs,
                                        const __nv_bfloat16 *__restrict__ u, long u_rs,
                                        __nv_bfloat16 *__restrict__ out, long o_rs,
                                        long rows, long cols) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    long r = idx / cols, c = idx - r * cols;

    float xf = __bfloat162float(g[r * g_rs + c]);
    float uf = __bfloat162float(u[r * u_rs + c]);
    float s = sigmoidf_rn(xf);
    // eager: act = silu(gate) -> bf16 store (rounding point #1)
    __nv_bfloat16 act_b = __float2bfloat16(__fmul_rn(xf, s));
    // eager: out = act * up -> bf16 store (rounding point #2)
    out[r * o_rs + c] = __float2bfloat16(__fmul_rn(__bfloat162float(act_b), uf));
}

__global__ void swiglu_exact_bwd_kernel(const __nv_bfloat16 *__restrict__ go, long go_rs,
                                        const __nv_bfloat16 *__restrict__ g, long g_rs,
                                        const __nv_bfloat16 *__restrict__ u, long u_rs,
                                        __nv_bfloat16 *__restrict__ dg, long dg_rs,
                                        __nv_bfloat16 *__restrict__ du, long du_rs,
                                        long rows, long cols) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    long r = idx / cols, c = idx - r * cols;

    float gof = __bfloat162float(go[r * go_rs + c]);
    float xf = __bfloat162float(g[r * g_rs + c]);
    float uf = __bfloat162float(u[r * u_rs + c]);

    float s = sigmoidf_rn(xf);
    // recompute the forward activation; identical bf16 value to what eager saved
    __nv_bfloat16 act_b = __float2bfloat16(__fmul_rn(xf, s));

    // eager mul-backward: d_up = grad_out * act (bf16 rounded)
    du[r * du_rs + c] = __float2bfloat16(__fmul_rn(gof, __bfloat162float(act_b)));

    // eager mul-backward: d_act = grad_out * up (bf16 rounded)
    __nv_bfloat16 dact_b = __float2bfloat16(__fmul_rn(gof, uf));

    // eager silu_backward, ATen association: ((dy * s) * (1 + x * (1 - s)))
    float dy = __bfloat162float(dact_b);
    float t = __fadd_rn(1.0f, __fmul_rn(xf, __fadd_rn(1.0f, -s)));
    dg[r * dg_rs + c] = __float2bfloat16(__fmul_rn(__fmul_rn(dy, s), t));
}

void SwigluExactFwd(const at::Tensor &gate, const at::Tensor &up, const at::Tensor &out) {
    check_bf16(gate, "gate");
    check_bf16(up, "up");
    check_bf16(out, "out");
    TORCH_CHECK(gate.sizes() == up.sizes(), "swiglu_exact: gate/up shape mismatch");
    TORCH_CHECK(gate.sizes() == out.sizes(), "swiglu_exact: gate/out shape mismatch");
    Shape2D sg = collapse(gate), su = collapse(up), so = collapse(out);
    long total = sg.rows * sg.cols;
    if (total == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    swiglu_exact_fwd_kernel<<<grid_for(total), kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(gate.data_ptr()), sg.row_stride,
        reinterpret_cast<const __nv_bfloat16 *>(up.data_ptr()), su.row_stride,
        reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()), so.row_stride, sg.rows, sg.cols);
    sync_check_cuda_error();
}

void SwigluExactBwd(const at::Tensor &grad_out, const at::Tensor &gate, const at::Tensor &up,
                    const at::Tensor &dgate, const at::Tensor &dup) {
    check_bf16(grad_out, "grad_out");
    check_bf16(gate, "gate");
    check_bf16(up, "up");
    Shape2D sgo = collapse(grad_out), sg = collapse(gate), su = collapse(up);
    Shape2D sdg = collapse(dgate), sdu = collapse(dup);
    long total = sg.rows * sg.cols;
    if (total == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    swiglu_exact_bwd_kernel<<<grid_for(total), kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(grad_out.data_ptr()), sgo.row_stride,
        reinterpret_cast<const __nv_bfloat16 *>(gate.data_ptr()), sg.row_stride,
        reinterpret_cast<const __nv_bfloat16 *>(up.data_ptr()), su.row_stride,
        reinterpret_cast<__nv_bfloat16 *>(dgate.data_ptr()), sdg.row_stride,
        reinterpret_cast<__nv_bfloat16 *>(dup.data_ptr()), sdu.row_stride, sg.rows, sg.cols);
    sync_check_cuda_error();
}

}  // namespace wallx_cuda_swiglu_exact
