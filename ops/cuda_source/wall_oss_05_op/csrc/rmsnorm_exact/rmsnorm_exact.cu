// Copyright 2026 The LoongForge Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Bitwise-exact partially-fused RMSNorm (regular branch, bf16 or fp32 in/out).
//
// Design: the three reductions stay in ATen, so their accumulation order is
// bit-identical to eager by construction; only the pointwise glue is fused here.
// Replicating ATen's own reduction order in a hand-written kernel is not viable --
// the split for the dw column reduction depends on the SM count.
//
// eager reference (Qwen2RMSNorm, cond=None):
//     x32    = x.to(fp32)
//     var    = x32.pow(2).mean(-1, keepdim=True)     <- ATen reduction
//     inv    = rsqrt(var + eps)                      <- ATen elementwise
//     out    = (w * (x32 * inv)).to(input_dtype)
//
// fp32 inputs take the same path: `x.to(fp32)` and the trailing `.to(input_dtype)`
// both become no-ops, so `Cvt<float>` uses identity load/store. The LLM norms
// (input_layernorm / post_attention_layernorm / model.norm) are deliberately kept
// in fp32 by `convert_to_mix_precision`, which is why the fp32 path matters.
//
// Every intermediate is fp32, so materializing vs keeping in registers makes no
// difference (unlike SwiGLU, where the bf16 rounding points are load-bearing).
// The only reassociation risk is fp32 contraction in `dx_a + dx_b` (an add of two
// products), so all fp32 arithmetic uses __fmul_rn/__fadd_rn and the result does
// not depend on the build passing -fmad=false.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../common/cuda_utils.h"

namespace wallx_cuda_rmsnorm_exact {

namespace {

struct Shape2D {
    long rows;
    long cols;
    long row_stride;
};

Shape2D collapse(const at::Tensor &t) {
    TORCH_CHECK(t.dim() >= 1, "rmsnorm_exact: expected >=1D tensor");
    TORCH_CHECK(t.stride(-1) == 1, "rmsnorm_exact: last dim must be contiguous");
    long cols = t.size(-1);
    long rows = t.numel() / cols;
    long rs = t.dim() >= 2 ? t.stride(-2) : cols;
    for (int d = t.dim() - 3; d >= 0; --d) {
        TORCH_CHECK(t.stride(d) == t.size(d + 1) * t.stride(d + 1),
                    "rmsnorm_exact: leading dims must nest over the row stride");
    }
    return {rows, cols, rs};
}

// Element conversion: fp32 loads/stores are the identity, so the same kernel body
// reproduces eager for both dtypes.
template <typename T>
struct Cvt;

template <>
struct Cvt<__nv_bfloat16> {
    __device__ static float load(const __nv_bfloat16 &v) { return __bfloat162float(v); }
    __device__ static __nv_bfloat16 store(float v) { return __float2bfloat16(v); }
};

template <>
struct Cvt<float> {
    __device__ static float load(const float &v) { return v; }
    __device__ static float store(float v) { return v; }
};

void check_elem(const at::Tensor &t, at::ScalarType dtype, const char *name) {
    TORCH_CHECK(t.scalar_type() == dtype, "rmsnorm_exact: ", name,
                " dtype mismatch (all element tensors must share one dtype)");
    TORCH_CHECK(t.is_cuda(), "rmsnorm_exact: ", name, " must be a CUDA tensor");
}

at::ScalarType elem_dtype(const at::Tensor &t) {
    TORCH_CHECK(t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kFloat,
                "rmsnorm_exact: only bfloat16 and float32 are supported, got ", t.scalar_type());
    TORCH_CHECK(t.is_cuda(), "rmsnorm_exact: expected a CUDA tensor");
    return t.scalar_type();
}

void check_rowvec(const at::Tensor &t, long rows, const char *name) {
    TORCH_CHECK(t.scalar_type() == at::kFloat, "rmsnorm_exact: ", name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), "rmsnorm_exact: ", name, " must be contiguous");
    TORCH_CHECK(t.numel() == rows, "rmsnorm_exact: ", name, " must have one entry per row");
}

constexpr int kThreads = 256;

long grid_for(long total) { return (total + kThreads - 1) / kThreads; }

}  // namespace

// Dispatch over the two supported element types.
#define WALLX_RMSNORM_DISPATCH(DTYPE, ...)                                   \
    do {                                                                     \
        if ((DTYPE) == at::kBFloat16) {                                      \
            using scalar_t = __nv_bfloat16;                                  \
            __VA_ARGS__;                                                     \
        } else {                                                             \
            using scalar_t = float;                                          \
            __VA_ARGS__;                                                     \
        }                                                                    \
    } while (0)

// sq = x32 * x32; consumed by at::mean afterwards. Matches eager's
// `x.to(fp32).pow(2)` (ATen lowers pow(t, 2) to t*t).
template <typename T>
__global__ void rmsnorm_exact_square_kernel(const T *__restrict__ x, long x_rs,
                                            float *__restrict__ sq, long s_rs,
                                            long rows, long cols) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    long r = idx / cols, c = idx - r * cols;
    float xf = Cvt<T>::load(x[r * x_rs + c]);
    sq[r * s_rs + c] = __fmul_rn(xf, xf);
}

// out = (w * (x32 * inv)).to(input_dtype)
template <typename T>
__global__ void rmsnorm_exact_fwd_out_kernel(const T *__restrict__ x, long x_rs,
                                             const float *__restrict__ inv,
                                             const T *__restrict__ w,
                                             T *__restrict__ out, long o_rs,
                                             long rows, long cols) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    long r = idx / cols, c = idx - r * cols;
    float xf = Cvt<T>::load(x[r * x_rs + c]);
    float wf = Cvt<T>::load(w[c]);
    float normed = __fmul_rn(xf, inv[r]);
    out[r * o_rs + c] = Cvt<T>::store(__fmul_rn(wf, normed));
}

// The two fp32 products consumed by the ATen reductions:
//   p_dw  = g32 * normed       -> sum over leading dims = dw
//   p_inv = (g32 * w32) * x32  -> sum over last dim     = grad wrt inv
template <typename T>
__global__ void rmsnorm_exact_bwd_prod_kernel(const T *__restrict__ go, long g_rs,
                                              const T *__restrict__ x, long x_rs,
                                              const float *__restrict__ inv,
                                              const T *__restrict__ w,
                                              float *__restrict__ p_dw, long pd_rs,
                                              float *__restrict__ p_inv, long pi_rs,
                                              long rows, long cols) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    long r = idx / cols, c = idx - r * cols;
    float gf = Cvt<T>::load(go[r * g_rs + c]);
    float xf = Cvt<T>::load(x[r * x_rs + c]);
    float wf = Cvt<T>::load(w[c]);
    float invf = inv[r];
    p_dw[r * pd_rs + c] = __fmul_rn(gf, __fmul_rn(xf, invf));
    p_inv[r * pi_rs + c] = __fmul_rn(__fmul_rn(gf, wf), xf);
}

// dx = ((g32*w32) * inv + g_sq2 * x32).to(input_dtype), g_sq2 = 2*g_var/N from ATen.
template <typename T>
__global__ void rmsnorm_exact_bwd_dx_kernel(const T *__restrict__ go, long g_rs,
                                            const T *__restrict__ x, long x_rs,
                                            const float *__restrict__ inv,
                                            const float *__restrict__ g_sq2,
                                            const T *__restrict__ w,
                                            T *__restrict__ dx, long d_rs,
                                            long rows, long cols) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    long r = idx / cols, c = idx - r * cols;
    float gf = Cvt<T>::load(go[r * g_rs + c]);
    float xf = Cvt<T>::load(x[r * x_rs + c]);
    float wf = Cvt<T>::load(w[c]);
    float dx_a = __fmul_rn(__fmul_rn(gf, wf), inv[r]);
    float dx_b = __fmul_rn(g_sq2[r], xf);
    dx[r * d_rs + c] = Cvt<T>::store(__fadd_rn(dx_a, dx_b));
}

void RmsNormExactSquare(const at::Tensor &x, const at::Tensor &sq) {
    at::ScalarType dt = elem_dtype(x);
    TORCH_CHECK(sq.scalar_type() == at::kFloat, "rmsnorm_exact: sq must be float32");
    Shape2D sx = collapse(x), ss = collapse(sq);
    long total = sx.rows * sx.cols;
    if (total == 0) return;
    WALLX_RMSNORM_DISPATCH(dt,
        rmsnorm_exact_square_kernel<scalar_t>
            <<<grid_for(total), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const scalar_t *>(x.data_ptr()), sx.row_stride,
                sq.data_ptr<float>(), ss.row_stride, sx.rows, sx.cols));
    sync_check_cuda_error();
}

void RmsNormExactFwdOut(const at::Tensor &x, const at::Tensor &inv, const at::Tensor &weight,
                        const at::Tensor &out) {
    at::ScalarType dt = elem_dtype(x);
    check_elem(weight, dt, "weight");
    check_elem(out, dt, "out");
    Shape2D sx = collapse(x), so = collapse(out);
    check_rowvec(inv, sx.rows, "inv");
    TORCH_CHECK(weight.numel() == sx.cols && weight.is_contiguous(),
                "rmsnorm_exact: weight must be contiguous with cols entries");
    long total = sx.rows * sx.cols;
    if (total == 0) return;
    WALLX_RMSNORM_DISPATCH(dt,
        rmsnorm_exact_fwd_out_kernel<scalar_t>
            <<<grid_for(total), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const scalar_t *>(x.data_ptr()), sx.row_stride,
                inv.data_ptr<float>(), reinterpret_cast<const scalar_t *>(weight.data_ptr()),
                reinterpret_cast<scalar_t *>(out.data_ptr()), so.row_stride, sx.rows, sx.cols));
    sync_check_cuda_error();
}

void RmsNormExactBwdProd(const at::Tensor &grad_out, const at::Tensor &x, const at::Tensor &inv,
                         const at::Tensor &weight, const at::Tensor &p_dw,
                         const at::Tensor &p_inv) {
    at::ScalarType dt = elem_dtype(x);
    check_elem(grad_out, dt, "grad_out");
    check_elem(weight, dt, "weight");
    Shape2D sgo = collapse(grad_out), sx = collapse(x);
    Shape2D spd = collapse(p_dw), spi = collapse(p_inv);
    check_rowvec(inv, sx.rows, "inv");
    long total = sx.rows * sx.cols;
    if (total == 0) return;
    WALLX_RMSNORM_DISPATCH(dt,
        rmsnorm_exact_bwd_prod_kernel<scalar_t>
            <<<grid_for(total), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const scalar_t *>(grad_out.data_ptr()), sgo.row_stride,
                reinterpret_cast<const scalar_t *>(x.data_ptr()), sx.row_stride,
                inv.data_ptr<float>(), reinterpret_cast<const scalar_t *>(weight.data_ptr()),
                p_dw.data_ptr<float>(), spd.row_stride, p_inv.data_ptr<float>(), spi.row_stride,
                sx.rows, sx.cols));
    sync_check_cuda_error();
}

void RmsNormExactBwdDx(const at::Tensor &grad_out, const at::Tensor &x, const at::Tensor &inv,
                       const at::Tensor &g_sq2, const at::Tensor &weight, const at::Tensor &dx) {
    at::ScalarType dt = elem_dtype(x);
    check_elem(grad_out, dt, "grad_out");
    check_elem(weight, dt, "weight");
    check_elem(dx, dt, "dx");
    Shape2D sgo = collapse(grad_out), sx = collapse(x), sd = collapse(dx);
    check_rowvec(inv, sx.rows, "inv");
    check_rowvec(g_sq2, sx.rows, "g_sq2");
    long total = sx.rows * sx.cols;
    if (total == 0) return;
    WALLX_RMSNORM_DISPATCH(dt,
        rmsnorm_exact_bwd_dx_kernel<scalar_t>
            <<<grid_for(total), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const scalar_t *>(grad_out.data_ptr()), sgo.row_stride,
                reinterpret_cast<const scalar_t *>(x.data_ptr()), sx.row_stride,
                inv.data_ptr<float>(), g_sq2.data_ptr<float>(),
                reinterpret_cast<const scalar_t *>(weight.data_ptr()),
                reinterpret_cast<scalar_t *>(dx.data_ptr()), sd.row_stride, sx.rows, sx.cols));
    sync_check_cuda_error();
}

}  // namespace wallx_cuda_rmsnorm_exact
