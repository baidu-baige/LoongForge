// Copyright 2026 The LoongForge Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Modified from Wall-X (https://github.com/X-Square-Robot/wall-x)
// under the Apache-2.0 License.
//
// Bindings for the bitwise-exact fused kernels.
//
// These live in a SEPARATE extension module (``_cuda_ext_exact_bin``) from the
// rest of the ops, because they must be compiled WITHOUT ``--use_fast_math``:
//   * fast math substitutes the approximate ``__expf`` for ``expf``, which breaks
//     the bitwise match with ``torch.sigmoid`` that the SwiGLU kernel relies on;
//   * it also enables ``--ftz=true``, which zeroes fp32 denormals that ATen keeps.
// The main extension keeps its existing flags, so its numerics are untouched.
//
// See setup.py for the flags, and EXPERIMENTS.md (R15/R16) for
// the bitwise argument these kernels implement.

#include <torch/extension.h>

namespace wallx_cuda_swiglu_exact {
void SwigluExactFwd(const at::Tensor& gate, const at::Tensor& up, const at::Tensor& out);
void SwigluExactBwd(const at::Tensor& grad_out, const at::Tensor& gate, const at::Tensor& up, const at::Tensor& dgate, const at::Tensor& dup);
}  // namespace wallx_cuda_swiglu_exact

namespace wallx_cuda_rmsnorm_exact {
void RmsNormExactSquare(const at::Tensor& x, const at::Tensor& sq);
void RmsNormExactInv(const at::Tensor& var, const at::Tensor& inv, double eps);
void RmsNormExactGsq2(const at::Tensor& g_inv, const at::Tensor& inv, const at::Tensor& g_sq2, long cols);
void RmsNormExactFwdOut(const at::Tensor& x, const at::Tensor& inv, const at::Tensor& weight, const at::Tensor& out);
void RmsNormExactBwdProd(const at::Tensor& grad_out, const at::Tensor& x, const at::Tensor& inv, const at::Tensor& weight, const at::Tensor& p_dw, const at::Tensor& p_inv);
void RmsNormExactBwdDx(const at::Tensor& grad_out, const at::Tensor& x, const at::Tensor& inv, const at::Tensor& g_sq2, const at::Tensor& weight, const at::Tensor& dx);
}  // namespace wallx_cuda_rmsnorm_exact

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_exact_fwd", &wallx_cuda_swiglu_exact::SwigluExactFwd);
    m.def("swiglu_exact_bwd", &wallx_cuda_swiglu_exact::SwigluExactBwd);
    m.def("rmsnorm_exact_square", &wallx_cuda_rmsnorm_exact::RmsNormExactSquare);
    m.def("rmsnorm_exact_inv", &wallx_cuda_rmsnorm_exact::RmsNormExactInv);
    m.def("rmsnorm_exact_gsq2", &wallx_cuda_rmsnorm_exact::RmsNormExactGsq2);
    m.def("rmsnorm_exact_fwd_out", &wallx_cuda_rmsnorm_exact::RmsNormExactFwdOut);
    m.def("rmsnorm_exact_bwd_prod", &wallx_cuda_rmsnorm_exact::RmsNormExactBwdProd);
    m.def("rmsnorm_exact_bwd_dx", &wallx_cuda_rmsnorm_exact::RmsNormExactBwdDx);
}
