#pragma once

#include "params.h"

namespace sm100::bwd::head128_2kernels::dq {

template<int D_QK>
void run_bwd_dq_phase_kernel(const SparseAttnBwdParams& params);

}  // namespace sm100::bwd::head128_2kernels::dq

namespace sm100::bwd::head128_2kernels::dkv {

template<int D_QK>
void run_bwd_dkv_phase_kernel(const SparseAttnBwdParams& params);

}  // namespace sm100::bwd::head128_2kernels::dkv

namespace sm100::bwd::head128_2kernels::fused {

template<int D_QK>
void run_bwd_fused_phase_kernel(const SparseAttnBwdParams& params);

}  // namespace sm100::bwd::head128_2kernels::fused
