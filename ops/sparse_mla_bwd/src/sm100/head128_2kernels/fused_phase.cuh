#pragma once

#include "phase.h"

namespace sm100::bwd::head128_2kernels::fused {

template<int D_QK>
void run_bwd_fused_phase_kernel(const SparseAttnBwdParams& params) {
    dq::run_bwd_dq_phase_kernel<D_QK>(params);
    dkv::run_bwd_dkv_phase_kernel<D_QK>(params);
}

}  // namespace sm100::bwd::head128_2kernels::fused
