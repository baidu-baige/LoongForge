#pragma once

#include "params.h"

namespace sm100::bwd::head128 {

// Backward kernel declarations
template<int D_QK>
void run_bwd_preprocess_delta_kernel(const SparseAttnBwdParams& params);

template<int D_QK>
void run_bwd_phase1_kernel(const SparseAttnBwdParams& params);

}  // namespace sm100::bwd::head128

namespace sm100::bwd::head64 {

// Backward kernel declarations
template<int D_QK>
void run_bwd_preprocess_delta_kernel(const SparseAttnBwdParams& params);

template<int D_QK>
void run_bwd_phase1_kernel(const SparseAttnBwdParams& params);

}  // namespace sm100::bwd::head64