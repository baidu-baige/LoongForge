#include "phase1.h"
#include "phase1.cuh"
#include "preprocess_delta.cuh"

namespace sm100::bwd::head128 {

template void run_bwd_preprocess_delta_kernel<576>(const SparseAttnBwdParams& params);
template void run_bwd_phase1_kernel<576>(const SparseAttnBwdParams& params);

}

namespace sm100::bwd::head64 {

template void run_bwd_preprocess_delta_kernel<576>(const SparseAttnBwdParams& params);
template void run_bwd_phase1_kernel<576>(const SparseAttnBwdParams& params);

}