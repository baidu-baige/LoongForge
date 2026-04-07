#include "fused_phase.cuh"

namespace sm100::bwd::head128_2kernels::fused {

template void run_bwd_fused_phase_kernel<576>(const SparseAttnBwdParams& params);

}  // namespace sm100::bwd::head128_2kernels::fused
