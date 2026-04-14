#include "dkv_phase.cuh"

namespace sm100::bwd::head128_2kernels::dkv {

template void run_bwd_dkv_phase_kernel<576>(const SparseAttnBwdParams& params);

}  // namespace sm100::bwd::head128_2kernels::dkv
