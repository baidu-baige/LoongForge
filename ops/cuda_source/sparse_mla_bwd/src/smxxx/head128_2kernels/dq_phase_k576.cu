#include "phase.h"
#include "dq_phase.cuh"

namespace sm100::bwd::head128_2kernels::dq {

template void run_bwd_dq_phase_kernel<576>(const SparseAttnBwdParams& params);

}  // namespace sm100::bwd::head128_2kernels::dq
