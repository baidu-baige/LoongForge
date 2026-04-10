#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "apis/attention_bwd.hpp"
#include "../vendor/deep_gemm_csrc/apis/runtime.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "lightning_indexer_bwd C++ library";

    deep_gemm::attention_bwd::register_apis(m);
    deep_gemm::runtime::register_apis(m);
}
