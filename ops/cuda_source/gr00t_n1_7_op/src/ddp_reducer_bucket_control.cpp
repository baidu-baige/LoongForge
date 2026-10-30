// Copyright 2026 The LoongForge Authors.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <torch/csrc/distributed/c10d/reducer.hpp>
#include <memory>
#include <vector>

static void ddp_reducer_initialize_buckets(
    const std::shared_ptr<c10d::Reducer>& reducer,
    std::vector<std::vector<size_t>> bucket_indices) {
    reducer->initialize_buckets(std::move(bucket_indices));
}

std::vector<c10d::GradBucket> ddp_reducer_get_buckets(
    const std::shared_ptr<c10d::Reducer>& reducer) {
    return reducer->get_grad_buckets(false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("ddp_reducer_initialize_buckets", &ddp_reducer_initialize_buckets);
    module.def("ddp_reducer_get_buckets", &ddp_reducer_get_buckets);
}
