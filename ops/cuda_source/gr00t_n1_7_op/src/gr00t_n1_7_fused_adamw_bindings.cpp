// Copyright 2026 The LoongForge Authors.
// Copyright (c) Meta Platforms, Inc. and affiliates.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//
// Modified from PyTorch (https://github.com/pytorch/pytorch)
// under the BSD-3-Clause License.

#include <torch/extension.h>

#include <vector>

void groot_n1_7_fused_adamw_eager_cuda(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    double decay_factor,
    double beta2,
    double first_moment_weight,
    double second_moment_weight,
    double eps,
    double bias_correction1,
    double bias_correction2_sqrt,
    double lr);

void groot_n1_7_fused_adamw_capturable_cuda(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    at::Tensor lr,
    at::Tensor step,
    at::Tensor bias_correction1,
    at::Tensor bias_correction2_sqrt,
    double beta2,
    double first_moment_weight,
    double second_moment_weight,
    double eps,
    double weight_decay);

void groot_n1_7_fused_adamw_capturable_grad_scaled_cuda(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    at::Tensor lr,
    at::Tensor step,
    at::Tensor bias_correction1,
    at::Tensor bias_correction2_sqrt,
    at::Tensor grad_scale,
    double beta2,
    double first_moment_weight,
    double second_moment_weight,
    double eps,
    double weight_decay);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "groot_n1_7_fused_adamw_eager_step",
        &groot_n1_7_fused_adamw_eager_cuda,
        "Precision-compatible fused AdamW eager step");
    module.def(
        "groot_n1_7_fused_adamw_capturable_step",
        &groot_n1_7_fused_adamw_capturable_cuda,
        "Precision-compatible fused AdamW capturable step");
    module.def(
        "groot_n1_7_fused_adamw_capturable_grad_scaled_step",
        &groot_n1_7_fused_adamw_capturable_grad_scaled_cuda,
        "Precision-compatible fused AdamW capturable step with fused gradient scale");
}
