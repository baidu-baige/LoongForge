// Copyright 2026 The LoongForge Authors.
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified from Isaac-GR00T (https://github.com/NVIDIA/Isaac-GR00T)
// under the Apache-2.0 License.

#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> qwen3_vl_fused_vision_rope_cuda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& cos,
    const at::Tensor& sin);

std::vector<at::Tensor> qwen3_vl_fused_text_rope_cuda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& cos,
    const at::Tensor& sin);

at::Tensor qwen3_vl_fused_text_rms_norm_square_cuda(const at::Tensor& input);

at::Tensor qwen3_vl_fused_text_rms_norm_finish_cuda(
    const at::Tensor& input,
    const at::Tensor& variance,
    const at::Tensor& weight,
    double epsilon);

at::Tensor qwen3_vl_fused_text_silu_mul_cuda(
    const at::Tensor& gate,
    const at::Tensor& up);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "qwen3_vl_fused_vision_rope",
        &qwen3_vl_fused_vision_rope_cuda,
        "FP32-rounding-compatible Qwen3-VL vision RoPE");
    module.def(
        "qwen3_vl_fused_text_rope",
        &qwen3_vl_fused_text_rope_cuda,
        "Dtype-rounding-compatible Qwen3-VL text RoPE");
    module.def(
        "qwen3_vl_fused_text_rms_norm_square",
        &qwen3_vl_fused_text_rms_norm_square_cuda,
        "FP32 Qwen3-VL RMSNorm square stage");
    module.def(
        "qwen3_vl_fused_text_rms_norm_finish",
        &qwen3_vl_fused_text_rms_norm_finish_cuda,
        "Dtype-rounding-compatible Qwen3-VL RMSNorm finish stage");
    module.def(
        "qwen3_vl_fused_text_silu_mul",
        &qwen3_vl_fused_text_silu_mul_cuda,
        "Dtype-rounding-compatible Qwen3-VL SiLU multiply");
}
