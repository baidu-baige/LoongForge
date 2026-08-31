#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

namespace at::native {
namespace {

__device__ __forceinline__ float foreach_lerp(
    const float exp_avg,
    const float grad,
    const float weight) {
  const float delta = __fsub_rn(grad, exp_avg);
  return __fmaf_rn(weight, delta, exp_avg);
}

__device__ __forceinline__ float foreach_second_moment(
    const float exp_avg_sq,
    const float grad,
    const float beta2) {
  const float base = __fmul_rn(exp_avg_sq, beta2);
  const float grad_sq = __fmul_rn(grad, grad);
  return __fmaf_rn(__fsub_rn(1.0F, beta2), grad_sq, base);
}

__device__ __forceinline__ void foreach_compatible_update(
    float& param,
    const float grad,
    float& exp_avg,
    float& exp_avg_sq,
    const float decay_factor,
    const float beta2,
    const float first_moment_weight,
    const float second_moment_weight,
    const float eps,
    const float bias_correction2_sqrt,
    const float step_size) {
  param = __fmul_rn(param, decay_factor);
  exp_avg = foreach_lerp(exp_avg, grad, first_moment_weight);
  const float base = __fmul_rn(exp_avg_sq, beta2);
  const float grad_sq = __fmul_rn(grad, grad);
  exp_avg_sq = __fmaf_rn(second_moment_weight, grad_sq, base);

  // denominator = sqrt(exp_avg_sq) / bias_correction2_sqrt + eps
  float denominator = sqrtf(exp_avg_sq);
  denominator = __fdiv_rn(denominator, bias_correction2_sqrt);
  denominator = __fadd_rn(denominator, eps);
  const float quotient = __fdiv_rn(exp_avg, denominator);
  param = __fmaf_rn(step_size, quotient, param);
}

struct GrootN17FusedAdamWCapturableFunctor {
  __device__ __forceinline__ void operator()(
      const int64_t chunk_size,
      TensorListMetadata<4>& metadata,
      const double* lr,
      const int64_t* step,
      const double* bias_correction1,
      const double* bias_correction2_sqrt,
      const float beta2,
      const float first_moment_weight,
      const float second_moment_weight,
      const float eps,
      const double weight_decay) const {
    const auto step_index = *step;
    const double lr_value = *lr;
    const float decay_factor =
        static_cast<float>(1.0 - lr_value * weight_decay);
    const float step_size =
        static_cast<float>(-lr_value / bias_correction1[step_index]);
    const float correction2 =
        static_cast<float>(bias_correction2_sqrt[step_index]);

    const auto tensor = metadata.block_to_tensor[blockIdx.x];
    const auto chunk = metadata.block_to_chunk[blockIdx.x];
    const auto count = metadata.numel_for_tensor[tensor] - chunk * chunk_size;
    auto* param = static_cast<float*>(
        const_cast<void*>(metadata.addresses[0][tensor])) + chunk * chunk_size;
    auto* grad = static_cast<float*>(
        const_cast<void*>(metadata.addresses[1][tensor])) + chunk * chunk_size;
    auto* exp_avg = static_cast<float*>(
        const_cast<void*>(metadata.addresses[2][tensor])) + chunk * chunk_size;
    auto* exp_avg_sq = static_cast<float*>(
        const_cast<void*>(metadata.addresses[3][tensor])) + chunk * chunk_size;

    for (int64_t index = threadIdx.x;
         index < count && index < chunk_size;
         index += blockDim.x) {
      float p = param[index];
      float m = exp_avg[index];
      float v = exp_avg_sq[index];
      foreach_compatible_update(
          p,
          grad[index],
          m,
          v,
          decay_factor,
          beta2,
          first_moment_weight,
          second_moment_weight,
          eps,
          correction2,
          step_size);
      param[index] = p;
      exp_avg[index] = m;
      exp_avg_sq[index] = v;
    }
  }
};

struct GrootN17FusedAdamWEagerFunctor {
  __device__ __forceinline__ void operator()(
      const int64_t chunk_size,
      TensorListMetadata<4>& metadata,
      const float decay_factor,
      const float beta2,
      const float first_moment_weight,
      const float second_moment_weight,
      const float eps,
      const float bias_correction2_sqrt,
      const float step_size) const {
    const auto tensor = metadata.block_to_tensor[blockIdx.x];
    const auto chunk = metadata.block_to_chunk[blockIdx.x];
    const auto count = metadata.numel_for_tensor[tensor] - chunk * chunk_size;
    auto* param = static_cast<float*>(
        const_cast<void*>(metadata.addresses[0][tensor])) + chunk * chunk_size;
    auto* grad = static_cast<float*>(
        const_cast<void*>(metadata.addresses[1][tensor])) + chunk * chunk_size;
    auto* exp_avg = static_cast<float*>(
        const_cast<void*>(metadata.addresses[2][tensor])) + chunk * chunk_size;
    auto* exp_avg_sq = static_cast<float*>(
        const_cast<void*>(metadata.addresses[3][tensor])) + chunk * chunk_size;

    for (int64_t index = threadIdx.x;
         index < count && index < chunk_size;
         index += blockDim.x) {
      float p = param[index];
      float m = exp_avg[index];
      float v = exp_avg_sq[index];
      foreach_compatible_update(
          p,
          grad[index],
          m,
          v,
          decay_factor,
          beta2,
          first_moment_weight,
          second_moment_weight,
          eps,
          bias_correction2_sqrt,
          step_size);
      param[index] = p;
      exp_avg[index] = m;
      exp_avg_sq[index] = v;
    }
  }
};

struct GrootN17FusedAdamWCapturableGradScaledFunctor {
  __device__ __forceinline__ void operator()(
      const int64_t chunk_size,
      TensorListMetadata<4>& metadata,
      const double* lr,
      const int64_t* step,
      const double* bias_correction1,
      const double* bias_correction2_sqrt,
      const float* grad_scale,
      const float beta2,
      const float first_moment_weight,
      const float second_moment_weight,
      const float eps,
      const double weight_decay) const {
    const auto step_index = *step;
    const double lr_value = *lr;
    const float decay_factor =
        static_cast<float>(1.0 - lr_value * weight_decay);
    const float step_size =
        static_cast<float>(-lr_value / bias_correction1[step_index]);
    const float correction2 =
        static_cast<float>(bias_correction2_sqrt[step_index]);
    const float scale = *grad_scale;

    const auto tensor = metadata.block_to_tensor[blockIdx.x];
    const auto chunk = metadata.block_to_chunk[blockIdx.x];
    const auto count = metadata.numel_for_tensor[tensor] - chunk * chunk_size;
    auto* param = static_cast<float*>(
        const_cast<void*>(metadata.addresses[0][tensor])) + chunk * chunk_size;
    auto* grad = static_cast<float*>(
        const_cast<void*>(metadata.addresses[1][tensor])) + chunk * chunk_size;
    auto* exp_avg = static_cast<float*>(
        const_cast<void*>(metadata.addresses[2][tensor])) + chunk * chunk_size;
    auto* exp_avg_sq = static_cast<float*>(
        const_cast<void*>(metadata.addresses[3][tensor])) + chunk * chunk_size;

    for (int64_t index = threadIdx.x;
         index < count && index < chunk_size;
         index += blockDim.x) {
      float p = param[index];
      float m = exp_avg[index];
      float v = exp_avg_sq[index];
      const float scaled_grad = __fmul_rn(grad[index], scale);
      foreach_compatible_update(
          p,
          scaled_grad,
          m,
          v,
          decay_factor,
          beta2,
          first_moment_weight,
          second_moment_weight,
          eps,
          correction2,
          step_size);
      param[index] = p;
      exp_avg[index] = m;
      exp_avg_sq[index] = v;
    }
  }
};

void validate_tensor_lists(
    const std::vector<at::Tensor>& params,
    const std::vector<at::Tensor>& grads,
    const std::vector<at::Tensor>& exp_avgs,
    const std::vector<at::Tensor>& exp_avg_sqs) {
  TORCH_CHECK(!params.empty(), "Fused AdamW requires at least one parameter.");
  TORCH_CHECK(
      params.size() == grads.size() && params.size() == exp_avgs.size() &&
          params.size() == exp_avg_sqs.size(),
      "Fused AdamW tensor lists must have equal lengths.");
  const auto device = params.front().device();
  for (size_t index = 0; index < params.size(); ++index) {
    for (const auto& tensor :
         {params[index], grads[index], exp_avgs[index], exp_avg_sqs[index]}) {
      TORCH_CHECK(tensor.is_cuda(), "Fused AdamW only supports CUDA tensors.");
      TORCH_CHECK(
          tensor.scalar_type() == at::kFloat,
          "Precision-compatible fused AdamW only supports float32 tensors.");
      TORCH_CHECK(
          tensor.is_contiguous(),
          "Precision-compatible fused AdamW requires contiguous tensors.");
      TORCH_CHECK(
          tensor.device() == device,
          "Fused AdamW tensor lists must be on one CUDA device.");
    }
    TORCH_CHECK(
        params[index].numel() == grads[index].numel() &&
            params[index].numel() == exp_avgs[index].numel() &&
            params[index].numel() == exp_avg_sqs[index].numel(),
        "Fused AdamW tensors at each index must have equal sizes.");
  }
}

}  // namespace
}  // namespace at::native

void groot_n1_7_fused_adamw_eager_cuda(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    const double decay_factor,
    const double beta2,
    const double first_moment_weight,
    const double second_moment_weight,
    const double eps,
    const double bias_correction1,
    const double bias_correction2_sqrt,
    const double lr) {
  at::native::validate_tensor_lists(params, grads, exp_avgs, exp_avg_sqs);
  c10::cuda::CUDAGuard guard(params.front().device());
  std::vector<std::vector<at::Tensor>> tensor_lists{
      std::move(params), std::move(grads), std::move(exp_avgs),
      std::move(exp_avg_sqs)};
  at::native::multi_tensor_apply<4>(
      tensor_lists,
      at::native::GrootN17FusedAdamWEagerFunctor(),
      static_cast<float>(decay_factor),
      static_cast<float>(beta2),
      static_cast<float>(first_moment_weight),
      static_cast<float>(second_moment_weight),
      static_cast<float>(eps),
      static_cast<float>(bias_correction2_sqrt),
      static_cast<float>(-lr / bias_correction1));
}

void groot_n1_7_fused_adamw_capturable_cuda(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    at::Tensor lr,
    at::Tensor step,
    at::Tensor bias_correction1,
    at::Tensor bias_correction2_sqrt,
    const double beta2,
    const double first_moment_weight,
    const double second_moment_weight,
    const double eps,
    const double weight_decay) {
  at::native::validate_tensor_lists(params, grads, exp_avgs, exp_avg_sqs);
  TORCH_CHECK(lr.is_cuda() && lr.scalar_type() == at::kDouble && lr.numel() == 1);
  TORCH_CHECK(step.is_cuda() && step.scalar_type() == at::kLong && step.numel() == 1);
  TORCH_CHECK(
      bias_correction1.is_cuda() &&
      bias_correction1.scalar_type() == at::kDouble);
  TORCH_CHECK(
      bias_correction2_sqrt.is_cuda() &&
      bias_correction2_sqrt.scalar_type() == at::kDouble);
  c10::cuda::CUDAGuard guard(params.front().device());
  std::vector<std::vector<at::Tensor>> tensor_lists{
      std::move(params), std::move(grads), std::move(exp_avgs),
      std::move(exp_avg_sqs)};
  at::native::multi_tensor_apply<4>(
      tensor_lists,
      at::native::GrootN17FusedAdamWCapturableFunctor(),
      lr.const_data_ptr<double>(),
      step.const_data_ptr<int64_t>(),
      bias_correction1.const_data_ptr<double>(),
      bias_correction2_sqrt.const_data_ptr<double>(),
      static_cast<float>(beta2),
      static_cast<float>(first_moment_weight),
      static_cast<float>(second_moment_weight),
      static_cast<float>(eps),
      weight_decay);
}

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
    const double beta2,
    const double first_moment_weight,
    const double second_moment_weight,
    const double eps,
    const double weight_decay) {
  at::native::validate_tensor_lists(params, grads, exp_avgs, exp_avg_sqs);
  TORCH_CHECK(lr.is_cuda() && lr.scalar_type() == at::kDouble && lr.numel() == 1);
  TORCH_CHECK(step.is_cuda() && step.scalar_type() == at::kLong && step.numel() == 1);
  TORCH_CHECK(
      bias_correction1.is_cuda() &&
      bias_correction1.scalar_type() == at::kDouble);
  TORCH_CHECK(
      bias_correction2_sqrt.is_cuda() &&
      bias_correction2_sqrt.scalar_type() == at::kDouble);
  TORCH_CHECK(
      grad_scale.is_cuda() && grad_scale.scalar_type() == at::kFloat &&
      grad_scale.numel() == 1);
  TORCH_CHECK(grad_scale.device() == params.front().device());
  c10::cuda::CUDAGuard guard(params.front().device());
  std::vector<std::vector<at::Tensor>> tensor_lists{
      std::move(params), std::move(grads), std::move(exp_avgs),
      std::move(exp_avg_sqs)};
  at::native::multi_tensor_apply<4>(
      tensor_lists,
      at::native::GrootN17FusedAdamWCapturableGradScaledFunctor(),
      lr.const_data_ptr<double>(),
      step.const_data_ptr<int64_t>(),
      bias_correction1.const_data_ptr<double>(),
      bias_correction2_sqrt.const_data_ptr<double>(),
      grad_scale.const_data_ptr<float>(),
      static_cast<float>(beta2),
      static_cast<float>(first_moment_weight),
      static_cast<float>(second_moment_weight),
      static_cast<float>(eps),
      weight_decay);
}
