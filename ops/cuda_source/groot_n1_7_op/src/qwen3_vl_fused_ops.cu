#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

template <typename scalar_t>
__global__ void qwen3_vl_fused_vision_rope_kernel(
    const scalar_t* query,
    const scalar_t* key,
    const float* cos,
    const float* sin,
    scalar_t* query_out,
    scalar_t* key_out,
    const int64_t elements,
    const int64_t heads,
    const int64_t head_dim,
    const int64_t query_stride0,
    const int64_t query_stride1,
    const int64_t query_stride2,
    const int64_t key_stride0,
    const int64_t key_stride1,
    const int64_t key_stride2,
    const int64_t cos_stride0,
    const int64_t cos_stride1,
    const int64_t sin_stride0,
    const int64_t sin_stride1) {
  const int64_t half_dim = head_dim / 2;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t dim = index % head_dim;
    const int64_t row = index / head_dim;
    const int64_t head = row % heads;
    const int64_t sequence = row / heads;
    const int64_t rotated_dim = dim < half_dim ? dim + half_dim : dim - half_dim;

    const int64_t query_offset = sequence * query_stride0 +
        head * query_stride1 + dim * query_stride2;
    const int64_t query_rotated_offset = sequence * query_stride0 +
        head * query_stride1 + rotated_dim * query_stride2;
    const int64_t key_offset = sequence * key_stride0 +
        head * key_stride1 + dim * key_stride2;
    const int64_t key_rotated_offset = sequence * key_stride0 +
        head * key_stride1 + rotated_dim * key_stride2;
    const int64_t cos_offset = sequence * cos_stride0 + dim * cos_stride1;
    const int64_t sin_offset = sequence * sin_stride0 + dim * sin_stride1;

    const float cos_value = cos[cos_offset];
    const float sin_value = sin[sin_offset];
    const float query_value = static_cast<float>(query[query_offset]);
    const float key_value = static_cast<float>(key[key_offset]);
    float query_rotated = static_cast<float>(query[query_rotated_offset]);
    float key_rotated = static_cast<float>(key[key_rotated_offset]);
    if (dim < half_dim) {
      query_rotated = __fsub_rn(0.0F, query_rotated);
      key_rotated = __fsub_rn(0.0F, key_rotated);
    }

    const float query_left = __fmul_rn(query_value, cos_value);
    const float query_right = __fmul_rn(query_rotated, sin_value);
    const float key_left = __fmul_rn(key_value, cos_value);
    const float key_right = __fmul_rn(key_rotated, sin_value);
    query_out[index] = static_cast<scalar_t>(__fadd_rn(query_left, query_right));
    key_out[index] = static_cast<scalar_t>(__fadd_rn(key_left, key_right));
  }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t rounded_mul(float left, float right) {
  return static_cast<scalar_t>(__fmul_rn(left, right));
}

template <typename input_t, typename output_t>
__device__ __forceinline__ output_t text_rope_value(
    const input_t* input,
    const input_t* cos,
    const input_t* sin,
    int64_t input_offset,
    int64_t rotated_offset,
    int64_t cos_offset,
    int64_t sin_offset,
    bool negate_rotated) {
  // cos/sin are broadcast across all heads.
  const float value = static_cast<float>(input[input_offset]);
  float rotated = static_cast<float>(input[rotated_offset]);
  if (negate_rotated) {
    rotated = __fsub_rn(0.0F, rotated);
  }
  const input_t left = rounded_mul<input_t>(
      value, static_cast<float>(cos[cos_offset]));
  const input_t right = rounded_mul<input_t>(
      rotated, static_cast<float>(sin[sin_offset]));
  return static_cast<output_t>(__fadd_rn(
      static_cast<float>(left), static_cast<float>(right)));
}

template <typename input_t, typename output_t>
__global__ void qwen3_vl_fused_text_rope_kernel(
    const input_t* query,
    const input_t* key,
    const input_t* cos,
    const input_t* sin,
    output_t* query_out,
    output_t* key_out,
    int64_t query_elements,
    int64_t key_elements,
    int64_t query_heads,
    int64_t key_heads,
    int64_t sequence_length,
    int64_t head_dim,
    int64_t query_stride0,
    int64_t query_stride1,
    int64_t query_stride2,
    int64_t query_stride3,
    int64_t key_stride0,
    int64_t key_stride1,
    int64_t key_stride2,
    int64_t key_stride3,
    int64_t cos_stride0,
    int64_t cos_stride1,
    int64_t cos_stride2,
    int64_t sin_stride0,
    int64_t sin_stride1,
    int64_t sin_stride2) {
  const int64_t half_dim = head_dim / 2;
  // Query loop
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < query_elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t dim = index % head_dim;
    const int64_t sequence = (index / head_dim) % sequence_length;
    const int64_t head = (index / (head_dim * sequence_length)) % query_heads;
    const int64_t batch = index / (head_dim * sequence_length * query_heads);
    const int64_t rotated_dim = dim < half_dim ? dim + half_dim : dim - half_dim;
    const int64_t input_offset = batch * query_stride0 + head * query_stride1 +
        sequence * query_stride2 + dim * query_stride3;
    const int64_t rotated_offset = batch * query_stride0 + head * query_stride1 +
        sequence * query_stride2 + rotated_dim * query_stride3;
    const int64_t cos_offset = batch * cos_stride0 + sequence * cos_stride1 +
        dim * cos_stride2;
    const int64_t sin_offset = batch * sin_stride0 + sequence * sin_stride1 +
        dim * sin_stride2;
    query_out[index] = text_rope_value<input_t, output_t>(
        query, cos, sin, input_offset, rotated_offset,
        cos_offset, sin_offset, dim < half_dim);
  }
  // Key loop
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < key_elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t dim = index % head_dim;
    const int64_t sequence = (index / head_dim) % sequence_length;
    const int64_t head = (index / (head_dim * sequence_length)) % key_heads;
    const int64_t batch = index / (head_dim * sequence_length * key_heads);
    const int64_t rotated_dim = dim < half_dim ? dim + half_dim : dim - half_dim;
    const int64_t input_offset = batch * key_stride0 + head * key_stride1 +
        sequence * key_stride2 + dim * key_stride3;
    const int64_t rotated_offset = batch * key_stride0 + head * key_stride1 +
        sequence * key_stride2 + rotated_dim * key_stride3;
    const int64_t cos_offset = batch * cos_stride0 + sequence * cos_stride1 +
        dim * cos_stride2;
    const int64_t sin_offset = batch * sin_stride0 + sequence * sin_stride1 +
        dim * sin_stride2;
    key_out[index] = text_rope_value<input_t, output_t>(
        key, cos, sin, input_offset, rotated_offset,
        cos_offset, sin_offset, dim < half_dim);
  }
}

template <typename scalar_t>
__global__ void qwen3_vl_fused_text_rms_norm_square_kernel(
    const scalar_t* input,
    float* output,
    int64_t elements) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float value = static_cast<float>(input[index]);
    output[index] = __fmul_rn(value, value);
  }
}

// Forward declaration of helper used by the finish kernel below.
template <typename scalar_t>
__device__ __forceinline__ float qwen3_vl_fused_text_rms_norm_round_to_input(float value) {
  return static_cast<float>(static_cast<scalar_t>(value));
}

template <>
__device__ __forceinline__ float qwen3_vl_fused_text_rms_norm_round_to_input<float>(float value) {
  return value;
}

template <typename scalar_t>
__global__ void qwen3_vl_fused_text_rms_norm_finish_kernel(
    const scalar_t* input,
    const float* variance,
    const float* weight,
    float* output,
    int64_t elements,
    int64_t hidden_size,
    float epsilon) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = index / hidden_size;
    const int64_t column = index - row * hidden_size;
    const float inverse_rms = rsqrtf(__fadd_rn(variance[row], epsilon));
    const float normalized = __fmul_rn(
        static_cast<float>(input[index]), inverse_rms);
    const float rounded = qwen3_vl_fused_text_rms_norm_round_to_input<scalar_t>(normalized);
    output[index] = __fmul_rn(weight[column], rounded);
  }
}

template <typename scalar_t>
__global__ void qwen3_vl_fused_text_silu_mul_kernel(
    const scalar_t* gate,
    const scalar_t* up,
    scalar_t* output,
    int64_t elements) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float value = static_cast<float>(gate[index]);
    const float denominator = __fadd_rn(1.0F, expf(-value));
    const scalar_t activated = static_cast<scalar_t>(__fdiv_rn(value, denominator));
    output[index] = static_cast<scalar_t>(__fmul_rn(
        static_cast<float>(activated),
        static_cast<float>(up[index])));
  }
}

void validate_inputs(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& cos,
    const at::Tensor& sin) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda(), "Qwen vision RoPE q/k must be CUDA tensors.");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda(), "Qwen vision RoPE cos/sin must be CUDA tensors.");
  TORCH_CHECK(query.device() == key.device() && query.device() == cos.device() &&
      query.device() == sin.device(), "Qwen vision RoPE tensors must share one CUDA device.");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3, "Qwen vision RoPE q/k must be rank 3.");
  TORCH_CHECK(query.sizes() == key.sizes(), "Qwen vision RoPE q/k shapes must match.");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "Qwen vision RoPE q/k dtypes must match.");
  TORCH_CHECK(
      query.scalar_type() == at::kFloat || query.scalar_type() == at::kHalf ||
          query.scalar_type() == at::kBFloat16,
      "Qwen vision RoPE supports fp32, fp16, and bf16 q/k tensors.");
  TORCH_CHECK(cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat,
      "Qwen vision RoPE cos/sin must be fp32.");
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2, "Qwen vision RoPE cos/sin must be rank 2.");
  TORCH_CHECK(cos.sizes() == sin.sizes(), "Qwen vision RoPE cos/sin shapes must match.");
  TORCH_CHECK(cos.size(0) == query.size(0) && cos.size(1) == query.size(2),
      "Qwen vision RoPE cos/sin shape must be [sequence, head_dim].");
  TORCH_CHECK(query.size(2) % 2 == 0, "Qwen vision RoPE head_dim must be even.");
}

void validate_text_inputs(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& cos,
    const at::Tensor& sin) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda(), "Qwen text RoPE q/k must be CUDA tensors.");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda(), "Qwen text RoPE cos/sin must be CUDA tensors.");
  TORCH_CHECK(query.device() == key.device() && query.device() == cos.device() &&
      query.device() == sin.device(), "Qwen text RoPE tensors must share one CUDA device.");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4, "Qwen text RoPE q/k must be rank 4.");
  TORCH_CHECK(query.size(0) == key.size(0) && query.size(2) == key.size(2) &&
      query.size(3) == key.size(3), "Qwen text RoPE q/k batch, sequence, and head dimensions must match.");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() && query.scalar_type() == cos.scalar_type() &&
      query.scalar_type() == sin.scalar_type(), "Qwen text RoPE tensors must share one dtype.");
  TORCH_CHECK(
      query.scalar_type() == at::kFloat || query.scalar_type() == at::kHalf ||
          query.scalar_type() == at::kBFloat16,
      "Qwen text RoPE supports fp32, fp16, and bf16 tensors.");
  TORCH_CHECK(cos.dim() == 3 && sin.dim() == 3, "Qwen text RoPE cos/sin must be rank 3.");
  TORCH_CHECK(cos.sizes() == sin.sizes(), "Qwen text RoPE cos/sin shapes must match.");
  TORCH_CHECK(cos.size(0) == query.size(0) && cos.size(1) == query.size(2) &&
      cos.size(2) == query.size(3), "Qwen text RoPE cos/sin shape must be [batch, sequence, head_dim].");
  TORCH_CHECK(query.size(3) % 2 == 0, "Qwen text RoPE head_dim must be even.");
}

void validate_rmsnorm_input(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "Qwen RMSNorm input must be a CUDA tensor.");
  TORCH_CHECK(input.is_contiguous(), "Qwen RMSNorm input must be contiguous.");
  TORCH_CHECK(input.dim() >= 1 && input.size(-1) > 0,
      "Qwen RMSNorm input must have a non-empty final dimension.");
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf ||
          input.scalar_type() == at::kBFloat16,
      "Qwen RMSNorm supports fp32, fp16, and bf16 inputs.");
}

}  // namespace

std::vector<at::Tensor> qwen3_vl_fused_vision_rope_cuda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& cos,
    const at::Tensor& sin) {
  validate_inputs(query, key, cos, sin);
  c10::cuda::CUDAGuard guard(query.device());
  auto query_out = at::empty(query.sizes(), query.options());
  auto key_out = at::empty(key.sizes(), key.options());
  const int64_t elements = query.numel();
  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>(
      (elements + threads - 1) / threads, 65535));
  const auto stream = c10::cuda::getCurrentCUDAStream(query.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "qwen3_vl_fused_vision_rope",
      [&] {
        qwen3_vl_fused_vision_rope_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            query.const_data_ptr<scalar_t>(),
            key.const_data_ptr<scalar_t>(),
            cos.const_data_ptr<float>(),
            sin.const_data_ptr<float>(),
            query_out.mutable_data_ptr<scalar_t>(),
            key_out.mutable_data_ptr<scalar_t>(),
            elements,
            query.size(1),
            query.size(2),
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            cos.stride(0),
            cos.stride(1),
            sin.stride(0),
            sin.stride(1));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {query_out, key_out};
}

std::vector<at::Tensor> qwen3_vl_fused_text_rope_cuda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& cos,
    const at::Tensor& sin) {
  validate_text_inputs(query, key, cos, sin);
  c10::cuda::CUDAGuard guard(query.device());
  auto query_out = at::empty(query.sizes(), query.options());
  auto key_out = at::empty(key.sizes(), key.options());
  const int64_t query_elements = query.numel();
  const int64_t key_elements = key.numel();
  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>(
      (std::max(query_elements, key_elements) + threads - 1) / threads, 65535));
  const auto stream = c10::cuda::getCurrentCUDAStream(query.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "qwen3_vl_fused_text_rope",
      [&] {
        qwen3_vl_fused_text_rope_kernel<scalar_t, scalar_t><<<blocks, threads, 0, stream>>>(
            query.const_data_ptr<scalar_t>(),
            key.const_data_ptr<scalar_t>(),
            cos.const_data_ptr<scalar_t>(),
            sin.const_data_ptr<scalar_t>(),
            query_out.mutable_data_ptr<scalar_t>(),
            key_out.mutable_data_ptr<scalar_t>(),
            query_elements,
            key_elements,
            query.size(1),
            key.size(1),
            query.size(2),
            query.size(3),
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            cos.stride(0),
            cos.stride(1),
            cos.stride(2),
            sin.stride(0),
            sin.stride(1),
            sin.stride(2));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {query_out, key_out};
}

at::Tensor qwen3_vl_fused_text_rms_norm_square_cuda(const at::Tensor& input) {
  validate_rmsnorm_input(input);
  c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty(input.sizes(), input.options().dtype(at::kFloat));
  const int64_t elements = input.numel();
  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>(
      (elements + threads - 1) / threads, 65535));
  const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "qwen3_vl_fused_text_rms_norm_square",
      [&] {
        qwen3_vl_fused_text_rms_norm_square_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            input.const_data_ptr<scalar_t>(),
            output.mutable_data_ptr<float>(),
            elements);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor qwen3_vl_fused_text_rms_norm_finish_cuda(
    const at::Tensor& input,
    const at::Tensor& variance,
    const at::Tensor& weight,
    double epsilon) {
  validate_rmsnorm_input(input);
  TORCH_CHECK(variance.is_cuda() && weight.is_cuda(),
      "Qwen RMSNorm variance and weight must be CUDA tensors.");
  TORCH_CHECK(input.device() == variance.device() && input.device() == weight.device(),
      "Qwen RMSNorm tensors must share one CUDA device.");
  TORCH_CHECK(variance.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat,
      "Qwen RMSNorm variance and weight must be fp32.");
  TORCH_CHECK(variance.is_contiguous() && weight.is_contiguous(),
      "Qwen RMSNorm variance and weight must be contiguous.");
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == input.size(-1),
      "Qwen RMSNorm weight must match the final input dimension.");
  TORCH_CHECK(variance.numel() == input.numel() / input.size(-1),
      "Qwen RMSNorm variance must contain one value per input row.");

  c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty(input.sizes(), input.options().dtype(at::kFloat));
  const int64_t elements = input.numel();
  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>(
      (elements + threads - 1) / threads, 65535));
  const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "qwen3_vl_fused_text_rms_norm_finish",
      [&] {
        qwen3_vl_fused_text_rms_norm_finish_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            input.const_data_ptr<scalar_t>(),
            variance.const_data_ptr<float>(),
            weight.const_data_ptr<float>(),
            output.mutable_data_ptr<float>(),
            elements,
            input.size(-1),
            static_cast<float>(epsilon));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor qwen3_vl_fused_text_silu_mul_cuda(
    const at::Tensor& gate,
    const at::Tensor& up) {
  TORCH_CHECK(gate.is_cuda() && up.is_cuda(),
      "Qwen SiLU multiply inputs must be CUDA tensors.");
  TORCH_CHECK(gate.device() == up.device() && gate.sizes() == up.sizes(),
      "Qwen SiLU multiply inputs must share device and shape.");
  TORCH_CHECK(gate.scalar_type() == up.scalar_type(),
      "Qwen SiLU multiply inputs must share one dtype.");
  TORCH_CHECK(gate.scalar_type() == at::kHalf || gate.scalar_type() == at::kBFloat16,
      "Qwen SiLU multiply supports fp16 and bf16 inputs.");
  TORCH_CHECK(gate.is_contiguous() && up.is_contiguous(),
      "Qwen SiLU multiply inputs must be contiguous.");

  c10::cuda::CUDAGuard guard(gate.device());
  auto output = at::empty(gate.sizes(), gate.options());
  const int64_t elements = gate.numel();
  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>(
      (elements + threads - 1) / threads, 65535));
  const auto stream = c10::cuda::getCurrentCUDAStream(gate.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gate.scalar_type(),
      "qwen3_vl_fused_text_silu_mul",
      [&] {
        qwen3_vl_fused_text_silu_mul_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            gate.const_data_ptr<scalar_t>(),
            up.const_data_ptr<scalar_t>(),
            output.mutable_data_ptr<scalar_t>(),
            elements);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
