#pragma once

#include "../../vendor/deep_gemm_csrc/jit/compiler.hpp"
#include "../../vendor/deep_gemm_csrc/jit/device_runtime.hpp"
#include "../../vendor/deep_gemm_csrc/jit/kernel_runtime.hpp"
#include "../../vendor/deep_gemm_csrc/jit_kernels/heuristics/sm90.hpp"
#include "../../vendor/deep_gemm_csrc/jit_kernels/heuristics/sm100.hpp"
#include "../../vendor/deep_gemm_csrc/jit_kernels/impls/runtime_utils.hpp"

namespace deep_gemm
{

class SMXXFP8MQALogitsBwdRuntime final: public LaunchRuntime<SMXXFP8MQALogitsBwdRuntime>
{
public:
    struct Args
    {
        // Logits configs
        int seq_len;
        int seq_len_kv;
        int max_seqlen_k;
        int stride_logits;
        int num_heads, head_dim;
        int topk;
        bool is_compressed_logits;

        // Kernel configs
        int num_q_stages;
        int num_kv_stages;
        int block_q;
        int block_kv;

        // Kenrel inputs
        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        float* grad_q;
        float* grad_kv;
        float* grad_weights;
        float* grad_logits;
        cutlass::float_e4m3_t* kv;
        float* kv_scales;
        int* topk_indices;
        float softmax_scale;

        // Kernel TMAs
        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_weights;

        // Block configs
        int num_specialized_threads;
        int num_sparse_load_threads;
        int num_math_threads;

        // Kernel launch
        LaunchArgs launch_args;
    };

    static std::string generate_impl( const Args& args )
    {
        // TODO: optimize performance by tuning args
        // Block sizes are fixed in this kernel
        DG_HOST_ASSERT(128 % args.num_heads == 0);
        const auto& arch = device_runtime->get_arch(true);

        return fmt::format(R"(
#include <lightning_indexer_bwd/impls/sm{}_fp8_mqa_logits_bwd_sparse.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm{}_fp8_mqa_logits_bwd<
        {}, {},
        {},
        {}, {},
        {}, {},
        {}, {}, {}
    >);
}};
)",
            arch, arch,
            args.num_heads, args.head_dim,
            args.is_compressed_logits,
            args.block_q, args.block_kv,
            args.num_q_stages, args.num_kv_stages,
            args.num_specialized_threads, args.num_math_threads, args.num_sparse_load_threads
        );
    }

    static void launch_impl( const KernelHandle& kernel, const LaunchConfigHandle& config, Args args )
    {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config,
            // Parameters
            args.seq_len,
            args.seq_len_kv,
            args.max_seqlen_k,
            args.topk,
            static_cast<int64_t>(args.stride_logits),
            // Inputs
            args.kv,
            args.kv_scales,
            args.grad_logits,
            args.cu_seq_len_k_start,
            args.cu_seq_len_k_end,
            args.topk_indices,
            // Results
            args.grad_q,
            args.grad_kv,
            args.grad_weights,
            // TMA descs
            args.tensor_map_q,
            args.tensor_map_weights
        ));
    }
};

static void smxx_fp8_mqa_logits_bwd(
    const torch::Tensor& grad_logits,
    const torch::Tensor& q,
    const torch::Tensor& kv, const torch::Tensor& kv_scales,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end,
    const torch::Tensor& topk_indices,
    const torch::Tensor& grad_q,
    const torch::Tensor& grad_kv,
    const torch::Tensor& grad_weights,
    const int& seq_len, const int& seq_len_kv,
    const int& max_seqlen_k, const int& stride_logits,
    const int& num_heads, const int& head_dim,
    const int& topk,
    const int& seq_len_alignment
)
{
    // Now only support Blackwell with num_heads = 64 or 32, max_seqlen_k = 0.
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 10);
    DG_HOST_ASSERT(num_heads == 64 || num_heads == 32);
    DG_HOST_ASSERT(max_seqlen_k == 0);

    constexpr int block_q  = 1;
    const int block_qh = num_heads;  // block_q * num_heads = 1 * num_heads
    constexpr int block_kv = 128;
    constexpr int num_specialized_threads = 128;
    constexpr int num_sparse_load_threads = 128;
    constexpr int num_q_stages = 2, num_kv_stages = 2;
    const int num_math_threads = (device_runtime->get_arch_major() == 10 ? 128 : 512); // TODO: support Hopper if necessary.
    DG_HOST_ASSERT(block_qh % num_heads == 0);
    DG_HOST_ASSERT(seq_len_alignment % block_q == 0);

    // Not compressed for logits with no `max_seqlen_k`.
    const bool is_compressed_logits = false;

    // Gmem ptrs
    int* cu_seq_len_k_start_ptr   = cu_seq_len_k_start.data_ptr<int>();
    int* cu_seq_len_k_end_ptr     = cu_seq_len_k_end.data_ptr<int>();
    float* grad_q_ptr             = grad_q.data_ptr<float>();
    float* grad_kv_ptr            = grad_kv.data_ptr<float>();
    float* grad_weights_ptr       = grad_weights.data_ptr<float>();
    float* grad_logits_ptr        = grad_logits.data_ptr<float>();
    cutlass::float_e4m3_t* kv_ptr = reinterpret_cast<cutlass::float_e4m3_t*>(kv.data_ptr());
    float* kv_scales_ptr          = kv_scales.data_ptr<float>();
    int* topk_indices_ptr         = topk_indices.data_ptr<int>();

    // Construct TMAs for q/weights.
    const auto& tensor_map_q           = make_tma_2d_desc(q, head_dim, seq_len * num_heads, head_dim, block_qh, head_dim, head_dim);
    const auto& tensor_map_weights     = make_tma_2d_desc(weights, num_heads, seq_len, num_heads, block_q, num_heads, 0);

    // smem_q_bf16 and smem_reshape_d_logit_bf16 inner dim is padded to 64 for 128B UMMA alignment.
    const int num_heads_padded = num_heads < 64 ? 64 : num_heads;

    // Calculate shared memory size per stage.
    const int q_size_per_stage                    = block_q * num_heads * head_dim * static_cast<int>(q.element_size());
    const int q_bf16_per_stage                    = block_q * num_heads_padded * head_dim * static_cast<int>(sizeof(cutlass::bfloat16_t));
    const int weight_size_per_stage               = block_q * num_heads * static_cast<int>(weights.element_size());
    const int kv_size_per_stage                   = block_kv * head_dim * static_cast<int>(kv.element_size());
    const int kv_bf16_per_stage                   = block_kv * head_dim * static_cast<int>(sizeof(cutlass::bfloat16_t));
    const int kv_scale_size_per_stage             = block_kv * static_cast<int>(kv_scales.element_size());
    const int topk_indices_size_per_stage         = block_q * block_kv * static_cast<int>(grad_logits.element_size());
    const int grad_qk_logit_per_stage             = block_q * num_heads_padded * block_kv * static_cast<int>(sizeof(cutlass::bfloat16_t));
    const int reshape_grad_qk_logit_per_stage     = block_q * num_heads_padded * block_kv * static_cast<int>(sizeof(cutlass::bfloat16_t));

    // Calculate shared memory size.
    int smem_size = 0;
    smem_size += num_q_stages  * q_size_per_stage;
    smem_size += num_q_stages  * q_bf16_per_stage;
    smem_size += num_q_stages  * weight_size_per_stage;
    smem_size += num_kv_stages * kv_size_per_stage;
    smem_size += num_kv_stages * kv_bf16_per_stage;
    smem_size += num_kv_stages * kv_scale_size_per_stage;
    smem_size += num_kv_stages * topk_indices_size_per_stage;
    smem_size += num_kv_stages * grad_qk_logit_per_stage;
    smem_size += num_kv_stages * reshape_grad_qk_logit_per_stage;
    smem_size += num_q_stages  * 2 * 8; // barrier_q
    smem_size += num_kv_stages * 2 * 8; // barrier_kv
    smem_size += (num_math_threads / 128) * 2 * 8; // barrier_mma_fp8
    smem_size += num_kv_stages * 2 * 8; // barrier_mma_bf16
    // smem_size += 1 * 2 * 8; // barrier_tmem_dq, num_q_stages use the same one buffer
    smem_size += 4; // TMEM ptr.
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Launch
    const SMXXFP8MQALogitsBwdRuntime::Args& args = {
        .seq_len = seq_len,
        .seq_len_kv = seq_len_kv,
        .max_seqlen_k = max_seqlen_k,
        .stride_logits = stride_logits,
        .num_heads = num_heads, .head_dim = head_dim,
        .topk = topk,
        .is_compressed_logits = is_compressed_logits,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .block_q = block_q,
        .block_kv = block_kv,
        .cu_seq_len_k_start = cu_seq_len_k_start_ptr,
        .cu_seq_len_k_end = cu_seq_len_k_end_ptr,
        .grad_q = grad_q_ptr,
        .grad_kv = grad_kv_ptr,
        .grad_weights = grad_weights_ptr,
        .grad_logits = grad_logits_ptr,
        .kv = kv_ptr,
        .kv_scales = kv_scales_ptr,
        .topk_indices = topk_indices_ptr,
        .tensor_map_q = tensor_map_q,
        .tensor_map_weights = tensor_map_weights,
        .num_specialized_threads = num_specialized_threads,
        .num_sparse_load_threads = num_sparse_load_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(
            device_runtime->get_num_sms(),
            num_specialized_threads + num_math_threads + num_sparse_load_threads,
            smem_size
        )
    };
    const auto& code = SMXXFP8MQALogitsBwdRuntime::generate(args);
    const auto& runtime = compiler->build("smxx_fp8_mqa_logits_bwd", code);
    SMXXFP8MQALogitsBwdRuntime::launch(runtime, args);
}

} // namespace deep_gemm
