#pragma once

#include "cutlass/bfloat16.h"

struct SparseAttnFwdParams {
    int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
    int q_start_index_s;
    float sm_scale, sm_scale_div_log2;

    // Input tensors
    cutlass::bfloat16_t* __restrict__ q;    // [s_q, h_q, d_qk]
    cutlass::bfloat16_t* __restrict__ kv;   // [s_kv, h_kv, d_qk]
    int* __restrict__ indices;              // [s_q, h_kv, topk]
    float* __restrict__ attn_sink;          // [h_q], may be nullptr
    int* __restrict__ topk_length;          // [s_q], may be nullptr

    // Strides
    int stride_q_s_q; int stride_q_h_q;
    int stride_kv_s_kv; int stride_kv_h_kv;
    int stride_indices_s_q; int stride_indices_h_kv;

    // Output tensors
    cutlass::bfloat16_t* __restrict__ out;  // [s_q, h_q, d_v]
    float* __restrict__ max_logits;         // [s_q, h_q]
    float* __restrict__ lse;                // [s_q, h_q]
    float* __restrict__ p_out;             // [s_q, h_q, topk], may be nullptr

    int num_sm;
    cudaStream_t stream;
};
