#pragma once

#include "cutlass/bfloat16.h"

// Sparse attention backward parameter struct
struct SparseAttnBwdParams {
    int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
    int q_start_index_s;
    float sm_scale, sm_scale_div_log2;

    // Input tensors
    cutlass::bfloat16_t* __restrict__ q;     // [s_q, h_q, d_qk] Query
    cutlass::bfloat16_t* __restrict__ kv;    // [s_kv, h_kv, d_qk] Key/Value
    cutlass::bfloat16_t* __restrict__ o;     // [s_q, h_q, d_v] Forward output (used to compute delta)
    cutlass::bfloat16_t* __restrict__ dO;    // [s_q, h_q, d_v] Output gradient
    int* __restrict__ indices;               // [s_q, h_kv, topk] TopK indices
    float* __restrict__ lse;                 // [s_q, h_q] Log-Sum-Exp (from forward)
    int* __restrict__ topk_length;           // [s_q], may be nullptr

    // Strides
    int stride_q_s_q; int stride_q_h_q;
    int stride_kv_s_kv; int stride_kv_h_kv;
    int stride_o_s_q; int stride_o_h_q;
    int stride_dO_s_q; int stride_dO_h_q;
    int stride_indices_s_q; int stride_indices_h_kv;

    // Output tensors
    cutlass::bfloat16_t* __restrict__ dQ;    // [s_q, h_q, d_qk] Query gradient
    float* __restrict__ dKV;                 // [s_kv, h_kv, d_qk] KV gradient (float32 accumulation)
    float* __restrict__ delta;               // [s_q, h_q] Delta = sum(O * dO, dim=-1)
    int stride_dQ_s_q; int stride_dQ_h_q;
    int stride_dKV_s_kv; int stride_dKV_h_kv;
    int stride_delta_s_q; int stride_delta_h_q;

    int num_sm;
    cudaStream_t stream;
};