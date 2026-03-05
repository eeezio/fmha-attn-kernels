// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>
#include <random>
#include <iomanip>
#include <map>

// clang-format off
// /opt/rocm/llvm/bin/clang++ -O3 -x hip --offload-arch=gfx950 -o attn_bwd attn_bwd.cpp && ./attn_bwd
// clang-format on

#define HIP_CHECK(call)                                                                    \
    do                                                                                     \
    {                                                                                      \
        hipError_t err = call;                                                             \
        if(err != hipSuccess)                                                              \
        {                                                                                  \
            printf("HIP error %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(1);                                                                       \
        }                                                                                  \
    } while(0)

enum class CausalMaskType
{
    DISABLE      = 0,
    TOP_LEFT     = 1,
    BOTTOM_RIGHT = 2
};

std::map<CausalMaskType, std::string> CausalMaskTypeName = {
    {CausalMaskType::DISABLE, "DISABLE"},
    {CausalMaskType::TOP_LEFT, "TOP_LEFT"},
    {CausalMaskType::BOTTOM_RIGHT, "BOTTOM_RIGHT"}};

template <int BS,
          int HEAD_NUM,
          int MAX_SEQ_KV,
          int HEAD_DIM,
          int STEP2_BLOCK_SIZE     = 256,
          bool ENABLE_DROPOUT_MASK = true,
          CausalMaskType MAKS_TYPE = CausalMaskType::DISABLE>
struct FmhaKernelConfig
{
    static constexpr int bs                        = BS;
    static constexpr int head_num                  = HEAD_NUM;
    // Each batch's Q seq length is either 0 or 1 at runtime.
    // Storage is always allocated for seq_q=1 (padded), so the static
    // layout dimension is 1.
    static constexpr int seq_q                     = 1;
    static constexpr int max_seq_kv                = MAX_SEQ_KV;
    static constexpr int head_dim                  = HEAD_DIM;
    static constexpr int step2_block_size          = STEP2_BLOCK_SIZE;
    static constexpr bool enable_dropout_mask      = ENABLE_DROPOUT_MASK;
    static constexpr enum CausalMaskType mask_type = MAKS_TYPE;
};

// Kernel 1: Compute grad_V = attn_weights^T @ grad_O
template <typename T, typename Config, int TASKS_PER_BLOCK = 1, int BLOCK_K = 16>
__global__ void compute_grad_v_kernel(const T* attn_weights,
                                      const T* grad_O,
                                      T* grad_V,
                                      const int* cu_seqlens_q,
                                      const int* cu_seqlens_q_padded,
                                      const int* cu_seqlens_kv,
                                      const int* cu_seqlens_kv_padded)
{
    constexpr int seq_q                 = Config::seq_q; // == 1
    constexpr int max_seq_kv            = Config::max_seq_kv;
    constexpr int head_dim              = Config::head_dim;
    constexpr int block_k               = BLOCK_K;
    constexpr int dwordx4_load_elt      = 16 / sizeof(T);
    constexpr int warp_size             = 64;
    constexpr int process_head_per_warp = warp_size / (head_dim / block_k);
    constexpr int tasks_per_block       = TASKS_PER_BLOCK;

    int base_block_offset   = blockIdx.x * process_head_per_warp * tasks_per_block;
    int thread_id           = threadIdx.x;
    int thread_batch_offset = thread_id / (head_dim / block_k);
    int thread_head_offset  = thread_id % (head_dim / block_k) * block_k;

    uint4 load_dwordx4_tmp_var[block_k / dwordx4_load_elt];
    T attn[max_seq_kv];

    for(int task = 0; task < tasks_per_block; task++)
    {
        int block_batch_head_idx = base_block_offset + task * process_head_per_warp;
        int cur_idx              = block_batch_head_idx + thread_batch_offset;

        int batch_idx    = cur_idx / (Config::seq_q * Config::head_num);
        int seq_head_idx = cur_idx % (Config::seq_q * Config::head_num);
        int head_idx     = seq_head_idx % Config::head_num;

        if(batch_idx >= Config::bs)
            continue;

        // Skip batches where actual Q seq is 0 — no grad_O to read from.
        int actual_seq_q = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx];
        if(actual_seq_q == 0)
            continue;

        int seq_kv           = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
        int q_storage_offset = cu_seqlens_q_padded[batch_idx]; // seq_q_idx == 0

        // attn_weights layout: [total_padded_q, head_num, max_seq_kv]
        int attn_offset = (q_storage_offset * Config::head_num + head_idx) * max_seq_kv;
#pragma unroll
        for(int i = 0; i < max_seq_kv; i++)
            attn[i] = attn_weights[attn_offset + i];

        // Compute grad_V = attn_weights^T @ grad_O
        for(int j = 0; j < seq_kv; j++)
        {
            uint4 store_dwordx4_tmp_var[block_k / dwordx4_load_elt];
#pragma unroll
            for(int i = 0; i < block_k / dwordx4_load_elt; i++)
            {
                store_dwordx4_tmp_var[i].x = 0;
                store_dwordx4_tmp_var[i].y = 0;
                store_dwordx4_tmp_var[i].z = 0;
                store_dwordx4_tmp_var[i].w = 0;
            }

            // grad_O layout: [total_padded_seq_q, head_num, head_dim]
#pragma unroll
            for(int i = 0; i < block_k / dwordx4_load_elt; i++)
            {
                load_dwordx4_tmp_var[i] =
                    *((uint4*)&grad_O[(q_storage_offset * Config::head_num + head_idx) * head_dim +
                                      thread_head_offset + i * dwordx4_load_elt]);
            }

#pragma unroll
            for(int b = 0; b < block_k; b++)
            {
                ((T*)&store_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt] +=
                    attn[j] *
                    ((T*)&load_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt];
            }

#pragma unroll
            for(int i = 0; i < block_k / dwordx4_load_elt; i++)
            {
                int grad_v_idx =
                    (cu_seqlens_kv_padded[batch_idx] + j) * Config::head_num * head_dim +
                    head_idx * head_dim + thread_head_offset + i * dwordx4_load_elt;
                *((uint4*)&grad_V[grad_v_idx]) = store_dwordx4_tmp_var[i];
            }
        }
    }
}

// Kernel 2: Compute grad_attn = grad_O @ V^T (similar to compute_scores_kernel)
template <typename T, typename Config, int TASKS_PER_BLOCK = 16>
__global__ void compute_grad_attn_kernel(const T* grad_O,
                                         const T* V,
                                         T* grad_attn,
                                         const int* cu_seqlens_q,
                                         const int* cu_seqlens_q_padded,
                                         const int* cu_seqlens_kv,
                                         const int* cu_seqlens_kv_padded)
{
    constexpr int seq_q = Config::seq_q; // == 1
    static_assert(seq_q == 1, "seq_q must be 1 for this kernel implementation.");
    constexpr int max_seq_kv        = Config::max_seq_kv;
    constexpr int head_dim          = Config::head_dim;
    constexpr int block_k           = 64;
    constexpr int thread_block_size = 64;
    constexpr int tasks_per_block   = TASKS_PER_BLOCK;

    int base_block_offset = blockIdx.x * thread_block_size * tasks_per_block;
    int thread_id         = threadIdx.x;

    for(int task = 0; task < tasks_per_block; task++)
    {
        int cur_batch_idx = base_block_offset + task * thread_block_size + thread_id;
        int batch_idx     = cur_batch_idx / (Config::seq_q * Config::head_num);
        int seq_head_idx  = cur_batch_idx % (Config::seq_q * Config::head_num);
        int head_idx      = seq_head_idx % Config::head_num;

        if(batch_idx >= Config::bs)
            continue;

        // Skip batches where actual Q seq is 0 — no row exists in workspace for them.
        int actual_seq_q = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx];
        if(actual_seq_q == 0)
            continue;

        int seq_kv           = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
        int q_storage_offset = cu_seqlens_q_padded[batch_idx]; // seq_idx == 0

        float results[max_seq_kv];
        T fetch_grad_O[block_k];
        T fetch_V[block_k];

        // grad_O layout: [total_padded_seq_q, head_num, head_dim]
        T* grad_O_ptr =
            (T*)&grad_O[(q_storage_offset * Config::head_num + head_idx) * head_dim];

        const T* V_base =
            &V[cu_seqlens_kv_padded[batch_idx] * Config::head_num * head_dim + head_idx * head_dim];
        int V_stride = Config::head_num * head_dim;

        // workspace layout: [total_padded_q, head_num, max_seq_kv]
        T* grad_attn_ptr = (T*)&grad_attn[(q_storage_offset * Config::head_num + head_idx) * max_seq_kv];

        uint4 ls_dwordx4_tmp_var;

        for(int i = 0; i < seq_kv; i++)
            results[i] = 0.0f;

        for(int dim_offset = 0; dim_offset < head_dim; dim_offset += block_k)
        {
            if constexpr(std::is_same<T, hip_bfloat16>::value)
            {
                for(int k = 0; k < block_k / 8; k++)
                {
                    ls_dwordx4_tmp_var      = *((uint4*)&grad_O_ptr[dim_offset + k * 8]);
                    fetch_grad_O[k * 8 + 0] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.x)[0];
                    fetch_grad_O[k * 8 + 1] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.x)[1];
                    fetch_grad_O[k * 8 + 2] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.y)[0];
                    fetch_grad_O[k * 8 + 3] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.y)[1];
                    fetch_grad_O[k * 8 + 4] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.z)[0];
                    fetch_grad_O[k * 8 + 5] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.z)[1];
                    fetch_grad_O[k * 8 + 6] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.w)[0];
                    fetch_grad_O[k * 8 + 7] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.w)[1];
                }

                for(int kv_idx = 0; kv_idx < seq_kv; kv_idx++)
                {
                    for(int k = 0; k < block_k / 8; k++)
                    {
                        ls_dwordx4_tmp_var =
                            *((uint4*)&V_base[kv_idx * V_stride + dim_offset + k * 8]);
                        fetch_V[k * 8 + 0] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.x)[0];
                        fetch_V[k * 8 + 1] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.x)[1];
                        fetch_V[k * 8 + 2] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.y)[0];
                        fetch_V[k * 8 + 3] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.y)[1];
                        fetch_V[k * 8 + 4] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.z)[0];
                        fetch_V[k * 8 + 5] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.z)[1];
                        fetch_V[k * 8 + 6] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.w)[0];
                        fetch_V[k * 8 + 7] = ((hip_bfloat16*)&ls_dwordx4_tmp_var.w)[1];
                    }
#pragma unroll
                    for(int k = 0; k < block_k; k++)
                    {
                        results[kv_idx] +=
                            static_cast<float>(fetch_grad_O[k]) * static_cast<float>(fetch_V[k]);
                    }
                }
            }
            else
            {
                for(int k = 0; k < block_k / 4; k++)
                {
                    ls_dwordx4_tmp_var      = *((uint4*)&grad_O_ptr[dim_offset + k * 4]);
                    fetch_grad_O[k * 4 + 0] = *((T*)&ls_dwordx4_tmp_var.x);
                    fetch_grad_O[k * 4 + 1] = *((T*)&ls_dwordx4_tmp_var.y);
                    fetch_grad_O[k * 4 + 2] = *((T*)&ls_dwordx4_tmp_var.z);
                    fetch_grad_O[k * 4 + 3] = *((T*)&ls_dwordx4_tmp_var.w);
                }

                for(int kv_idx = 0; kv_idx < seq_kv; kv_idx++)
                {
                    for(int k = 0; k < block_k / 4; k++)
                    {
                        ls_dwordx4_tmp_var =
                            *((uint4*)&V_base[kv_idx * V_stride + dim_offset + k * 4]);
                        fetch_V[k * 4 + 0] = *((T*)&ls_dwordx4_tmp_var.x);
                        fetch_V[k * 4 + 1] = *((T*)&ls_dwordx4_tmp_var.y);
                        fetch_V[k * 4 + 2] = *((T*)&ls_dwordx4_tmp_var.z);
                        fetch_V[k * 4 + 3] = *((T*)&ls_dwordx4_tmp_var.w);
                    }
#pragma unroll
                    for(int k = 0; k < block_k; k++)
                    {
                        results[kv_idx] += fetch_grad_O[k] * fetch_V[k];
                    }
                }
            }
        }

        for(int i = 0; i < seq_kv; i++)
        {
            grad_attn_ptr[i] = T(results[i]);
        }
        // Zero out padding positions beyond seq_kv
        for(int i = seq_kv; i < max_seq_kv; i++)
        {
            grad_attn_ptr[i] = T(0.0f);
        }
    }
}

// softmax_backward_kernel
//
// grad_attn / attn_weights layout: [total_padded_q, head_num, max_seq_kv]
// Empty-Q batches have no row in this buffer (kernel 2 skips them).
// padded_q_to_batch[padded_q_slot] -> batch_idx  (host-precomputed)
template <typename T, typename Config>
__global__ void softmax_backward_kernel(const T* attn_weights,
                                        const T* dropout_mask,
                                        T* grad_attn,
                                        float dropout_scale,
                                        const int* cu_seqlens_kv,
                                        const int* padded_q_to_batch,
                                        uint32_t total_elt)
{
    const uint32_t block_id          = blockIdx.x;
    const uint32_t thread_id         = threadIdx.x;
    constexpr int max_seq_kv         = Config::max_seq_kv;
    constexpr int block_size         = Config::step2_block_size;
    constexpr int per_grad_attn_size = max_seq_kv; // seq_q == 1
    constexpr int valid_thread_range = block_size / per_grad_attn_size * per_grad_attn_size;
    const uint32_t cur_block_offset  = block_id * valid_thread_range + thread_id;
    bool is_tail                     = block_id * valid_thread_range + block_size >= total_elt;
    int real_row_num = is_tail ? (total_elt - block_id * valid_thread_range) / max_seq_kv
                               : valid_thread_range / max_seq_kv;

    if(cur_block_offset < total_elt && thread_id < valid_thread_range)
    {
        __shared__ T tmp_grad_score[valid_thread_range];
        constexpr int row_num = valid_thread_range / max_seq_kv;
        __shared__ T reduce_grad_score[row_num];

        // [total_padded_q, head_num, max_seq_kv] flat layout
        int global_row_idx = cur_block_offset / max_seq_kv;
        int padded_q_slot  = global_row_idx / Config::head_num;
        int k_idx          = cur_block_offset % max_seq_kv;

        // All rows in the buffer belong to active batches (empty-Q batches have no row).
        int batch_idx = padded_q_to_batch[padded_q_slot];
        int seq_kv    = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];

        T grad_attn_value = grad_attn[cur_block_offset];
        if constexpr(Config::enable_dropout_mask)
        {
            grad_attn_value = grad_attn_value * dropout_mask[cur_block_offset] * dropout_scale;
        }
        T attn_weight             = attn_weights[cur_block_offset];
        T grad_score              = grad_attn_value * attn_weight;
        tmp_grad_score[thread_id] = grad_score;
        __syncthreads();

        // Reduce within block
        if(thread_id < real_row_num)
        {
            T sum = T(0.0f);
#pragma unroll
            for(int i = 0; i < max_seq_kv; i++)
                sum += tmp_grad_score[thread_id * max_seq_kv + i];
            reduce_grad_score[thread_id] = sum;
        }
        __syncthreads();

        grad_score -= attn_weight * reduce_grad_score[thread_id / max_seq_kv];

        // Apply causal mask and KV-padding mask
        if constexpr(Config::mask_type == CausalMaskType::TOP_LEFT)
        {
            // q_idx == 0; mask: k_idx > 0 || k_idx >= seq_kv
            if(k_idx > 0 || k_idx >= seq_kv)
                grad_score = T(0.0f);
        }
        else if constexpr(Config::mask_type == CausalMaskType::BOTTOM_RIGHT)
        {
            if(k_idx >= seq_kv)
                grad_score = T(0.0f);
        }
        else
        {
            if(k_idx >= seq_kv)
                grad_score = T(0.0f);
        }

        grad_attn[cur_block_offset] = grad_score;
    }
}

// Kernel 4: Compute grad_Q and grad_K
template <typename T, typename Config, int TASKS_PER_BLOCK = 1, int BLOCK_K = 16>
__global__ void compute_grad_qk_kernel(const T* grad_scores,
                                       const T* Q,
                                       const T* K,
                                       T* grad_Q,
                                       T* grad_K,
                                       float scale,
                                       const int* cu_seqlens_q,
                                       const int* cu_seqlens_q_padded,
                                       const int* cu_seqlens_kv,
                                       const int* cu_seqlens_kv_padded)
{
    constexpr int seq_q                 = Config::seq_q;
    constexpr int max_seq_kv            = Config::max_seq_kv;
    constexpr int head_dim              = Config::head_dim;
    constexpr int block_k               = BLOCK_K;
    constexpr int dwordx4_load_elt      = 16 / sizeof(T);
    constexpr int warp_size             = 64;
    constexpr int process_head_per_warp = warp_size / (head_dim / block_k);
    constexpr int tasks_per_block       = TASKS_PER_BLOCK;

    int base_block_offset   = blockIdx.x * process_head_per_warp * tasks_per_block;
    int thread_id           = threadIdx.x;
    int thread_batch_offset = thread_id / (head_dim / block_k);
    int thread_head_offset  = thread_id % (head_dim / block_k) * block_k;

    uint4 load_dwordx4_tmp_var[block_k / dwordx4_load_elt];
    T grad_score_vals[max_seq_kv];

    for(int task = 0; task < tasks_per_block; task++)
    {
        int block_batch_head_idx = base_block_offset + task * process_head_per_warp;
        int cur_idx              = block_batch_head_idx + thread_batch_offset;

        int batch_idx    = cur_idx / (Config::seq_q * Config::head_num);
        int seq_head_idx = cur_idx % (Config::seq_q * Config::head_num);
        int seq_q_idx    = seq_head_idx / Config::head_num;
        int head_idx     = seq_head_idx % Config::head_num;

        if(batch_idx >= Config::bs)
            continue;

        // Skip batches where actual Q seq is 0 — no grad_Q/grad_K to compute.
        int actual_seq_q = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx];
        if(actual_seq_q == 0)
            continue;

        int seq_kv           = cu_seqlens_kv[batch_idx + 1] - cu_seqlens_kv[batch_idx];
        int q_storage_offset = cu_seqlens_q_padded[batch_idx]; // seq_q_idx == 0

        // workspace layout: [total_padded_q, head_num, max_seq_kv]
        int gs_offset = (q_storage_offset * Config::head_num + head_idx) * max_seq_kv;
#pragma unroll
        for(int i = 0; i < max_seq_kv; i++)
            grad_score_vals[i] = grad_scores[gs_offset + i];

        // Compute grad_Q = grad_scores @ K * scale
        uint4 store_dwordx4_tmp_var[block_k / dwordx4_load_elt];
#pragma unroll
        for(int i = 0; i < block_k / dwordx4_load_elt; i++)
        {
            store_dwordx4_tmp_var[i].x = 0;
            store_dwordx4_tmp_var[i].y = 0;
            store_dwordx4_tmp_var[i].z = 0;
            store_dwordx4_tmp_var[i].w = 0;
        }

        for(int j = 0; j < seq_kv; j++)
        {
#pragma unroll
            for(int i = 0; i < block_k / dwordx4_load_elt; i++)
            {
                int k_idx = (cu_seqlens_kv_padded[batch_idx] + j) * Config::head_num * head_dim +
                            head_idx * head_dim + thread_head_offset + i * dwordx4_load_elt;
                load_dwordx4_tmp_var[i] = *((uint4*)&K[k_idx]);
            }
#pragma unroll
            for(int b = 0; b < block_k; b++)
            {
                ((T*)&store_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt] +=
                    grad_score_vals[j] *
                    ((T*)&load_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt];
            }
        }

#pragma unroll
        for(int i = 0; i < block_k / dwordx4_load_elt; i++)
        {
            // grad_Q layout: [total_padded_seq_q, head_num, head_dim]
            T* grad_Q_ptr = &grad_Q[(q_storage_offset * Config::head_num + head_idx) * head_dim +
                                    thread_head_offset + i * dwordx4_load_elt];
            for(int b = 0; b < dwordx4_load_elt; b++)
            {
                grad_Q_ptr[b] = ((T*)&store_dwordx4_tmp_var[i])[b] * scale;
            }
        }

        // Compute grad_K = grad_scores^T @ Q * scale
        // Q layout: [total_padded_seq_q, head_num, head_dim]
#pragma unroll
        for(int i = 0; i < block_k / dwordx4_load_elt; i++)
        {
            load_dwordx4_tmp_var[i] =
                *((uint4*)&Q[(q_storage_offset * Config::head_num + head_idx) * head_dim +
                             thread_head_offset + i * dwordx4_load_elt]);
        }

        for(int j = 0; j < seq_kv; j++)
        {
#pragma unroll
            for(int b = 0; b < block_k; b++)
            {
                T val = grad_score_vals[j] *
                        ((T*)&load_dwordx4_tmp_var[b / dwordx4_load_elt])[b % dwordx4_load_elt] *
                        T(scale);
                // DYNAMIC layout: use cu_seqlens_kv_padded for offset
                int grad_k_idx =
                    (cu_seqlens_kv_padded[batch_idx] + j) * Config::head_num * head_dim +
                    head_idx * head_dim + thread_head_offset + b;
                grad_K[grad_k_idx] = val;
            }
        }
    }
}

template <typename T, typename Config>
struct AttnBackwardKernelLauncher
{

    static size_t calc_workspace_size(int total_padded_q)
    {
        constexpr int head_num   = Config::head_num;
        constexpr int max_seq_kv = Config::max_seq_kv;
        // workspace layout: [total_padded_q, head_num, max_seq_kv]
        return (size_t)total_padded_q * head_num * max_seq_kv * sizeof(T);
    }

    static void run_attn_bwd_kernel(const T* Q,
                                    const T* K,
                                    const T* V,
                                    const T* grad_O,
                                    const T* attn_weights,
                                    const T* dropout_mask,
                                    float dropout_p,
                                    float sqr_dk_scale,
                                    T* grad_Q,
                                    T* grad_K,
                                    T* grad_V,
                                    T* workspace,
                                    const int* cu_seqlens_q,
                                    const int* cu_seqlens_q_padded,
                                    const int* cu_seqlens_kv,
                                    const int* cu_seqlens_kv_padded,
                                    const int* padded_q_to_batch,
                                    int total_padded_q)
    {
        constexpr int bs         = Config::bs;
        constexpr int head_num   = Config::head_num;
        constexpr int seq_q      = Config::seq_q;
        constexpr int max_seq_kv = Config::max_seq_kv;
        constexpr int head_dim   = Config::head_dim;
        constexpr int warp_size  = 64;

        constexpr int merge_bs = bs * head_num;
        float scale            = sqr_dk_scale;
        float dropout_scale    = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;

        dim3 block(warp_size);

        // Step 1: Compute grad_V = attn_weights^T @ grad_O — grid covers all (bs * head_num) tasks
        constexpr int tasks_per_block_v = 16;
        dim3 grid_v((bs * seq_q * head_num + tasks_per_block_v - 1) / tasks_per_block_v);
        compute_grad_v_kernel<T, Config, tasks_per_block_v><<<grid_v, block>>>(
            attn_weights, grad_O, grad_V, cu_seqlens_q, cu_seqlens_q_padded, cu_seqlens_kv,
            cu_seqlens_kv_padded);

        // Step 2: Compute grad_attn = grad_O @ V^T — grid covers all (bs * head_num) tasks
        constexpr int tasks_per_block_attn  = 16;
        constexpr int process_head_per_warp = warp_size / (head_dim / 64);
        dim3 grid_grad_attn(
            (bs * seq_q * head_num + tasks_per_block_attn * process_head_per_warp - 1) /
            (tasks_per_block_attn * process_head_per_warp));
        compute_grad_attn_kernel<T, Config, tasks_per_block_attn><<<grid_grad_attn, block>>>(
            grad_O, V, workspace, cu_seqlens_q, cu_seqlens_q_padded, cu_seqlens_kv,
            cu_seqlens_kv_padded);

        // Step 3: Softmax backward — grid covers [total_padded_q, head_num, max_seq_kv] elements
        constexpr int work_thread_num = Config::step2_block_size / max_seq_kv * max_seq_kv;
        uint32_t total_elt = (uint32_t)total_padded_q * head_num * max_seq_kv;
        dim3 grid_softmax((total_elt + work_thread_num - 1) / work_thread_num);
        dim3 block_softmax(Config::step2_block_size);
        softmax_backward_kernel<T, Config><<<grid_softmax, block_softmax>>>(
            attn_weights, dropout_mask, workspace, dropout_scale, cu_seqlens_kv,
            padded_q_to_batch, total_elt);

        // Step 4: Compute grad_Q and grad_K — grid covers all (bs * head_num) tasks
        constexpr int tasks_per_block_qk = 4;
        dim3 grid_qk((bs * seq_q * head_num + tasks_per_block_qk - 1) / tasks_per_block_qk);
        compute_grad_qk_kernel<T, Config, tasks_per_block_qk><<<grid_qk, block>>>(
            workspace, Q, K, grad_Q, grad_K, scale, cu_seqlens_q, cu_seqlens_q_padded,
            cu_seqlens_kv, cu_seqlens_kv_padded);
    }
};

// Helper function: Matrix multiplication C = A @ B
// A: [rows_a, cols_a], B: [cols_a, cols_b], C: [rows_a, cols_b]
template <typename T>
void matmul(const T* A, const T* B, T* C, int rows_a, int cols_a, int cols_b)
{
    for(int i = 0; i < rows_a; i++)
    {
        for(int j = 0; j < cols_b; j++)
        {
            float sum = 0.0f;
            for(int k = 0; k < cols_a; k++)
            {
                sum += float(A[i * cols_a + k]) * float(B[k * cols_b + j]);
            }
            C[i * cols_b + j] = T(sum);
        }
    }
}

// Helper function: Matrix transpose
template <typename T>
void transpose(const T* A, T* A_T, int rows, int cols)
{
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            A_T[j * rows + i] = A[i * cols + j];
        }
    }
}

// Helper function: Sum along last dimension
template <typename T>
void sum_last_dim(const T* A, T* sums, int rows, int cols)
{
    for(int i = 0; i < rows; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < cols; j++)
        {
            sum += float(A[i * cols + j]);
        }
        sums[i] = T(sum);
    }
}

/**
 * Multi-Head Attention Backward Pass (CPU Reference Implementation)
 *
 * Q/grad_O/grad_Q layout: [total_padded_seq_q, head_num, head_dim]
 * K/V/grad_K/grad_V layout: [total_padded_seq_kv, head_num, head_dim]
 * attn_weights/dropout_mask: [bs, head_num, max_kv_seq] (batch-strided, no Q padding)
 *
 * Batches where actual Q seq = 0 are skipped.
 */
template <typename T>
void attn_backward(const T* Q,
                   const T* K,
                   const T* V,
                   const T* grad_O,
                   const T* attn_weights,
                   const T* dropout_mask,
                   float dropout_p,
                   T* grad_Q,
                   T* grad_K,
                   T* grad_V,
                   int batch,
                   int head_num,
                   int max_kv_seq,
                   int head_dim,
                   CausalMaskType mask_type,
                   const int* cu_seqlens_q,
                   const int* cu_seqlens_q_padded,
                   const int* cu_seqlens_kv,
                   const int* cu_seqlens_kv_padded,
                   int total_padded_q,
                   int total_padded_kv_seq)
{
    float scale         = 1.0f / std::sqrt(static_cast<float>(head_dim));
    float dropout_scale = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;

    // Temporary buffers (seq_q is always 1 when active)
    std::vector<T> V_T(max_kv_seq * head_dim);
    std::vector<T> grad_attn(max_kv_seq);
    std::vector<T> grad_scores(max_kv_seq);
    std::vector<T> K_cont_buf(max_kv_seq * head_dim);
    std::vector<T> V_cont_buf(max_kv_seq * head_dim);
    std::vector<T> grad_K_cont_buf(max_kv_seq * head_dim);
    std::vector<T> grad_V_cont_buf(max_kv_seq * head_dim);

    // Initialize gradients to zero
    std::memset(grad_Q, 0, total_padded_q * head_num * head_dim * sizeof(T));
    std::memset(grad_K, 0, total_padded_kv_seq * head_num * head_dim * sizeof(T));
    std::memset(grad_V, 0, total_padded_kv_seq * head_num * head_dim * sizeof(T));

    for(int b = 0; b < batch; b++)
    {
        // Skip batches where actual Q seq is 0
        int actual_q_seq = cu_seqlens_q[b + 1] - cu_seqlens_q[b];
        if(actual_q_seq == 0)
            continue;

        int kv_seq       = cu_seqlens_kv[b + 1] - cu_seqlens_kv[b];
        int q_off        = cu_seqlens_q_padded[b];  // padded Q storage offset
        int kv_off       = cu_seqlens_kv_padded[b]; // padded KV storage offset
        int kv_stride    = head_num * head_dim;

        for(int h = 0; h < head_num; h++)
        {
            // Q/grad_O/grad_Q: [total_padded_seq_q, head_num, head_dim]
            int offset_Q    = (q_off * head_num + h) * head_dim;
            // attn_weights/dropout_mask: [total_padded_q, head_num, max_kv_seq]
            int offset_attn = (q_off * head_num + h) * max_kv_seq;
            int offset_drop = dropout_mask ? (q_off * head_num + h) * max_kv_seq : 0;

            const T* Q_bh       = Q + offset_Q;
            const T* grad_O_bh  = grad_O + offset_Q; // same offset as Q
            const T* attn_bh    = attn_weights + offset_attn;
            const T* dropout_bh = dropout_mask ? dropout_mask + offset_drop : nullptr;
            T* grad_Q_bh        = grad_Q + offset_Q;

            // K/V: [total_padded_seq_kv, head_num, head_dim]
            int offset_kv_base = kv_off * head_num * head_dim + h * head_dim;
            const T* K_bh      = K + offset_kv_base;
            const T* V_bh      = V + offset_kv_base;
            T* grad_K_bh       = grad_K + offset_kv_base;
            T* grad_V_bh       = grad_V + offset_kv_base;

            // Flatten K/V into contiguous row-major buffers [kv_seq, head_dim]
            for(int i = 0; i < kv_seq; i++)
                for(int j = 0; j < head_dim; j++)
                {
                    K_cont_buf[i * head_dim + j] = K_bh[i * kv_stride + j];
                    V_cont_buf[i * head_dim + j] = V_bh[i * kv_stride + j];
                }

            // Step 1: grad_V[:, j] += attn[j] * grad_O  (attn^T @ grad_O)
            // attn_bh is [max_kv_seq]; grad_O_bh is [head_dim] (single Q token)
            for(int j = 0; j < kv_seq; j++)
            {
                for(int d = 0; d < head_dim; d++)
                    grad_V_cont_buf[j * head_dim + d] = T(float(attn_bh[j]) * float(grad_O_bh[d]));
            }

            // Step 2: grad_attn[j] = grad_O . V[j]
            for(int j = 0; j < kv_seq; j++)
            {
                float s = 0.0f;
                for(int d = 0; d < head_dim; d++)
                    s += float(grad_O_bh[d]) * float(V_cont_buf[j * head_dim + d]);
                grad_attn[j] = T(s);
            }

            // Step 3: Dropout backward
            if(dropout_p > 0.0f && dropout_bh != nullptr)
                for(int j = 0; j < kv_seq; j++)
                    grad_attn[j] = T(float(grad_attn[j]) * float(dropout_bh[j]) * dropout_scale);

            // Step 4: Softmax backward — grad_score[j] = attn[j] * (grad_attn[j] - sum)
            float dot_sum = 0.0f;
            for(int j = 0; j < kv_seq; j++)
                dot_sum += float(grad_attn[j]) * float(attn_bh[j]);
            for(int j = 0; j < kv_seq; j++)
                grad_scores[j] = T(float(attn_bh[j]) * (float(grad_attn[j]) - dot_sum));

            // Step 5: Mask backward
            if(mask_type == CausalMaskType::TOP_LEFT)
            {
                // q_idx == 0; mask: k_idx > 0 → grad_score = 0
                for(int j = 1; j < kv_seq; j++)
                    grad_scores[j] = T(0.0f);
            }
            else if(mask_type == CausalMaskType::BOTTOM_RIGHT)
            {
                // q_idx == 0; mask: k_idx < 0 → nothing masked for this row
            }

            // Step 6: grad_Q = grad_scores @ K * scale  ([kv_seq] @ [kv_seq, head_dim])
            for(int d = 0; d < head_dim; d++)
            {
                float s = 0.0f;
                for(int j = 0; j < kv_seq; j++)
                    s += float(grad_scores[j]) * float(K_cont_buf[j * head_dim + d]);
                grad_Q_bh[d] = T(s * scale);
            }

            // Step 7: grad_K[j] = grad_scores[j] * Q * scale
            for(int j = 0; j < kv_seq; j++)
                for(int d = 0; d < head_dim; d++)
                    grad_K_cont_buf[j * head_dim + d] =
                        T(float(grad_scores[j]) * float(Q_bh[d]) * scale);

            // Copy grad_K and grad_V back to strided layout
            for(int i = 0; i < kv_seq; i++)
                for(int j = 0; j < head_dim; j++)
                {
                    grad_K_bh[i * kv_stride + j] = grad_K_cont_buf[i * head_dim + j];
                    grad_V_bh[i * kv_stride + j] = grad_V_cont_buf[i * head_dim + j];
                }
        }
    }
}

/**
 * Test run_attn_bwd_kernel correctness and bandwidth
 *
 * Q/grad_O/grad_Q layout: [total_padded_q, head_num, head_dim]
 *   where total_padded_q == bs (each batch always occupies exactly 1 slot)
 * K/V/grad_K/grad_V layout: [total_padded_kv_seq, head_num, head_dim] (variable)
 * attn_weights/dropout_mask: [bs, head_num, max_seq_kv] (batch-strided, no Q axis)
 */
template <typename DataType, typename Config>
void test_run_attn_bwd_kernel(
    float dropout_p, int warmup_iters, int test_iters, bool check_correctness, bool dump_err)
{
    using Launcher = AttnBackwardKernelLauncher<DataType, Config>;

    constexpr int bs         = Config::bs;
    constexpr int head_num   = Config::head_num;
    constexpr int max_seq_kv = Config::max_seq_kv;
    constexpr int head_dim   = Config::head_dim;

    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> normal_dis(4.0f, 2.0f);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::uniform_int_distribution<int> pad_dis(0, 5);
    // Bernoulli for Q presence: 50% chance each batch has an active Q token
    std::bernoulli_distribution q_present_dis(0.5);

    // --- Generate cu_seqlens_q / cu_seqlens_q_padded ---
    // Actual Q seq per batch is 0 or 1.
    // Padded storage only advances by 1 for active batches (same as actual in this scheme).
    std::vector<int> h_cu_seqlens_q(bs + 1);
    std::vector<int> h_cu_seqlens_q_padded(bs + 1);
    h_cu_seqlens_q[0]        = 0;
    h_cu_seqlens_q_padded[0] = 0;
    int total_actual_q = 0;
    for(int b = 0; b < bs; b++)
    {
        int actual_q = q_present_dis(gen) ? 1 : 0;
        total_actual_q += actual_q;
        h_cu_seqlens_q[b + 1]        = total_actual_q;
        // Only allocate a padded slot for active batches.
        h_cu_seqlens_q_padded[b + 1] = h_cu_seqlens_q_padded[b] + actual_q;
    }
    int total_padded_q = h_cu_seqlens_q_padded[bs]; // == total_actual_q in this scheme

    // Build padded_q_to_batch reverse map (host-side, size = total_padded_q)
    std::vector<int> h_padded_q_to_batch(total_padded_q);
    for(int b = 0; b < bs; b++)
    {
        if(h_cu_seqlens_q_padded[b + 1] > h_cu_seqlens_q_padded[b])
            h_padded_q_to_batch[h_cu_seqlens_q_padded[b]] = b;
    }

    // --- Generate cu_seqlens_kv / cu_seqlens_kv_padded ---
    std::vector<int> h_cu_seqlens_kv(bs + 1);
    std::vector<int> h_cu_seqlens_kv_padded(bs + 1);
    h_cu_seqlens_kv[0]        = 0;
    h_cu_seqlens_kv_padded[0] = 0;
    int total_actual_kv_seq = 0;
    int total_padded_kv_seq = 0;
    for(int b = 0; b < bs; b++)
    {
        int kv_len     = static_cast<int>(std::round(normal_dis(gen)));
        kv_len         = std::max(2, std::min(max_seq_kv, kv_len));
        int random_pad = pad_dis(gen);
        int padded_len = kv_len + random_pad > max_seq_kv ? max_seq_kv : kv_len + random_pad;
        total_padded_kv_seq += padded_len;
        total_actual_kv_seq += kv_len;
        h_cu_seqlens_kv[b + 1]        = total_actual_kv_seq;
        h_cu_seqlens_kv_padded[b + 1] = total_padded_kv_seq;
    }

    // --- Buffer sizes ---
    // Q/grad_O/grad_Q: [total_padded_q, head_num, head_dim]
    // attn_weights/dropout_mask: [total_padded_q, head_num, max_seq_kv]
    size_t size_Q            = (size_t)total_padded_q * head_num * head_dim;
    size_t size_K            = (size_t)total_padded_kv_seq * head_num * head_dim;
    size_t size_V            = (size_t)total_padded_kv_seq * head_num * head_dim;
    size_t size_grad_O       = (size_t)total_padded_q * head_num * head_dim;
    size_t size_attn_weights = (size_t)total_padded_q * head_num * max_seq_kv;
    size_t size_dropout_mask = (size_t)total_padded_q * head_num * max_seq_kv;

    // Allocate host memory
    std::vector<DataType> h_Q(size_Q, DataType(0.0f));
    std::vector<DataType> h_K(size_K, DataType(0.0f));
    std::vector<DataType> h_V(size_V, DataType(0.0f));
    std::vector<DataType> h_grad_O(size_grad_O, DataType(0.0f));
    std::vector<DataType> h_attn_weights(size_attn_weights, DataType(0.0f));
    std::vector<DataType> h_dropout_mask(size_dropout_mask, DataType(1.0f));
    std::vector<DataType> h_grad_Q_gpu(size_Q, DataType(0.0f));
    std::vector<DataType> h_grad_K_gpu(size_K, DataType(0.0f));
    std::vector<DataType> h_grad_V_gpu(size_V, DataType(0.0f));
    std::vector<DataType> h_grad_Q_cpu(size_Q, DataType(0.0f));
    std::vector<DataType> h_grad_K_cpu(size_K, DataType(0.0f));
    std::vector<DataType> h_grad_V_cpu(size_V, DataType(0.0f));

    // --- Initialize Q and grad_O for active-Q batches ---
    // Layout: [total_padded_q, head_num, head_dim]; cu_seqlens_q_padded[b] == b
    for(int b = 0; b < bs; b++)
    {
        int actual_q = h_cu_seqlens_q[b + 1] - h_cu_seqlens_q[b];
        if(actual_q == 0)
            continue; // slot exists but we leave it zero (unused)
        int q_off = h_cu_seqlens_q_padded[b]; // == b
        for(int h = 0; h < head_num; h++)
        {
            int base = (q_off * head_num + h) * head_dim;
            for(int d = 0; d < head_dim; d++)
            {
                h_Q[base + d]      = DataType(dis(gen));
                h_grad_O[base + d] = DataType(dis(gen));
            }
        }
    }

    // --- Initialize K/V (variable-length KV, dynamic layout) ---
    for(int b = 0; b < bs; b++)
    {
        int kv_seq = h_cu_seqlens_kv[b + 1] - h_cu_seqlens_kv[b];
        int kv_off = h_cu_seqlens_kv_padded[b];
        for(int h = 0; h < head_num; h++)
        {
            for(int s = 0; s < kv_seq; s++)
            {
                int base = (kv_off + s) * head_num * head_dim + h * head_dim;
                for(int d = 0; d < head_dim; d++)
                {
                    h_K[base + d] = DataType(dis(gen));
                    h_V[base + d] = DataType(dis(gen));
                }
            }
        }
    }

    // attn_weights layout: [total_padded_q, head_num, max_seq_kv] — only active-Q batches.
    for(int b = 0; b < bs; b++)
    {
        int actual_q = h_cu_seqlens_q[b + 1] - h_cu_seqlens_q[b];
        if(actual_q == 0)
            continue; // empty-Q batches have no row in this buffer
        int kv_seq = h_cu_seqlens_kv[b + 1] - h_cu_seqlens_kv[b];
        int q_off  = h_cu_seqlens_q_padded[b];
        for(int h = 0; h < head_num; h++)
        {
            int base = (q_off * head_num + h) * max_seq_kv;
            // Normalized random softmax output
            float sum = 0.0f;
            for(int j = 0; j < kv_seq; j++)
            {
                h_attn_weights[base + j] = DataType(std::abs(dis(gen)));
                sum += float(h_attn_weights[base + j]);
            }
            for(int j = kv_seq; j < max_seq_kv; j++)
                h_attn_weights[base + j] = DataType(0.0f);
            if(sum > 0.0f)
                for(int j = 0; j < kv_seq; j++)
                    h_attn_weights[base + j] =
                        DataType(float(h_attn_weights[base + j]) / sum);
        }
    }

    // --- Initialize dropout mask: [total_padded_q, head_num, max_seq_kv] ---
    for(size_t i = 0; i < size_dropout_mask; i++)
    {
        h_dropout_mask[i] = Config::enable_dropout_mask
                                ? DataType(dis(gen) > dropout_p ? 1.0f : 0.0f)
                                : DataType(1.0f);
    }

    // --- CPU reference ---
    float sqr_dk_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if(check_correctness)
        attn_backward(h_Q.data(),
                      h_K.data(),
                      h_V.data(),
                      h_grad_O.data(),
                      h_attn_weights.data(),
                      Config::enable_dropout_mask ? h_dropout_mask.data() : nullptr,
                      dropout_p,
                      h_grad_Q_cpu.data(),
                      h_grad_K_cpu.data(),
                      h_grad_V_cpu.data(),
                      bs,
                      head_num,
                      max_seq_kv,
                      head_dim,
                      Config::mask_type,
                      h_cu_seqlens_q.data(),
                      h_cu_seqlens_q_padded.data(),
                      h_cu_seqlens_kv.data(),
                      h_cu_seqlens_kv_padded.data(),
                      total_padded_q,
                      total_padded_kv_seq);

    // --- Allocate device memory ---
    DataType *d_Q, *d_K, *d_V, *d_grad_O, *d_attn_weights;
    DataType* d_dropout_mask;
    DataType *d_grad_Q, *d_grad_K, *d_grad_V, *d_workspace;
    int *d_cu_seqlens_q, *d_cu_seqlens_q_padded;
    int *d_cu_seqlens_kv, *d_cu_seqlens_kv_padded;
    int* d_padded_q_to_batch;

    HIP_CHECK(hipMalloc(&d_Q, size_Q > 0 ? size_Q * sizeof(DataType) : 1));
    HIP_CHECK(hipMalloc(&d_K, size_K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_V, size_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_grad_O, size_grad_O > 0 ? size_grad_O * sizeof(DataType) : 1));
    HIP_CHECK(hipMalloc(&d_attn_weights, size_attn_weights > 0 ? size_attn_weights * sizeof(DataType) : 1));
    HIP_CHECK(hipMalloc(&d_dropout_mask, size_dropout_mask > 0 ? size_dropout_mask * sizeof(DataType) : 1));
    HIP_CHECK(hipMalloc(&d_grad_Q, size_Q > 0 ? size_Q * sizeof(DataType) : 1));
    HIP_CHECK(hipMalloc(&d_grad_K, size_K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_grad_V, size_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_q, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_q_padded, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_kv, (bs + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_cu_seqlens_kv_padded, (bs + 1) * sizeof(int)));
    if(total_padded_q > 0)
        HIP_CHECK(hipMalloc(&d_padded_q_to_batch, total_padded_q * sizeof(int)));
    else
        d_padded_q_to_batch = nullptr;

    // workspace: [total_padded_q, head_num, max_seq_kv]
    size_t workspace_size = Launcher::calc_workspace_size(total_padded_q);
    HIP_CHECK(hipMalloc(&d_workspace, workspace_size > 0 ? workspace_size : 1));

    // --- Copy to device ---
    if(size_Q > 0)
        HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), size_Q * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K.data(), size_K * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), size_V * sizeof(DataType), hipMemcpyHostToDevice));
    if(size_grad_O > 0)
        HIP_CHECK(hipMemcpy(d_grad_O, h_grad_O.data(), size_grad_O * sizeof(DataType), hipMemcpyHostToDevice));
    if(size_attn_weights > 0)
        HIP_CHECK(hipMemcpy(d_attn_weights, h_attn_weights.data(),
                            size_attn_weights * sizeof(DataType), hipMemcpyHostToDevice));
    if(size_dropout_mask > 0)
        HIP_CHECK(hipMemcpy(d_dropout_mask, h_dropout_mask.data(),
                            size_dropout_mask * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_cu_seqlens_q, h_cu_seqlens_q.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_q_padded,
                        h_cu_seqlens_q_padded.data(),
                        (bs + 1) * sizeof(int),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_cu_seqlens_kv, h_cu_seqlens_kv.data(), (bs + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_kv_padded,
                        h_cu_seqlens_kv_padded.data(),
                        (bs + 1) * sizeof(int),
                        hipMemcpyHostToDevice));
    if(total_padded_q > 0)
        HIP_CHECK(hipMemcpy(d_padded_q_to_batch, h_padded_q_to_batch.data(),
                            total_padded_q * sizeof(int), hipMemcpyHostToDevice));

    auto bwd_launch = [&]() {
        Launcher::run_attn_bwd_kernel(d_Q, d_K, d_V, d_grad_O, d_attn_weights,
                                      Config::enable_dropout_mask ? d_dropout_mask
                                                                  : static_cast<DataType*>(nullptr),
                                      dropout_p, sqr_dk_scale,
                                      d_grad_Q, d_grad_K, d_grad_V, d_workspace,
                                      d_cu_seqlens_q, d_cu_seqlens_q_padded,
                                      d_cu_seqlens_kv, d_cu_seqlens_kv_padded,
                                      d_padded_q_to_batch, total_padded_q);
    };

    // --- Warmup runs ---
    for(int i = 0; i < warmup_iters; i++)
        bwd_launch();
    HIP_CHECK(hipDeviceSynchronize());

    // --- Timed runs ---
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    for(int i = 0; i < test_iters; i++)
        bwd_launch();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    double avg_time_ms = elapsed_ms / test_iters;

    // --- TFLOPS (based on active-Q batches; seq_q=1 per active batch) ---
    double avg_kv_seq        = total_actual_kv_seq / double(bs);
    double avg_padded_kv_seq = total_padded_kv_seq / double(bs);
    // Each active batch contributes: grad_V + grad_attn + grad_Q + grad_K (4 x 2*kv*dim MACs)
    double flops_per_active_batch_head = 4.0 * 2.0 * avg_kv_seq * head_dim;
    double total_flops = flops_per_active_batch_head * total_actual_q * head_num;
    double tflops      = (total_flops / 1e12) / (avg_time_ms / 1000.0);

    // --- Copy results back ---
    HIP_CHECK(
        hipMemcpy(h_grad_Q_gpu.data(), d_grad_Q, size_Q * sizeof(DataType), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(h_grad_K_gpu.data(), d_grad_K, size_K * sizeof(DataType), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(h_grad_V_gpu.data(), d_grad_V, size_V * sizeof(DataType), hipMemcpyDeviceToHost));

    // --- Correctness check (only on active-Q slots for grad_Q; full arrays for grad_K/grad_V) ---
    auto check_grad_q = [&](float tolerance = 1e-1f) {
        float max_diff     = 0.0f;
        float max_rel_diff = 0.0f;
        size_t diff_count  = 0;
        size_t active_elems = 0;

        for(int b = 0; b < bs; b++)
        {
            int actual_q = h_cu_seqlens_q[b + 1] - h_cu_seqlens_q[b];
            if(actual_q == 0)
                continue; // skip empty slots
            int q_off = h_cu_seqlens_q_padded[b];
            for(int h = 0; h < head_num; h++)
            {
                int base = (q_off * head_num + h) * head_dim;
                for(int d = 0; d < head_dim; d++)
                {
                    size_t idx     = base + d;
                    float diff     = std::abs(float(h_grad_Q_gpu[idx]) - float(h_grad_Q_cpu[idx]));
                    float ref      = std::abs(float(h_grad_Q_cpu[idx])) + 1e-6f;
                    float rel_diff = diff / ref;
                    max_diff       = std::max(max_diff, diff);
                    max_rel_diff   = std::max(max_rel_diff, rel_diff);
                    if(rel_diff > tolerance)
                    {
                        if(dump_err)
                            std::cout << "grad_Q mismatch at [b=" << b << ",h=" << h << ",d=" << d
                                      << "]: GPU=" << static_cast<float>(h_grad_Q_gpu[idx])
                                      << " CPU=" << static_cast<float>(h_grad_Q_cpu[idx])
                                      << " rel=" << rel_diff << std::endl;
                        diff_count++;
                    }
                    active_elems++;
                }
            }
        }
        std::cout << "grad_Q check (active slots only):" << std::endl;
        std::cout << "  Active Q elements: " << active_elems << std::endl;
        std::cout << "  Max abs diff: " << max_diff << "  Max rel diff: " << max_rel_diff
                  << std::endl;
        std::cout << "  Exceeding tolerance: " << diff_count << " / " << active_elems << std::endl;
        std::cout << "  Status: " << (max_rel_diff < tolerance ? "PASS" : "FAIL") << std::endl;
    };

    auto check_array = [&](const std::vector<DataType>& gpu,
                           const std::vector<DataType>& cpu,
                           const std::string& name,
                           float tolerance = 1e-1f) {
        float max_diff     = 0.0f;
        float max_rel_diff = 0.0f;
        size_t diff_count  = 0;
        for(size_t i = 0; i < gpu.size(); i++)
        {
            float diff     = std::abs(float(gpu[i]) - float(cpu[i]));
            float rel_diff = diff / (std::abs(float(cpu[i])) + 1e-6f);
            max_diff       = std::max(max_diff, diff);
            max_rel_diff   = std::max(max_rel_diff, rel_diff);
            if(rel_diff > tolerance)
            {
                if(dump_err)
                    std::cout << name << " mismatch at " << i
                              << ": GPU=" << static_cast<float>(gpu[i])
                              << " CPU=" << static_cast<float>(cpu[i])
                              << " rel=" << rel_diff << std::endl;
                diff_count++;
            }
        }
        std::cout << name << " check:" << std::endl;
        std::cout << "  Max abs diff: " << max_diff << "  Max rel diff: " << max_rel_diff
                  << std::endl;
        std::cout << "  Exceeding tolerance: " << diff_count << " / " << gpu.size() << std::endl;
        std::cout << "  Status: " << (max_rel_diff < tolerance ? "PASS" : "FAIL") << std::endl;
    };

    // --- Bandwidth ---
    size_t bytes_read =
        (size_Q + size_K + size_V + size_grad_O + size_attn_weights) * sizeof(DataType);
    if(Config::enable_dropout_mask)
        bytes_read += size_dropout_mask * sizeof(DataType);
    size_t bytes_write    = (size_Q + size_K + size_V) * sizeof(DataType);
    size_t total_bytes    = bytes_read + bytes_write;
    double bandwidth_gbps = (total_bytes / 1e9) / (avg_time_ms / 1000.0);

    // --- Print results ---
    std::cout << "\n===== run_attn_bwd_kernel Test =====" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << bs << std::endl;
    std::cout << "  Heads: " << head_num << std::endl;
    std::cout << "  Active Q batches: " << total_actual_q << " / " << bs << std::endl;
    std::cout << "  KV max sequence length: " << max_seq_kv << std::endl;
    std::cout << "  KV avg sequence length: " << std::fixed << std::setprecision(2) << avg_kv_seq
              << std::endl;
    std::cout << "  KV avg padded length: " << std::fixed << std::setprecision(2)
              << avg_padded_kv_seq << std::endl;
    std::cout << "  Total actual KV seq: " << total_actual_kv_seq << std::endl;
    std::cout << "  Total padded KV seq: " << total_padded_kv_seq << std::endl;
    std::cout << "  Head dimension: " << head_dim << std::endl;
    std::cout << "  Dropout: " << (Config::enable_dropout_mask ? "enabled" : "disabled")
              << std::endl;
    std::cout << "  Mask: " << (CausalMaskTypeName[Config::mask_type]) << std::endl;
    std::cout << std::endl;

    if(check_correctness)
    {
        std::cout << "Correctness:" << std::endl;
        check_grad_q();
        check_array(h_grad_K_gpu, h_grad_K_cpu, "grad_K");
        check_array(h_grad_V_gpu, h_grad_V_cpu, "grad_V");
        std::cout << std::endl;
    }
    std::cout << "Memory:" << std::endl;
    std::cout << "  Total data read: " << std::fixed << std::setprecision(2) << bytes_read / 1e6
              << " MB" << std::endl;
    std::cout << "  Total data write: " << bytes_write / 1e6 << " MB" << std::endl;
    std::cout << "  Total data transfer: " << total_bytes / 1e6 << " MB" << std::endl;
    std::cout << "  Workspace size: " << workspace_size / 1e6 << " MB" << std::endl;
    std::cout << std::endl;

    std::cout << "Performance:" << std::endl;
    std::cout << "  Average time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms"
              << std::endl;
    std::cout << "  Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth_gbps << " GB/s"
              << std::endl;
    std::cout << "  TFLOPS: " << std::fixed << std::setprecision(2) << tflops << std::endl;
    std::cout << "====================================\n" << std::endl;

    // --- Cleanup ---
    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_K));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_grad_O));
    HIP_CHECK(hipFree(d_attn_weights));
    HIP_CHECK(hipFree(d_dropout_mask));
    HIP_CHECK(hipFree(d_grad_Q));
    HIP_CHECK(hipFree(d_grad_K));
    HIP_CHECK(hipFree(d_grad_V));
    HIP_CHECK(hipFree(d_workspace));
    HIP_CHECK(hipFree(d_cu_seqlens_q));
    HIP_CHECK(hipFree(d_cu_seqlens_q_padded));
    HIP_CHECK(hipFree(d_cu_seqlens_kv));
    HIP_CHECK(hipFree(d_cu_seqlens_kv_padded));
    if(d_padded_q_to_batch)
        HIP_CHECK(hipFree(d_padded_q_to_batch));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

// Template metaprogramming: recursive template to iterate over SEQ_KV values
template <int SEQ_KV, int MAX_SEQ_KV>
struct TestRunner
{
    template <typename DataType,
              int BS,
              int HEAD_NUM,
              int HEAD_DIM,
              int STEP2_BLOCK_SIZE,
              bool ENABLE_DROPOUT_MASK,
              CausalMaskType MASK_TYPE>
    static void
    run(float dropout_p, int warmup_iters, int test_iters, bool check_correctness, bool dump_err)
    {
        using KernelConfig = FmhaKernelConfig<BS,
                                              HEAD_NUM,
                                              SEQ_KV,
                                              HEAD_DIM,
                                              STEP2_BLOCK_SIZE,
                                              ENABLE_DROPOUT_MASK,
                                              MASK_TYPE>;
        test_run_attn_bwd_kernel<DataType, KernelConfig>(
            dropout_p, warmup_iters, test_iters, check_correctness, dump_err);

        // Recursive call for next SEQ_KV value
        TestRunner<SEQ_KV + 1, MAX_SEQ_KV>::template run<DataType,
                                                         BS,
                                                         HEAD_NUM,
                                                         HEAD_DIM,
                                                         STEP2_BLOCK_SIZE,
                                                         ENABLE_DROPOUT_MASK,
                                                         MASK_TYPE>(
            dropout_p, warmup_iters, test_iters, check_correctness, dump_err);
    }
};

// Termination condition: when SEQ_KV reaches MAX_SEQ_KV
template <int MAX_SEQ_KV>
struct TestRunner<MAX_SEQ_KV, MAX_SEQ_KV>
{
    template <typename DataType,
              int BS,
              int HEAD_NUM,
              int HEAD_DIM,
              int STEP2_BLOCK_SIZE,
              bool ENABLE_DROPOUT_MASK,
              CausalMaskType MASK_TYPE>
    static void
    run(float dropout_p, int warmup_iters, int test_iters, bool check_correctness, bool dump_err)
    {
        using KernelConfig = FmhaKernelConfig<BS,
                                              HEAD_NUM,
                                              MAX_SEQ_KV,
                                              HEAD_DIM,
                                              STEP2_BLOCK_SIZE,
                                              ENABLE_DROPOUT_MASK,
                                              MASK_TYPE>;
        test_run_attn_bwd_kernel<DataType, KernelConfig>(
            dropout_p, warmup_iters, test_iters, check_correctness, dump_err);
    }
};

int main(int argc, char const* argv[])
{
    std::cout << "\n========== correctness test ==========" << std::endl;

    // BS=30720, HEAD_NUM=32, MAX_SEQ_KV=16, HEAD_DIM=128, STEP2=128, no dropout, no mask
    using CorrConfig = FmhaKernelConfig<30720, 32, 16, 128, 128, false, CausalMaskType::DISABLE>;
    test_run_attn_bwd_kernel<float, CorrConfig>(0,    // dropout_p
                                                10,   // warmup_iters
                                                10,   // test_iters
                                                true, // check_correctness
                                                true  // dump_err
    );

    std::cout << "\n========== performance test ==========" << std::endl;

    // BS=30720, HEAD_NUM=32, MAX_SEQ_KV=16, HEAD_DIM=128, STEP2=128, no dropout, top-left mask
    using PerfConfig = FmhaKernelConfig<30720, 32, 16, 128, 128, false, CausalMaskType::TOP_LEFT>;
    test_run_attn_bwd_kernel<hip_bfloat16, PerfConfig>(0,     // dropout_p
                                                       3,     // warmup_iters
                                                       5,     // test_iters
                                                       false, // check_correctness
                                                       false  // dump_err
    );

    std::cout << "\n========== mixed-Q test (cu_seqlens_q with 0/1 tokens per batch) ==========" << std::endl;

    // BS=128: each batch has 0 or 1 active Q tokens (bernoulli); tests cu_seqlens_q path
    using MixedConfig = FmhaKernelConfig<128, 4, 8, 64, 256, false, CausalMaskType::DISABLE>;
    test_run_attn_bwd_kernel<float, MixedConfig>(0,    // dropout_p
                                                 2,    // warmup_iters
                                                 5,    // test_iters
                                                 true, // check_correctness
                                                 true  // dump_err
    );

    return 0;
}