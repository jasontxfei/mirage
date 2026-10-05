/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "tasks/common/common_header.cuh"
#include <cutlass/arch/barrier.h>

namespace kernel {

// MiniMax-M2.5 Q/K preprocessing for one token.
//
// The fused row is grouped by KV head, matching PersistentKernel attention:
//   [q_group_0, k_0, v_0, q_group_1, k_1, v_1, ...]
// where each q_group contains NUM_Q_HEADS / NUM_KV_HEADS heads. MiniMax
// applies RMSNorm across the complete Q projection and complete K projection,
// not independently per head. RoPE then rotates the first ROTARY_DIM channels
// of every Q and K head. V is copied unchanged.
//
// Input and output must not alias: the RoPE pass rereads the unnormalized input
// to preserve the checkpoint's bf16 round-trip between RMSNorm and RoPE.
template <typename T,
          int NUM_Q_HEADS,
          int NUM_KV_HEADS,
          int HEAD_DIM,
          int ROTARY_DIM,
          int QKV_STRIDE>
__device__ __forceinline__ void minimax_qk_norm_rope_sm100(
    void const *qkv_ptr,
    void const *q_norm_weight_ptr,
    void const *k_norm_weight_ptr,
    void const *cos_ptr,
    void const *sin_ptr,
    void *output_ptr,
    float eps) {
  static_assert(NUM_Q_HEADS % NUM_KV_HEADS == 0);
  static_assert(ROTARY_DIM > 0 && ROTARY_DIM <= HEAD_DIM);
  static_assert(ROTARY_DIM % 2 == 0);
  constexpr int Q_HEADS_PER_KV = NUM_Q_HEADS / NUM_KV_HEADS;
  constexpr int GROUP_STRIDE = (Q_HEADS_PER_KV + 2) * HEAD_DIM;
  constexpr int Q_DIM = NUM_Q_HEADS * HEAD_DIM;
  constexpr int K_DIM = NUM_KV_HEADS * HEAD_DIM;
  static_assert(QKV_STRIDE == (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM);

  // Blackwell worker blocks have 256 threads, while MPK task bodies use the
  // first 128. A named barrier keeps the inactive half out of synchronization.
  cutlass::arch::NamedBarrier task_barrier(NUM_THREADS, /*bar-id=*/6);
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }

  T const *qkv = static_cast<T const *>(qkv_ptr);
  T const *q_weight = static_cast<T const *>(q_norm_weight_ptr);
  T const *k_weight = static_cast<T const *>(k_norm_weight_ptr);
  T const *cos = static_cast<T const *>(cos_ptr);
  T const *sin = static_cast<T const *>(sin_ptr);
  T *output = static_cast<T *>(output_ptr);

  float q_ss = 0.0f;
  for (int idx = threadIdx.x; idx < Q_DIM; idx += NUM_THREADS) {
    int const head = idx / HEAD_DIM;
    int const dim = idx % HEAD_DIM;
    int const group = head / Q_HEADS_PER_KV;
    int const head_in_group = head % Q_HEADS_PER_KV;
    int const col = group * GROUP_STRIDE + head_in_group * HEAD_DIM + dim;
    float const value = static_cast<float>(qkv[col]);
    q_ss += value * value;
  }

  float k_ss = 0.0f;
  for (int idx = threadIdx.x; idx < K_DIM; idx += NUM_THREADS) {
    int const group = idx / HEAD_DIM;
    int const dim = idx % HEAD_DIM;
    int const col =
        group * GROUP_STRIDE + Q_HEADS_PER_KV * HEAD_DIM + dim;
    float const value = static_cast<float>(qkv[col]);
    k_ss += value * value;
  }

  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    q_ss += __shfl_down_sync(0xffffffff, q_ss, offset);
    k_ss += __shfl_down_sync(0xffffffff, k_ss, offset);
  }

  __shared__ float reductions[2 * NUM_WARPS];
  int const warp = threadIdx.x / NUM_THREADS_PER_WARP;
  int const lane = threadIdx.x % NUM_THREADS_PER_WARP;
  if (lane == 0) {
    reductions[warp] = q_ss;
    reductions[NUM_WARPS + warp] = k_ss;
  }
  task_barrier.arrive_and_wait();

  if (warp == 0) {
    q_ss = lane < NUM_WARPS ? reductions[lane] : 0.0f;
    k_ss = lane < NUM_WARPS ? reductions[NUM_WARPS + lane] : 0.0f;
    for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
      q_ss += __shfl_down_sync(0xffffffff, q_ss, offset);
      k_ss += __shfl_down_sync(0xffffffff, k_ss, offset);
    }
    if (lane == 0) {
      reductions[0] = rsqrtf(q_ss / static_cast<float>(Q_DIM) + eps);
      reductions[1] = rsqrtf(k_ss / static_cast<float>(K_DIM) + eps);
    }
  }
  task_barrier.arrive_and_wait();
  float const q_rms_rcp = reductions[0];
  float const k_rms_rcp = reductions[1];

  // Write the normalized projections first. Conversion to T here is the
  // checkpoint's dtype round-trip before RoPE.
  for (int idx = threadIdx.x; idx < Q_DIM; idx += NUM_THREADS) {
    int const head = idx / HEAD_DIM;
    int const dim = idx % HEAD_DIM;
    int const group = head / Q_HEADS_PER_KV;
    int const head_in_group = head % Q_HEADS_PER_KV;
    int const col = group * GROUP_STRIDE + head_in_group * HEAD_DIM + dim;
    output[col] = static_cast<T>(static_cast<float>(qkv[col]) * q_rms_rcp *
                                 static_cast<float>(q_weight[idx]));
  }
  for (int idx = threadIdx.x; idx < K_DIM; idx += NUM_THREADS) {
    int const group = idx / HEAD_DIM;
    int const dim = idx % HEAD_DIM;
    int const col =
        group * GROUP_STRIDE + Q_HEADS_PER_KV * HEAD_DIM + dim;
    output[col] = static_cast<T>(static_cast<float>(qkv[col]) * k_rms_rcp *
                                 static_cast<float>(k_weight[idx]));
  }

  // V is already in the attention layout and bypasses normalization/RoPE.
  for (int idx = threadIdx.x; idx < K_DIM; idx += NUM_THREADS) {
    int const group = idx / HEAD_DIM;
    int const dim = idx % HEAD_DIM;
    int const col =
        group * GROUP_STRIDE + (Q_HEADS_PER_KV + 1) * HEAD_DIM + dim;
    output[col] = qkv[col];
  }

  // Do not let a late normalized write race with the RoPE overwrite of the
  // same Q/K channel from another warp.
  task_barrier.arrive_and_wait();

  constexpr int ROTARY_HALF = ROTARY_DIM / 2;
  for (int idx = threadIdx.x; idx < NUM_Q_HEADS * ROTARY_DIM;
       idx += NUM_THREADS) {
    int const head = idx / ROTARY_DIM;
    int const dim = idx % ROTARY_DIM;
    int const pair_dim =
        dim < ROTARY_HALF ? dim + ROTARY_HALF : dim - ROTARY_HALF;
    int const group = head / Q_HEADS_PER_KV;
    int const head_in_group = head % Q_HEADS_PER_KV;
    int const base = group * GROUP_STRIDE + head_in_group * HEAD_DIM;
    int const weight_base = head * HEAD_DIM;
    T const value_t = static_cast<T>(
        static_cast<float>(qkv[base + dim]) * q_rms_rcp *
        static_cast<float>(q_weight[weight_base + dim]));
    T const pair_t = static_cast<T>(
        static_cast<float>(qkv[base + pair_dim]) * q_rms_rcp *
        static_cast<float>(q_weight[weight_base + pair_dim]));
    float const rotated = dim < ROTARY_HALF ? -static_cast<float>(pair_t)
                                             : static_cast<float>(pair_t);
    output[base + dim] = static_cast<T>(static_cast<float>(value_t) *
                                           static_cast<float>(cos[dim]) +
                                       rotated * static_cast<float>(sin[dim]));
  }

  for (int idx = threadIdx.x; idx < NUM_KV_HEADS * ROTARY_DIM;
       idx += NUM_THREADS) {
    int const group = idx / ROTARY_DIM;
    int const dim = idx % ROTARY_DIM;
    int const pair_dim =
        dim < ROTARY_HALF ? dim + ROTARY_HALF : dim - ROTARY_HALF;
    int const base = group * GROUP_STRIDE + Q_HEADS_PER_KV * HEAD_DIM;
    int const weight_base = group * HEAD_DIM;
    T const value_t = static_cast<T>(
        static_cast<float>(qkv[base + dim]) * k_rms_rcp *
        static_cast<float>(k_weight[weight_base + dim]));
    T const pair_t = static_cast<T>(
        static_cast<float>(qkv[base + pair_dim]) * k_rms_rcp *
        static_cast<float>(k_weight[weight_base + pair_dim]));
    float const rotated = dim < ROTARY_HALF ? -static_cast<float>(pair_t)
                                             : static_cast<float>(pair_t);
    output[base + dim] = static_cast<T>(static_cast<float>(value_t) *
                                           static_cast<float>(cos[dim]) +
                                       rotated * static_cast<float>(sin[dim]));
  }
  task_barrier.arrive_and_wait();
}

} // namespace kernel
