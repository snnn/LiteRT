// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"

namespace ynn {
namespace {

using subgraph_ptr =
    std::unique_ptr<ynn_subgraph, decltype(&ynn_delete_subgraph)>;
using runtime_ptr = std::unique_ptr<ynn_runtime, decltype(&ynn_delete_runtime)>;
using threadpool_ptr =
    std::unique_ptr<ynn_threadpool, decltype(&ynn_delete_threadpool)>;

constexpr size_t kGroups = 8;
constexpr size_t kQueryHeadsPerGroup = 2;
constexpr size_t kHeadDim = 128;

enum class KLayout {
  kDotReady,
  kDepthMajor,
};

void Check(ynn_status status, const char* what) {
  if (status != ynn_status_success) {
    std::cerr << what << " failed with status " << static_cast<int>(status)
              << std::endl;
    std::abort();
  }
}

uint32_t DefineScalar(ynn_subgraph_t subgraph, float value) {
  uint32_t id = YNN_INVALID_VALUE_ID;
  Check(ynn_define_tensor(subgraph, ynn_type_fp32, 0, nullptr, &value,
                          YNN_VALUE_FLAG_COPY_DATA, &id),
        "ynn_define_tensor(scalar)");
  return id;
}

struct Buffers {
  std::vector<float> q;
  std::vector<float> k;
  std::vector<float> v;
  std::vector<float> out;
  std::vector<float> ref;
};

struct AttentionRuntime {
  threadpool_ptr threadpool;
  runtime_ptr runtime;
};

size_t QIndex(size_t group, size_t q_head, size_t d) {
  return (group * kQueryHeadsPerGroup + q_head) * kHeadDim + d;
}

size_t KIndex(KLayout layout, size_t depth, size_t group, size_t d, size_t t) {
  if (layout == KLayout::kDepthMajor) {
    return (group * depth + t) * kHeadDim + d;
  }
  return (group * kHeadDim + d) * depth + t;
}

size_t VIndex(size_t depth, size_t group, size_t t, size_t d) {
  return (group * depth + t) * kHeadDim + d;
}

size_t OutIndex(size_t group, size_t q_head, size_t d) {
  return (group * kQueryHeadsPerGroup + q_head) * kHeadDim + d;
}

Buffers MakeBuffers(size_t depth, KLayout k_layout) {
  Buffers buffers;
  buffers.q.resize(kGroups * kQueryHeadsPerGroup * kHeadDim);
  buffers.k.resize(kGroups * kHeadDim * depth);
  buffers.v.resize(kGroups * depth * kHeadDim);
  buffers.out.resize(kGroups * kQueryHeadsPerGroup * kHeadDim);
  buffers.ref.resize(buffers.out.size());

  std::mt19937 rng(0x1234);
  std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
  for (float& x : buffers.q) x = dist(rng);
  for (size_t group = 0; group < kGroups; ++group) {
    for (size_t t = 0; t < depth; ++t) {
      for (size_t d = 0; d < kHeadDim; ++d) {
        buffers.k[KIndex(k_layout, depth, group, d, t)] = dist(rng);
      }
    }
  }
  for (float& x : buffers.v) x = dist(rng);
  return buffers;
}

void ReferenceAttention(size_t depth, KLayout k_layout, const Buffers& buffers,
                        std::vector<float>& ref) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  std::vector<float> scores(depth);
  std::vector<float> probs(depth);

  for (size_t group = 0; group < kGroups; ++group) {
    for (size_t q_head = 0; q_head < kQueryHeadsPerGroup; ++q_head) {
      float max_score = -std::numeric_limits<float>::infinity();
      for (size_t t = 0; t < depth; ++t) {
        float score = 0.0f;
        for (size_t d = 0; d < kHeadDim; ++d) {
          score += buffers.q[QIndex(group, q_head, d)] *
                   buffers.k[KIndex(k_layout, depth, group, d, t)];
        }
        score *= scale;
        scores[t] = score;
        max_score = std::max(max_score, score);
      }

      float denom = 0.0f;
      for (size_t t = 0; t < depth; ++t) {
        const float p = std::exp(scores[t] - max_score);
        probs[t] = p;
        denom += p;
      }

      for (size_t d = 0; d < kHeadDim; ++d) {
        float acc = 0.0f;
        for (size_t t = 0; t < depth; ++t) {
          acc += (probs[t] / denom) * buffers.v[VIndex(depth, group, t, d)];
        }
        ref[OutIndex(group, q_head, d)] = acc;
      }
    }
  }
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  double max_abs = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    max_abs = std::max(max_abs, static_cast<double>(std::abs(a[i] - b[i])));
  }
  return max_abs;
}

AttentionRuntime CreateAttentionRuntime(TestScheduler& scheduler,
                                        KLayout k_layout) {
  ynn_threadpool_t threadpool_raw = nullptr;
  Check(ynn_create_threadpool(TestScheduler::scheduler(), &scheduler, 0,
                              &threadpool_raw),
        "ynn_create_threadpool");
  threadpool_ptr threadpool(threadpool_raw, &ynn_delete_threadpool);

  ynn_subgraph_t subgraph_raw = nullptr;
  Check(ynn_create_subgraph(4, YNN_FLAG_FAST_MATH, &subgraph_raw),
        "ynn_create_subgraph");
  subgraph_ptr subgraph(subgraph_raw, &ynn_delete_subgraph);

  const uint32_t q_id = 0;
  const uint32_t k_id = 1;
  const uint32_t v_id = 2;
  const uint32_t out_id = 3;

  const size_t q_dims[] = {kGroups, kQueryHeadsPerGroup, kHeadDim};
  const size_t k_dot_ready_dims[] = {kGroups, kHeadDim, 0};
  const size_t k_depth_major_dims[] = {kGroups, 0, kHeadDim};
  const size_t v_dims[] = {kGroups, 0, kHeadDim};
  const size_t out_dims[] = {kGroups, kQueryHeadsPerGroup, kHeadDim};
  const size_t* k_dims = k_layout == KLayout::kDepthMajor ? k_depth_major_dims
                                                          : k_dot_ready_dims;

  uint32_t defined_q_id = q_id;
  uint32_t defined_k_id = k_id;
  uint32_t defined_v_id = v_id;
  uint32_t defined_out_id = out_id;
  Check(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 3, q_dims, nullptr,
                          YNN_VALUE_FLAG_EXTERNAL_INPUT, &defined_q_id),
        "define q");
  Check(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 3, k_dims, nullptr,
                          YNN_VALUE_FLAG_EXTERNAL_INPUT, &defined_k_id),
        "define k");
  Check(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 3, v_dims, nullptr,
                          YNN_VALUE_FLAG_EXTERNAL_INPUT, &defined_v_id),
        "define v");
  Check(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 3, out_dims, nullptr,
                          YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &defined_out_id),
        "define output");

  const uint32_t scale_id =
      DefineScalar(subgraph.get(), 1.0f / std::sqrt(static_cast<float>(kHeadDim)));

  uint32_t k_for_qk_id = k_id;
  if (k_layout == KLayout::kDepthMajor) {
    const int32_t perm[] = {0, 2, 1};
    k_for_qk_id = YNN_INVALID_VALUE_ID;
    Check(ynn_define_static_transpose(subgraph.get(), 3, perm, k_id,
                                      &k_for_qk_id, 0),
          "transpose k");
  }

  uint32_t qk_id = YNN_INVALID_VALUE_ID;
  Check(ynn_define_dot(subgraph.get(), 1, q_id, k_for_qk_id,
                       YNN_INVALID_VALUE_ID,
                       &qk_id, 0),
        "qk dot");

  uint32_t scaled_id = YNN_INVALID_VALUE_ID;
  Check(ynn_define_binary(subgraph.get(), ynn_binary_multiply, qk_id, scale_id,
                          &scaled_id, 0),
        "scale scores");

  const int32_t depth_axis = 2;
  uint32_t max_id = YNN_INVALID_VALUE_ID;
  Check(ynn_define_reduce(subgraph.get(), ynn_reduce_max, 1, &depth_axis,
                          scaled_id, YNN_INVALID_VALUE_ID, &max_id,
                          YNN_NODE_FLAG_KEEP_DIMS),
        "reduce max");

  uint32_t centered_id = YNN_INVALID_VALUE_ID;
  Check(ynn_define_binary(subgraph.get(), ynn_binary_subtract, scaled_id,
                          max_id, &centered_id, 0),
        "subtract max");

  uint32_t exp_id = YNN_INVALID_VALUE_ID;
  Check(ynn_define_unary(subgraph.get(), ynn_unary_exp, centered_id, &exp_id,
                         0),
        "exp");

  uint32_t sum_id = YNN_INVALID_VALUE_ID;
  Check(ynn_define_reduce(subgraph.get(), ynn_reduce_sum, 1, &depth_axis,
                          exp_id, YNN_INVALID_VALUE_ID, &sum_id,
                          YNN_NODE_FLAG_KEEP_DIMS),
        "reduce sum");

  uint32_t probs_id = YNN_INVALID_VALUE_ID;
  Check(ynn_define_binary(subgraph.get(), ynn_binary_divide, exp_id, sum_id,
                          &probs_id, 0),
        "normalize");

  uint32_t final_out_id = out_id;
  Check(ynn_define_dot(subgraph.get(), 1, probs_id, v_id,
                       YNN_INVALID_VALUE_ID, &final_out_id, 0),
        "pv dot");

  Check(ynn_optimize_subgraph(subgraph.get(), threadpool.get(), 0),
        "ynn_optimize_subgraph");

  ynn_runtime_t runtime_raw = nullptr;
  Check(ynn_create_runtime(subgraph.get(), threadpool.get(), 0, &runtime_raw),
        "ynn_create_runtime");
  runtime_ptr runtime(runtime_raw, &ynn_delete_runtime);

  return AttentionRuntime{std::move(threadpool), std::move(runtime)};
}

void RunAttentionDecodeLayer(benchmark::State& state, KLayout k_layout) {
  const size_t depth = static_cast<size_t>(state.range(0));
  const int thread_count = static_cast<int>(state.range(1));

  Buffers buffers = MakeBuffers(depth, k_layout);
  // `thread_count` includes the runtime's invoking thread, which participates
  // as a worker. Only create the remaining background threads.
  TestScheduler scheduler(thread_count - 1);
  AttentionRuntime attention_runtime =
      CreateAttentionRuntime(scheduler, k_layout);
  ynn_runtime_t runtime = attention_runtime.runtime.get();

  const size_t q_dims[] = {kGroups, kQueryHeadsPerGroup, kHeadDim};
  const size_t k_dot_ready_dims[] = {kGroups, kHeadDim, depth};
  const size_t k_depth_major_dims[] = {kGroups, depth, kHeadDim};
  const size_t v_dims[] = {kGroups, depth, kHeadDim};
  const size_t* k_dims = k_layout == KLayout::kDepthMajor ? k_depth_major_dims
                                                          : k_dot_ready_dims;
  Check(ynn_set_external_value_shape(runtime, 0, 3, q_dims),
        "set q shape");
  Check(ynn_set_external_value_shape(runtime, 1, 3, k_dims),
        "set k shape");
  Check(ynn_set_external_value_shape(runtime, 2, 3, v_dims),
        "set v shape");
  Check(ynn_reshape_runtime(runtime), "reshape");
  Check(ynn_set_external_value_data(runtime, 0, buffers.q.data()),
        "set q data");
  Check(ynn_set_external_value_data(runtime, 1, buffers.k.data()),
        "set k data");
  Check(ynn_set_external_value_data(runtime, 2, buffers.v.data()),
        "set v data");
  Check(ynn_set_external_value_data(runtime, 3, buffers.out.data()),
        "set out data");

  Check(ynn_invoke_runtime(runtime), "warmup invoke");
  ReferenceAttention(depth, k_layout, buffers, buffers.ref);
  const double max_abs = MaxAbsDiff(buffers.out, buffers.ref);
  if (max_abs > 1.0e-4) {
    state.SkipWithError(("max_abs mismatch: " + std::to_string(max_abs)).c_str());
    return;
  }

  for (auto _ : state) {
    Check(ynn_invoke_runtime(runtime), "invoke");
  }

  state.counters["depth"] = static_cast<double>(depth);
  state.counters["threads"] = static_cast<double>(thread_count);
  state.counters["max_abs"] = max_abs;
  state.counters["logical_scores_mb"] =
      static_cast<double>(kGroups * kQueryHeadsPerGroup * depth *
                          sizeof(float)) /
      (1024.0 * 1024.0);
}

void BM_attention_decode_layer_fp32_k_dot_ready(benchmark::State& state) {
  RunAttentionDecodeLayer(state, KLayout::kDotReady);
}

void BM_attention_decode_layer_fp32_k_depth_major(benchmark::State& state) {
  RunAttentionDecodeLayer(state, KLayout::kDepthMajor);
}

static void Configure(benchmark::Benchmark* b) {
  b->ArgNames({"depth", "thread_count"});
  b->UseRealTime();
  b->MeasureProcessCPUTime();
  for (int depth : {64, 256, 1024, 4096, 8192, 16384}) {
    for (int thread_count : {1, 4, 8, 16}) {
      b->Args({depth, thread_count});
    }
  }
}

BENCHMARK(BM_attention_decode_layer_fp32_k_dot_ready)->Apply(Configure);
BENCHMARK(BM_attention_decode_layer_fp32_k_depth_major)->Apply(Configure);

}  // namespace
}  // namespace ynn
