/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/runners/xnnpack/runner.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/runners/common_nnpack/runner_test_suite.h"
#include "tensor/tensor.h"

namespace litert::tensor {

struct XnnpackTestTraits {
  using Tag = XnnpackMixinTag;
  using Runner = XnnpackRunner;

  static constexpr uint32_t kFlagExternalInput = XNN_VALUE_FLAG_EXTERNAL_INPUT;
  static constexpr uint32_t kFlagExternalOutput =
      XNN_VALUE_FLAG_EXTERNAL_OUTPUT;

  static constexpr bool kSupportsConv2D = true;
  static constexpr bool kSupportsDepthwiseConv2D = true;
  static constexpr bool kSupportsTransposeConv2D = true;
  static constexpr bool kSupportsResize = true;
  static constexpr bool kSupportsSoftmaxBeta = false;
  // TODO(b/574438217): Enable once the XNNPACK backend is fixed. Its squeeze
  // freezes the dimensions it keeps, so reshaping the input fails in
  // `xnn_reshape_runtime`.
  static constexpr bool kSupportsSqueezeOfReshapedInput = false;
  // TODO(b/574438217): Enable once the XNNPACK backend rejects squeezing a
  // dimension whose extent isn't 1.
  static constexpr bool kRejectsInvalidSqueeze = false;
};

INSTANTIATE_TYPED_TEST_SUITE_P(Xnnpack, NnpackRunnerTest, XnnpackTestTraits);

namespace {

using XnnTensor = Tensor<XnnpackMixinTag>;

constexpr uint32_t kRuntimeFlags =
    XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC | XNN_FLAG_BASIC_PROFILING;

void ExpectConfiguredRun(XnnpackRunner& runner, const XnnTensor& output,
                        const std::array<float, 3>& input_values) {
  ASSERT_THAT(runner.Run(), IsOk());
  EXPECT_EQ(runner.runtime_flags(), kRuntimeFlags);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto values,
                                runner.ReadOutputAs<float>(output));
  ASSERT_EQ(values.size(), input_values.size());
  for (size_t i = 0; i < input_values.size(); ++i) {
    EXPECT_FLOAT_EQ(values.data()[i], 1.0f / std::sqrt(input_values[i]));
  }

  // Profiling succeeds only if the configured flags reached the actual runtime.
  size_t operator_count = 0;
  size_t required_size = 0;
  ASSERT_EQ(xnn_get_runtime_profiling_info(
                runner.runtime(), xnn_profile_info_num_operators,
                sizeof(operator_count), &operator_count, &required_size),
            xnn_status_success);
  EXPECT_GT(operator_count, 0);
}

TEST(XnnpackRunnerTest, RuntimeFlagsApplyToLazyCreationAndRepeatedRuns) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {3}});
  XnnTensor output = Rsqrt(input);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto runner, XnnpackRunner::Create({output}, kRuntimeFlags));
  EXPECT_EQ(runner.runtime(), nullptr);

  std::array<float, 3> data = {0.9646427035331726f, 1.0f, 4.0f};
  ASSERT_THAT(runner.SetInput(input, data), IsOk());
  ExpectConfiguredRun(runner, output, data);
  xnn_runtime_t runtime = runner.runtime();

  data = {9.0f, 16.0f, 25.0f};
  ExpectConfiguredRun(runner, output, data);
  EXPECT_EQ(runner.runtime(), runtime);
}

TEST(XnnpackRunnerTest, MoveBeforePreparationPreservesRuntimeFlags) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {3}});
  XnnTensor output = Rsqrt(input);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto runner, XnnpackRunner::Create({output}, kRuntimeFlags));
  const std::array<float, 3> data = {0.9646427035331726f, 1.0f, 4.0f};
  ASSERT_THAT(runner.SetInput(input, data), IsOk());

  XnnpackRunner moved(std::move(runner));
  EXPECT_EQ(moved.runtime(), nullptr);
  ExpectConfiguredRun(moved, output, data);
}

TEST(XnnpackRunnerTest, MoveAssignmentReplacesPreparedRuntimeAndFlags) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {3}});
  XnnTensor output = Rsqrt(input);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));
  XnnpackRunner source(std::move(graph), kRuntimeFlags);
  const std::array<float, 3> data = {0.9646427035331726f, 1.0f, 4.0f};
  ASSERT_THAT(source.SetInput(input, data), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto destination,
                                XnnpackRunner::Create({output}));
  ASSERT_THAT(destination.SetInput(input, data), IsOk());
  ASSERT_THAT(destination.Run(), IsOk());
  EXPECT_EQ(destination.runtime_flags(), 0);
  size_t operator_count = 0;
  size_t required_size = 0;
  EXPECT_EQ(xnn_get_runtime_profiling_info(
                destination.runtime(), xnn_profile_info_num_operators,
                sizeof(operator_count), &operator_count, &required_size),
            xnn_status_invalid_state);

  destination = std::move(source);
  EXPECT_EQ(destination.runtime(), nullptr);
  ExpectConfiguredRun(destination, output, data);
}

}  // namespace

TEST(XnnpackRunnerTest, SoftmaxRejectsBetaNotEqualToOne) {
  Tensor<XnnpackMixinTag> input({.name = "input",
                                 .type = Type::kFP32,
                                 .shape = {2},
                                 .buffer = std::vector<float>{0.f, 0.f}});
  Tensor<XnnpackMixinTag> output = Softmax(input, /*beta=*/2.0f);

  auto runner = XnnpackRunner::Create({output});
  EXPECT_TRUE(absl::IsUnimplemented(runner.status()));
}

TEST(XnnpackRunnerTest, SharesWeightsCacheAndWorkspaceAcrossRunners) {
  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);
  xnn_weights_cache_t raw_weights_cache = nullptr;
  ASSERT_EQ(xnn_create_weights_cache(&raw_weights_cache), xnn_status_success);
  std::unique_ptr<xnn_weights_cache_provider,
                  decltype(&xnn_delete_weights_cache)>
      weights_cache(raw_weights_cache, &xnn_delete_weights_cache);
  xnn_workspace_t raw_workspace = nullptr;
  ASSERT_EQ(xnn_create_workspace(&raw_workspace), xnn_status_success);
  std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> workspace(
      raw_workspace, &xnn_release_workspace);

  std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor<XnnpackMixinTag> w(
      {.name = "w", .type = Type::kFP32, .shape = {2, 3}, .buffer = weights});

  Tensor<XnnpackMixinTag> x_prefill(
      {.name = "x_prefill", .type = Type::kFP32, .shape = {2, 3}});
  Tensor<XnnpackMixinTag> y_prefill = FullyConnected(x_prefill, w);

  Tensor<XnnpackMixinTag> x_decode(
      {.name = "x_decode", .type = Type::kFP32, .shape = {1, 3}});
  Tensor<XnnpackMixinTag> y_decode = FullyConnected(x_decode, w);

  auto prefill_runner = XnnpackRunner::Create({y_prefill});
  ASSERT_TRUE(prefill_runner.ok());
  prefill_runner->SetWeightsCache(weights_cache.get());
  prefill_runner->SetWorkspace(workspace.get());
  ASSERT_TRUE(prefill_runner->PrepareRuntime().ok());

  auto decode_runner = XnnpackRunner::Create({y_decode});
  ASSERT_TRUE(decode_runner.ok());
  decode_runner->SetWeightsCache(weights_cache.get());
  decode_runner->SetWorkspace(workspace.get());
  ASSERT_TRUE(decode_runner->PrepareRuntime().ok());
  EXPECT_EQ(decode_runner->workspace(), workspace.get());
  ASSERT_EQ(xnn_finalize_weights_cache(
                weights_cache.get(), xnn_weights_cache_finalization_kind_soft),
            xnn_status_success);

  const std::vector<float> prefill_in = {1.0f, 0.0f, 2.0f, 0.5f, 1.0f, -1.0f};
  ASSERT_TRUE(prefill_runner->SetInput(x_prefill, prefill_in).ok());
  ASSERT_TRUE(prefill_runner->Run().ok());
  auto prefill_out = prefill_runner->ReadOutputAs<float>(y_prefill);
  ASSERT_TRUE(prefill_out.ok());
  ASSERT_EQ(prefill_out->size(), 4u);
  EXPECT_FLOAT_EQ(prefill_out->data()[0], 7.0f);
  EXPECT_FLOAT_EQ(prefill_out->data()[1], 16.0f);
  EXPECT_FLOAT_EQ(prefill_out->data()[2], -0.5f);
  EXPECT_FLOAT_EQ(prefill_out->data()[3], 1.0f);

  const std::vector<float> decode_in = {0.5f, 1.0f, -1.0f};
  ASSERT_TRUE(decode_runner->SetInput(x_decode, decode_in).ok());
  ASSERT_TRUE(decode_runner->Run().ok());
  auto decode_out = decode_runner->ReadOutputAs<float>(y_decode);
  ASSERT_TRUE(decode_out.ok());
  ASSERT_EQ(decode_out->size(), 2u);
  EXPECT_FLOAT_EQ(decode_out->data()[0], -0.5f);
  EXPECT_FLOAT_EQ(decode_out->data()[1], 1.0f);
}

}  // namespace litert::tensor
