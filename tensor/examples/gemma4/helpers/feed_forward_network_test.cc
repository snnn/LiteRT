/* Copyright 2026 Google LLC.

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

#include "tensor/examples/gemma4/helpers/feed_forward_network.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples::gemma4 {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;
using XnnTensor = Tensor<XnnpackMixinTag>;

TEST(Gemma4GraphTest, FeedForwardNetworkTest) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 1, 2}});

  XnnTensor gate_proj(
      {.name = "gate_proj",
       .type = Type::kFP32,
       .shape = {3, 2},
       .buffer = std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f, 1.0f, -1.0f}});
  XnnTensor up_proj(
      {.name = "up_proj",
       .type = Type::kFP32,
       .shape = {3, 2},
       .buffer = std::vector<float>{0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 0.5f}});
  XnnTensor down_proj(
      {.name = "down_proj",
       .type = Type::kFP32,
       .shape = {2, 3},
       .buffer = std::vector<float>{1.0f, 0.0f, 0.5f, 0.0f, 1.0f, -0.5f}});

  XnnTensor output = FeedForwardNetwork(input, gate_proj, up_proj, down_proj);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(XnnpackRunner runner,
                                  XnnpackRunner::Create({output}));

  const std::array<float, 2> input_data = {1.0f, 2.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> result,
                                  runner.ReadOutput(output));
  LockedBufferSpan<const float> floats = std::move(result).As<const float>();
  ASSERT_EQ(floats.size(), 2);

  // Expected data computed using the script in
  // `./reference/feed_forward_network.py`.
  const std::array<float, 2> expected_data = {1.182384f, 2.034002f};

  EXPECT_THAT(floats, Pointwise(FloatNear(1e-5f), expected_data));
}

TEST(Gemma4GraphTest, MobileScalesAffectEveryFeedForwardProjection) {
  constexpr int channels = 32;
  // Identity projections produce GELU(1) without activation quantization.
  // Give one projection an input scale of 4: its input rounds to zero, and so
  // does the final output. A generic FeedForward would ignore these scales.
  for (const std::string module :
       {"", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}) {
    SCOPED_TRACE(module);
    XnnTensor input({.type = Type::kFP32, .shape = {1, 1, channels}});
    auto projection = [](const char* name) {
      std::vector<int8_t> identity(channels * channels, 0);
      for (int i = 0; i < channels; ++i) identity[i * channels + i] = 1;
      return XnnTensor(
          {.name = name,
           .type = Type::kI8,
           .shape = {channels, channels},
           .buffer = identity,
           .quantization = std::make_shared<PerChannelAffineQuantization>(
               std::vector<float>(channels, 1.0f), std::vector<int64_t>{0},
               0)});
    };
    absl::flat_hash_map<std::string, XnnTensor> weights;
    if (!module.empty()) {
      weights.emplace(
          module + ".input_scale",
          XnnTensor({.type = Type::kFP32, .shape = {}, .buffer = 4.0f}));
      weights.emplace(
          module + ".output_scale",
          XnnTensor({.type = Type::kFP32, .shape = {}, .buffer = 1.0f}));
    }

    auto output =
        FeedForwardNetwork(input, projection("mlp.gate_proj.weight"),
                           projection("mlp.up_proj.weight"),
                           projection("mlp.down_proj.weight"), &weights);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner,
                                    XnnpackRunner::Create({output}));
    const std::vector<float> input_data(channels, 1.0f);
    ASSERT_THAT(runner.SetInput(input, input_data), IsOk());
    ASSERT_THAT(runner.Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto actual,
                                    runner.ReadOutputAs<float>(output));
    EXPECT_THAT(actual,
                Pointwise(FloatNear(1e-5f),
                          std::vector<float>(
                              channels, module.empty() ? 0.841192f : 0.0f)));
  }
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4
