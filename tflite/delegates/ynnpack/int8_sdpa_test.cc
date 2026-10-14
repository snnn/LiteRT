/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tflite/c/common.h"
#include "tflite/delegates/ynnpack/ynnpack_delegate.h"
#include "tflite/kernels/test_util.h"
#include "tflite/schema/schema_generated.h"

namespace tflite {
namespace ynnpack {
namespace {

// Canonical data uses [head, row, channel]; only the model's boundary buffers
// are transposed. The reference does not call YNN or the production dot helper.
struct Inputs {
  int heads = 2;
  int rows;
  static constexpr int kCapacity = 64;
  static constexpr int kChannels = 16;
  float key_scale = 0.015625f;
  float value_scale = 0.0234375f;
  int key_zero_point = -9;
  int value_zero_point = 13;
  float attention_scale = 0.25f;
  float logit_cap = 0.0f;
  bool boolean_mask = false;
  std::vector<float> query;
  std::vector<int8_t> key;
  std::vector<int8_t> value;
  std::vector<float> mask;

  explicit Inputs(int query_rows) : rows(query_rows) {
    query.resize(heads * rows * kChannels);
    key.resize(heads * kCapacity * kChannels);
    value.resize(key.size());
    mask.resize(rows * kCapacity);
    for (size_t i = 0; i < query.size(); ++i) {
      query[i] = 0.65f * std::sin(0.31f * static_cast<float>(i + 1));
    }
    for (size_t i = 0; i < key.size(); ++i) {
      const size_t head = i / (kCapacity * kChannels);
      key[i] = static_cast<int8_t>(
          static_cast<int>((17 * i + 29 * head) % 256) - 128);
      value[i] = static_cast<int8_t>(
          static_cast<int>((43 * i + 71 * head) % 256) - 128);
    }
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < kCapacity; ++col) {
        mask[row * kCapacity + col] =
            (col + row) % 7 == 0 ? -std::numeric_limits<float>::infinity()
                                 : 0.03f * static_cast<float>(col % 3);
      }
    }
  }
};

template <typename T>
std::vector<T> Pack(const std::vector<T>& source, int heads, int rows,
                    bool sequence_major, bool transpose_channels = false) {
  std::vector<T> result(source.size());
  for (int head = 0; head < heads; ++head) {
    for (int row = 0; row < rows; ++row) {
      for (int channel = 0; channel < Inputs::kChannels; ++channel) {
        const int src = (head * rows + row) * Inputs::kChannels + channel;
        const int dst = sequence_major
                            ? (row * heads + head) * Inputs::kChannels + channel
                        : transpose_channels
                            ? (head * Inputs::kChannels + channel) * rows + row
                            : src;
        result[dst] = source[src];
      }
    }
  }
  return result;
}

TfLiteRegistration* RegisterUnimplementedSdpa() {
  static TfLiteRegistration registration = [] {
    TfLiteRegistration result{};
    result.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      const TfLiteTensor& query = context->tensors[node->inputs->data[0]];
      return context->ResizeTensor(context,
                                   &context->tensors[node->outputs->data[0]],
                                   TfLiteIntArrayCopy(query.dims));
    };
    // Accidental fallback must fail rather than silently produce zero output.
    result.invoke = [](TfLiteContext*, TfLiteNode*) { return kTfLiteError; };
    return result;
  }();
  return &registration;
}

class Int8SdpaModel : public MultiOpModel {
 public:
  Int8SdpaModel(const Inputs& inputs, bool sequence_major,
                bool constant_cache = false)
      : sequence_major_(sequence_major),
        constant_cache_(constant_cache),
        boolean_mask_(inputs.boolean_mask) {
    const int n = inputs.heads;
    const int t = inputs.rows;
    const int s = Inputs::kCapacity;
    const int d = Inputs::kChannels;
    const std::vector<int> q_shape = sequence_major
                                         ? std::vector<int>{1, t, n, d}
                                         : std::vector<int>{1, n, t, d};
    const std::vector<int> k_shape = sequence_major
                                         ? std::vector<int>{1, s, n, d}
                                         : std::vector<int>{1, n, s, d};
    const std::vector<int> v_shape = sequence_major
                                         ? std::vector<int>{1, s, n, d}
                                         : std::vector<int>{1, n, d, s};
    query_ = AddInput({TensorType_FLOAT32, q_shape});
    const TensorData k_type(TensorType_INT8, k_shape, 0, 0, inputs.key_scale,
                            inputs.key_zero_point);
    const TensorData v_type(TensorType_INT8, v_shape, 0, 0, inputs.value_scale,
                            inputs.value_zero_point);
    std::vector<std::vector<int>> input_shapes{q_shape};
    if (constant_cache) {
      auto k = Pack(inputs.key, n, s, sequence_major);
      auto v = Pack(inputs.value, n, s, sequence_major, !sequence_major);
      key_ = AddConstInput<int8_t>(k_type, k.data(), k.size());
      value_ = AddConstInput<int8_t>(v_type, v.data(), v.size());
      input_shapes.push_back({});
      input_shapes.push_back({});
    } else {
      key_ = AddInput(k_type);
      value_ = AddInput(v_type);
      input_shapes.push_back(k_shape);
      input_shapes.push_back(v_shape);
    }
    mask_ = AddInput(
        {boolean_mask_ ? TensorType_BOOL : TensorType_FLOAT32, {1, 1, t, s}});
    active_ = AddInput({TensorType_INT32, {1}});
    input_shapes.push_back({1, 1, t, s});
    input_shapes.push_back({1});
    output_ = AddOutput({TensorType_FLOAT32, q_shape});
    flexbuffers::Builder options;
    options.Map([&] {
      options.Float("scale", inputs.attention_scale);
      if (inputs.logit_cap > 0) options.Float("logit_cap", inputs.logit_cap);
    });
    options.Finish();
    AddCustomOp(sequence_major ? "odml.scaled_dot_product_attention"
                               : "odml.sdpa_transposed",
                options.GetBuffer(), RegisterUnimplementedSdpa,
                {query_, key_, value_, mask_, active_}, {output_});
    BuildInterpreter(input_shapes, -1, false, /*apply_delegate=*/false,
                     /*allocate_and_delegate=*/false);
    EXPECT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  }

  TfLiteStatus Delegate() {
    auto options = TfLiteYNNPackDelegateOptionsDefault();
    options.num_threads = 1;
    options.static_shape = true;
    SetDelegate(TfLiteYNNPackDelegateCreateUnique(&options));
    return ApplyDelegate();
  }
  void Populate(const Inputs& inputs, int active) {
    PopulateTensor(
        query_, Pack(inputs.query, inputs.heads, inputs.rows, sequence_major_));
    if (!constant_cache_) {
      PopulateTensor(key_, Pack(inputs.key, inputs.heads, Inputs::kCapacity,
                                sequence_major_));
      PopulateTensor(value_, Pack(inputs.value, inputs.heads, Inputs::kCapacity,
                                  sequence_major_, !sequence_major_));
    }
    if (boolean_mask_) {
      bool* mask = interpreter_->typed_tensor<bool>(mask_);
      for (size_t i = 0; i < inputs.mask.size(); ++i) {
        mask[i] = std::isfinite(inputs.mask[i]);
      }
    } else {
      PopulateTensor(mask_, inputs.mask);
    }
    PopulateTensor<int32_t>(active_, {active});
  }
  std::vector<float> Output() { return ExtractVector<float>(output_); }
  TfLiteTensor& Cache(bool value) {
    return *interpreter_->tensor(value ? value_ : key_);
  }

 private:
  bool sequence_major_;
  bool constant_cache_;
  bool boolean_mask_;
  int query_, key_, value_, mask_, active_, output_;
};

std::vector<float> Reference(const Inputs& inputs, int active,
                             bool sequence_major) {
  std::vector<float> result(inputs.query.size());
  for (int head = 0; head < inputs.heads; ++head) {
    for (int row = 0; row < inputs.rows; ++row) {
      std::vector<double> scores(active);
      for (int col = 0; col < active; ++col) {
        double dot = 0;
        for (int channel = 0; channel < Inputs::kChannels; ++channel) {
          const int q =
              (head * inputs.rows + row) * Inputs::kChannels + channel;
          const int k =
              (head * Inputs::kCapacity + col) * Inputs::kChannels + channel;
          dot += inputs.query[q] *
                 (static_cast<int>(inputs.key[k]) - inputs.key_zero_point) *
                 inputs.key_scale;
        }
        dot *= inputs.attention_scale;
        if (inputs.logit_cap > 0)
          dot = inputs.logit_cap * std::tanh(dot / inputs.logit_cap);
        scores[col] = dot + inputs.mask[row * Inputs::kCapacity + col];
      }
      const double maximum = *std::max_element(scores.begin(), scores.end());
      double sum = 0;
      for (double& score : scores) {
        score = std::exp(score - maximum);
        sum += score;
      }
      for (int channel = 0; channel < Inputs::kChannels; ++channel) {
        double value = 0;
        for (int col = 0; col < active; ++col) {
          const int v =
              (head * Inputs::kCapacity + col) * Inputs::kChannels + channel;
          value +=
              scores[col] / sum *
              (static_cast<int>(inputs.value[v]) - inputs.value_zero_point) *
              inputs.value_scale;
        }
        result[(head * inputs.rows + row) * Inputs::kChannels + channel] =
            value;
      }
    }
  }
  return Pack(result, inputs.heads, inputs.rows, sequence_major);
}

using LayoutAndRows = std::tuple<bool, int>;
class Int8SdpaTest : public testing::TestWithParam<LayoutAndRows> {};

TEST_P(Int8SdpaTest, AffineCacheAndChangingActivePrefix) {
  const auto [sequence_major, rows] = GetParam();
  Inputs inputs(rows);
  if (rows > 32) inputs.logit_cap = 1.5f;
  if (rows == 32) inputs.key_zero_point = inputs.value_zero_point = 0;
  Int8SdpaModel model(inputs, sequence_major);
  ASSERT_EQ(model.Delegate(), kTfLiteOk);
  ASSERT_EQ(model.CountNumberOfDelegatedPartitions(), 1);
  ASSERT_EQ(model.CountOpsExecutedByCpuKernel(), 0);
  float maximum_absolute_error = 0;
  for (int active : {13, 47, 64, 13}) {
    SCOPED_TRACE(active);
    model.Populate(inputs, active);
    ASSERT_EQ(model.Invoke(), kTfLiteOk);
    const auto expected = Reference(inputs, active, sequence_major);
    const auto actual = model.Output();
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
      ASSERT_TRUE(std::isfinite(actual[i]));
      maximum_absolute_error =
          std::max(maximum_absolute_error, std::abs(actual[i] - expected[i]));
      // Q and probabilities are dynamically quantized in this path; the
      // independent reference deliberately retains their original precision.
      // This absolute bound is for these bounded fixture inputs, not a model
      // logit-parity threshold.
      EXPECT_NEAR(actual[i], expected[i], 0.01f) << "element " << i;
    }
    if (active == 64) continue;
    Inputs poisoned = inputs;
    for (int head = 0; head < inputs.heads; ++head) {
      for (int col = active; col < Inputs::kCapacity; ++col) {
        for (int channel = 0; channel < Inputs::kChannels; ++channel) {
          const int i =
              (head * Inputs::kCapacity + col) * Inputs::kChannels + channel;
          poisoned.key[i] = 127;
          poisoned.value[i] = -128;
        }
      }
    }
    model.Populate(poisoned, active);
    ASSERT_EQ(model.Invoke(), kTfLiteOk);
    const auto after = model.Output();
    for (size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(after[i], actual[i], 1e-6f);
  }
  RecordProperty("maximum_absolute_error",
                 std::to_string(maximum_absolute_error));
}

INSTANTIATE_TEST_SUITE_P(LayoutAndQueryBranches, Int8SdpaTest,
                         testing::Combine(testing::Bool(),
                                          testing::Values(1, 32, 33)));

TEST(Int8SdpaTest, BooleanMaskAndLogitCap) {
  for (bool sequence_major : {false, true}) {
    SCOPED_TRACE(sequence_major);
    Inputs inputs(1);
    inputs.boolean_mask = true;
    inputs.logit_cap = 1.5f;
    for (float& mask : inputs.mask) {
      if (std::isfinite(mask)) mask = 0.0f;
    }
    Int8SdpaModel model(inputs, sequence_major);
    ASSERT_EQ(model.Delegate(), kTfLiteOk);
    ASSERT_EQ(model.CountNumberOfDelegatedPartitions(), 1);
    ASSERT_EQ(model.CountOpsExecutedByCpuKernel(), 0);
    float maximum_absolute_error = 0;
    for (int active : {13, 47}) {
      model.Populate(inputs, active);
      ASSERT_EQ(model.Invoke(), kTfLiteOk);
      const auto actual = model.Output();
      const auto expected = Reference(inputs, active, sequence_major);
      ASSERT_EQ(actual.size(), expected.size());
      for (size_t i = 0; i < actual.size(); ++i) {
        ASSERT_TRUE(std::isfinite(actual[i]));
        maximum_absolute_error =
            std::max(maximum_absolute_error, std::abs(actual[i] - expected[i]));
        EXPECT_NEAR(actual[i], expected[i], 0.01f);
      }
    }
    RecordProperty(sequence_major ? "sequence_major_maximum_absolute_error"
                                  : "transposed_maximum_absolute_error",
                   std::to_string(maximum_absolute_error));
  }
}

TEST(Int8SdpaTest, FullInt8RangeForConstantAndMutableCaches) {
  for (bool sequence_major : {false, true}) {
    for (bool constant_cache : {false, true}) {
      for (bool zero_zero_points : {false, true}) {
        SCOPED_TRACE(sequence_major);
        SCOPED_TRACE(constant_cache);
        SCOPED_TRACE(zero_zero_points);
        Inputs inputs(1);
        if (zero_zero_points)
          inputs.key_zero_point = inputs.value_zero_point = 0;
        std::fill(inputs.key.begin(), inputs.key.end(), int8_t{-128});
        std::fill(inputs.value.begin(), inputs.value.end(), int8_t{-128});
        std::fill(inputs.mask.begin(), inputs.mask.end(), 0.0f);
        Int8SdpaModel model(inputs, sequence_major, constant_cache);
        ASSERT_EQ(model.Delegate(), kTfLiteOk);
        ASSERT_EQ(model.CountNumberOfDelegatedPartitions(), 1);
        ASSERT_EQ(model.CountOpsExecutedByCpuKernel(), 0);
        for (int active : {13, 47}) {
          model.Populate(inputs, active);
          ASSERT_EQ(model.Invoke(), kTfLiteOk);
          const float expected =
              (-128 - inputs.value_zero_point) * inputs.value_scale;
          for (float value : model.Output()) {
            ASSERT_TRUE(std::isfinite(value));
            EXPECT_NEAR(value, expected, 1e-3f);
          }
        }
      }
    }
  }
}

TEST(Int8SdpaTest, ConstantCacheFullRangeAffectsAttentionScores) {
  for (bool sequence_major : {false, true}) {
    SCOPED_TRACE(sequence_major);
    Inputs inputs(1);
    inputs.key_scale = 0.125f;
    inputs.key_zero_point = inputs.value_zero_point = 0;
    std::fill(inputs.query.begin(), inputs.query.end(), 1.0f);
    std::fill(inputs.mask.begin(), inputs.mask.end(), 0.0f);
    for (int head = 0; head < inputs.heads; ++head) {
      for (int row = 0; row < Inputs::kCapacity; ++row) {
        const bool higher_key = (row + head) % 2 != 0;
        for (int channel = 0; channel < Inputs::kChannels; ++channel) {
          const int index =
              (head * Inputs::kCapacity + row) * Inputs::kChannels + channel;
          inputs.key[index] = higher_key ? -127 : -128;
          inputs.value[index] = higher_key ? 64 + 16 * head : -64 - 16 * head;
        }
      }
    }
    // The two score levels differ by 16 * 0.125 * 0.25 = 0.5.
    // Mishandling -128 as -127 would erase the distinction, changing the
    // weighted nonconstant V output by much more than the quantization bound.
    Int8SdpaModel model(inputs, sequence_major, /*constant_cache=*/true);
    ASSERT_EQ(model.Delegate(), kTfLiteOk);
    ASSERT_EQ(model.CountNumberOfDelegatedPartitions(), 1);
    ASSERT_EQ(model.CountOpsExecutedByCpuKernel(), 0);
    float maximum_absolute_error = 0;
    for (int active : {13, 47}) {
      model.Populate(inputs, active);
      ASSERT_EQ(model.Invoke(), kTfLiteOk);
      const auto actual = model.Output();
      const auto expected = Reference(inputs, active, sequence_major);
      ASSERT_EQ(actual.size(), expected.size());
      for (size_t i = 0; i < actual.size(); ++i) {
        ASSERT_TRUE(std::isfinite(actual[i]));
        maximum_absolute_error =
            std::max(maximum_absolute_error, std::abs(actual[i] - expected[i]));
        EXPECT_NEAR(actual[i], expected[i], 0.01f);
      }
    }
    RecordProperty(sequence_major ? "sequence_major_maximum_absolute_error"
                                  : "transposed_maximum_absolute_error",
                   std::to_string(maximum_absolute_error));
  }
}

// Restore the original allocation-owned metadata before the model is destroyed.
class QuantizationOverride {
 public:
  explicit QuantizationOverride(TfLiteTensor& tensor)
      : tensor_(tensor), saved_(tensor.quantization) {
    affine = *static_cast<TfLiteAffineQuantization*>(saved_.params);
    scales = TfLiteFloatArrayCreate(2);
    zero_points = TfLiteIntArrayCreate(2);
    scales->data[0] = scales->data[1] = affine.scale->data[0];
    zero_points->data[0] = zero_points->data[1] = affine.zero_point->data[0];
    scales->size = zero_points->size = 1;
    affine.scale = scales;
    affine.zero_point = zero_points;
    tensor_.quantization = {kTfLiteAffineQuantization, &affine};
  }
  ~QuantizationOverride() {
    tensor_.quantization = saved_;
    TfLiteFloatArrayFree(scales);
    TfLiteIntArrayFree(zero_points);
  }
  TfLiteAffineQuantization affine;
  TfLiteFloatArray* scales;
  TfLiteIntArray* zero_points;

 private:
  TfLiteTensor& tensor_;
  TfLiteQuantization saved_;
};

TEST(Int8SdpaTest, RejectsUnsupportedCacheQuantization) {
  for (bool value_cache : {false, true}) {
    for (int kind = 0; kind < 15; ++kind) {
      SCOPED_TRACE(value_cache);
      SCOPED_TRACE(kind);
      Inputs inputs(1);
      Int8SdpaModel model(inputs, false);
      TfLiteTensor& tensor = model.Cache(value_cache);
      QuantizationOverride modified(tensor);
      switch (kind) {
        case 0:
          modified.scales->data[0] = 0;
          break;
        case 1:
          modified.scales->data[0] = -0.1f;
          break;
        case 2:
          modified.scales->data[0] = std::numeric_limits<float>::quiet_NaN();
          break;
        case 3:
          modified.scales->data[0] = std::numeric_limits<float>::infinity();
          break;
        case 4:
          tensor.quantization.type = kTfLiteNoQuantization;
          break;
        case 5:
          tensor.quantization.params = nullptr;
          break;
        case 6:
          modified.affine.scale = nullptr;
          break;
        case 7:
          modified.scales->size = 0;
          break;
        case 8:
          modified.scales->size = 2;
          break;
        case 9:
          modified.affine.zero_point = nullptr;
          break;
        case 10:
          modified.zero_points->size = 2;
          break;
        case 11:
          modified.zero_points->data[0] = -129;
          break;
        case 12:
          modified.zero_points->data[0] = 128;
          break;
        case 13:
          modified.zero_points->size = 0;
          break;
        case 14:
          tensor.type = kTfLiteUInt8;
          break;
      }
      ASSERT_EQ(model.Delegate(), kTfLiteOk);
      EXPECT_EQ(model.CountNumberOfDelegatedPartitions(), 0);
      EXPECT_EQ(model.CountOpsExecutedByCpuKernel(), 1);
    }
  }
}

}  // namespace
}  // namespace ynnpack
}  // namespace tflite
