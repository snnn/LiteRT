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

#include "tensor/examples/utils/safetensor_loader.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/utils/safetensor_test_util.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::FloatEq;
using ::testing::Not;

TEST(SafetensorLoaderTest, AbslStringifyMethodAndStrategy) {
  EXPECT_EQ(absl::StrCat(QuantizationConfig::Method::kCompressedTensors),
            "compressed-tensors");
  EXPECT_EQ(absl::StrCat(QuantizationConfig::Method::kUnknown), "unknown");

  EXPECT_EQ(absl::StrCat(QuantizationConfig::Strategy::kTensor), "tensor");
  EXPECT_EQ(absl::StrCat(QuantizationConfig::Strategy::kChannel), "channel");
  EXPECT_EQ(absl::StrCat(QuantizationConfig::Strategy::kGroup), "group");
  EXPECT_EQ(absl::StrCat(QuantizationConfig::Strategy::kUnknown), "unknown");
}

TEST(SafetensorLoaderTest, ParseTopLevelConfig) {
  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
      "group_0": {
        "num_bits": 4,
        "group_size": 128,
        "symmetric": true
      }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  EXPECT_EQ(quant_config->quant_method,
            QuantizationConfig::Method::kCompressedTensors);
  EXPECT_EQ(quant_config->num_bits, 4);
  EXPECT_EQ(quant_config->group_size, 128);
  EXPECT_TRUE(quant_config->symmetric);
}

TEST(SafetensorLoaderTest, ParseNestedConfigGroups) {
  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "quantization_status": "compressed",
    "config_groups": {
      "group_0": {
        "weights": {
          "num_bits": 4,
          "type": "int",
          "symmetric": true,
          "strategy": "group",
          "group_size": 128
        },
        "targets": ["Linear"]
      }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  EXPECT_EQ(quant_config->quant_method,
            QuantizationConfig::Method::kCompressedTensors);
  EXPECT_EQ(quant_config->num_bits, 4);
  EXPECT_EQ(quant_config->group_size, 128);
  EXPECT_TRUE(quant_config->symmetric);
}

TEST(SafetensorLoaderTest, ParseInt8Config) {
  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "config_groups": {
      "group_0": {
        "weights": {
          "num_bits": 8
        }
      }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  EXPECT_EQ(quant_config->num_bits, 8);
}

TEST(SafetensorLoaderTest, ParsesMultipleConfigGroups) {
  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "ignore": ["model.vision_tower", "relative_k_proj"],
    "config_groups": {
      "group_0": {
        "weights": { "num_bits": 2, "strategy": "channel", "group_size": null },
        "targets": ["model.embed_tokens", "re:.*lm_head$"]
      },
      "group_1": {
        "weights": { "num_bits": 4, "strategy": "group", "group_size": 128 },
        "targets": ["model.layers.0.mlp.down_proj"]
      }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  ASSERT_EQ(quant_config->schemes.size(), 2);

  // A group without a group size quantizes per channel.
  const QuantizationConfig::Scheme* embed =
      quant_config->FindScheme("model.embed_tokens");
  ASSERT_NE(embed, nullptr);
  EXPECT_EQ(embed->num_bits, 2);
  EXPECT_EQ(embed->strategy, QuantizationConfig::Strategy::kChannel);

  // Targets also match as regular expressions.
  EXPECT_EQ(quant_config->FindScheme("model.lm_head"), embed);

  const QuantizationConfig::Scheme* down_proj =
      quant_config->FindScheme("model.layers.0.mlp.down_proj");
  ASSERT_NE(down_proj, nullptr);
  EXPECT_EQ(down_proj->num_bits, 4);
  EXPECT_EQ(down_proj->strategy, QuantizationConfig::Strategy::kGroup);
  EXPECT_EQ(down_proj->group_size, 128);

  // Unclaimed modules have no scheme, since every group names its targets.
  EXPECT_EQ(quant_config->FindScheme("model.layers.0.mlp.up_proj"), nullptr);

  // Ignored modules are never quantized, whether named as a parent or a leaf.
  EXPECT_TRUE(quant_config->IsIgnored("model.vision_tower.layers.0.self_attn"));
  EXPECT_TRUE(
      quant_config->IsIgnored("model.layers.0.self_attn.relative_k_proj"));
  EXPECT_FALSE(quant_config->IsIgnored("model.embed_tokens"));
}

TEST(SafetensorLoaderTest, TargetsNamingAModuleClassApplyToEveryModule) {
  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
      "group_0": {
        "weights": { "num_bits": 4, "group_size": 128 },
        "targets": ["Linear"]
      }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(json);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  const auto& quant_config = loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  ASSERT_EQ(quant_config->schemes.size(), 1);
  EXPECT_TRUE(quant_config->schemes.front().matches_any_module);
  EXPECT_NE(quant_config->FindScheme("model.layers.3.mlp.up_proj"), nullptr);
}

TEST(SafetensorLoaderTest, LoadsPackedInt4WeightPerChannel) {
  // Two rows of eight 4-bit values, spanning the whole signed range.
  auto values = OwningCpuBuffer::Copy<Type::kI4>(
      {-8, -1, 0, 7, 3, -4, 5, -6, 1, 2, 3, 4, 5, 6, 7, -7});
  auto scales = OwningCpuBuffer::Copy<Type::kFP32>({0.5f, 0.25f});

  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
      "group_0": {
        "weights": { "num_bits": 4, "strategy": "channel" },
        "targets": ["model.layers.0.mlp.down_proj"]
      }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.layers.0.mlp.down_proj.weight_packed",
           .type = Type::kI4,
           .shape = {2, 8},
           .buffer = values},
          {.name = "model.layers.0.mlp.down_proj.weight_scale",
           .type = Type::kFP32,
           .shape = {2, 1},
           .buffer = scales},
      },
      json);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));
  // The weight is asked for by the name it would have when uncompressed.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      TensorHandle tensor,
      loader.LoadTensor("model.layers.0.mlp.down_proj.weight"));

  EXPECT_EQ(tensor.GetType(), Type::kI4);
  EXPECT_THAT(tensor.GetShape(), ElementsAre(2, 8));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, tensor.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const int4_t>(),
              ElementsAreArray(values->Span<int4_t>()));

  ASSERT_NE(tensor.GetQuantization(), nullptr);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      const PerChannelAffineQuantization& quantization,
      tensor.GetQuantization()->As<const PerChannelAffineQuantization>());
  EXPECT_THAT(quantization.scales, ElementsAre(FloatEq(0.5f), FloatEq(0.25f)));
  EXPECT_THAT(quantization.zero_points, ElementsAre(0));
  EXPECT_EQ(quantization.quantized_dimension, 0);
}

TEST(SafetensorLoaderTest, LoadsPackedInt2WeightBlockwise) {
  // One row of sixteen 2-bit values, i.e. one int32, in two blocks of eight.
  auto values = OwningCpuBuffer::Copy<Type::kI2>(
      {-2, -1, 0, 1, 1, 0, -1, -2, 0, 1, 1, 1, -2, -2, 0, 1});
  auto scales = OwningCpuBuffer::Copy<Type::kFP32>({0.125f, 0.0625f});

  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
      "group_0": {
        "weights": { "num_bits": 2, "strategy": "group", "group_size": 8 },
        "targets": ["model.embed_tokens_per_layer"]
      }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.embed_tokens_per_layer.weight_packed",
           .type = Type::kI2,
           .shape = {1, 16},
           .buffer = values},
          {.name = "model.embed_tokens_per_layer.weight_scale",
           .type = Type::kFP32,
           .shape = {1, 2},
           .buffer = scales},
      },
      json);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      TensorHandle tensor,
      loader.LoadTensor("model.embed_tokens_per_layer.weight"));

  EXPECT_EQ(tensor.GetType(), Type::kI2);
  EXPECT_THAT(tensor.GetShape(), ElementsAre(1, 16));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, tensor.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const int2_t>(),
              ElementsAreArray(values->Span<int2_t>()));

  ASSERT_NE(tensor.GetQuantization(), nullptr);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      const BlockwiseQuantization& quantization,
      tensor.GetQuantization()->As<const BlockwiseQuantization>());
  EXPECT_EQ(quantization.block_size, 8);
  EXPECT_THAT(quantization.scales,
              ElementsAre(FloatEq(0.125f), FloatEq(0.0625f)));
}

TEST(SafetensorLoaderTest, WeightShapeTrimsPackingPadding) {
  // Six values, i.e. one int32 worth of 4-bit fields with two to spare.
  auto values = OwningCpuBuffer::Copy<Type::kI4>({1, 2, 3, 4, 5, 6});
  auto logical_shape = OwningCpuBuffer::Copy<Type::kI64>({1, 6});
  auto scales = OwningCpuBuffer::Copy<Type::kFP32>({1});

  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
      "group_0": { "weights": { "num_bits": 4, "strategy": "channel" } }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.proj.weight_packed",
           .type = Type::kI4,
           .shape = {1, 6},
           .buffer = values},
          {.name = "model.proj.weight_scale",
           .type = Type::kFP32,
           .shape = {1, 1},
           .buffer = scales},
          {.name = "model.proj.weight_shape",
           .type = Type::kI64,
           .shape = {2},
           .buffer = logical_shape},
      },
      json);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle tensor,
                                  loader.LoadTensor("model.proj.weight"));

  EXPECT_THAT(tensor.GetShape(), ElementsAre(1, 6));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, tensor.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const int4_t>(),
              ElementsAreArray(values->Span<int4_t>()));
}

TEST(SafetensorLoaderTest, WeightShapeTrimsPackingPaddingOfEveryRow) {
  // Two rows of six values, each padded to its own int32.
  auto values = OwningCpuBuffer::Copy<Type::kI4>(
      {1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6});
  auto logical_shape = OwningCpuBuffer::Copy<Type::kI64>({2, 6});
  auto scales = OwningCpuBuffer::Copy<Type::kFP32>({1, 1});

  std::string json = R"({
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
      "group_0": { "weights": { "num_bits": 4, "strategy": "channel" } }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.proj.weight_packed",
           .type = Type::kI4,
           .shape = {2, 6},
           .buffer = values},
          {.name = "model.proj.weight_scale",
           .type = Type::kFP32,
           .shape = {2, 1},
           .buffer = scales},
          {.name = "model.proj.weight_shape",
           .type = Type::kI64,
           .shape = {2},
           .buffer = logical_shape},
      },
      json);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TensorHandle tensor,
                                  loader.LoadTensor("model.proj.weight"));

  EXPECT_THAT(tensor.GetShape(), ElementsAre(2, 6));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, tensor.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const int4_t>(),
              ElementsAreArray(values->Span<int4_t>()));
}

TEST(SafetensorLoaderTest, ReadsQuantizationConfigFromConfigJson) {
  // QAT checkpoints exported by HuggingFace carry no header metadata and
  // describe their quantization in config.json instead.
  auto values = OwningCpuBuffer::Copy<Type::kI4>({1, -2, 3, -4, 5, -6, 7, -8});
  auto scales = OwningCpuBuffer::Copy<Type::kFP32>({0.5});

  SafetensorFileGuard file = CreateTempSafetensor(
      {
          {.name = "model.layers.0.mlp.up_proj.weight_packed",
           .type = Type::kI4,
           .shape = {1, 8},
           .buffer = values},
          {.name = "model.layers.0.mlp.up_proj.weight_scale",
           .type = Type::kFP32,
           .shape = {1, 1},
           .buffer = scales},
      },
      /*quant_config_json=*/"");

  {
    std::ofstream config(file.GetConfigPath());
    config << R"({
      "model_type": "test",
      "quantization_config": {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
          "group_0": {
            "weights": { "num_bits": 4, "strategy": "channel" },
            "targets": ["Linear"]
          }
        }
      }
    })";
  }

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));

  const std::optional<QuantizationConfig>& quant_config =
      loader.GetQuantizationConfig();
  ASSERT_TRUE(quant_config.has_value());
  EXPECT_NE(quant_config->FindScheme("model.layers.0.mlp.up_proj"), nullptr);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      TensorHandle tensor,
      loader.LoadTensor("model.layers.0.mlp.up_proj.weight"));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, tensor.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const int4_t>(),
              ElementsAreArray(values->Span<int4_t>()));
}

TEST(SafetensorLoaderTest, PackedWeightWithoutQuantizationConfigFails) {
  auto values = OwningCpuBuffer::Copy<Type::kI4>({1, -2, 3, -4, 5, -6, 7, -8});

  SafetensorFileGuard file =
      CreateTempSafetensor({{.name = "model.proj.weight_packed",
                             .type = Type::kI4,
                             .shape = {1, 8},
                             .buffer = values}},
                           /*quant_config_json=*/"");

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(SafetensorLoader loader,
                                  SafetensorLoader::Load(file.GetPath()));
  EXPECT_FALSE(loader.LoadTensor("model.proj.weight").ok());
}

TEST(SafetensorLoaderTest, RejectNonPositiveNumBits) {
  std::string json = R"({
    "quant_method": "compressed-tensors",
    "config_groups": {
      "group_0": {
        "weights": {
          "num_bits": 0,
          "group_size": 128
        }
      }
    }
  })";

  SafetensorFileGuard file = CreateTempSafetensor(json);
  EXPECT_THAT(SafetensorLoader::Load(file.GetPath()), Not(IsOk()));
}

TEST(SafetensorLoaderTest, MalformedJsonFails) {
  std::string invalid_json =
      R"({ "quant_method": "compressed-tensors", format: })";

  SafetensorFileGuard file = CreateTempSafetensor(invalid_json);
  EXPECT_THAT(SafetensorLoader::Load(file.GetPath()), Not(IsOk()));
}

const char kMixedConfig[] = R"json({
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
      "two": {
        "targets": ["model.two", "re:.*lm_head$"],
        "weights": {"num_bits": 2, "type": "int", "dynamic": false,
                    "symmetric": true, "strategy": "channel"}
      },
      "four": {
        "targets": ["re:model\\.four$"],
        "weights": {"num_bits": 4, "type": "int", "dynamic": false,
                    "symmetric": true, "strategy": "group", "group_size": 4}
      },
      "eight": {
        "format": "int-quantized", "targets": ["model.eight"],
        "weights": {"num_bits": 8, "type": "int", "dynamic": false,
                    "symmetric": true, "strategy": "channel"},
        "input_activations": {"num_bits": 8, "type": "int", "dynamic": false,
                              "symmetric": true, "strategy": "tensor"},
        "output_activations": {"num_bits": 8, "type": "int", "dynamic": false,
                               "symmetric": true, "strategy": "tensor"}
      }
    }
  }
})json";

class CompressedTensorFixture {
 public:
  CompressedTensorFixture() {
    static std::atomic<int> counter = 0;
    directory_ = std::filesystem::path(testing::TempDir()) /
                 absl::StrCat("tensor_ct_", counter++);
    std::filesystem::create_directories(directory_);
  }
  ~CompressedTensorFixture() { std::filesystem::remove_all(directory_); }

  template <class T>
  void Add(const std::string& name, safetensors::dtype type,
           std::vector<size_t> shape, const std::vector<T>& values) {
    const size_t start = file_.storage.size();
    file_.storage.resize(start + values.size() * sizeof(T));
    std::memcpy(file_.storage.data() + start, values.data(),
                values.size() * sizeof(T));
    safetensors::tensor_t tensor;
    tensor.dtype = type;
    tensor.shape = std::move(shape);
    tensor.data_offsets = {start, file_.storage.size()};
    file_.tensors.insert(name, tensor);
  }

  void TwoBit(bool add_scale = true) {
    Add<uint32_t>("model.two.weight_packed", safetensors::kINT32, {1, 1},
                  {0xE4E4E4E4});
    Add<int64_t>("model.two.weight_shape", safetensors::kINT64, {2}, {1, 16});
    if (add_scale)
      Add<uint16_t>("model.two.weight_scale", safetensors::kBFLOAT16, {1, 1},
                    {0x3F00});
  }

  void EightBit(bool activation_scales = true) {
    Add<int8_t>("model.eight.weight", safetensors::kINT8, {2, 3},
                {-128, -1, 127, 0, 64, -64});
    Add<float>("model.eight.weight_scale", safetensors::kFLOAT32, {2, 1},
               {0.25f, 0.5f});
    if (activation_scales) {
      Add<float>("model.eight.input_scale", safetensors::kFLOAT32, {},
                 {0.125f});
      Add<float>("model.eight.output_scale", safetensors::kFLOAT32, {},
                 {0.25f});
    }
  }

  void Save(const std::string& config = kMixedConfig) {
    std::string warning, error;
    ASSERT_TRUE(safetensors::save_to_file(file_, FilePath(), &warning, &error))
        << error;
    std::ofstream output(directory_ / "config.json");
    output << config;
    ASSERT_TRUE(output.good());
  }

  std::string FilePath() const {
    return (directory_ / "model.safetensors").string();
  }
  std::string Directory() const { return directory_.string(); }

 private:
  std::filesystem::path directory_;
  safetensors::safetensors_t file_;
};

std::vector<uint8_t> Bytes(const TensorHandle& tensor) {
  auto lock = tensor.GetBufferPtr()->Lock();
  const auto* data = reinterpret_cast<const uint8_t*>(lock.data());
  return {data, data + lock.size()};
}

TEST(SafetensorLoaderTest, HeaderConfigTakesPrecedenceOverCompanionConfig) {
  SafetensorFileGuard file = CreateTempSafetensor(R"({
    "quant_method": "compressed-tensors",
    "config_groups": {"group_0": {"num_bits": 8}}
  })");
  {
    std::ofstream config(file.GetConfigPath());
    config << kMixedConfig;
  }
  for (const auto& path : {file.GetPath(), file.GetFolder()}) {
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                    SafetensorLoader::Load(path.string()));
    ASSERT_TRUE(loader.GetQuantizationConfig().has_value());
    EXPECT_EQ(loader.GetQuantizationConfig()->num_bits, 8);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto tensor,
                                    loader.LoadTensor("dummy_tensor"));
    EXPECT_EQ(tensor.GetType(), Type::kFP32);
  }
}

TEST(SafetensorLoaderTest, SparseClassConfigLoadsPackedAndFloatingWeights) {
  CompressedTensorFixture fixture;
  fixture.TwoBit();
  fixture.Add<float>("norm.weight", safetensors::kFLOAT32, {1}, {1.0f});
  fixture.Save(R"({"quantization_config": {
    "quant_method": "compressed-tensors",
    "config_groups": {"group_0": {
      "weights": {"num_bits": 2}, "targets": ["Linear"]
    }}
  }})");
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto packed,
                                  loader.LoadTensor("model.two.weight"));
  EXPECT_EQ(packed.GetType(), Type::kI4);
  EXPECT_EQ(Bytes(packed), std::vector<uint8_t>({0xFE, 0x10, 0xFE, 0x10, 0xFE,
                                                 0x10, 0xFE, 0x10}));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto floating,
                                  loader.LoadTensor("norm.weight"));
  EXPECT_EQ(floating.GetType(), Type::kFP32);
  EXPECT_EQ(floating.GetQuantization(), nullptr);
}

TEST(SafetensorLoaderTest, SpecificCompanionTargetWinsOverClassTarget) {
  CompressedTensorFixture fixture;
  fixture.TwoBit();
  fixture.EightBit();
  fixture.Save(R"({"quantization_config": {
    "quant_method": "compressed-tensors",
    "config_groups": {
      "default": {"weights": {"num_bits": 8}, "targets": ["Linear"]},
      "two": {"weights": {"num_bits": 2}, "targets": ["model.two"]}
    }
  }})");
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto two,
                                  loader.LoadTensor("model.two.weight"));
  EXPECT_EQ(two.GetType(), Type::kI4);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto eight,
                                  loader.LoadTensor("model.eight.weight"));
  EXPECT_EQ(eight.GetType(), Type::kI8);
}

TEST(SafetensorLoaderTest, InvalidCompanionRegexReturnsError) {
  CompressedTensorFixture fixture;
  fixture.TwoBit();
  fixture.Save(R"({"quantization_config": {
    "quant_method": "compressed-tensors",
    "config_groups": {"group_0": {
      "weights": {"num_bits": 2}, "targets": ["re:["]
    }}
  }})");
  EXPECT_THAT(SafetensorLoader::Load(fixture.FilePath()), Not(IsOk()));
}

TEST(SafetensorLoaderTest, MixedCompanionConfigDecodesSignedWeightsAndScales) {
  CompressedTensorFixture fixture;
  fixture.TwoBit();
  fixture.EightBit();
  fixture.Add<uint32_t>("model.four.weight_packed", safetensors::kINT32, {2, 1},
                        {0x76543210, 0xFEDCBA98});
  fixture.Add<int64_t>("model.four.weight_shape", safetensors::kINT64, {2},
                       {2, 8});
  fixture.Add<uint16_t>("model.four.weight_scale", safetensors::kBFLOAT16,
                        {2, 2}, {0x3F80, 0x4000, 0x4080, 0x4100});
  fixture.Save();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.Directory()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto two,
                                  loader.LoadTensor("model.two.weight"));
  EXPECT_EQ(two.GetType(), Type::kI4);
  EXPECT_EQ(two.GetShape(), Shape({1, 16}));
  EXPECT_EQ(Bytes(two), std::vector<uint8_t>(
                            {0xFE, 0x10, 0xFE, 0x10, 0xFE, 0x10, 0xFE, 0x10}));
  ASSERT_NE(two.GetQuantization(), nullptr);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      const auto& channel,
      two.GetQuantization()->As<PerChannelAffineQuantization>());
  EXPECT_EQ(channel.scales, std::vector<float>({0.5f}));
  EXPECT_EQ(channel.zero_points, std::vector<int64_t>({0}));
  EXPECT_EQ(channel.quantized_dimension, 0);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto four,
                                  loader.LoadTensor("model.four.weight"));
  EXPECT_EQ(four.GetShape(), Shape({2, 8}));
  EXPECT_EQ(Bytes(four), std::vector<uint8_t>(
                             {0x98, 0xBA, 0xDC, 0xFE, 0x10, 0x32, 0x54, 0x76}));
  ASSERT_NE(four.GetQuantization(), nullptr);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      const auto& grouped, four.GetQuantization()->As<BlockwiseQuantization>());
  EXPECT_EQ(grouped.block_size, 4);
  EXPECT_EQ(grouped.scales, std::vector<float>({1, 2, 4, 8}));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto eight,
                                  loader.LoadTensor("model.eight.weight"));
  EXPECT_EQ(eight.GetType(), Type::kI8);
  EXPECT_EQ(Bytes(eight), std::vector<uint8_t>({128, 255, 127, 0, 64, 192}));
}

TEST(SafetensorLoaderTest, MappingIncludesActivationScalesAndExplicitLmHead) {
  CompressedTensorFixture fixture;
  fixture.EightBit();
  fixture.Add<uint32_t>("lm_head.weight_packed", safetensors::kINT32, {1, 1},
                        {0xE4E4E4E4});
  fixture.Add<int64_t>("lm_head.weight_shape", safetensors::kINT64, {2},
                       {1, 16});
  fixture.Add<float>("lm_head.weight_scale", safetensors::kFLOAT32, {1, 1},
                     {1.0f});
  fixture.Save();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto mapped,
      loader.LoadWeightsWithMapping({{"model.eight.weight", "renamed.weight"},
                                     {"lm_head.weight", "lm_head.weight"}}));
  ASSERT_TRUE(mapped.contains("renamed.weight"));
  ASSERT_TRUE(mapped.contains("renamed.input_scale"));
  ASSERT_TRUE(mapped.contains("renamed.output_scale"));
  ASSERT_TRUE(mapped.contains("lm_head.weight"));
  EXPECT_EQ(mapped.at("lm_head.weight").GetType(), Type::kI4);
  EXPECT_EQ(mapped.at("renamed.input_scale").GetName(), "renamed.input_scale");
  const auto scale =
      mapped.at("renamed.input_scale").GetBufferPtr()->Lock().As<const float>();
  EXPECT_FLOAT_EQ(scale.data()[0], 0.125f);
}

TEST(SafetensorLoaderTest, PackedRowsTrimPaddingAtOriginalShape) {
  CompressedTensorFixture fixture;
  fixture.Add<uint32_t>("model.two.weight_packed", safetensors::kINT32, {2, 1},
                        {0x000000E4, 0x0000031B});
  fixture.Add<int64_t>("model.two.weight_shape", safetensors::kINT64, {2},
                       {2, 5});
  fixture.Add<float>("model.two.weight_scale", safetensors::kFLOAT32, {2, 1},
                     {1, 1});
  fixture.Save();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto tensor,
                                  loader.LoadTensor("model.two.weight"));
  EXPECT_EQ(Bytes(tensor),
            std::vector<uint8_t>({0xFE, 0x10, 0x1E, 0xF0, 0x1E}));
  EXPECT_EQ(tensor.GetShape(), Shape({2, 5}));
}

TEST(SafetensorLoaderTest,
     ConvertsFloatWeightsButKeepsPerLayerEmbeddingMapped) {
  CompressedTensorFixture fixture;
  fixture.Add<uint16_t>("norm.weight", safetensors::kBFLOAT16, {3},
                        {0x3F80, 0xC000, 0x3F00});
  fixture.Add<uint16_t>("projection.weight", safetensors::kFLOAT16, {3},
                        {0x3C00, 0xC000, 0x3800});
  fixture.Add<uint16_t>("model.embed_tokens_per_layer.weight",
                        safetensors::kBFLOAT16, {1, 3},
                        {0x3F80, 0xC000, 0x3F00});
  fixture.Save("{}");
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  for (const char* name : {"norm.weight", "projection.weight"}) {
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto tensor, loader.LoadTensor(name));
    EXPECT_EQ(tensor.GetType(), Type::kFP32);
    const auto lock = tensor.GetBufferPtr()->Lock().As<const float>();
    EXPECT_EQ(std::vector<float>(lock.begin(), lock.end()),
              std::vector<float>({1, -2, 0.5f}));
  }
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto embedding, loader.LoadTensor("model.embed_tokens_per_layer.weight"));
  EXPECT_EQ(embedding.GetType(), Type::kBF16);
}

TEST(SafetensorLoaderTest, MissingWeightScaleIsAnErrorIncludingMappedLoad) {
  CompressedTensorFixture fixture;
  fixture.TwoBit(false);
  fixture.Save();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  EXPECT_FALSE(loader.LoadTensor("model.two.weight").ok());
  EXPECT_FALSE(
      loader.LoadWeightsWithMapping({{"model.two.weight", "two.weight"}}).ok());
}

TEST(SafetensorLoaderTest, RejectsPackedShapeDisagreement) {
  CompressedTensorFixture fixture;
  fixture.Add<uint32_t>("model.two.weight_packed", safetensors::kINT32, {1, 1},
                        {0});
  fixture.Add<int64_t>("model.two.weight_shape", safetensors::kINT64, {2},
                       {1, 17});
  fixture.Add<float>("model.two.weight_scale", safetensors::kFLOAT32, {1, 1},
                     {1});
  fixture.Save();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  EXPECT_FALSE(loader.LoadTensor("model.two.weight").ok());
}

TEST(SafetensorLoaderTest, RejectsWrongPackedDtype) {
  CompressedTensorFixture fixture;
  fixture.Add<float>("model.two.weight_packed", safetensors::kFLOAT32, {1, 1},
                     {0});
  fixture.Add<int64_t>("model.two.weight_shape", safetensors::kINT64, {2},
                       {1, 16});
  fixture.Add<float>("model.two.weight_scale", safetensors::kFLOAT32, {1, 1},
                     {1});
  fixture.Save();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  EXPECT_FALSE(loader.LoadTensor("model.two.weight").ok());
}

TEST(SafetensorLoaderTest, RejectsNonpositiveScalesAndNonzeroZeroPoints) {
  for (bool bad_scale : {true, false}) {
    CompressedTensorFixture fixture;
    fixture.Add<uint32_t>("model.two.weight_packed", safetensors::kINT32,
                          {1, 1}, {0});
    fixture.Add<int64_t>("model.two.weight_shape", safetensors::kINT64, {2},
                         {1, 16});
    fixture.Add<float>("model.two.weight_scale", safetensors::kFLOAT32, {1, 1},
                       {bad_scale ? 0.0f : 1.0f});
    fixture.Add<int8_t>("model.two.weight_zero_point", safetensors::kINT8,
                        {1, 1}, {bad_scale ? int8_t{0} : int8_t{1}});
    fixture.Save();
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                    SafetensorLoader::Load(fixture.FilePath()));
    EXPECT_FALSE(loader.LoadTensor("model.two.weight").ok());
  }
}

TEST(SafetensorLoaderTest, RejectsMissingStaticActivationScale) {
  CompressedTensorFixture fixture;
  fixture.EightBit(false);
  fixture.Save();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  EXPECT_FALSE(loader.LoadTensor("model.eight.weight").ok());
}

TEST(SafetensorLoaderTest, RejectsAmbiguousTargetGroups) {
  CompressedTensorFixture fixture;
  fixture.TwoBit();
  std::string config = kMixedConfig;
  const auto position = config.find("re:model\\\\.four$");
  ASSERT_NE(position, std::string::npos);
  config.replace(position, std::string("re:model\\\\.four$").size(),
                 "model.two");
  fixture.Save(config);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  EXPECT_FALSE(loader.LoadTensor("model.two.weight").ok());
}

TEST(SafetensorLoaderTest, MappedInt8BufferOutlivesLoaderAndFile) {
  TensorHandle tensor;
  {
    CompressedTensorFixture fixture;
    fixture.EightBit();
    fixture.Save();
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                    SafetensorLoader::Load(fixture.FilePath()));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(tensor,
                                    loader.LoadTensor("model.eight.weight"));
  }
  EXPECT_EQ(Bytes(tensor), std::vector<uint8_t>({128, 255, 127, 0, 64, 192}));
}

TEST(SafetensorLoaderTest, TwoBitGroupedWeightsPreserveEachGroupScale) {
  CompressedTensorFixture fixture;
  fixture.Add<uint32_t>("model.four.weight_packed", safetensors::kINT32, {1, 1},
                        {0xE4E4E4E4});
  fixture.Add<int64_t>("model.four.weight_shape", safetensors::kINT64, {2},
                       {1, 16});
  fixture.Add<float>("model.four.weight_scale", safetensors::kFLOAT32, {1, 4},
                     {1, 2, 4, 8});
  std::string config = kMixedConfig;
  const std::string before = "\"num_bits\": 4";
  const auto position = config.find(before);
  ASSERT_NE(position, std::string::npos);
  config.replace(position, before.size(), "\"num_bits\": 2");
  fixture.Save(config);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto tensor,
                                  loader.LoadTensor("model.four.weight"));
  EXPECT_EQ(tensor.GetType(), Type::kI4);
  EXPECT_EQ(Bytes(tensor), std::vector<uint8_t>({0xFE, 0x10, 0xFE, 0x10, 0xFE,
                                                 0x10, 0xFE, 0x10}));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      const auto& quantization,
      tensor.GetQuantization()->As<BlockwiseQuantization>());
  EXPECT_EQ(quantization.block_size, 4);
  EXPECT_EQ(quantization.scales, std::vector<float>({1, 2, 4, 8}));
}

TEST(SafetensorLoaderTest, RejectsScaleShapeDisagreement) {
  CompressedTensorFixture fixture;
  fixture.Add<uint32_t>("model.two.weight_packed", safetensors::kINT32, {2, 1},
                        {0, 0});
  fixture.Add<int64_t>("model.two.weight_shape", safetensors::kINT64, {2},
                       {2, 16});
  fixture.Add<float>("model.two.weight_scale", safetensors::kFLOAT32, {1, 2},
                     {1, 1});
  fixture.Save();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                  SafetensorLoader::Load(fixture.FilePath()));
  EXPECT_FALSE(loader.LoadTensor("model.two.weight").ok());
}

TEST(SafetensorLoaderTest, CompanionConfigAcceptsScientificNotation) {
  for (const char* number : {"1e-06", "2E+3", "3e4", "-4.5E-2"}) {
    CompressedTensorFixture fixture;
    fixture.TwoBit();
    std::string config = kMixedConfig;
    config.insert(1, absl::StrCat("\"rms_norm_eps\":", number, ","));
    fixture.Save(config);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto loader,
                                    SafetensorLoader::Load(fixture.FilePath()));
    EXPECT_TRUE(loader.LoadTensor("model.two.weight").ok());
    const char* position = number;
    minijson::value parsed;
    ASSERT_EQ(minijson::parse(position, parsed), minijson::no_error);
    ASSERT_NE(parsed.as<minijson::number>(), nullptr);
    EXPECT_DOUBLE_EQ(*parsed.as<minijson::number>(), std::stod(number));
  }
}

TEST(SafetensorLoaderTest, CompanionConfigRejectsIncompleteExponent) {
  for (const char* number : {"1e", "1e+", "1e-", "1E", "1E+", "1E-"}) {
    CompressedTensorFixture fixture;
    fixture.TwoBit();
    std::string config = kMixedConfig;
    config.insert(1, absl::StrCat("\"rms_norm_eps\":", number, ","));
    fixture.Save(config);
    EXPECT_FALSE(SafetensorLoader::Load(fixture.FilePath()).ok());
  }
}

}  // namespace
}  // namespace litert::tensor::examples
