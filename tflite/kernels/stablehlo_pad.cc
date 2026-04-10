/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
         //
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>

#include "tflite/c/c_api_types.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/core/c/common.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_pad {
namespace {

static constexpr int kMaxDims = TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT;

// Fills a buffer with the given data.
//
// WARNING: This expects buffer_bytes to be a multiple of data_bytes.
void FillBuffer(char* buffer, int64_t buffer_bytes, const char* data,
                int64_t data_bytes) {
  if (buffer_bytes == 0) {
    return;
  }
  assert(buffer_bytes % data_bytes == 0);
  std::memcpy(buffer, data, data_bytes);
  buffer_bytes -= data_bytes;
  while (buffer_bytes) {
    const int64_t bytes = std::min(buffer_bytes, data_bytes);
    std::memcpy(buffer + data_bytes, buffer, bytes);
    buffer_bytes -= bytes;
    data_bytes += bytes;
  }
}

// Recursive implementation of a strided copy of a tensor.
void StridedCopy(const int rank, const char* input, const int64_t* input_shape,
                 const int64_t* input_strides, char* output,
                 const int64_t* output_strides, const int64_t element_size,
                 const int depth) {
  if (depth + 1 == rank) {
    for (int64_t i = 0; i < input_shape[depth]; ++i) {
      std::memcpy(output, input, element_size);
      input += input_strides[depth];
      output += output_strides[depth];
    }
  } else {
    for (int64_t i = 0; i < input_shape[depth]; ++i) {
      StridedCopy(rank, input, input_shape, input_strides, output,
                  output_strides, element_size, depth + 1);
      input += input_strides[depth];
      output += output_strides[depth];
    }
  }
}

// Holds the main implementation of the Pad operation.
//
// The StableHLO pad operation can add interior padding and edge padding to a
// tensor. The edge padding may be negative in which case it is considered as a
// cropping specification.
//
// This is implemented as a strided copy where:
//
// - interior padding affects the output strides.
// - positive edge padding affects the output shape, strides and initial offset.
// - negative edge padding affects the input shape and initial offset as well as
// the output initial offset.
//
// See https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad for more
// information.
class PadData {
 public:
  enum { kInput, kPaddingValue, kInputTensorCount };
  enum { kOutput, kOutputTensorCount };

  explicit PadData(const TfLiteStablehloPadParams& params) {
    std::memcpy(
        edge_pad_low_, params.edge_padding_low,
        TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT * sizeof(int64_t));
    std::memcpy(
        edge_pad_high_, params.edge_padding_high,
        TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT * sizeof(int64_t));
    std::memcpy(
        interior_pad_, params.interior_padding,
        TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT * sizeof(int64_t));
  }

  // Computes the shapes and strides that are needed for the final strided copy.
  TfLiteStatus Setup(TfLiteContext* context, const int* dims, const int rank,
                     const int64_t element_size) {
    rank_ = rank;
    element_size_ = element_size;
    input_offset_ = 0;
    output_offset_ = 0;
    output_size_ = 0;
    if (rank == 0) {
      output_size_ = element_size;
      return kTfLiteOk;
    }

    // Compute the output shape.
    for (int i = 0; i < rank; ++i) {
      TF_LITE_ENSURE_MSG(context, interior_pad_[i] >= 0,
                         "StableHLO Pad interior padding must be non-negative.");
      const auto output_dim =
          (CheckedInt<int64_t>(dims[i]) - 1) *
              (CheckedInt<int64_t>(interior_pad_[i]) + 1) +
          1 + edge_pad_low_[i] + edge_pad_high_[i];
      TF_LITE_ENSURE_MSG(context, !output_dim.Overflow(),
                         "StableHLO Pad output dimension overflowed.");
      output_shape_[i] = output_dim.Value();
    }
    if (std::any_of(output_shape_, output_shape_ + rank,
                    [](auto s) { return s <= 0; })) {
      std::memset(input_shape_, 0, sizeof(input_shape_));
      std::memset(output_shape_, 0, sizeof(output_shape_));
      output_size_ = 0;
      return kTfLiteOk;
    }
    // Compute the output size for each dimension.
    //
    // This is different from the output strides because of the interior
    // padding: the output strides take it into account to "jump" over the
    // interior padding elements.
    output_dimension_sizes_[rank - 1] = element_size;
    for (int i = rank - 2; i >= 0; --i) {
      const auto output_dimension_size =
          CheckedInt<int64_t>(output_shape_[i + 1]) *
          output_dimension_sizes_[i + 1];
      TF_LITE_ENSURE_MSG(context, !output_dimension_size.Overflow(),
                         "StableHLO Pad output dimension size overflowed.");
      output_dimension_sizes_[i] = output_dimension_size.Value();
    }
    // Compute the output stride for each dimension.
    //
    // This is the stride between two elements that are copied from the input
    // tensor (i.e. not generated by interior padding).
    const auto last_output_stride =
        CheckedInt<int64_t>(element_size) *
        (CheckedInt<int64_t>(interior_pad_[rank - 1]) + 1);
    TF_LITE_ENSURE_MSG(context, !last_output_stride.Overflow(),
                       "StableHLO Pad output stride overflowed.");
    output_strides_[rank - 1] = last_output_stride.Value();
    for (int i = rank - 2; i >= 0; --i) {
      const auto output_stride = CheckedInt<int64_t>(output_dimension_sizes_[i]) *
                                 (CheckedInt<int64_t>(interior_pad_[i]) + 1);
      TF_LITE_ENSURE_MSG(context, !output_stride.Overflow(),
                         "StableHLO Pad output stride overflowed.");
      output_strides_[i] = output_stride.Value();
    }
    // Compute the output offset from the eventual pads.
    CheckedInt<int64_t> output_offset = 0;
    for (int i = 0; i < rank; ++i) {
      output_offset = output_offset + CheckedInt<int64_t>(
                                          std::max<int64_t>(edge_pad_low_[i], 0)) *
                                          output_dimension_sizes_[i];
      TF_LITE_ENSURE_MSG(context, !output_offset.Overflow(),
                         "StableHLO Pad output offset overflowed.");
    }
    output_offset_ = output_offset.Value();
    // Compute the final output size.
    CheckedInt<int64_t> output_size = element_size;
    for (int i = 0; i < rank; ++i) {
      output_size = output_size * output_shape_[i];
      TF_LITE_ENSURE_MSG(context, !output_size.Overflow(),
                         "StableHLO Pad output size overflowed.");
    }
    output_size_ = output_size.Value();
    // Compute input strides.
    input_strides_[rank - 1] = element_size;
    for (int i = rank - 1; i >= 1; --i) {
      const auto input_stride =
          CheckedInt<int64_t>(dims[i]) * input_strides_[i];
      TF_LITE_ENSURE_MSG(context, !input_stride.Overflow(),
                         "StableHLO Pad input stride overflowed.");
      input_strides_[i - 1] = input_stride.Value();
    }
    // Helper that computes the division between a negative num and a positive
    // denum, rounding away from 0, or returns 0 if num is positive.
    auto DivNegRoundAwayOrZero =
        [context](int64_t num, int64_t denum, int64_t* result) -> TfLiteStatus {
      assert(denum > 0);
      if (num >= 0) {
        *result = 0;
        return kTfLiteOk;
      }
      const auto adjusted = CheckedInt<int64_t>(num) - denum + 1;
      TF_LITE_ENSURE_MSG(context, !adjusted.Overflow(),
                         "StableHLO Pad crop computation overflowed.");
      *result = adjusted.Value() / denum;
      return kTfLiteOk;
    };
    // Compute the input bounds from the eventual crops.
    //
    // If negative padding is applied, we can treat this as copying a subtensor
    // of the input. We modify the input shape in place as we don't use it for
    // anything else.
    for (int i = 0; i < rank; ++i) {
      int64_t low_crop = 0;
      int64_t high_crop = 0;
      const auto padded_step = CheckedInt<int64_t>(interior_pad_[i]) + 1;
      TF_LITE_ENSURE_MSG(context, !padded_step.Overflow(),
                         "StableHLO Pad crop computation overflowed.");
      TF_LITE_ENSURE_OK(
          context, DivNegRoundAwayOrZero(edge_pad_low_[i], padded_step.Value(),
                                         &low_crop));
      TF_LITE_ENSURE_OK(
          context, DivNegRoundAwayOrZero(edge_pad_high_[i], padded_step.Value(),
                                         &high_crop));
      const auto input_shape =
          CheckedInt<int64_t>(dims[i]) + low_crop + high_crop;
      TF_LITE_ENSURE_MSG(context, !input_shape.Overflow(),
                         "StableHLO Pad input shape overflowed.");
      input_shape_[i] = input_shape.Value();
    }
    // Compute the input offset from the eventual crops.
    //
    // When computing the subtensor from the negative padding, we need to find
    // out the offset to its first element in addition to its shape (see
    // previous comment).
    //
    // Cropping also means that the interior padding can become edge padding so
    // we also need to update the output offset:
    //
    // > `1 0 0 0 2 0 0 0 3` cropped by 1 low element becomes `0 0 0 2 0 0 0 3`
    // > which effectlvely means pad `2 3` with an interior padding of 3 and a
    // > low edge padding of 3.
    CheckedInt<int64_t> input_offset = 0;
    CheckedInt<int64_t> adjusted_output_offset = output_offset_;
    for (int i = 0; i < rank; ++i) {
      int64_t low_crop = 0;
      const auto padded_step = CheckedInt<int64_t>(interior_pad_[i]) + 1;
      TF_LITE_ENSURE_MSG(context, !padded_step.Overflow(),
                         "StableHLO Pad crop computation overflowed.");
      TF_LITE_ENSURE_OK(
          context, DivNegRoundAwayOrZero(edge_pad_low_[i], padded_step.Value(),
                                         &low_crop));
      input_offset =
          input_offset - CheckedInt<int64_t>(low_crop) * input_strides_[i];
      TF_LITE_ENSURE_MSG(context, !input_offset.Overflow(),
                         "StableHLO Pad input offset overflowed.");
      if (edge_pad_low_[i] < 0) {
        const auto padded_offset =
            padded_step + CheckedInt<int64_t>(edge_pad_low_[i]);
        TF_LITE_ENSURE_MSG(context, !padded_offset.Overflow(),
                           "StableHLO Pad crop computation overflowed.");
        int64_t tmp_offset = padded_offset.Value() % padded_step.Value();
        if (tmp_offset < 0) {
          tmp_offset += padded_step.Value();
        }
        adjusted_output_offset =
            adjusted_output_offset +
            CheckedInt<int64_t>(tmp_offset) * output_dimension_sizes_[i];
        TF_LITE_ENSURE_MSG(context, !adjusted_output_offset.Overflow(),
                           "StableHLO Pad output offset overflowed.");
      }
    }
    input_offset_ = input_offset.Value();
    output_offset_ = adjusted_output_offset.Value();
    return kTfLiteOk;
  }

  void Apply(const char* input, const char* padding_value, char* output) const {
    if (rank_ == 0) {
      std::memcpy(output, input, element_size_);
      return;
    }
    // Fill the output tensor with the padding value.
    FillBuffer(output, output_size_, padding_value, element_size_);
    StridedCopy(rank_, input + input_offset_, input_shape_, input_strides_,
                output + output_offset_, output_strides_, element_size_,
                /*depth=*/0);
  }

  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)>
  BuildOutputTensorDims(TfLiteContext* context) const {
    std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> dims(
        TfLiteIntArrayCreate(static_cast<int>(rank_)), TfLiteIntArrayFree);
    if (dims == nullptr) {
      return {nullptr, TfLiteIntArrayFree};
    }
    for (int64_t i = 0; i < rank_; ++i) {
      if (CheckedShapeDimension(context, output_shape_[i],
                                "StableHLO Pad output dimension overflowed.",
                                &dims->data[i]) != kTfLiteOk) {
        return {nullptr, TfLiteIntArrayFree};
      }
    }
    return dims;
  }

 private:
  int64_t edge_pad_low_[kMaxDims];
  int64_t edge_pad_high_[kMaxDims];
  int64_t interior_pad_[kMaxDims];
  int64_t rank_ = 0;
  int64_t element_size_ = 0;
  int64_t input_shape_[kMaxDims];
  int64_t output_shape_[kMaxDims];
  int64_t input_strides_[kMaxDims];
  int64_t output_strides_[kMaxDims];
  int64_t output_dimension_sizes_[kMaxDims];
  int64_t input_offset_ = 0;
  int64_t output_offset_ = 0;
  int64_t output_size_ = 0;
};

void* Init(TfLiteContext* context, const char* options, size_t options_len) {
  return new PadData(
      *reinterpret_cast<const TfLiteStablehloPadParams*>(options));
}

void Free(TfLiteContext* context, void* node_data) {
  delete reinterpret_cast<PadData*>(node_data);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // Input checks.
  const TfLiteTensor* input_tensor = GetInput(context, node, PadData::kInput);
  const TfLiteTensor* padding_value_tensor =
      GetInput(context, node, PadData::kPaddingValue);
  TF_LITE_ENSURE(context, input_tensor->type == padding_value_tensor->type);
  // PadData computations.
  size_t element_size;
  TF_LITE_ENSURE(context, GetSizeOfType(context, input_tensor->type,
                                        &element_size) == kTfLiteOk);
  PadData& pad_data = *reinterpret_cast<PadData*>(node->user_data);
  TF_LITE_ENSURE_OK(context, pad_data.Setup(context, input_tensor->dims->data,
                                            input_tensor->dims->size,
                                            element_size));
  // Output tensor setup.
  TfLiteTensor* output_tensor = GetOutput(context, node, PadData::kOutput);
  TF_LITE_ENSURE(context, input_tensor->type == output_tensor->type);
  auto output_dims = pad_data.BuildOutputTensorDims(context);
  if (output_dims == nullptr) {
    return kTfLiteError;
  }
  TF_LITE_ENSURE_OK(
      context,
      context->ResizeTensor(context, output_tensor, output_dims.release()));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_tensor = GetInput(context, node, PadData::kInput);
  const TfLiteTensor* padding_value_tensor =
      GetInput(context, node, PadData::kPaddingValue);
  TfLiteTensor* output_tensor = GetOutput(context, node, PadData::kOutput);
  // Pad using PadData
  PadData& pad_data = *reinterpret_cast<PadData*>(node->user_data);
  pad_data.Apply(input_tensor->data.raw_const,
                 padding_value_tensor->data.raw_const, output_tensor->data.raw);
  return kTfLiteOk;
}

}  // namespace
}  // namespace stablehlo_pad

TfLiteRegistration* Register_STABLEHLO_PAD() {
  static TfLiteRegistration r = {/*.init=*/stablehlo_pad::Init,
                                 /*.free=*/stablehlo_pad::Free,
                                 /*.prepare=*/stablehlo_pad::Prepare,
                                 /*.invoke=*/stablehlo_pad::Eval};
  return &r;
}
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
