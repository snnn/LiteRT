/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tflite/kernels/internal/reference/slice.h"

#include <stdint.h>

#include <algorithm>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "tflite/context_util.h"
#include "tflite/core/c/common.h"
#include "tflite/kernels/internal/compatibility.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/reference/reference_ops.h"
#include "tflite/kernels/internal/tensor.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/internal/types.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/string_type.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace slice {

enum KernelType {
  kReference,
  kGenericOptimized,
};

constexpr int kInputTensor = 0;
constexpr int kBeginTensor = 1;
constexpr int kSizeTensor = 2;
constexpr int kOutputTensor = 0;

// The optimized helper supports up to 5D. Higher-rank tensors use the
// runtime-rank reference path.
const int kMaxDim = 5;

template <typename T>
TfLiteStatus GetBeginAndSizeVectorsImpl(TfLiteContext* context,
                                        const TfLiteTensor* input,
                                        const TfLiteTensor* begin,
                                        const TfLiteTensor* size,
                                        std::vector<int>* begins,
                                        std::vector<int>* sizes) {
  const T* begin_data = GetTensorData<T>(begin);
  const T* size_data = GetTensorData<T>(size);
  for (int idx = 0; idx < NumDimensions(input); ++idx) {
    const int input_dim = SizeOfDimension(input, idx);
    const T begin_value = begin_data[idx];
    if (begin_value < 0 || begin_value > input_dim) {
      TF_LITE_KERNEL_LOG(context,
                         "Slice begin is out of range in dimension %d.", idx);
      return kTfLiteError;
    }

    const int checked_begin = static_cast<int>(begin_value);
    const T size_value = size_data[idx];
    int checked_size;
    if (size_value == -1) {
      checked_size = input_dim - checked_begin;
    } else {
      if (size_value < 0 || size_value > input_dim - checked_begin) {
        TF_LITE_KERNEL_LOG(context,
                           "Slice size is out of range in dimension %d.", idx);
        return kTfLiteError;
      }
      // input_dim and checked_begin are ints and the comparison above proves
      // that size_value is in their non-negative difference's range.
      checked_size = static_cast<int>(size_value);
    }

    begins->push_back(checked_begin);
    sizes->push_back(checked_size);
  }
  return kTfLiteOk;
}

TfLiteStatus GetBeginAndSizeVectors(TfLiteContext* context,
                                    const TfLiteTensor* input,
                                    const TfLiteTensor* begin,
                                    const TfLiteTensor* size,
                                    std::vector<int>* begins,
                                    std::vector<int>* sizes) {
  TF_LITE_ENSURE_TYPES_EQ(context, begin->type, size->type);
  TF_LITE_ENSURE_EQ(context, NumElements(begin), NumDimensions(input));
  TF_LITE_ENSURE_EQ(context, NumElements(size), NumDimensions(input));

  begins->clear();
  sizes->clear();
  begins->reserve(NumDimensions(input));
  sizes->reserve(NumDimensions(input));
  if (begin->type == kTfLiteInt32) {
    return GetBeginAndSizeVectorsImpl<int32_t>(context, input, begin, size,
                                               begins, sizes);
  }
  if (begin->type == kTfLiteInt64) {
    return GetBeginAndSizeVectorsImpl<int64_t>(context, input, begin, size,
                                               begins, sizes);
  }
  TF_LITE_KERNEL_LOG(context, "Type %d is currently not supported by Slice.",
                     begin->type);
  return kTfLiteError;
}

TfLiteStatus ResizeOutputShape(TfLiteContext* context,
                               const std::vector<int>& output_shape_vector,
                               TfLiteTensor* output) {
  TfLiteIntArray* output_shape =
      TfLiteIntArrayCreate(output_shape_vector.size());
  std::copy(output_shape_vector.begin(), output_shape_vector.end(),
            output_shape->data);
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus ValidateOutputShape(TfLiteContext* context,
                                 const std::vector<int>& expected_shape,
                                 const TfLiteTensor* output) {
  if (static_cast<size_t>(NumDimensions(output)) != expected_shape.size()) {
    TF_LITE_KERNEL_LOG(context,
                       "Slice output rank does not match the inferred rank.");
    return kTfLiteError;
  }
  for (int idx = 0; idx < NumDimensions(output); ++idx) {
    if (SizeOfDimension(output, idx) != expected_shape[idx]) {
      TF_LITE_KERNEL_LOG(
          context,
          "Slice output shape does not match the inferred shape in dimension "
          "%d.",
          idx);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

bool ShapeHasRank(const TfLiteIntArray* shape) {
  // Note that we consider scalar as false here because there is
  // no differentiation between scalar and dynamic properly supported.
  if (shape == nullptr || shape->size == 0) return false;
  return true;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* begin;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kBeginTensor, &begin));
  const TfLiteTensor* size;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kSizeTensor, &size));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Ensure validity of input tensor and its dimension.
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE(context,
                 begin->type == kTfLiteInt32 || begin->type == kTfLiteInt64);
  TF_LITE_ENSURE(context,
                 size->type == kTfLiteInt32 || size->type == kTfLiteInt64);
  TF_LITE_ENSURE_TYPES_EQ(context, begin->type, size->type);
  TF_LITE_ENSURE_EQ(context, NumDimensions(begin), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(size), 1);
  TF_LITE_ENSURE_EQ(context, NumElements(begin), NumDimensions(input));
  TF_LITE_ENSURE_EQ(context, NumElements(size), NumDimensions(input));
  // A declared static output keeps its allocation, but Eval still validates
  // the runtime input extent and index tensor values before using it.
  if (!HasUnspecifiedDimension(output) && ShapeHasRank(output->dims)) {
    if (IsConstantOrPersistentTensor(begin) &&
        IsConstantOrPersistentTensor(size) && !HasUnspecifiedDimension(input)) {
      std::vector<int> begins;
      std::vector<int> sizes;
      TF_LITE_ENSURE_STATUS(
          GetBeginAndSizeVectors(context, input, begin, size, &begins, &sizes));
      TF_LITE_ENSURE_STATUS(ValidateOutputShape(context, sizes, output));
    }
    return kTfLiteOk;
  }
  // Postpone allocation of output if any of the indexing tensors is not
  // constant, or the input tensor has dynamic dimension.
  if (!(IsConstantOrPersistentTensor(begin) &&
        IsConstantOrPersistentTensor(size)) ||
      HasUnspecifiedDimension(input)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }

  std::vector<int> begins;
  std::vector<int> sizes;
  TF_LITE_ENSURE_STATUS(
      GetBeginAndSizeVectors(context, input, begin, size, &begins, &sizes));
  return ResizeOutputShape(context, sizes, output);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* begin;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kBeginTensor, &begin));
  const TfLiteTensor* size;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kSizeTensor, &size));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  const int input_dims = NumDimensions(input);
  std::vector<int> begins;
  std::vector<int> sizes;
  TF_LITE_ENSURE_STATUS(
      GetBeginAndSizeVectors(context, input, begin, size, &begins, &sizes));

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputShape(context, sizes, output));
  } else {
    TF_LITE_ENSURE_STATUS(ValidateOutputShape(context, sizes, output));
  }

  if (input_dims > kMaxDim) {
    if (input->type == kTfLiteString) {
      reference_ops::Slice<string>(begins, GetTensorShape(input), input,
                                   GetTensorShape(output), output);
      return kTfLiteOk;
    } else if (input->type == kTfLiteInt4) {
      reference_ops::SliceInt4(begins, GetTensorShape(input), input,
                               GetTensorShape(output), output);
      return kTfLiteOk;
    }

#define TF_LITE_SLICE_DYNAMIC(data_type)                                  \
  {                                                                       \
    reference_ops::Slice<data_type>(begins, GetTensorShape(input), input, \
                                    GetTensorShape(output), output);      \
  }
    switch (TfLiteTypeGetSizeBits(output->type)) {
      case 8:
        TF_LITE_SLICE_DYNAMIC(int8_t);
        break;
      case 16:
        TF_LITE_SLICE_DYNAMIC(int16_t);
        break;
      case 32:
        TF_LITE_SLICE_DYNAMIC(int32_t);
        break;
      case 64:
        TF_LITE_SLICE_DYNAMIC(int64_t);
        break;
      default:
        TF_LITE_KERNEL_LOG(context,
                           "Type %d is currently not supported by Slice.",
                           input->type);
        return kTfLiteError;
    }
#undef TF_LITE_SLICE_DYNAMIC
    return kTfLiteOk;
  }

  std::vector<int> padded_begins;
  padded_begins.reserve(kMaxDim);
  std::vector<int> padded_sizes;
  padded_sizes.reserve(kMaxDim);
  for (int i = input_dims; i < kMaxDim; ++i) {
    padded_begins.push_back(0);
    padded_sizes.push_back(1);
  }
  padded_begins.insert(padded_begins.end(), begins.begin(), begins.end());
  padded_sizes.insert(padded_sizes.end(), sizes.begin(), sizes.end());

#define TF_LITE_SLICE_INT4()                                            \
  {                                                                     \
    TF_LITE_ENSURE_EQ(context, padded_begins.size(), kMaxDim);          \
    TF_LITE_ENSURE_EQ(context, padded_sizes.size(), kMaxDim);           \
    tflite::SliceParams op_params;                                      \
    op_params.begin_count = kMaxDim;                                    \
    op_params.size_count = kMaxDim;                                     \
    for (int i = 0; i < kMaxDim; ++i) {                                 \
      op_params.begin[i] = padded_begins[i];                            \
      op_params.size[i] = padded_sizes[i];                              \
    }                                                                   \
                                                                        \
    if (kernel_type == kGenericOptimized) {                             \
      optimized_ops::SliceInt4(op_params, GetTensorShape(input), input, \
                               GetTensorShape(output), output);         \
    } else {                                                            \
      reference_ops::SliceInt4(op_params, GetTensorShape(input), input, \
                               GetTensorShape(output), output);         \
    }                                                                   \
  }

#define TF_LITE_SLICE(data_type)                                               \
  {                                                                            \
    TF_LITE_ENSURE_EQ(context, padded_begins.size(), kMaxDim);                 \
    TF_LITE_ENSURE_EQ(context, padded_sizes.size(), kMaxDim);                  \
    tflite::SliceParams op_params;                                             \
    op_params.begin_count = kMaxDim;                                           \
    op_params.size_count = kMaxDim;                                            \
    for (int i = 0; i < kMaxDim; ++i) {                                        \
      op_params.begin[i] = padded_begins[i];                                   \
      op_params.size[i] = padded_sizes[i];                                     \
    }                                                                          \
                                                                               \
    if (kernel_type == kGenericOptimized) {                                    \
      optimized_ops::Slice<data_type>(op_params, GetTensorShape(input), input, \
                                      GetTensorShape(output), output);         \
    } else {                                                                   \
      reference_ops::Slice<data_type>(op_params, GetTensorShape(input), input, \
                                      GetTensorShape(output), output);         \
    }                                                                          \
  }

  if (input->type == kTfLiteString) {
    TF_LITE_SLICE(string);
    return kTfLiteOk;
  } else if (input->type == kTfLiteInt4) {
    TF_LITE_SLICE_INT4();
    return kTfLiteOk;
  }

  switch (TfLiteTypeGetSizeBits(output->type)) {
    case 8:
      TF_LITE_SLICE(int8_t);
      break;
    case 16:
      TF_LITE_SLICE(int16_t);
      break;
    case 32:
      TF_LITE_SLICE(int32_t);
      break;
    case 64:
      TF_LITE_SLICE(int64_t);
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Type %d is currently not supported by Slice.", input->type);
      return kTfLiteError;
  }
#undef TF_LITE_SLICE
  return kTfLiteOk;
}

}  // namespace slice

TfLiteRegistration* Register_SLICE_REF() {
  static TfLiteRegistration r = {nullptr, nullptr, slice::Prepare,
                                 slice::Eval<slice::kReference>};
  return &r;
}

TfLiteRegistration* Register_SLICE() {
  static TfLiteRegistration r = {nullptr, nullptr, slice::Prepare,
                                 slice::Eval<slice::kGenericOptimized>};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
