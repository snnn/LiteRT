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

#ifndef TENSORFLOW_LITE_DELEGATES_YNNPACK_DOT_H_
#define TENSORFLOW_LITE_DELEGATES_YNNPACK_DOT_H_

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "tflite/core/c/common.h"
#include "tflite/delegates/ynnpack/utils.h"

namespace tflite {
namespace ynnpack {

bool IsRuntimeBmm(const TfLiteRegistration* registration,
                  const TfLiteNode* node);

// Defines FP32 A times per-tensor affine INT8 B with FP32 output, using the same
// dynamic activation quantization as runtime_bmm. The caller validates B's
// scalar quantization metadata; value IDs/ranks describe operands after views.
// Unlike constant weights, B may use the full INT8 range, including -128.
TfLiteStatus DefineDynamicallyQuantizedMatMul(
    TfLiteContext* context, ynn_subgraph_t subgraph, int rank_a, int rank_b,
    uint32_t a_id, uint32_t b_id, const TfLiteTensor& b_tensor, bool adj_y,
    uint32_t* output_id);

TfLiteStatus IsBatchMatMulSupported(const TfLiteRegistration* registration,
                                    const TfLiteNode* node,
                                    TfLiteContext* context,
                                    bool is_runtime_bmm = false);

TfLiteStatus IsRuntimeBatchedMatMulSupported(
    const TfLiteRegistration* registration, const TfLiteNode* node,
    TfLiteContext* context);

TfLiteStatus IsFullyConnectedSupported(const TfLiteRegistration* registration,
                                       const TfLiteNode* node,
                                       TfLiteContext* context);

TfLiteStatus DefineBatchMatMulNode(TfLiteContext* context,
                                   ynn_subgraph_t subgraph,
                                   TensorToValueIdMap& tensor_to_value_id,
                                   const NodeInfo& node);

TfLiteStatus DefineRuntimeBatchedMatMulNode(
    TfLiteContext* context, ynn_subgraph_t subgraph,
    TensorToValueIdMap& tensor_to_value_id, uint32_t& next_external_id,
    std::vector<DummyInputInfo>& dummy_inputs, const NodeInfo& node);

TfLiteStatus DefineFullyConnectedNode(TfLiteContext* context,
                                      ynn_subgraph_t subgraph,
                                      TensorToValueIdMap& tensor_to_value_id,
                                      const NodeInfo& node);

TfLiteStatus IsConvSupported(const TfLiteRegistration* registration,
                             const TfLiteNode* node, TfLiteContext* context);

TfLiteStatus IsDepthwiseConvSupported(const TfLiteRegistration* registration,
                                      const TfLiteNode* node,
                                      TfLiteContext* context);

TfLiteStatus DefineConvNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                            TensorToValueIdMap& tensor_to_value_id,
                            const NodeInfo& node);

TfLiteStatus DefineDepthwiseConvNode(TfLiteContext* context,
                                     ynn_subgraph_t subgraph,
                                     TensorToValueIdMap& tensor_to_value_id,
                                     const NodeInfo& node);

}  // namespace ynnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_YNNPACK_DOT_H_
