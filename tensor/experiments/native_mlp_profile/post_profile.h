/* Copyright 2026 Google LLC.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef LITERT_TENSOR_EXPERIMENTS_NATIVE_MLP_PROFILE_POST_PROFILE_H_
#define LITERT_TENSOR_EXPERIMENTS_NATIVE_MLP_PROFILE_POST_PROFILE_H_

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "tensor/examples/gemma4/native/memory_snapshot.h"
#include "tensor/examples/gemma4/native/stage_runner.h"
#include "xnnpack/datatype.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator-utils.h"
#include "xnnpack/operator.h"

namespace litert::tensor::examples::gemma4::native {

// Diagnostic only. Uses the pinned Linux XNNPACK profiler's nanosecond
// timestamps, avoiding the public profile API's integer-microsecond rounding.
// Samples include operator dispatch/threadpool synchronization inside invoke.
class PostOperatorProfile {
 public:
  struct Operator {
    size_t op_index, object_index;
    std::string name, input_dtype, output_dtype, weight, weight_dtype;
    std::vector<size_t> weight_shape;
    size_t source_bytes = 0;
    std::vector<uint64_t> samples_ns;
  };
  struct Layer {
    xnn_runtime_t runtime = nullptr;
    std::vector<Operator> operators;
    std::vector<uint64_t> wall_ns;
  };

  bool collecting = false;

  template <class WeightMap>
  absl::Status Register(int layer, StageRunner& stage, const WeightMap& weights) {
    Layer data;
    data.runtime = stage.runtime();
    if (!data.runtime->profiling)
      return absl::InternalError("Post runtime profiling flag was not applied");
    std::map<const void*, std::pair<std::string, size_t>> names;
    for (const auto& entry : weights) {
      auto buffer = entry.second.GetBufferPtr();
      if (!buffer) continue;
      auto lock = buffer->Lock();
      LRT_TENSOR_ASSIGN_OR_RETURN(auto bytes, buffer->ByteSize());
      names.emplace(lock.data(), std::make_pair(entry.first, bytes));
    }
    for (size_t i = 0; i < data.runtime->num_ops; ++i) {
      const auto& op = data.runtime->opdata[i];
      for (size_t j = 0; j < XNN_MAX_OPERATOR_OBJECTS; ++j) {
        auto* object = op.operator_objects[j];
        if (!object) continue;
        Operator item{};
        item.op_index = i;
        item.object_index = j;
        item.name = xnn_operator_type_to_string_v2(object);
        if (op.num_inputs)
          item.input_dtype = xnn_datatype_to_string(data.runtime->values[op.inputs[0]].datatype);
        if (op.num_outputs)
          item.output_dtype = xnn_datatype_to_string(data.runtime->values[op.outputs[0]].datatype);
        if (op.type == xnn_node_type_fully_connected) {
          const auto& value = data.runtime->values[op.inputs[1]];
          const auto found = names.find(value.data);
          if (found == names.end())
            return absl::InternalError("FC source pointer was not identified");
          item.weight = found->second.first;
          item.source_bytes = found->second.second;
          item.weight_dtype = xnn_datatype_to_string(value.datatype);
          item.weight_shape.assign(value.shape.dim, value.shape.dim + value.shape.num_dims);
        }
        item.samples_ns.reserve(1024);
        data.operators.push_back(std::move(item));
      }
    }
    data.wall_ns.reserve(1024);
    if (!layers_.emplace(layer, std::move(data)).second)
      return absl::InternalError("Duplicate post layer registration");
    return absl::OkStatus();
  }

  absl::Status Sample(int layer, double wall_ms) {
    if (!collecting) return absl::OkStatus();
    auto& data = layers_.at(layer);
    int64_t previous = Nanoseconds(data.runtime->start_ts);
    uint64_t total = 0;
    for (auto& item : data.operators) {
      const int64_t end = Nanoseconds(data.runtime->opdata[item.op_index].end_ts[item.object_index]);
      if (end < previous) return absl::InternalError("Invalid operator timestamp order");
      item.samples_ns.push_back(end - previous);
      total += end - previous;
      previous = end;
    }
    const uint64_t wall_ns = static_cast<uint64_t>(wall_ms * 1000000.0 + 0.5);
    if (wall_ns < total) return absl::InternalError("Operator times exceed enclosing stage time");
    data.wall_ns.push_back(wall_ns);
    return absl::OkStatus();
  }

  absl::Status Write(const std::string& directory) const {
    std::ofstream ops(directory + "/post_operators.jsonl");
    std::ofstream stages(directory + "/post_stages.jsonl");
    if (layers_.size() != 35) return absl::InternalError("Expected 35 post stages");
    for (const auto& [layer, data] : layers_) {
      stages << "{\"layer\":" << layer << ",\"wall_ns\":";
      WriteArray(stages, data.wall_ns); stages << "}\n";
      for (const auto& item : data.operators) {
        if (item.samples_ns.size() != data.wall_ns.size() || item.samples_ns.empty())
          return absl::InternalError("Inconsistent profile sample counts");
        ops << "{\"layer\":" << layer << ",\"op_index\":" << item.op_index
            << ",\"object_index\":" << item.object_index
            << ",\"name\":" << gemma_memory::Quote(item.name)
            << ",\"input_dtype\":" << gemma_memory::Quote(item.input_dtype)
            << ",\"output_dtype\":" << gemma_memory::Quote(item.output_dtype)
            << ",\"weight\":" << gemma_memory::Quote(item.weight)
            << ",\"weight_dtype\":" << gemma_memory::Quote(item.weight_dtype)
            << ",\"source_bytes\":" << item.source_bytes << ",\"weight_shape\":";
        WriteArray(ops, item.weight_shape);
        ops << ",\"samples_ns\":"; WriteArray(ops, item.samples_ns); ops << "}\n";
      }
    }
    return ops && stages ? absl::OkStatus() : absl::InternalError("Profile write failed");
  }

 private:
  static int64_t Nanoseconds(const xnn_timestamp& ts) {
    return static_cast<int64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
  }
  template <class T>
  static void WriteArray(std::ostream& out, const std::vector<T>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
      if (i) out << ',';
      out << values[i];
    }
    out << ']';
  }
  std::map<int, Layer> layers_;
};

inline PostOperatorProfile post_operator_profile;
}  // namespace litert::tensor::examples::gemma4::native
#endif
