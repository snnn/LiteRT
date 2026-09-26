// Copyright 2026 Google LLC.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef LITERT_EXPERIMENT_JOINED_MAPPED_RUNNER_H_
#define LITERT_EXPERIMENT_JOINED_MAPPED_RUNNER_H_

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/examples/gemma4/native/memory_snapshot.h"
#include "tensor/examples/gemma4/native/model/helpers/bundle_matched_ops.h"
#include "tensor/experiments/litert_chunked_prefill/runtime_observer.h"
#include "tensor/internal/graph_traversal.h"
#include "tflite/builtin_ops.h"
#include "tflite/core/subgraph.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/cpu_backend_context.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/subgraph.h"

namespace litert::tensor::examples::gemma4::native {

struct SharedCpuResources {
  tflite::ExternalCpuBackendContext context;
  pthreadpool_t Initialize(int threads) {
    auto backend = std::make_unique<tflite::CpuBackendContext>();
    backend->SetMaxNumThreads(threads);
    auto pool = backend->get_xnnpack_threadpool();
    context.set_internal_backend_context(std::move(backend));
    return pool;
  }
};

inline void WriteKernelInventory(xnn_runtime_t runtime,
                                 const std::string& directory,
                                 const std::string& phase) {
  std::map<std::string, int> counts;
  for (size_t i = 0; i < runtime->num_ops; ++i)
    for (const auto* object : runtime->opdata[i].operator_objects)
      if (object) ++counts[xnn_operator_type_to_string(object->type)];
  std::ofstream out(directory + "/" + phase + "_kernel_inventory.json");
  out << "{\n";
  bool first = true;
  for (const auto& [name, count] : counts) {
    if (!first) out << ",\n";
    first = false;
    out << "  \"" << name << "\": " << count;
  }
  out << "\n}\n";
}

inline absl::Status TfStatus(TfLiteStatus status, const char* what) {
  return status == kTfLiteOk ? absl::OkStatus() : absl::InternalError(what);
}

// These are serialization rules for existing LiteRT builtins, not new ops.
class QuantizedCastExport final : public graph::TfLiteOperation {
 public:
  absl::StatusOr<graph::TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(op));
    const TensorHandle input(op.inputs.at(0)), output(outputs.at(0));
    if (input.GetQuantization() && output.GetType() == Type::kFP32)
      return graph::TfLiteOpBuildInfo(tflite::BuiltinOperator_DEQUANTIZE);
    if (output.GetQuantization())
      return graph::TfLiteOpBuildInfo(tflite::BuiltinOperator_QUANTIZE);
    return absl::InvalidArgumentError("Expected quantized cast");
  }
};

class FullyConnectedExport final : public graph::TfLiteOperation {
 public:
  absl::StatusOr<graph::TfLiteOpBuildInfo> ToTfLite(
      const graph::Operation& op) const override {
    return graph::TfLiteOpBuildInfo(
        tflite::BuiltinOperator_FULLY_CONNECTED,
        tflite::FullyConnectedOptionsT{
            .fused_activation_function = tflite::ActivationFunctionType_NONE,
            .keep_num_dims = true,
            .asymmetric_quantize_inputs = true});
  }
};

inline absl::Status PrepareLiteRtExport(
    const std::vector<TensorHandle>& roots) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto plan,
                              GetExecutionPlan(absl::MakeConstSpan(roots)));
  // Use exactly the primitive expansion of the native RMSNorm lowering.
  for (const auto* raw : plan) {
    if (!raw->IsA(internal::TypeId::Get<graph::RmsNormOperation>())) continue;
    auto group = raw->outputs_group.lock();
    auto keep_old_op_alive = group->producer;
    using T = Tensor<TfLiteMixinTag>;
    T x(raw->inputs[0]), eps(raw->inputs[2]);
    T squared = Square(x);
    T mean = Mean(squared, {int(x.GetShape().size()) - 1}, true);
    T result = Mul(x, Rsqrt(Add(mean, eps)));
    if (raw->inputs[1].group) result = Mul(result, T(raw->inputs[1]));
    LRT_TENSOR_RETURN_IF_ERROR(result.GetStatus());
    group->producer = result.GetRaw().group->producer;
    group->producer->outputs_group = group;
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(plan,
                              GetExecutionPlan(absl::MakeConstSpan(roots)));
  TfLiteMixinRegistrar registrar;
  std::map<std::string, int> counts;
  for (const auto* raw : plan) {
    auto group = raw->outputs_group.lock();
    auto op = group->producer;
    ++counts[std::string(op->GetName())];
    if (op->IsA(internal::TypeId::Get<
                graph::BundleDynamicInt2FullyConnectedOperation>())) {
      // The native helper compacts this original INT2 head from an INT4
      // transport container. Serialize those same original two-bit codes.
      TensorHandle source(op->inputs[1]);
      const size_t n = size_t(source.GetShape()[0]) * source.GetShape()[1];
      auto bytes = source.GetBufferPtr()->Lock();
      if (bytes.size() != n / 2)
        return absl::InvalidArgumentError("INT2 head size");
      const auto* in = reinterpret_cast<const uint8_t*>(bytes.data());
      std::vector<uint8_t> packed(n / 4, 0);
      for (size_t i = 0; i < n; ++i) {
        const int nibble = (in[i / 2] >> (4 * (i % 2))) & 15;
        const int value = nibble >= 8 ? nibble - 16 : nibble;
        if (value < -2 || value > 1)
          return absl::InvalidArgumentError("INT2 head code");
        packed[i / 4] |= (value & 3) << (2 * (i % 4));
      }
      TensorHandle compact({.name = "joined_int2_head",
                            .type = Type::kI2,
                            .shape = source.GetShape(),
                            .buffer = OwningCpuBuffer::Copy<Type::kU8>(packed),
                            .quantization = source.GetQuantization()});
      op->inputs[1] = compact.GetRaw();
      LRT_TENSOR_ASSIGN_OR_RETURN(auto& info, graph::GetInfo(op->inputs[1]));
      info.consumers.push_back(op);
      op->extensions.insert(op->extensions.begin(),
                            std::make_unique<FullyConnectedExport>());
    } else if (op->IsA(internal::TypeId::Get<
                       graph::StaticInt2FullyConnectedOperation>())) {
      op->extensions.insert(op->extensions.begin(),
                            std::make_unique<FullyConnectedExport>());
    } else if (op->IsA(internal::TypeId::Get<graph::CastOperation>())) {
      LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*op));
      if (TensorHandle(op->inputs[0]).GetQuantization() ||
          TensorHandle(outputs[0]).GetQuantization())
        op->extensions.insert(op->extensions.begin(),
                              std::make_unique<QuantizedCastExport>());
    }
    if (!op->GetExtension<graph::TfLiteOperation>()) registrar.Register(op);
    if (!op->GetExtension<graph::TfLiteOperation>())
      return absl::UnimplementedError("No LiteRT export for " +
                                      std::string(op->GetName()));
  }
  for (const auto& [name, count] : counts)
    std::cerr << "Export " << name << "=" << count << "\n";
  return absl::OkStatus();
}

class MappedRunner {
 public:
  // All signatures in a model share one delegate, hence one XNNPACK scratch
  // workspace. Invocation is serialized; KV outputs live outside that scratch.
  struct ModelState {
    std::vector<LockedBufferSpan<const std::byte>> constant_locks;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
        delegate{nullptr, &TfLiteXNNPackDelegateDelete};
    std::unique_ptr<tflite::Interpreter> interpreter;
  };

  static absl::StatusOr<std::unique_ptr<MappedRunner>> Create(
      std::vector<TensorHandle> roots, pthreadpool_t pool,
      xnn_weights_cache_t cache, TfLiteExternalContext* cpu_context,
      const std::string& directory, const std::string& phase) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto runners, CreateGroup({std::move(roots)}, {phase}, pool, cache,
                                  cpu_context, directory));
    return std::move(runners.front());
  }

  static absl::StatusOr<std::vector<std::unique_ptr<MappedRunner>>> CreateGroup(
      std::vector<std::vector<TensorHandle>> roots,
      const std::vector<std::string>& phases, pthreadpool_t pool,
      xnn_weights_cache_t cache, TfLiteExternalContext* cpu_context,
      const std::string& directory) {
    if (roots.empty() || roots.size() != phases.size())
      return absl::InvalidArgumentError("Invalid LiteRT signature group");
    auto state = std::make_shared<ModelState>();
    std::vector<std::unique_ptr<MappedRunner>> runners;
    std::vector<graph::Tensor> tensors;
    std::vector<std::vector<graph::Tensor>> signature_tensors(roots.size());
    absl::flat_hash_map<graph::Tensor, size_t> tensor_ids;
    for (size_t i = 0; i < roots.size(); ++i) {
      auto runner = std::unique_ptr<MappedRunner>(new MappedRunner);
      runner->phase_ = phases[i];
      runner->roots_ = std::move(roots[i]);
      runner->state_ = state;
      LRT_TENSOR_RETURN_IF_ERROR(PrepareLiteRtExport(runner->roots_));
      LRT_TENSOR_ASSIGN_OR_RETURN(
          auto plan, GetExecutionPlan(absl::MakeConstSpan(runner->roots_)));
      absl::flat_hash_set<graph::Tensor> seen;
      for (const auto* op : plan) {
        LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*op));
        for (const auto& group : {op->inputs, outputs})
          for (const auto& tensor : group) {
            if (seen.insert(tensor).second)
              signature_tensors[i].push_back(tensor);
            if (tensor_ids.emplace(tensor, tensors.size()).second)
              tensors.push_back(tensor);
          }
      }
      runners.push_back(std::move(runner));
    }
    std::vector<std::string> original_names;
    std::vector<std::shared_ptr<Quantization>> original_quantization;
    for (size_t i = 0; i < tensors.size(); ++i) {
      LRT_TENSOR_ASSIGN_OR_RETURN(auto& info, graph::GetInfo(tensors[i]));
      original_names.push_back(info.name);
      original_quantization.push_back(info.quantization);
      info.name = "map_t" + std::to_string(i);
      if (info.quantization) {
        auto q = info.quantization->As<const PerChannelAffineQuantization>();
        if (q.ok() && q->zero_points.size() == 1 && q->scales.size() > 1)
          info.quantization = std::make_shared<PerChannelAffineQuantization>(
              q->scales,
              std::vector<int64_t>(q->scales.size(), q->zero_points[0]),
              q->quantized_dimension);
      }
    }
    static int serial = 0;
    const std::string path =
        directory + "/joined_model_" + std::to_string(serial++) + ".tflite";
    // Release builder metadata before constructing/packing the interpreter.
    const auto save_status = [&]() -> absl::Status {
      ModelFactory factory;
      for (size_t i = 0; i < runners.size(); ++i)
        LRT_TENSOR_RETURN_IF_ERROR(
            factory.AddSignature(runners[i]->roots_, phases[i]));
      return factory.Save(path);
    }();
    for (size_t i = 0; i < tensors.size(); ++i) {
      LRT_TENSOR_ASSIGN_OR_RETURN(auto& info, graph::GetInfo(tensors[i]));
      info.name = original_names[i];
      // Expanded zero-point vectors are a serialization requirement. Restore
      // the compact authoring metadata after writing the model; the interpreter
      // owns the expanded affine arrays it needs from the FlatBuffer.
      info.quantization = std::move(original_quantization[i]);
    }
    LRT_TENSOR_RETURN_IF_ERROR(save_status);
    state->model = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    if (!state->model) return absl::InternalError("Load exported FlatBuffer");
    tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
    LRT_TENSOR_RETURN_IF_ERROR(
        TfStatus(tflite::InterpreterBuilder(*state->model,
                                            resolver)(&state->interpreter),
                 "Build LiteRT interpreter"));
    auto& interpreter = *state->interpreter;
    size_t rebound_bytes = 0;
    for (size_t i = 0; i < runners.size(); ++i) {
      auto& runner = *runners[i];
      runner.subgraph_ = interpreter.subgraph(i);
      auto& subgraph = *runner.subgraph_;
      std::map<std::string, int> name_to_index;
      for (int j = 0; j < subgraph.tensors_size(); ++j)
        name_to_index[subgraph.tensor(j)->name] = j;
      for (const auto& tensor : signature_tensors[i]) {
        auto it =
            name_to_index.find("map_t" + std::to_string(tensor_ids.at(tensor)));
        if (it == name_to_index.end())
          return absl::InternalError("Missing exported tensor");
        runner.indices_[tensor] = it->second;
        LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, graph::GetInfo(tensor));
        LRT_TENSOR_ASSIGN_OR_RETURN(const auto& producer,
                                    graph::GetProducer(tensor));
        if (info.buffer && !producer) {
          auto lock = info.buffer->Lock();
          auto* t = subgraph.tensor(it->second);
          if (t->bytes != lock.size())
            return absl::InternalError(
                "Constant size changed in serialization");
          TfLiteQuantization quantization{kTfLiteNoQuantization, nullptr};
          if (t->quantization.type == kTfLiteAffineQuantization) {
            auto* source =
                static_cast<TfLiteAffineQuantization*>(t->quantization.params);
            auto* q = static_cast<TfLiteAffineQuantization*>(
                std::malloc(sizeof(TfLiteAffineQuantization)));
            q->scale = TfLiteFloatArrayCopy(source->scale);
            q->zero_point = TfLiteIntArrayCopy(source->zero_point);
            q->quantized_dimension = source->quantized_dimension;
            quantization = {kTfLiteAffineQuantization, q};
          }
          const std::vector<int> dims(t->dims->data,
                                      t->dims->data + t->dims->size);
          LRT_TENSOR_RETURN_IF_ERROR(TfStatus(
              subgraph.SetTensorParametersReadOnly(
                  it->second, t->type, t->name, dims, quantization,
                  reinterpret_cast<const char*>(lock.data()), lock.size()),
              "Bind constant"));
          rebound_bytes += lock.size();
          state->constant_locks.push_back(std::move(lock));
        }
      }
    }
    auto options = TfLiteXNNPackDelegateOptionsDefault();
    options.num_threads = pool ? pthreadpool_get_threads_count(pool) : 1;
    options.flags = TFLITE_XNNPACK_DELEGATE_FLAG_QS8 |
                    TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
    options.runtime_flags = 0;
    options.weights_cache =
        reinterpret_cast<TfLiteXNNPackDelegateWeightsCache*>(cache);
    interpreter.SetExternalContext(kTfLiteCpuBackendContext, cpu_context);
    state->delegate.reset(TfLiteXNNPackDelegateCreateWithThreadpool(
        &options, interpreter.primary_subgraph().context()));
    if (!state->delegate) return absl::InternalError("Create XNNPACK delegate");
    if (TfLiteXNNPackDelegateGetThreadPool(state->delegate.get()) != pool)
      return absl::InternalError(
          "Delegate did not reuse the shared thread pool");
    std::vector<RuntimeCreation> runtimes;
    observed_runtime_creations = &runtimes;
    const auto delegate_status =
        interpreter.ModifyGraphWithDelegate(state->delegate.get());
    observed_runtime_creations = nullptr;
    LRT_TENSOR_RETURN_IF_ERROR(
        TfStatus(delegate_status, "Delegate exported graph"));
    if (runtimes.size() != runners.size())
      return absl::FailedPreconditionError("Require one runtime per signature");
    // Interpreter delegates subgraphs in ascending index order. Require one
    // execution node per subgraph; no CPU fallback or additional partitions.
    for (size_t i = 0; i < runners.size(); ++i) {
      auto& runner = *runners[i];
      auto& subgraph = *runner.subgraph_;
      const auto& plan = subgraph.execution_plan();
      if (plan.size() != 1 ||
          subgraph.node_and_registration(plan[0])->second.builtin_code !=
              kTfLiteBuiltinDelegate)
        return absl::FailedPreconditionError(
            "Require one delegated partition per signature, with no fallback");
      const auto& creation = runtimes[i];
      if (creation.flags != XNN_FLAG_DONT_SPIN_WORKERS ||
          creation.pool != pool || creation.cache != cache)
        return absl::InternalError(
            "Unexpected delegate execution resources/flags");
      runner.runtime_ = creation.runtime;
      if (creation.runtime->workspace != runtimes[0].runtime->workspace)
        return absl::InternalError("Signatures did not share delegate scratch");
      WriteKernelInventory(runner.runtime_, directory, runner.phase_);
      std::cerr << "LiteRT " << runner.phase_
                << " partitions=1 runtimes=1 fallback=0 shared_workspace=1\n";
    }
    std::cerr << "LiteRT model signatures=" << runners.size()
              << " delegates=1 rebound_constant_bytes=" << rebound_bytes
              << "\n";
    return runners;
  }

  absl::StatusOr<size_t> Lookup(const TensorHandle& h) const {
    auto it = indices_.find(h.GetRaw());
    if (it == indices_.end()) return absl::NotFoundError("No exported tensor");
    return it->second;
  }
  const MappedRunner& graph() const { return *this; }
  xnn_runtime_t runtime() const { return runtime_; }

  absl::Status ReshapeInput(const TensorHandle& h, const Shape& shape) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto index, Lookup(h));
    auto* t = subgraph_->tensor(index);
    if (t->dims->size == shape.size() &&
        std::equal(shape.begin(), shape.end(), t->dims->data))
      return absl::OkStatus();
    return TfStatus(subgraph_->ResizeInputTensor(index, shape),
                    "Resize active input");
  }
  absl::Status SetInput(const TensorHandle& h,
                        absl::Span<const std::byte> bytes) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto index, Lookup(h));
    bindings_[index] = bytes;
    return absl::OkStatus();
  }
  template <class Data>
  absl::Status SetInput(const TensorHandle& h, const Data& data) {
    return SetInput(h, absl::Span<const std::byte>(
                           reinterpret_cast<const std::byte*>(data.data()),
                           data.size() * sizeof(*data.data())));
  }
  absl::Status Run() {
    for (int i : subgraph_->inputs()) {
      auto it = bindings_.find(i);
      if (it == bindings_.end())
        return absl::InvalidArgumentError("Missing input binding");
      if (it->second.size() != subgraph_->tensor(i)->bytes)
        return absl::InvalidArgumentError("Input byte count mismatch");
      auto* data = const_cast<std::byte*>(it->second.data());
      if (!data) data = &empty_;
      const TfLiteCustomAllocation allocation{data, it->second.size()};
      LRT_TENSOR_RETURN_IF_ERROR(TfStatus(
          subgraph_->SetCustomAllocationForTensor(
              i, allocation, kTfLiteCustomAllocationFlagsSkipAlignCheck),
          "Bind active input"));
    }
    LRT_TENSOR_RETURN_IF_ERROR(
        TfStatus(subgraph_->AllocateTensors(), "Allocate/reshape LiteRT"));
    int invocations = 0;
    observed_runtime_invocations = &invocations;
    const auto invoke_status = subgraph_->Invoke();
    observed_runtime_invocations = nullptr;
    LRT_TENSOR_RETURN_IF_ERROR(TfStatus(invoke_status, "Invoke LiteRT"));
    if (invocations != 1)
      return absl::InternalError(
          "Expected one XNNPACK invocation through LiteRT");
    return absl::OkStatus();
  }
  template <class T>
  absl::StatusOr<LockedBufferSpan<const T>> ReadOutputAs(
      const TensorHandle& h) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto index, Lookup(h));
    const auto* t = subgraph_->tensor(index);
    LockedBufferSpan<const std::byte> out(
        reinterpret_cast<const std::byte*>(t->data.raw),
        [](const std::byte*) {}, t->bytes);
    return std::move(out).template As<const T>();
  }
  void AddMemoryCounters(gemma_memory::Counters& counts,
                         std::set<const void*>& workspaces) const {
    if (workspaces.insert(runtime_->workspace).second) {
      counts["unique_workspace_count"]++;
      counts["unique_workspace_bytes"] += runtime_->workspace->size;
    }
    counts["litert_" + phase_ + "_runtime_count"] = 1;
    counts["litert_" + phase_ + "_xnn_workspace_bytes"] =
        runtime_->workspace->size;
    size_t bytes = 0;
    std::set<const void*> seen;
    for (int i = 0; i < subgraph_->tensors_size(); ++i) {
      const auto* t = subgraph_->tensor(i);
      if ((t->allocation_type == kTfLiteArenaRw ||
           t->allocation_type == kTfLiteArenaRwPersistent ||
           t->allocation_type == kTfLiteDynamic) &&
          t->data.raw && seen.insert(t->data.raw).second)
        bytes += t->bytes;
    }
    counts["litert_" + phase_ + "_unique_rw_tensor_payload_bytes"] = bytes;
  }

 private:
  std::string phase_;
  std::vector<TensorHandle> roots_;
  std::shared_ptr<ModelState> state_;
  tflite::Subgraph* subgraph_ = nullptr;
  absl::flat_hash_map<graph::Tensor, int> indices_;
  std::map<int, absl::Span<const std::byte>> bindings_;
  xnn_runtime_t runtime_ = nullptr;
  std::byte empty_{};
};
}  // namespace litert::tensor::examples::gemma4::native
#endif
