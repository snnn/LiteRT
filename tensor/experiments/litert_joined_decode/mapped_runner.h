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
#include "tensor/experiments/litert_joined_decode/runtime_observer.h"
#include "tensor/internal/graph_traversal.h"
#include "tflite/interpreter.h"
#include "tflite/model_builder.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/kernels/cpu_backend_context.h"
#include "tflite/builtin_ops.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/core/subgraph.h"
#include "xnnpack/subgraph.h"
#include "xnnpack/operator-type.h"

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
                                 const std::string& directory) {
  std::map<std::string, int> counts;
  for (size_t i=0; i<runtime->num_ops; ++i)
    for (const auto* object : runtime->opdata[i].operator_objects)
      if (object) ++counts[xnn_operator_type_to_string(object->type)];
  std::ofstream out(directory + "/decode_kernel_inventory.json");
  out << "{\n";
  bool first=true;
  for (const auto& [name, count] : counts) {
    if(!first) out << ",\n";
    first=false;
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
    return graph::TfLiteOpBuildInfo(tflite::BuiltinOperator_FULLY_CONNECTED,
        tflite::FullyConnectedOptionsT{
            .fused_activation_function=tflite::ActivationFunctionType_NONE,
            .keep_num_dims=true,
            .asymmetric_quantize_inputs=true});
  }
};

inline absl::Status PrepareLiteRtExport(const std::vector<TensorHandle>& roots) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto plan, GetExecutionPlan(absl::MakeConstSpan(roots)));
  // Use exactly the primitive expansion of the native RMSNorm lowering.
  for (const auto* raw : plan) {
    if (!raw->IsA(internal::TypeId::Get<graph::RmsNormOperation>())) continue;
    auto group = raw->outputs_group.lock();
    auto keep_old_op_alive = group->producer;
    using T = Tensor<TfLiteMixinTag>;
    T x(raw->inputs[0]), eps(raw->inputs[2]);
    T squared = Square(x);
    T mean = Mean(squared, {int(x.GetShape().size())-1}, true);
    T result = Mul(x, Rsqrt(Add(mean, eps)));
    if (raw->inputs[1].group) result = Mul(result, T(raw->inputs[1]));
    LRT_TENSOR_RETURN_IF_ERROR(result.GetStatus());
    group->producer = result.GetRaw().group->producer;
    group->producer->outputs_group = group;
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(plan, GetExecutionPlan(absl::MakeConstSpan(roots)));
  TfLiteMixinRegistrar registrar;
  std::map<std::string, int> counts;
  for (const auto* raw : plan) {
    auto group = raw->outputs_group.lock();
    auto op = group->producer;
    ++counts[std::string(op->GetName())];
    if (op->IsA(internal::TypeId::Get<graph::BundleDynamicInt2FullyConnectedOperation>())) {
      // The native helper compacts this original INT2 head from an INT4
      // transport container. Serialize those same original two-bit codes.
      TensorHandle source(op->inputs[1]);
      const size_t n = size_t(source.GetShape()[0]) * source.GetShape()[1];
      auto bytes = source.GetBufferPtr()->Lock();
      if (bytes.size() != n/2) return absl::InvalidArgumentError("INT2 head size");
      const auto* in = reinterpret_cast<const uint8_t*>(bytes.data());
      std::vector<uint8_t> packed(n/4, 0);
      for (size_t i=0; i<n; ++i) {
        const int nibble = (in[i/2] >> (4*(i%2))) & 15;
        const int value = nibble >= 8 ? nibble-16 : nibble;
        if (value < -2 || value > 1) return absl::InvalidArgumentError("INT2 head code");
        packed[i/4] |= (value & 3) << (2*(i%4));
      }
      TensorHandle compact({.name="joined_int2_head", .type=Type::kI2,
          .shape=source.GetShape(), .buffer=OwningCpuBuffer::Copy<Type::kU8>(packed),
          .quantization=source.GetQuantization()});
      op->inputs[1] = compact.GetRaw();
      LRT_TENSOR_ASSIGN_OR_RETURN(auto& info, graph::GetInfo(op->inputs[1]));
      info.consumers.push_back(op);
      op->extensions.insert(op->extensions.begin(), std::make_unique<FullyConnectedExport>());
    } else if(op->IsA(internal::TypeId::Get<graph::StaticInt2FullyConnectedOperation>())) {
      op->extensions.insert(op->extensions.begin(), std::make_unique<FullyConnectedExport>());
    } else if(op->IsA(internal::TypeId::Get<graph::CastOperation>())) {
      LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*op));
      if (TensorHandle(op->inputs[0]).GetQuantization() || TensorHandle(outputs[0]).GetQuantization())
        op->extensions.insert(op->extensions.begin(), std::make_unique<QuantizedCastExport>());
    }
    if (!op->GetExtension<graph::TfLiteOperation>()) registrar.Register(op);
    if (!op->GetExtension<graph::TfLiteOperation>())
      return absl::UnimplementedError("No LiteRT export for " + std::string(op->GetName()));
  }
  for (const auto& [name, count] : counts) std::cerr << "Export " << name << "=" << count << "\n";
  return absl::OkStatus();
}

class MappedRunner {
 public:
  static absl::StatusOr<std::unique_ptr<MappedRunner>> Create(
      std::vector<TensorHandle> roots, pthreadpool_t pool,
      xnn_weights_cache_t cache, TfLiteExternalContext* cpu_context,
      const std::string& directory) {
    auto result = std::unique_ptr<MappedRunner>(new MappedRunner);
    result->roots_ = std::move(roots);
    LRT_TENSOR_RETURN_IF_ERROR(PrepareLiteRtExport(result->roots_));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto plan, GetExecutionPlan(absl::MakeConstSpan(result->roots_)));
    std::vector<graph::Tensor> tensors;
    absl::flat_hash_set<graph::Tensor> seen;
    for (const auto* op : plan) {
      LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, graph::GetOutputs(*op));
      for(const auto& group : {op->inputs, outputs}) for(const auto& tensor : group)
        if(seen.insert(tensor).second) tensors.push_back(tensor);
    }
    std::vector<std::string> original_names;
    for(size_t i=0; i<tensors.size(); ++i) {
      LRT_TENSOR_ASSIGN_OR_RETURN(auto& info, graph::GetInfo(tensors[i]));
      original_names.push_back(info.name);
      info.name = "map_t" + std::to_string(i);
      // XNNPACK permits a broadcast zero point; the FlatBuffer interpreter
      // requires one zero point per channel when scales are per-channel.
      if (info.quantization) {
        auto q = info.quantization->As<const PerChannelAffineQuantization>();
        if(q.ok() && q->zero_points.size()==1 && q->scales.size()>1)
          info.quantization = std::make_shared<PerChannelAffineQuantization>(
              q->scales, std::vector<int64_t>(q->scales.size(), q->zero_points[0]),
              q->quantized_dimension);
      }
    }
    static int serial = 0;
    const std::string path = directory + "/joined_model_" + std::to_string(serial++) + ".tflite";
    const auto save_status = Save(result->roots_, path);
    for(size_t i=0; i<tensors.size(); ++i) {
      LRT_TENSOR_ASSIGN_OR_RETURN(auto& info, graph::GetInfo(tensors[i]));
      info.name = original_names[i];
    }
    LRT_TENSOR_RETURN_IF_ERROR(save_status);
    result->model_ = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    if(!result->model_) return absl::InternalError("Load exported FlatBuffer");
    tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
    LRT_TENSOR_RETURN_IF_ERROR(TfStatus(tflite::InterpreterBuilder(*result->model_, resolver)(
        &result->interpreter_), "Build LiteRT interpreter"));
    auto& interpreter = *result->interpreter_;
    std::map<std::string,int> name_to_index;
    for(int i=0; i<interpreter.tensors_size(); ++i)
      name_to_index[interpreter.tensor(i)->name] = i;
    size_t rebound_bytes=0;
    for(size_t i=0; i<tensors.size(); ++i) {
      auto it = name_to_index.find("map_t" + std::to_string(i));
      if(it == name_to_index.end()) return absl::InternalError("Missing exported tensor");
      result->indices_[tensors[i]] = it->second;
      LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, graph::GetInfo(tensors[i]));
      LRT_TENSOR_ASSIGN_OR_RETURN(const auto& producer, graph::GetProducer(tensors[i]));
      if(info.buffer && !producer) {
        auto lock = info.buffer->Lock();
        auto* t = interpreter.tensor(it->second);
        if(t->bytes != lock.size()) return absl::InternalError("Constant size changed in serialization");
        TfLiteQuantization quantization{kTfLiteNoQuantization,nullptr};
        if(t->quantization.type == kTfLiteAffineQuantization) {
          auto* source = static_cast<TfLiteAffineQuantization*>(t->quantization.params);
          auto* q = static_cast<TfLiteAffineQuantization*>(std::malloc(sizeof(TfLiteAffineQuantization)));
          q->scale=TfLiteFloatArrayCopy(source->scale);
          q->zero_point=TfLiteIntArrayCopy(source->zero_point);
          q->quantized_dimension=source->quantized_dimension;
          quantization={kTfLiteAffineQuantization,q};
        }
        const std::vector<int> dims(t->dims->data,t->dims->data+t->dims->size);
        // Public caller-owned constant binding keeps pointer-based packed
        // weight sharing equivalent to native staged prefill.
        LRT_TENSOR_RETURN_IF_ERROR(TfStatus(interpreter.SetTensorParametersReadOnly(
            it->second,t->type,t->name,dims,quantization,
            reinterpret_cast<const char*>(lock.data()),lock.size()),"Bind constant"));
        rebound_bytes += lock.size();
        result->constant_locks_.push_back(std::move(lock));
      }
    }
    auto options = TfLiteXNNPackDelegateOptionsDefault();
    options.num_threads = pool ? pthreadpool_get_threads_count(pool) : 1;
    options.flags = TFLITE_XNNPACK_DELEGATE_FLAG_QS8 | TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
    options.runtime_flags = 0;
    options.weights_cache = reinterpret_cast<TfLiteXNNPackDelegateWeightsCache*>(cache);
    interpreter.SetExternalContext(kTfLiteCpuBackendContext, cpu_context);
    result->delegate_.reset(TfLiteXNNPackDelegateCreateWithThreadpool(
        &options,interpreter.primary_subgraph().context()));
    if(!result->delegate_) return absl::InternalError("Create XNNPACK delegate");
    if(TfLiteXNNPackDelegateGetThreadPool(result->delegate_.get()) != pool)
      return absl::InternalError("Delegate did not reuse the shared thread pool");
    std::vector<RuntimeCreation> runtimes;
    observed_runtime_creations = &runtimes;
    const auto delegate_status = interpreter.ModifyGraphWithDelegate(result->delegate_.get());
    observed_runtime_creations = nullptr;
    LRT_TENSOR_RETURN_IF_ERROR(TfStatus(delegate_status,"Delegate exported graph"));
    int delegated=0;
    for(int n : interpreter.execution_plan()) {
      const auto* node = interpreter.node_and_registration(n);
      if(node->second.builtin_code == kTfLiteBuiltinDelegate) ++delegated;
      else std::cerr << "UNDELEGATED node=" << n << " op=" << tflite::EnumNameBuiltinOperator(
          static_cast<tflite::BuiltinOperator>(node->second.builtin_code)) << "\n";
    }
    std::cerr << "LiteRT partitions=" << delegated << " execution_nodes=" << interpreter.execution_plan().size()
              << " observed_runtimes=" << runtimes.size() << " rebound_constant_bytes=" << rebound_bytes << "\n";
    if(delegated != 1 || interpreter.execution_plan().size() != 1 || runtimes.size() != 1)
      return absl::FailedPreconditionError("Require one delegated runtime with no fallback operators");
    const auto& creation = runtimes.front();
    std::cerr << "LiteRT runtime_flags=" << creation.flags
              << " shared_pool=" << (creation.pool == pool)
              << " shared_weight_cache=" << (creation.cache == cache) << "\n";
    if(creation.flags != XNN_FLAG_DONT_SPIN_WORKERS || creation.pool != pool || creation.cache != cache)
      return absl::InternalError("Unexpected delegate execution resources/flags");
    result->runtime_ = creation.runtime;
    WriteKernelInventory(result->runtime_, directory);
    return result;
  }

  absl::StatusOr<size_t> Lookup(const TensorHandle& h) const {
    auto it=indices_.find(h.GetRaw());
    if(it==indices_.end()) return absl::NotFoundError("No exported tensor");
    return it->second;
  }
  const MappedRunner& graph() const { return *this; }
  xnn_runtime_t runtime() const { return runtime_; }

  absl::Status ReshapeInput(const TensorHandle& h, const Shape& shape) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto index, Lookup(h));
    auto* t=interpreter_->tensor(index);
    if(t->dims->size==shape.size() && std::equal(shape.begin(),shape.end(),t->dims->data)) return absl::OkStatus();
    return TfStatus(interpreter_->ResizeInputTensor(index,shape),"Resize active input");
  }
  absl::Status SetInput(const TensorHandle& h, absl::Span<const std::byte> bytes) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto index, Lookup(h));
    bindings_[index]=bytes;
    return absl::OkStatus();
  }
  template<class Data>
  absl::Status SetInput(const TensorHandle& h, const Data& data) {
    return SetInput(h,absl::Span<const std::byte>(
        reinterpret_cast<const std::byte*>(data.data()),data.size()*sizeof(*data.data())));
  }
  absl::Status Run() {
    for(int i : interpreter_->inputs()) {
      auto it=bindings_.find(i);
      if(it==bindings_.end()) return absl::InvalidArgumentError("Missing input binding");
      if(it->second.size()!=interpreter_->tensor(i)->bytes) return absl::InvalidArgumentError("Input byte count mismatch");
      auto* data=const_cast<std::byte*>(it->second.data());
      if(!data) data=&empty_;
      const TfLiteCustomAllocation allocation{data,it->second.size()};
      LRT_TENSOR_RETURN_IF_ERROR(TfStatus(interpreter_->SetCustomAllocationForTensor(
          i,allocation,kTfLiteCustomAllocationFlagsSkipAlignCheck),"Bind active input"));
    }
    LRT_TENSOR_RETURN_IF_ERROR(TfStatus(interpreter_->AllocateTensors(),"Allocate/reshape LiteRT"));
    int invocations=0;
    observed_runtime_invocations=&invocations;
    const auto invoke_status=interpreter_->Invoke();
    observed_runtime_invocations=nullptr;
    LRT_TENSOR_RETURN_IF_ERROR(TfStatus(invoke_status,"Invoke LiteRT"));
    if(invocations!=1) return absl::InternalError("Expected one XNNPACK invocation through LiteRT");
    return absl::OkStatus();
  }
  template<class T>
  absl::StatusOr<LockedBufferSpan<const T>> ReadOutputAs(const TensorHandle& h) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto index,Lookup(h));
    const auto* t=interpreter_->tensor(index);
    LockedBufferSpan<const std::byte> out(reinterpret_cast<const std::byte*>(t->data.raw),
        [](const std::byte*){},t->bytes);
    return std::move(out).template As<const T>();
  }
  void AddMemoryCounters(gemma_memory::Counters& counts) const {
    counts["litert_decode_runtime_count"]=1;
    counts["litert_decode_xnn_workspace_bytes"]=runtime_->workspace->size;
    size_t bytes=0;
    std::set<const void*> seen;
    for(int i=0;i<interpreter_->tensors_size();++i) {
      const auto* t=interpreter_->tensor(i);
      if((t->allocation_type==kTfLiteArenaRw || t->allocation_type==kTfLiteArenaRwPersistent || t->allocation_type==kTfLiteDynamic)
          && t->data.raw && seen.insert(t->data.raw).second) bytes+=t->bytes;
    }
    counts["litert_decode_unique_rw_tensor_payload_bytes"]=bytes;
  }
 private:
  std::vector<TensorHandle> roots_;
  std::vector<LockedBufferSpan<const std::byte>> constant_locks_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<TfLiteDelegate,decltype(&TfLiteXNNPackDelegateDelete)> delegate_{nullptr,&TfLiteXNNPackDelegateDelete};
  std::unique_ptr<tflite::Interpreter> interpreter_;
  absl::flat_hash_map<graph::Tensor,int> indices_;
  std::map<int,absl::Span<const std::byte>> bindings_;
  xnn_runtime_t runtime_=nullptr;
  std::byte empty_{};
};
}
#endif
