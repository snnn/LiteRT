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
#include "tensor/experiments/litert_chunked_prefill/runtime_observer.h"
#include <pthreadpool.h>
namespace litert::tensor::examples::gemma4::native {
thread_local std::vector<RuntimeCreation>* observed_runtime_creations = nullptr;
thread_local int* observed_runtime_invocations = nullptr;
}
extern "C" xnn_status __real_xnn_create_runtime_v4(
    xnn_subgraph_t, xnn_weights_cache_t, xnn_workspace_t, pthreadpool_t,
    uint32_t, xnn_runtime_t*);
extern "C" xnn_status __wrap_xnn_create_runtime_v4(
    xnn_subgraph_t graph, xnn_weights_cache_t cache, xnn_workspace_t workspace,
    pthreadpool_t pool, uint32_t flags, xnn_runtime_t* out) {
  const auto status = __real_xnn_create_runtime_v4(graph, cache, workspace,
                                                 pool, flags, out);
  using namespace litert::tensor::examples::gemma4::native;
  if(status == xnn_status_success && observed_runtime_creations)
    observed_runtime_creations->push_back({*out, cache, workspace, pool, flags});
  return status;
}
extern "C" xnn_status __real_xnn_invoke_runtime(xnn_runtime_t);
extern "C" xnn_status __wrap_xnn_invoke_runtime(xnn_runtime_t runtime) {
  using namespace litert::tensor::examples::gemma4::native;
  if(observed_runtime_invocations) ++*observed_runtime_invocations;
  return __real_xnn_invoke_runtime(runtime);
}
