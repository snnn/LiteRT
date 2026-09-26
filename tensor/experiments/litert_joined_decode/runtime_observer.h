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
#ifndef LITERT_EXPERIMENT_JOINED_RUNTIME_OBSERVER_H_
#define LITERT_EXPERIMENT_JOINED_RUNTIME_OBSERVER_H_
#include <vector>
#include "xnnpack.h"
namespace litert::tensor::examples::gemma4::native {
// Link-time observation only: wrappers forward every argument unchanged.
struct RuntimeCreation {
  xnn_runtime_t runtime;
  xnn_weights_cache_t cache;
  xnn_workspace_t workspace;
  pthreadpool_t pool;
  uint32_t flags;
};
extern thread_local std::vector<RuntimeCreation>* observed_runtime_creations;
extern thread_local int* observed_runtime_invocations;
}
#endif
