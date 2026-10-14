// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LITERT_ML_DRIFT_DELEGATE_WEBGPU_COMMAND_CAPTURE_H_
#define LITERT_ML_DRIFT_DELEGATE_WEBGPU_COMMAND_CAPTURE_H_

#include <cstdint>
#include <functional>
#include <memory>

#include "ml_drift/webgpu/command_buffer_capture.h"  // from @ml_drift

namespace litert::ml_drift {

// Optional coarse diagnostics for normal dispatch, independent of per-operation
// profiling. Configure before creating inference contexts. Callbacks may run on
// the command-preparation thread: they must be thread-safe, bounded and avoid
// waiting for GPU completion. The caller controls nonblocking collection;
// final drain and export belong outside timed requests.
// IDs share a process-local sequence; preparation IDs are distinct from actual
// invocation IDs, since speculative command buffers can be discarded.
struct WebGpuCommandCapture {
  std::shared_ptr<::ml_drift::webgpu::CommandBufferCapturePool> pool;
  std::function<void(char phase, const char* scope, uint64_t id,
                     uint64_t context_id)> event;
  std::function<void(uint64_t invocation_id, uint64_t context_id,
                     uint64_t preparation_id, bool from_cache, bool captured)>
      submitted;
};

}  // namespace litert::ml_drift

#endif  // LITERT_ML_DRIFT_DELEGATE_WEBGPU_COMMAND_CAPTURE_H_
