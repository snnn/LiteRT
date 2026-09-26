# Copyright 2026 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

cc_binary(
    name = "runner",
    testonly = True,
    srcs = [
        "driver.cc", "driver_support.h", "mapped_runner.h",
        "runtime_observer.cc", "runtime_observer.h",
        "model/gemma4_graph.h", "model/helpers/active_graph_context.h",
        "model/helpers/attention.h",
        "model/helpers/transformer.h",
    ],
    copts = ["-ffp-contract=off"],
    linkopts = ["-Wl,--wrap=xnn_create_runtime_v4", "-Wl,--wrap=xnn_invoke_runtime"],
    deps = [
        "//tensor/examples/gemma4/native:decode_trace",
        "//tensor/examples/gemma4/native:native_support",
        "//tensor/backends/tflite:arithmetic_tflite",
        "//tensor/backends/tflite:tflite_flatbuffer_conversion",
        "//tensor/internal:graph_traversal",
        "//tflite:framework",
        "//tflite/core:framework",
        "//tflite/kernels:builtin_ops",
        "//tflite/kernels:cpu_backend_context",
        "//tflite/delegates/xnnpack:xnnpack_delegate",
    ],
)
