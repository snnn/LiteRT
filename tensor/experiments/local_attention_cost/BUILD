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
    name = "attention_cost_probe",
    testonly = True,
    srcs = ["attention_cost_probe.cc"],
    copts = ["-ffp-contract=off"],
    deps = ["//tensor/examples/gemma4/native:active_attention_test_source"],
)
