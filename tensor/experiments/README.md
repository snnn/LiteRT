<!--
Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Gemma4 runner experiments

Both supported Gemma4 runner implementations now live in this checkout:

- [Native XNNPACK runner](../examples/gemma4/native/README.md).
- [Production LiteRT runner](../examples/gemma4/litert/README.md), including the
  exporter, bundle tools, and the shared `LitertDynamicRunner` implementation.

The experiment sources and small result summaries below are committed with
these implementations. The production runner was consolidated from the former
`gemma4-litert-chunked-prefill` branch. No separate PR worktree is needed.

| Directory | Purpose |
| --- | --- |
| [native_runner_lifetimes](native_runner_lifetimes/README.md) | Packed-source release and redundant reshape experiments. |
| [native_mlp_profile](native_mlp_profile/README.md) | MLP/FC profiles, CPU thread sweeps, and phone follow-ups. |
| [native_single_graph_decode](native_single_graph_decode/README.md) | Joined native decode graph and comparison against staged execution. |
| [litert_joined_decode](litert_joined_decode/README.md) | Controlled native versus LiteRT joined decode. |
| [litert_chunked_prefill](litert_chunked_prefill/README.md) | Joined native and LiteRT chunked prefill with correctness and memory checks. |
| [litert_production_phones](litert_production_phones/README.md) | Original three-phone production/native campaign controllers and analysis. |
| [runner_memory_accounting](runner_memory_accounting/README.md) | Container accounting, embedding residency, and pageout diagnostics. |
| [local_attention_cost](local_attention_cost/README.md) | Local-attention cost probe. |
| [ynnpack_attention_layout](ynnpack_attention_layout/README.md) | YNNPACK attention layout experiment and patch. |
| [history](history/README.md) | Earlier source snapshots and migration provenance. |

For a new phone measurement, use the
[reproduction guide](../../local_docs/gemma4_phone_reproduction/README.md).
The [memory model](../../local_docs/gemma4_runner_memory_model.md) and
[model-size analysis](../../local_docs/gemma4_e2b_model_size_vs_memory.md)
explain the allocation and residency accounting.

Large model files, packed caches, raw captures, and compiled dependency archives
remain in the artifact directories recorded by each experiment. Some historical
native relink scripts deliberately require those archived build outputs to keep
kernel and compiler choices identical to the original measurement. Ordinary
runner builds use the Bazel targets in the two example directories.

[Consolidation provenance](consolidation-20260925.json) records source locations
and hashes for files recovered from temporary artifact directories. Historical
hashes and build paths describe their original captures, not new builds.
