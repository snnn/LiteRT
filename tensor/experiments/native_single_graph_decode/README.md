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

# One XNNPACK graph per Gemma4 decode invocation

This experiment tests whether C++ must intervene between the native runner's
decode layers. See the [report](../../docs/gemma4_single_graph_decode.md) for
results and limits. Prefill remains unchanged; MTP is excluded.

`build.py` copies the driver and three model-authoring headers into an isolated
overlay. It conditionally keeps the hidden-state, per-layer embedding, and
attention edges connected. `joined_attention.inc` authors active past/new/pad
KV concatenations, shared by consumers of each owner. `joined_run.inc` binds
inputs before one invocation and commits the 15 new owner rows afterward.
It also audits the actual attention operator types and extents during full-logit
diagnostics. No production source or XNNPACK library is edited.

The executable supports both staged and joined modes, so they use identical
libraries and compiler settings. This reproduces the archived Linux build by
relinking its driver object; it does not invoke CMake or rebuild dependencies.
Use the repository's Bazelisk workflow for ordinary production builds.

Use a fresh output root; captures and builds refuse to overwrite existing
directories. These commands reproduce the reported desktop setup and retain
its fixture, bundle, archived-build, and baseline-reference paths:

```bash
python3 tensor/experiments/native_single_graph_decode/build.py \
  --baseline /data/bt/tmp/gemma4-native-next-20260923/baseline \
  --out /data/bt/tmp/gemma4-native-single-graph-new/build-v3

python3 tensor/experiments/native_single_graph_decode/run.py \
  --out /data/bt/tmp/gemma4-native-single-graph-new \
  --binary /data/bt/tmp/gemma4-native-single-graph-new/build-v3/runner --mode smoke

python3 tensor/experiments/native_single_graph_decode/run.py \
  --out /data/bt/tmp/gemma4-native-single-graph-new \
  --binary /data/bt/tmp/gemma4-native-single-graph-new/build-v3/runner --mode correctness

python3 tensor/experiments/native_single_graph_decode/run.py \
  --out /data/bt/tmp/gemma4-native-single-graph-new \
  --binary /data/bt/tmp/gemma4-native-single-graph-new/build-v3/runner --mode timing

python3 tensor/experiments/native_single_graph_decode/run.py \
  --out /data/bt/tmp/gemma4-native-single-graph-new \
  --binary /data/bt/tmp/gemma4-native-single-graph-new/build-v3/runner --mode memory
```

The CPU mask `0,2,4,6` is specific to the audited i9 desktop. Correctness
compares all vocabulary vectors and committed owner K/V, including empty
history, empty padding, window/alignment boundaries, long contexts, and
runtime reuse after long-to-short session changes. Timing reverses mode order
in two blocks with one warmup and three measured sessions per process.

`analyze.py` validates the reported captures, compares the unchanged mode with
the previous baseline, and produces `analysis.json`. The builder also verifies
the linked archives against the baseline's `inputs.json` and records
`archive-identity.json` at the experiment root. Run:

```bash
python3 tensor/experiments/native_single_graph_decode/analyze.py \
  --out /data/bt/tmp/gemma4-native-single-graph-new
```

The final reported captures are in
`/data/bt/tmp/gemma4-native-single-graph-20260923`; [results.json](results.json)
preserves the validated summary and identities. Memory/profiling/dump timings
are not ordinary benchmark results. Per-stage attention/projection/post fields
are zero in joined mode because those stages are internal to one invocation;
use the whole decode and `forward_ms` fields for this comparison.
