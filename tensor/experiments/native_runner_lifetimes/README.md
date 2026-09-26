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

# Gemma4 native runner: packing lifetimes and redundant reshape

These isolated experiments investigate two changes to the existing optimized
native runner. Production sources and defaults are unchanged. The results and
next priorities are in
[the experiment report](../../docs/gemma4_native_runner_improvements.md).
The checked-in [results.json](results.json) contains the validated memory,
timing, correctness, and executable-identity summaries from the reported run.

[probe.patch](probe.patch) adds two independently selectable switches:

- `--experiment_release_packed_sources=true`: after both signatures are
  compiled and the weights cache is hard-finalized, release the 60 compact INT2
  MLP source buffers and the head's INT2 conversion buffer. It checks that the
  runtime consumers are static QC2 fully connected kernels, removes retained
  graph references, and verifies that all 60 source owners expire. Quantization
  scales, packed weights, embeddings, and execution constants remain live. This
  is a one-way transition, requiring `--reuse_runtimes=true` and
  `--preserve_static_int2=true`; recompilation would require reloading sources.
- `--experiment_cache_stage_shapes=true`: cache each stage's external input
  shapes, skip `xnn_reshape_runtime` while they are unchanged, and always run
  external pointer binding/setup before invocation. The pinned XNNPACK repairs
  attached runtimes when a shared workspace relocates. The common runner's
  output queries, reservations, locks, and temporary binding vectors still run.

The release probe deliberately recognizes the E2B head's 96 MiB conversion
buffer. This is a controlled ownership experiment, not a generic constant
release API. A production implementation needs explicit compile-time versus
execution-time buffer lifetimes and must account for all runtime consumers.

The scripts run processes sequentially with four threads pinned to CPUs
`0,2,4,6`, 128-row prefill, KV alignment 32, capacity 8,448, shared workspace,
compact INT2 weights, and reused runtimes. These CPU IDs describe the tested
Linux desktop, not a portable affinity recommendation.

## Reproduce on the audited desktop

The experiment directory used for the report is
`/data/bt/tmp/gemma4-native-next-20260923`. Use a fresh output directory for a
new experiment; capture directories cannot be overwritten.

```bash
python3 tensor/experiments/native_runner_lifetimes/build_probe.py \
  --build .native-tensor-build/host \
  --out /data/bt/tmp/gemma4-native-next-new

python3 tensor/experiments/native_runner_lifetimes/run_experiments.py prepare \
  --out /data/bt/tmp/gemma4-native-next-new \
  --bundle /data/bt/tmp/gemma4-linux-xnn-memory-20260923/matched-bundle

python3 tensor/experiments/native_runner_lifetimes/run_experiments.py smoke \
  --out /data/bt/tmp/gemma4-native-next-new \
  --bundle /data/bt/tmp/gemma4-linux-xnn-memory-20260923/matched-bundle \
  --variants archived baseline release reshape combined

python3 tensor/experiments/native_runner_lifetimes/analyze.py smoke \
  --out /data/bt/tmp/gemma4-native-next-new \
  --variants archived baseline release reshape combined
```

Then run and analyze each of `correctness`, `memory`, and `timing` in that order:

```bash
python3 tensor/experiments/native_runner_lifetimes/run_experiments.py correctness \
  --out /data/bt/tmp/gemma4-native-next-new \
  --bundle /data/bt/tmp/gemma4-linux-xnn-memory-20260923/matched-bundle
python3 tensor/experiments/native_runner_lifetimes/analyze.py correctness \
  --out /data/bt/tmp/gemma4-native-next-new

python3 tensor/experiments/native_runner_lifetimes/run_experiments.py memory \
  --out /data/bt/tmp/gemma4-native-next-new \
  --bundle /data/bt/tmp/gemma4-linux-xnn-memory-20260923/matched-bundle
python3 tensor/experiments/native_runner_lifetimes/analyze.py memory \
  --out /data/bt/tmp/gemma4-native-next-new

python3 tensor/experiments/native_runner_lifetimes/run_experiments.py timing \
  --out /data/bt/tmp/gemma4-native-next-new \
  --bundle /data/bt/tmp/gemma4-linux-xnn-memory-20260923/matched-bundle
python3 tensor/experiments/native_runner_lifetimes/analyze.py timing \
  --out /data/bt/tmp/gemma4-native-next-new
```

`build_probe.py` reads the preserved compilation database and link command,
compiles two isolated drivers, and links both against the same existing static
libraries. It does not invoke CMake, change the production checkout, or rebuild
dependencies. This preserves the previously audited XNNPACK kernels. Normal
repository development should use the documented Bazelisk targets. The script
records source, archive, patch, and executable SHA-256 identities and the exact
compiler/link commands.

`correctness` checks ten prompt lengths from 8 through 8,192, including partial
chunks, alignment boundaries, local-window boundaries, and resets from long to
short contexts in a reused runtime. Each case runs a warmup and a measured
session. The first measured session dumps every vocabulary logit and the
committed INT8 K/V for every owner after each pass. The analyzer requires
bit-identical dumps and matching input histories and predictions.

`memory` uses the previous Linux comparison's 1,024-token prompt and 64 forced
tokens, with one warmup and one measured session. Phase RSS/allocation snapshots
are diagnostic; their timings are not benchmark results. Peak RSS comes from
the child's `wait4` result. No allocator trimming or cache-dropping is used.

`timing` disables memory reports and all dumps. Each process warms up each of
the 128-, 1,024-, and 4,096-token prompts, then measures three sessions with 64
forced tokens. There are three process blocks; the middle block reverses the
configuration order. The analyzer reports per-process medians and paired
changes against the baseline in that block. This small sample describes this
CPU/configuration and does not establish performance on Android.
