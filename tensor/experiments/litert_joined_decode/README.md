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

# Joined Gemma4 decode through LiteRT

The [report](../../docs/gemma4_litert_joined_decode.md) describes results,
allocation accounting, and scope. This is a desktop experiment using the
LiteRT TFLite Interpreter and XNNPACK delegate. Prefill remains native and MTP
is excluded. It does not replace the production Tensor API runner.

The generated driver and model headers are included here so the Bazel target
can build directly. `prepare.py` regenerates them from the archived joined
native overlay and current native support/transformer headers, recording source
hashes in `source_identity.json`. Its `--source` argument selects a different
overlay. The default is
`/data/bt/tmp/gemma4-native-single-graph-20260923/build-v3/overlay`.

`mapped_runner.h` expands native helpers, serializes existing LiteRT operators,
loads the resulting file, rebinds constants for shared packing, and executes
with active shapes and borrowed inputs. Link-time wrappers in
`runtime_observer.cc` only observe calls; they forward all arguments unchanged.
Partial delegation, changed attention kernels/extents, changed INT2 FC counts,
or multiple XNNPACK invocations cause the experiment to fail.

Build from the repository root:

```bash
mkdir -p .bazelisk-cache .cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  //tensor/experiments/litert_joined_decode:runner
```

Use a new artifact directory. The commands preserve the previous experiment's
fixture and bundle locations and pin CPUs `0,2,4,6` on this desktop:

```bash
python3 tensor/experiments/litert_joined_decode/run.py \
  --out=/data/bt/tmp/gemma4-litert-joined-new --label=final- --mode=correctness
python3 tensor/experiments/litert_joined_decode/run.py \
  --out=/data/bt/tmp/gemma4-litert-joined-new --label=final- --mode=timing
python3 tensor/experiments/litert_joined_decode/run.py \
  --out=/data/bt/tmp/gemma4-litert-joined-new --label=final- --mode=memory
python3 tensor/experiments/litert_joined_decode/analyze.py \
  --out=/data/bt/tmp/gemma4-litert-joined-new --label=final-
```

`--binary` and `--bundle` override the executable and model assets. `--only`
selects one variant for diagnosis. `--mode=minimal` exercises four decode
passes starting with empty history; `--mode=smoke` uses the earlier smoke
fixture. Diagnostic and memory runs are marked ineligible for timing.

Both modes default to matching the delegate's worker policy; use
`--match_delegate_worker_policy=false` directly on the binary only to diagnose
the old native scheduling policy. The matched comparison retains separate
native/delegate workspaces, and the report accounts for that difference.

The final captures and frozen binary are in
`/data/bt/tmp/gemma4-litert-single-graph-20260923`. The artifact root contains
`build_identity.json`, the build log, `export_metadata.json`, and the serialized
models. [results.json](results.json) records the validated comparison and the
single binary hash shared by all final correctness, timing, and memory runs.

The subsequent empty-dimension fix to `CompiledModel` and tensor-buffer size
validation is independent of the measured Interpreter binary. Its regression
commands are:

```bash
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" test -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  //litert/cc:litert_compiled_model_test \
  //litert/c:litert_tensor_buffer_test \
  //litert/core/util:tensor_type_util_test --test_output=errors
```
