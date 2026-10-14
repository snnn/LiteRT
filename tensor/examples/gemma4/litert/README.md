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

# Gemma4 through the LiteRT production runner

This example exports two Tensor API graphs to one `.tflite` model and runs it
through `LitertDynamicRunner` and the public `CompiledModel` API. The CPU backend
uses LiteRT's existing XNNPACK delegate. Inference contains no direct XNNPACK
runtime calls, custom attention operators, or per-layer execution loop.

The original C++ exporter is specialized to the published `gemma-4-E2B-it.litertlm` bundle in
`litert-community/gemma-4-E2B-it-litert-lm`, revision
`616f4124e6ff216292f16e7f73ff33b5ba9a4dd4`. Its source SHA-256 is
`ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42`
(2,583,085,056 bytes). The loader checks this identity and the expected layout;
the preparation tools validate the actual exported coefficients against the
source. The shared Python/Core IR author additionally supports the published
E4B architecture, selected by bundle metadata, and the runner accepts
`--model_variant=e4b`. E4B requires a compact bundle and a Python-authored model;
the C++ exporter remains E2B-only. See the
[E4B instructions and numerical limitations](../../../experiments/explicit_dynamic_shapes/E4B.md).
The example does not load arbitrary checkpoints. MTP is outside its scope.

The execution and original preparation instructions below describe the E2B
control. New extractions use schema version 2, which validates architecture and
source coefficients rather than requiring the original E2B source identity.
Existing schema-version-1 bundles retain the original identity checks.

## Execution

The model has a 128-row `prefill` signature and a one-row `decode` signature.
Both use a single `CompiledModel`, allowing the delegate to share its packed
weight cache and scratch workspace across signatures. The fully-delegated CPU
hint requires a single delegate partition per subgraph and fails model loading
if that requirement cannot be met.

Prefill processes all prompt tokens except the last. Its graph produces only
the 15 distinct owners' new K/V blocks: the first 14 complete layers and the
last owner's K/V projections. Later layers share existing KV and are unnecessary
for the prompt prefix. The final prompt token and subsequent decode tokens run
all 35 layers and the vocabulary head.

The host stores 30 aligned, token-major INT8 K/V arrays. Capacity controls only
persistent storage (74.25 MiB at 8,448 tokens). Each invocation binds the active
history, a small alignment pad, and local/global masks. Input shapes grow with
the actual history; local attention retains its window. The final partial
prefill chunk executes padded rows, masks them, and commits only valid rows.
An append transaction publishes the new length after every KV owner finishes.
Reset invalidates the logical history without reallocating or clearing storage.

Attention uses ordinary concatenation, BMM, mask, softmax and BMM operators.
The prefill masks broadcast across heads. Attention scores are still
materialized per chunk, and concatenation still copies the active history
inside the graph. This is chunked prefill, not tiled online-softmax attention.
Original static INT2 MLP weights, the dynamically quantized INT2 head, INT8 KV
scales and activation quantization boundaries are preserved.
The authoring uses explicit `Quantize()` and `Dequantize()` operations for
static FC activations and K/V conversions; numeric `Cast()` does not apply
quantization scales or zero points.
Per-channel weights explicitly carry one zero point per scale, so the TFLite
serializer does not need to expand shared zero points.

## Build and prepare

From the repository root:

```sh
mkdir -p .bazelisk-cache .cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  //tensor/examples/gemma4/litert:export_model \
  //tensor/examples/gemma4/litert:runner @flatbuffers//:flatc
```

Preparation needs Python, NumPy and FlatBuffers. Set `MODEL` to the source
`.litertlm` path, `SCHEMA_DIR` to a generated-Python schema directory, and
`PREPARED` to a new output directory. No model coefficients are checked in.

```sh
mkdir -p "$SCHEMA_DIR"
bazel-bin/external/flatbuffers/flatc --python -o "$SCHEMA_DIR" \
  tflite/converter/schema/schema.fbs
python3 tensor/examples/gemma4/litert/tools/prepare_bundle.py \
  --model "$MODEL" --schema-dir "$SCHEMA_DIR" --output-dir "$PREPARED"
```

Run Python without `-O`; source-validation assertions are part of the tools.
Preparation writes the audit and verified `weights/` directory. It includes
all 1,093 tensors, original quantization scales and source provenance. INT2
values travel losslessly in INT4 containers; the model exporter restores the
original compact INT2 tensors. The host embedding tables remain in their
lookup format.

```sh
bazel-bin/tensor/examples/gemma4/litert/export_model \
  --bundle_dir "$PREPARED/weights" --model_path /tmp/gemma4.tflite
bazel-bin/tensor/examples/gemma4/litert/runner \
  --bundle_dir "$PREPARED/weights" --model_path /tmp/gemma4.tflite \
  --cases_file tensor/examples/gemma4/litert/fixtures/smoke.tsv \
  --output_dir /tmp/gemma4-smoke --cache_capacity 8448 --num_threads 4 \
  --warmup_runs 1 --measured_runs 3
```

Export and results paths must be new. The runtime creates/reuses a packed weight
cache at `model_path.xnnpack_cache`; `--weight_cache_path` overrides it. Export
is a separate process, so inference retains only the host embedding tables and
the compiled model, not the graph builder or conversion weight buffers.

Cases have three tab-separated fields: a unique case ID, comma-separated prompt
token IDs, and comma-separated forced decode input IDs (`-` for none). This
makes numerical comparisons independent of generated token divergence. The
runner writes `timings.jsonl`, including prefix time, per-token decode time,
argmax, context length and process RSS high-water mark. Step zero consumes the
last prompt token. `decode_ms` stops when the runner returns logits; `token_ms`
also includes validation and greedy token selection. Add prefix time to the
step-zero duration to obtain first-logits or first-token latency respectively.
Use `--dump_outputs --warmup_runs=0 --measured_runs=1` for complete logits and
committed KV dumps. Dump runs are correctness runs, not latency measurements.
The reusable client is in [runner.h](runner.h).

## Android

Export the model on the host, then build the runner for Android. Set the SDK
and NDK paths to your installation; this configuration was tested with NDK r28c
and Android API 35:

```sh
ANDROID_SDK=/path/to/android_sdk
ANDROID_NDK="$ANDROID_SDK/ndk/28.2.13676358"
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  --repo_env=ANDROID_SDK_HOME="$ANDROID_SDK" \
  --repo_env=ANDROID_NDK_HOME="$ANDROID_NDK" \
  --repo_env=ANDROID_NDK_VERSION=28 \
  --repo_env=ANDROID_SDK_API_LEVEL=35 \
  --repo_env=ANDROID_NDK_API_LEVEL=35 \
  --repo_env=ANDROID_BUILD_TOOLS_VERSION=35.0.0 \
  --config=android_arm64 //tensor/examples/gemma4/litert:runner
```

Copy the runner, `.tflite` model, prepared weight directory and cases TSV into
`/data/local/tmp` on the device, then run the same command-line interface through
`adb shell`. Use device-local paths and let the device create its packed-weight
cache. The binary targets explicitly retain Abseil flag names, which Android
builds otherwise strip by default.

[Three-phone benchmark results](PHONE_BENCHMARKS.md) compare the production
runner with historical records and fresh runs of the native XNNPACK runner.
Samsung decode matches, TECNO is about 5% slower, and Pixel is about 14% slower;
first-token latency improves on all three phones.

## Validation and memory accounting

The CPU regression tests cover empty/growing/shrinking history across signatures,
borrowed buffer identity, output resizing, quantized export and transactional KV
appends. Run them from the repository root:

```sh
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" test -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  //litert/cc:litert_compiled_model_test \
  //litert/c:litert_tensor_buffer_test \
  //litert/core/util:tensor_type_util_test \
  //tensor/backends/tflite:tflite_flatbuffer_conversion_test \
  //tensor/runners/litert:lambda_model_runner_test \
  //tensor/examples/gemma4/litert:kv_bank_test
```

All six targets passed on Linux (229 individual tests passed, six skipped).
The production example was also checked against the previously
validated Gemma4 reference: all 5,053 files (1,227,569,152 bytes) were bit-identical.
These include 163 full-vocabulary vectors and 4,890 committed KV snapshots across
11 cases, prompt lengths through 8,192, partial chunks and long-to-short reuse.

On an Intel Core i9-12900K Linux desktop, four threads pinned to CPUs
`0,2,4,6`, chunk size 128 and KV capacity 8,448 produced the following medians.
Each implementation ran two blocks with one warmup and three measured runs per
case. Decode uses the two forced tokens after the prompt. The reference uses
the same Tensor API graphs through the raw TFLite interpreter. The production
path uses the public `CompiledModel` API and a warm packed-weight cache.

| Prompt tokens | Interpreter prefix ms | Production prefix ms | Interpreter decode ms | Production decode ms | Production peak RSS MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 106.9 | 106.3 | 26.05 | 25.42 | 1,092 |
| 1,024 | 968.7 | 965.3 | 28.02 | 27.36 | 1,227 |
| 4,096 | 4,443.3 | 4,410.8 | 31.80 | 31.09 | 1,262 |
| 8,192 | 10,140.5 | 10,088.3 | 36.69 | 36.19 | 1,316 |

Prefix time excludes the final prompt token; median production first-logits
latencies were 130.1, 990.9, 4,440.1 and 10,122.4 ms respectively. RSS is the
process high-water mark, including setup and earlier cases in the same process.
The interpreter's decode timer includes output copying and greedy selection,
whereas the production column reports `decode_ms`. Small differences between
these columns therefore do not isolate wrapper overhead. Subsequent comparisons
with generation runners use `token_ms` for a consistent timing boundary.
These are desktop observations with a warm cache.
Caching signature names and querying only the requested input tensor removed
approximately 2.3 ms of decode overhead observed in the initial integration.

The standard file-backed XNNPACK weight-cache provider may request
`madvise(MADV_PAGEOUT)` for source weights after packing. Warm cache loads can
also avoid touching those source pages. Consequently, RSS is process residency,
not total retained allocations, and it must not be compared directly with a
runner that keeps both source and packed weights resident. Model files and
packed-cache files remain mapped. Separate conversion/inference lifetimes and
active, shared graph scratch address different sources of memory use.

An untimed smoke audit, forwarding every `madvise` call unchanged, observed 277
`MADV_PAGEOUT` requests totaling 739.6 MiB of requested ranges while creating a
fresh packed cache, and zero on reuse. Both runs had zero sampled `VmSwap`.
Requested ranges are advisory and are not a measurement of freed allocations.
