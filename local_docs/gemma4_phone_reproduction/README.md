<!-- Copyright 2026 The ODML Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Reproducing Gemma4 E2B CPU performance on Android phones

This guide compares the Gemma4 **production Tensor API / LiteRT runner** with
**LiteRT-LM using XNNPACK**, with identical token IDs, four CPU threads, and a KV
capacity of 8,448 tokens. The primary comparison uses 128-row prefill in both
runners. A second comparison keeps LiteRT-LM's original signature selection to
measure the effect of choosing the smaller prefill graph.

Assume the LiteRT checkout contains the production Gemma4 runner, the developer
already manages source versions, and `TENSORFLOW_SHA` is up to date. No source
pinning or dependency-update procedure is needed here. The Android device is
already connected and usable through `adb`.

The guide and its small helper files are separate from the implementation
branch. They do not require the original native XNNPACK runner. MTP, speculative
decoding, GPU execution, and an SDPA graph rewrite are outside this procedure.

## 1. What is being compared

| Variant | Execution path | Prefill | Attention history |
| --- | --- | --- | --- |
| Production | Tensor API graphs → public LiteRT `CompiledModel` → XNNPACK | 128-row chunks | Active history, with alignment padding and a bounded local window |
| LM-128 | Original `.litertlm` → LiteRT-LM CPU executor → XNNPACK | Existing `prefill_128` signature | Original exported model's capacity-shaped behavior |
| LM-original | Same as LM-128, with the unmodified bundle | Original scheduler selection; verify the trace | Same as LM-128 |

Removing `prefill_1024` lets LiteRT-LM select its existing smaller prefill graph.
It does **not** turn the original exported model into the production runner's
active-history graph. Therefore a remaining performance or memory difference
cannot be attributed solely to prefill chunk size or the wrapper API.

Both runners use the same source coefficients. The production preparation step
preserves the original quantization, including compact INT2 weights and INT8
KV data. It exports a different graph, with separate prefill and decode
signatures. The prefill graph computes the distinct KV owners and omits later
shared-KV computation that the prompt prefix does not need. The decode graph
executes the full model. This is ordinary chunked attention: attention scores
are materialized per chunk, and it is not FlashAttention/online softmax.

### Workload and timing contract

Use the included [performance fixture](fixtures/performance_1024_64.tsv):

- 1,024 prompt token IDs, including BOS.
- 64 **forced decode input** token IDs, so numerical differences cannot change
  the sequence being benchmarked.
- 65 predictions: one after the complete prompt, then one after each forced
  decode input. The first prediction is not part of the 64-call decode mean.
- Four threads; capacity 8,448; production prefill chunk 128 and KV alignment 32.
- One warmup session and three measured sessions in each process.
- Two processes per variant, giving six measured session means.

The fixture is a synthetic performance sequence made from repeated text-token
segments. It is not a quality evaluation. Its SHA-256 is
`01009f2da16a6c32e7288323029d019b49e1507022fd711f471c8001824b4eef`.

The bundled [LiteRT-LM harness](litert_lm_harness/adapter.cc) calls the real CPU
executor with raw token IDs. This avoids tokenizer, chat-template, EOS, and
sampler differences. It applies the engine's model metadata settings and the
context-size rewrite, verifies the consumed token history, and checks that
every output logit is finite. It does not replace LiteRT-LM's inference graph.

## 2. Host paths and tools

Use a Linux host with `bazelisk`, `uv`, Python, an Android SDK/NDK, and `adb`.
Run the following snippets in the same Bash shell, adjusting the paths:

```bash
set -euo pipefail
export LITERT_SRC=/path/to/LiteRT
export LM_SRC=/path/to/LiteRT-LM
export GUIDE=/path/to/gemma4_phone_reproduction
export WORK=/path/to/gemma4-phone-work
export ANDROID_SDK=/path/to/android_sdk
export ANDROID_NDK="$ANDROID_SDK/ndk/28.2.13676358"

mkdir -p "$WORK/bin" "$WORK/models" "$WORK/schema" "$WORK/results"
uv venv --python 3.13 "$WORK/venv"
uv pip install --python "$WORK/venv/bin/python" numpy flatbuffers
export PY="$WORK/venv/bin/python"
```

The Android recipe below uses NDK r28c, SDK/build tools 35, and minimum API 35,
as in the production phone run. All three reference phones support that API.
For an older device, choose an appropriate minimum API for **both** binaries
and treat its results as a different configuration.

Allow room for Bazel outputs, the 2.58 GB source bundle, a same-size modified
copy, extracted weights, the exported model, and packed-weight caches. Check
device storage before copying the artifacts. Build the two executables against
the same LiteRT checkout using the repository override below.

## 3. Obtain the source model

The production example is specialized to
[the Gemma4 E2B instruction-tuned bundle](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm).
It does not accept arbitrary Gemma4/E4B checkpoints. This **model artifact** is
fixed because the loader validates its coefficients and layout:

```bash
uvx hf download litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  --revision 616f4124e6ff216292f16e7f73ff33b5ba9a4dd4 \
  --local-dir "$WORK/download"

export MODEL="$WORK/download/gemma-4-E2B-it.litertlm"
sha256sum "$MODEL"
```

Expected size: **2,583,085,056 bytes**. Expected SHA-256:

```text
ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42
```

An existing downloaded copy with this checksum is sufficient. Use the
**unmodified** bundle for production weight preparation. Create a separate
LM-128 copy in section 6.

## 4. Prepare and export the production model on the host

Build host tools from the LiteRT repository root:

```bash
cd "$LITERT_SRC"
mkdir -p .bazelisk-cache .cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  //tensor/examples/gemma4/litert:export_model \
  @flatbuffers//:flatc

cp bazel-bin/tensor/examples/gemma4/litert/export_model "$WORK/bin/export_model"
cp bazel-bin/external/flatbuffers/flatc "$WORK/bin/flatc"

"$WORK/bin/flatc" --python -o "$WORK/schema" \
  tflite/converter/schema/schema.fbs

"$PY" tensor/examples/gemma4/litert/tools/prepare_bundle.py \
  --model "$MODEL" \
  --schema-dir "$WORK/schema" \
  --output-dir "$WORK/prepared"

"$WORK/bin/export_model" \
  --bundle_dir "$WORK/prepared/weights" \
  --model_path "$WORK/models/gemma4.tflite" \
  --prefill_chunk_rows=128
```

The prepared/export paths must be new. Run Python without `-O`: the preparation
tools use assertions to validate the source data. The preparation output
contains `weights/` and audit files. Copy the entire `weights/` directory to the
phone, including its manifest and quantization constants.

Export runs as a separate host process. Its graph-builder and conversion
buffers are therefore absent from the phone inference process. Save host tools
before changing Bazel's target platform; `bazel-bin` can point to different
outputs after the Android build.

## 5. Build the two Android executables

Define common Android options once:

```bash
ANDROID_FLAGS=(
  --config=android_arm64
  --repo_env=HERMETIC_PYTHON_VERSION=3.13
  --repo_env=ANDROID_SDK_HOME="$ANDROID_SDK"
  --repo_env=ANDROID_NDK_HOME="$ANDROID_NDK"
  --repo_env=ANDROID_NDK_VERSION=28
  --repo_env=ANDROID_SDK_API_LEVEL=35
  --repo_env=ANDROID_NDK_API_LEVEL=35
  --repo_env=ANDROID_BUILD_TOOLS_VERSION=35.0.0
)

cd "$LITERT_SRC"
mkdir -p .bazelisk-cache .cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  "${ANDROID_FLAGS[@]}" \
  //tensor/examples/gemma4/litert:runner
cp bazel-bin/tensor/examples/gemma4/litert/runner "$WORK/bin/runner.android"
```

Install the local raw-token harness into a new package in the LiteRT-LM checkout:

```bash
test ! -e "$LM_SRC/local_phone_reproduction"
cp -R "$GUIDE/litert_lm_harness" "$LM_SRC/local_phone_reproduction"

cd "$LM_SRC"
mkdir -p .bazelisk-cache .cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  "${ANDROID_FLAGS[@]}" \
  --override_repository=litert="$LITERT_SRC" \
  --define=litert_runtime_link_mode=static \
  --define=litert_link_capi_so=false \
  --define=DISABLE_HUGGINGFACE_TOKENIZER=1 \
  --define=DISABLE_SENTENCEPIECE_TOKENIZER=1 \
  --define=litert_enable_ynnpack=true \
  //local_phone_reproduction:adapter
cp bazel-bin/local_phone_reproduction/adapter "$WORK/bin/litertlm.android"
```

The LM binary includes the YNNPACK option, but every primary run below explicitly
sets `--enable_ynnpack false`. Including that backend in the build does not
select it. These are static **Android** link options; do not reuse them as a
Linux shared-runtime build recipe. The disabled tokenizers are unnecessary for
this raw-token workload.

The harness uses LiteRT-LM executor APIs, so normal source compatibility applies
if those APIs change. A compilation error should be resolved against the
current executor API; it is not a reason to substitute a chat CLI with different
tokenization and timing semantics.

## 6. Select 128-row prefill in the `.litertlm` bundle

### Why remove a signature?

The original language-model section exposes `decode`, `prefill_1024`,
`prefill_128`, and `verify`. With the 1,024-token prompt, the observed static
prefill scheduler selected the larger prefill signature. Exposing only
`prefill_128` as a prefill candidate makes it process the prefix in smaller
chunks. The final prompt token still goes through decode to produce the first
prediction.

The `CpuConfig.prefill_chunk_size` option in the dynamic executor is not a
universal override for an exported model's static signatures. Verify the actual
selected signature instead of assuming a flag changed it. If a future executor
offers a supported static-signature selection option, that can replace this
bundle edit after the same trace check.

### Remove `prefill_1024` without repacking weights

Inspect the original bundle and create a separate copy:

```bash
"$PY" "$GUIDE/select_prefill128.py" \
  --mode inspect --input "$MODEL" \
  --schema-dir "$WORK/schema" --litert-src "$LITERT_SRC" \
  > "$WORK/original-signatures.json"

"$PY" "$GUIDE/select_prefill128.py" \
  --mode remove --input "$MODEL" \
  --output "$WORK/models/gemma4-prefill128.litertlm" \
  --schema-dir "$WORK/schema" --litert-src "$LITERT_SRC" \
  > "$WORK/prefill128-edit.json"
```

The helper rejects an existing output file and refuses to modify any model
other than the documented source bundle. It removes the `prefill_1024` entry
from the TFLite `SignatureDefs` vector, retaining `decode`, `prefill_128`,
`verify`, and the other model sections. It preserves subgraph indices, section
offsets, external-buffer offsets, coefficients, and total file size.

For this bundle, literal removal changes **five bytes**. Expected output SHA-256:

```text
3ae76d1b646a9a1b0400ab860762bf02c76041ee6a4cc7cf7f5373818dd7bad9
```

`prefill128-edit.json` records the complete before/after signature lists and
verifies that all other bytes match the original. Removing a signature entry
does not delete its underlying subgraph or shrink the file. The improvement
comes from the execution path selected at runtime, not from deleting weights.
The string `prefill_1024` can still appear in the file's unused data; `strings`
or a byte search is not a reliable test of the exposed signatures.

Avoid a naive whole-file FlatBuffer unpack/repack or byte deletion: a
`.litertlm` is a container, and its section offsets and external buffers must
remain valid. This helper is intentionally specific to this model. For a
different bundle, use a format-aware tool and validate its signatures, section
bounds, buffers, and inference separately.

### Reproduce the earlier memory-control file exactly

The previous memory audit used a same-length rename to **disable selection**,
not literal signature removal. It changed `prefill_1024` to `ignored_1024` in
two places, leaving the signature entry present but outside the `prefill`
name prefix. To recreate that control instead:

```bash
"$PY" "$GUIDE/select_prefill128.py" \
  --mode rename --input "$MODEL" \
  --output "$WORK/models/gemma4-prefill128-renamed.litertlm" \
  --schema-dir "$WORK/schema" --litert-src "$LITERT_SRC" \
  > "$WORK/prefill128-rename.json"
```

That file changes 14 bytes and has SHA-256
`1fb7ac959565ca4c8135533afaf7057e7d030b5acd81d113ef18ab4978f45090`.
Both methods aim to select `prefill_128`; do not assume their initialization
behavior or memory peaks are identical. Use rename mode when reproducing the
historical memory control, and report which method you used.

Smaller chunks reduce the largest prefill activations, but introduce more graph
invocations. They can improve CPU performance under memory/cache pressure;
**128 rows is not guaranteed to improve prefill latency on every backend or
phone**. Measure first-token latency as well as peak memory.

## 7. Install on a phone and choose its CPU affinity

The reference devices used these affinity masks:

| Phone | SoC | CPU mask | Selected logical CPUs |
| --- | --- | --- | --- |
| TECNO LJ9 | MT6897 | `f0` | 4–7 |
| Pixel 8 | Tensor G3 | `1e0` | 5–8 |
| Samsung SM-S937U1 | SM8750 | `f0` | 4–7 |

Masks are hexadecimal. Do not reuse them blindly on a different CPU topology.
Keep the same four cores and thread count for both variants. `taskset` pins the
process and its workers; it does not fix their clock frequency.

For each phone, set its serial and mask, and choose an experiment directory:

```bash
export SERIAL=YOUR_ADB_SERIAL
export MASK=f0
export PHONE_NAME=tecno
export REMOTE=/data/local/tmp/gemma4-phone-repro

adb -s "$SERIAL" shell getprop ro.product.model
adb -s "$SERIAL" shell getprop ro.soc.model
adb -s "$SERIAL" shell df -h /data/local/tmp
adb -s "$SERIAL" shell mkdir -p "$REMOTE"

adb -s "$SERIAL" push -Z "$WORK/bin/runner.android" "$REMOTE/runner.android"
adb -s "$SERIAL" push -Z "$WORK/bin/litertlm.android" "$REMOTE/litertlm.android"
adb -s "$SERIAL" push -Z "$WORK/models/gemma4.tflite" "$REMOTE/gemma4.tflite"
adb -s "$SERIAL" push -Z "$WORK/prepared/weights" "$REMOTE/"
adb -s "$SERIAL" push -Z "$GUIDE/fixtures/performance_1024_64.tsv" "$REMOTE/"
adb -s "$SERIAL" push -Z "$WORK/models/gemma4-prefill128.litertlm" \
  "$REMOTE/gemma4-prefill128.litertlm"
adb -s "$SERIAL" shell chmod 755 "$REMOTE/runner.android" "$REMOTE/litertlm.android"

sha256sum "$WORK/bin/runner.android" "$WORK/bin/litertlm.android" \
  "$WORK/models/gemma4.tflite" "$WORK/models/gemma4-prefill128.litertlm" \
  "$GUIDE/fixtures/performance_1024_64.tsv"
adb -s "$SERIAL" shell sha256sum "$REMOTE/runner.android" \
  "$REMOTE/litertlm.android" "$REMOTE/gemma4.tflite" \
  "$REMOTE/gemma4-prefill128.litertlm" "$REMOTE/performance_1024_64.tsv"
```

Match the local and remote hashes. The manifest in `weights/` also records the
coefficient files; keep the complete directory together. `adb push -Z` disables
transfer compression and was useful for large transfers in the reference run.
If space is tight, remove only your obsolete experiment artifacts, or run the
original and modified LM variants sequentially without keeping both bundles on
the phone. Do not delete an active model or cache during inference.

### Prime the production packed-weight cache

Run a separate untimed process to create the device's packed-weight cache:

```bash
adb -s "$SERIAL" shell "taskset $MASK $REMOTE/runner.android \
  --bundle_dir=$REMOTE/weights --model_path=$REMOTE/gemma4.tflite \
  --cases_file=$REMOTE/performance_1024_64.tsv \
  --output_dir=$REMOTE/prime-production \
  --weight_cache_path=$REMOTE/gemma4.xnnpack_cache \
  --num_threads=4 --cache_capacity=8448 --warmup_runs=0 --measured_runs=1" \
  > "$WORK/results/$PHONE_NAME-prime.log" 2>&1
```

Output directories must be new. Do not copy a packed-weight cache from the host
or from a different phone/build. Reuse this device-local cache for the timed
production processes. This matches the recorded warm-cache production policy.
LiteRT-LM's harness uses `:memory` for its packed-weight cache and reuses its
executor within a process; it does not persist that cache across processes.

### Confirm LM-128 actually invokes `prefill_128`

```bash
adb -s "$SERIAL" shell "taskset $MASK $REMOTE/litertlm.android \
  --model_path $REMOTE/gemma4-prefill128.litertlm \
  --cases_file $REMOTE/performance_1024_64.tsv \
  --output_dir $REMOTE/trace-lm128 \
  --num_threads 4 --max_num_tokens 8448 --reuse_runtimes true \
  --warmup_runs 0 --measured_runs 1 --dump_full_logits false \
  --enable_ynnpack false --enable_profiling false --trace_signatures true" \
  > "$WORK/results/$PHONE_NAME-trace-lm128.log" 2>&1

rg 'SIGNATURE|enable_ynnpack|magic-number' \
  "$WORK/results/$PHONE_NAME-trace-lm128.log"
```

Expect `enable_ynnpack=0`, a context rewrite `32003 -> 8448`, and
`SIGNATURE prefill_128` messages followed by decode. The 1,023-token prefix is
served by eight 128-row calls, with the last call partially filled. There
should be no invocation of `prefill_1024` or `verify`. The empty-input
`DecodeLogits` call flushes the pending last prompt token; do not manually feed
that token again.

Trace runs set `timings_valid_for_benchmark=false`. They verify scheduling, not
latency. The production model should delegate each signature into a single
XNNPACK partition; its fully-delegated hint rejects an unsupported graph rather
than silently benchmarking a partly delegated version. The recorded build
reported 1,815 prefill nodes and 3,830 decode nodes in their respective
partitions; node counts can change with graph/compiler changes.

## 8. Run the cooled paired comparison

Use the bundled [phone controller](run_phone.py):

```bash
python3 "$GUIDE/run_phone.py" \
  --serial "$SERIAL" --mask "$MASK" --remote "$REMOTE" \
  --lm-model gemma4-prefill128.litertlm \
  --output "$WORK/results/$PHONE_NAME-lm128"
```

The controller runs **production, LM-128, LM-128, production**, using new
processes and fresh result directories. Each process executes one warmup and
three measured sessions. It captures commands, exit status, process logs,
device information, timing files, and thermal readings.

Its starting-temperature gate matches the reference campaign:

- Two eligible live HAL samples, at least five seconds apart.
- CPU temperature at most 45 °C, skin at most 34 °C, and SoC at most 39 °C when
  exposed by the device.
- Exposed CPU cooling-device states must be zero.
- At least 15 seconds between benchmark processes.
- Thermal readings every five seconds during each process.

The controller stops if it cannot read the required CPU/skin sensors or cannot
reach the starting gate within ten minutes. If another phone reports sensors
differently, adapt the reader and document its equivalent gate. Do not silently
discard the gate. Keep charging, screen state, ambient conditions, and other
workloads comparable. Do not change thermal policy or force clock/governor
settings for these numbers.

The start gate does not keep a phone cool throughout a run. Inspect
`telemetry.json`, the per-session range, and the difference between the first
and second blocks. Zero exposed cooling states do not prove that DVFS or
throttling was absent. Repeat a thermally skewed comparison rather than
interpreting a small difference as a runtime improvement.

Repeat this section on the other phones with their own serial, affinity mask,
and result directory. Run only one benchmark workload at a time on each phone.

### Optional: measure the cost of original prefill selection

Copy the original bundle and run another paired experiment:

```bash
adb -s "$SERIAL" push -Z "$MODEL" "$REMOTE/gemma4-original.litertlm"

python3 "$GUIDE/run_phone.py" \
  --serial "$SERIAL" --mask "$MASK" --remote "$REMOTE" \
  --lm-model gemma4-original.litertlm \
  --output "$WORK/results/$PHONE_NAME-lm-original"
```

First run the section 7 trace command with the original model and a new trace
output path, to confirm which prefill signature it selects. Keep all other
settings unchanged. If the original bundle exceeds available phone memory,
record that failure and perform this control on a suitable device or desktop.
Reducing capacity to make it fit produces a different benchmark.

For a direct LM-original versus LM-128 claim, alternate their process order as
well and use the same thermal gate. Two separate production-paired experiments
alone do not fully remove drift between the two LM variants.

## 9. Summarize and check the results

```bash
export RUN="$WORK/results/$PHONE_NAME-lm128"
python3 "$GUIDE/summarize.py" \
  --production "$RUN/production-b0/capture" "$RUN/production-b1/capture" \
  --lm "$RUN/lm-b0/capture" "$RUN/lm-b1/capture" \
  --lm-label 'LiteRT-LM XNNPACK, prefill_128 (prefill_1024 removed)' \
  > "$RUN/summary.json"
```

The summary contains per-session measurements and these aggregates:

| Metric | Production source | LiteRT-LM source |
| --- | --- | --- |
| First-token latency | Step 0 `prefix_ms + token_ms` | First pass `elapsed_ms` |
| Decode ms/token | Mean of steps 1–64 `token_ms` | Mean of passes 1–64 `elapsed_ms` |
| Reported decode value | Median of the six session means | Same |
| Decode tokens/s | `1000 / reported_decode_ms` | Same |
| Complete request | First-token latency + sum of 64 decode calls, then median across sessions | Same |
| Peak RSS | Maximum `peak_rss_kib / 1024` across both processes | Maximum `peak_process_rss_kib / 1024` across both processes |

Use production **`token_ms`**, not `decode_ms`, for the inclusive comparison.
`token_ms` includes model invocation, output access, finite validation, and greedy
selection. LiteRT-LM's `elapsed_ms` includes analogous work plus creation of its
executor input tensor. Its `forward_ms` excludes that input construction and
the output check/selection. Comparing `decode_ms` with `elapsed_ms` would mix
timing scopes.

Production repeats `prefix_ms` on every row of a session; add it **once**.
Exclude repetition `-1` / `warmup_*.json`. Do not average the per-step RSS
high-water marks, and do not pool all token latencies and call that a median of
session means.

First-token latency here excludes process startup, model loading, delegate
creation, and the separate production cache-priming process. It is a warm
session measure, not launch-to-first-token application latency. The LM harness
includes public `Reset()` time in its first pass; production invalidates its KV
history logically. Those are real runner-policy differences and should be
reported when comparing first-token time. JSON/log-file writes are outside the
per-call timers.

The summarizer checks the expected context progression 1,024–1,088, the LM
fixture IDs, completed sessions, backend, thread/capacity settings, and absence
of tracing/profiling/logit dumps. It lists any argmax disagreement across all
65 predictions. Read the captured production command to verify its settings;
its timing file does not encode every CLI option.

Argmax agreement is only a smoke check. Differences between graph structures or
backends can occur even with the same inputs. Investigate them using a separate
single-session correctness run with production `--dump_outputs` and LM
`--dump_full_logits true`; compare corresponding full-vocabulary vectors and
use an appropriate numerical/quality criterion. Dump runs are excluded from
latency measurements. The prior production/native phone runs agreed on every
argmax; that does not establish bitwise equivalence with every LiteRT-LM
backend.

## 10. Interpret memory correctly

The reported RSS is a **process high-water mark** in MiB, where 1 MiB is
1,048,576 bytes. It includes setup, the warmup, all measured sessions, and pages
touched before a later release. Reusing an executor avoids paying setup inside
each timed request; it does not reset the process high-water mark.

The two paths also have different ownership and residency policies:

- Production exports in a separate process and uses a device-local file-backed
  packed-weight cache. It retains host embedding tables and runtime state,
  without retaining the export graph builder and all its conversion buffers.
- The LM harness keeps the normal `.litertlm` model resources and uses the
  in-memory packed-weight-cache policy. Its original graph has different
  prefill/decode activation requirements and KV handling.
- XNNPACK's Linux/Android mapped-data path can use
  `madvise(..., MADV_PAGEOUT)` for no-longer-needed mapped pages. This can reduce
  current resident memory without freeing a tensor allocation, unmapping the
  model, or reducing its virtual size. It is advisory, and a later access can
  fault pages back in. It does not erase an earlier RSS high-water mark.

Consequently, a smaller RSS number is useful evidence of lower process
residency for this execution policy, but it is not by itself a measurement of
shared scratch or arena savings. Separate model mappings, packed weights,
anonymous activations, and KV storage before assigning a cause. Do not use file
size or `/proc/PID/status` `VmSize` as a proxy for resident memory.

For a deeper audit, use a separate diagnostic run. The LM harness can stop at
decode boundaries with `--pause_at_decode_boundaries true --measured_runs 1`.
It prints its PID and stops itself immediately before and after the forced
decode calls. Where Android permissions allow, capture `/proc/PID/status`,
`smaps_rollup`, and `smaps`, then send `SIGCONT` to resume at each stop. Such a
run is explicitly marked unsuitable for timing. Do not compare a final
post-pageout RSS snapshot from one path with a peak from the other.

Keep the production cache-priming policy consistent. A first-ever process that
packs and writes weights is a distinct cold-start experiment, even if its
per-request timers exclude setup.

## 11. Previously recorded results: reference points, not a new three-way run

The latest three-phone campaign compared **production LiteRT versus the native
XNNPACK runner**. It did not include a fresh LM-128 timing measurement on all
three phones. Use the procedure above to fill in those cells; do not infer them
from the native result.

The recorded workload used the settings in section 1 and the order
production/native/native/production. Decode values below are inclusive
milliseconds per token, reported as the median of six session means for the
fresh runs:

| Phone | Earlier native record | Fresh native | Production LiteRT | Production vs earlier record |
| --- | ---: | ---: | ---: | ---: |
| TECNO LJ9 | 40.91 | 40.65 | 42.84 | +4.7% |
| Pixel 8 | 48.92 | 49.00 | 55.62 | +13.7% |
| Samsung SM-S937U1 | 24.07 | 24.00 | 24.13 | +0.2% |

| Phone | Fresh native first token, ms | Production first token, ms | Fresh native peak RSS, MiB | Production peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: |
| TECNO LJ9 | 3,359.5 | 3,246.8 | 2,065.3 | 1,168.8 |
| Pixel 8 | 4,861.7 | 3,290.1 | 2,058.0 | 1,148.0 |
| Samsung SM-S937U1 | 1,778.1 | 1,661.1 | 2,052.7 | 1,134.6 |

All 65 argmax outputs per session matched the historical native outputs.
The [saved result summary](historical-production-native.json) includes session
ranges and block medians. The phones heated during the runs: maximum sampled
CPU temperatures across the paired captures reached about 83.6 °C on TECNO,
98.0 °C on Pixel, and 61.4 °C on Samsung. No exposed CPU cooling state was
observed as active, but that is not proof of constant frequency. Treat the
numbers as reference measurements with thermal variation, not exact targets.

### Earlier LiteRT-LM evidence

An earlier TECNO capacity sweep used the original LiteRT-LM bundle, four CPU
threads, and the same 1,024+64 token workload. It produced these decode medians:

| KV capacity | Native XNNPACK | LiteRT-LM XNNPACK | LiteRT-LM YNNPACK |
| ---: | ---: | ---: | ---: |
| 2,048 | 41.421 ms | 50.903 ms | 49.031 ms |
| 8,448 | 41.042 ms | 75.223 ms | 51.125 ms |

This is evidence that the original XNNPACK-backed LM graph's decode cost grew
with configured capacity. It is a separate experiment, not a fresh comparison
against today's production binary. The YNNPACK column is a different backend;
keep it separate from the primary XNNPACK comparison.

An earlier **Linux desktop** memory control at capacity 8,448 selected the
existing `prefill_128` by the rename method in section 6:

| LiteRT-LM XNNPACK prefill selection | Peak RSS | Final-decode resident memory |
| --- | ---: | ---: |
| Original 1,024-row selection | 5,728.9 MiB | 5,417.6 MiB |
| 128-row selection | 4,137.8 MiB | 4,103.0 MiB |

That is a measured **1,591.1 MiB peak-RSS reduction** on that desktop. It is not
a phone memory target or a measured 128-row latency speedup. It includes
residency effects as well as graph allocation differences. A separate Pixel
audit using YNNPACK and an earlier SDPA graph rewrite reduced the prefill arena
from 687.252 to 85.907 MiB, but that changed backend/graph configuration is not
the original XNNPACK baseline above.

## 12. Troubleshooting and reporting

| Symptom | Check |
| --- | --- |
| Bazel reports no `arm64-v8a` toolchain | Supply `ANDROID_SDK_HOME` and `ANDROID_NDK_HOME` through `--repo_env`, not only shell `ANDROID_HOME`. |
| Runner rejects CLI flags | Use the production target with retained Abseil flag names; do not substitute an older experimental Android binary. |
| Bundle preparation rejects the model | Check the original artifact's size/checksum; do not prepare production weights from the modified LM-128 bundle. |
| LM trace still invokes `prefill_1024` | Check the model path/hash and exposed signatures; use a fresh output path and a separate modified-model cache location if your runner persists caches. |
| Unexpected LM speed or memory | Confirm CPU/XNNPACK, four threads, capacity 8,448, correct affinity, runtime reuse, no profiling/dumps, and actual prefill signature. |
| Delegate/fallback behavior differs | Inspect process logs and the production full-delegation requirement; record partitioning changes. |
| Process exits without results | Inspect the exit code/log and available device memory/storage; do not report a partial session as a completed benchmark. |
| Controller times out waiting to cool | Check live HAL readings and ambient/charging conditions; do not weaken the gate without documenting the changed experiment. |
| Small speed difference changes sign | Compare per-session ranges, reversed blocks, temperatures, and repeat the pair. |

For each phone, retain the fixture and model identity, exact commands, backend,
thread count, CPU mask, capacity, signature-selection method/trace, cache policy,
warmup/repetition counts, raw timing files, and thermal logs. Report decode,
first-token latency, complete-request latency, and peak RSS together. State
whether the correctness check was argmax-only or a separate full-logit check.
Source version control is left to the developer.

### Checks performed for this guide

The supplied Android LM harness was built against the production LiteRT
checkout. Both signature-edit modes were checked against the original bundle,
including complete-file verification of unchanged bytes and the output hashes
listed above. A TECNO LJ9 smoke run of the literal-removal copy invoked
`prefill_128` eight times and `decode` 65 times; all consumed contexts were
verified, logits were finite, and all 65 argmax predictions matched the saved
production run. This traced smoke run supplies no new latency result.

The thermal parser was checked against all three connected reference phones.
The production side of the summarizer reproduced every saved headline metric
for all three phones. The guide's Bash snippets passed syntax checks. A new
full production-versus-LM phone timing campaign was not run while writing this
guide; use sections 8–9 to perform it.
