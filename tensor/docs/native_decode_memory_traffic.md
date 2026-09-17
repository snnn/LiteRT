<!-- Copyright 2026 Google LLC.

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

# Native runner decode memory traffic

This workflow adds the [native Gemma4 E2B runner](../examples/gemma4/native/README.md)
to the converter's existing decode-window memory-controller measurement tool.
It measures DRAM reads and writes separately, then divides by the number of
decode calls inside the measured intervals. The existing LiteRT-LM path uses
the same collector. See also the [KV cache layout reference](kv_cache_layout.md).

The companion converter checkout supplies
[`measure_decode_bandwidth.py`](../../../litert-converter/tools/measure_decode_bandwidth.py)
and [`imc_freerun_probe.cc`](../../../litert-converter/tools/imc_freerun_probe.cc).
Its [bytes-per-token summary](../../../litert-converter/docs/LLM_DECODE_BYTES_PER_TOKEN_SUMMARY.md)
contains earlier measurements. The
[measurement plan](../../../litert-converter/docs/LLM_DECODE_BANDWIDTH_MEASUREMENT_PLAN.md)
records the original design; its early statement that LiteRT-LM had no direct
IMC measurement is historical.

## Measured results: September 16, 2026

All three configurations were measured: native XNNPACK, LiteRT-LM CPU with
YNN disabled, and LiteRT-LM CPU with YNN enabled. Native used fewer total DRAM
bytes per decode call, but that reduction explains only part of its throughput
advantage. Increasing unused cache capacity substantially hurt the XNN-only
LiteRT-LM configuration; YNN avoided most of its extra traffic.

The follow-up [mobile decode investigation](native_mobile_decode_analysis.md)
reproduces the larger phone speedup and explains why these DRAM totals alone
do not account for it. It includes matched capacity sweeps, decode-only
operator/CPU profiles, and controls for attention layout and compiler flags.

These are Intel i9-12900K results with four threads pinned to four distinct
P-cores (`0,2,4,6`), using the existing `powersave` governor. Each cell is the
median of three independent processes, each with one excluded warmup session
and 64 measured forced decode calls. MB and GB are decimal. Calls/s uses the
controller's gated interval, including decode bookkeeping. This is a comparison
of the recorded runtime configurations, not an isolated graph-authoring
ablation: their XNNPACK dependency revisions differ, as recorded below.

### Live context at fixed capacity

Cache capacity is 4,096 in every row. Prompt length is the starting history;
64 further inputs are consumed during measurement.

| Prompt tokens | Runtime | Read MB/call | Write MB/call | Total MB/call | Calls/s | DRAM GB/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | Native XNNPACK | 790.51 | 2.63 | 793.14 | 41.86 | 33.17 |
| 128 | LiteRT-LM + XNNPACK | 825.36 | 7.56 | 832.93 | 32.02 | 26.66 |
| 128 | LiteRT-LM + YNNPACK | 804.15 | 2.15 | 806.31 | 37.43 | 30.24 |
| 1,024 | Native XNNPACK | 801.56 | 4.45 | 806.01 | 39.53 | 31.87 |
| 1,024 | LiteRT-LM + XNNPACK | 826.62 | 7.55 | 834.42 | 31.95 | 26.65 |
| 1,024 | LiteRT-LM + YNNPACK | 826.52 | 2.63 | 829.15 | 35.56 | 29.40 |
| 3,072 | Native XNNPACK | 814.33 | 5.25 | 819.56 | 37.00 | 30.33 |
| 3,072 | LiteRT-LM + XNNPACK | 827.30 | 7.50 | 834.85 | 31.91 | 26.65 |
| 3,072 | LiteRT-LM + YNNPACK | 848.00 | 3.11 | 851.11 | 32.34 | 27.55 |

Columns are independently median-aggregated, so displayed read and write
medians need not sum exactly to the displayed total median.

At a 1,024-token prompt, native moves **3.4% fewer bytes than LM-XNNPACK**
while delivering **23.7% more calls/s**. Relative to LM-YNNPACK, it moves
**2.8% fewer bytes** and delivers **11.2% more calls/s**. Fewer DRAM bytes alone
therefore do not explain the speed difference. The higher achieved GB/s is
derived from bytes and elapsed time; it is not independent evidence that a
specific kernel, cache policy or scheduling choice caused the improvement.

Native and YNN traffic grow with live context, whereas XNN-only traffic is
nearly flat at fixed capacity. At the deepest context, YNN moves more bytes
than XNN but remains slightly faster. YNN also writes fewer bytes than native
in these captures. Thus native's advantage is not a universal reduction in
every category of traffic.

### Capacity at fixed live context

The prompt and all 64 forced inputs are identical across capacities. An
8,192-token capacity here is **not an 8,192-token prompt**.

| Runtime | Capacity | Read MB/call | Write MB/call | Total MB/call | Calls/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native XNNPACK | 2,048 | 800.42 | 4.30 | 804.72 | 39.66 |
| Native XNNPACK | 4,096 | 801.56 | 4.45 | 806.01 | 39.53 |
| Native XNNPACK | 8,192 | 800.82 | 4.65 | 805.47 | 39.59 |
| LiteRT-LM + XNNPACK | 2,048 | 814.51 | 8.42 | 822.71 | 34.99 |
| LiteRT-LM + XNNPACK | 4,096 | 826.62 | 7.55 | 834.42 | 31.95 |
| LiteRT-LM + XNNPACK | 8,192 | 868.39 | 17.40 | 886.03 | 26.89 |
| LiteRT-LM + YNNPACK | 2,048 | 818.54 | 2.62 | 821.16 | 36.45 |
| LiteRT-LM + YNNPACK | 4,096 | 826.52 | 2.63 | 829.15 | 35.56 |
| LiteRT-LM + YNNPACK | 8,192 | 824.00 | 2.74 | 826.74 | 34.95 |

From capacity 2,048 to 8,192, native total traffic changes by +0.09% and
throughput by -0.16%, both smaller than the largest within-cell ranges.
LM-XNNPACK traffic rises **7.7%** and throughput falls **23.1%**. LM-YNNPACK
traffic rises **0.7%** and throughput falls **4.1%**; its traffic is not
monotonic across all three capacities. YNN largely avoids the measured
capacity-dependent DRAM growth, but these results do not establish complete
capacity independence or identify its remaining timing cost.

Separate audits of the actual LiteRT-LM cache tensors verified 30 INT8
buffers at every capacity, for both backends: 18,874,368 bytes at 2,048,
37,748,736 at 4,096, and 75,497,472 at 8,192. The native payload has the same
capacity scaling. The capacity control therefore changes real cache storage,
not merely an ignored command-line limit. The combination of this control
and the context sweep supports a capacity-sensitive execution cost in the
XNN-only configuration. Whole-model counters do not isolate how much of that
cost belongs to attention, KV conversion/copies, scratch, or cache effects.

### Measurement quality and limits

All **45 captures** passed the marker, invocation-count, history and completion
checks. Each has exactly one measured window and 64 decode calls; warmup and
the prompt's first prediction are excluded. Across the 15 cells, the largest
three-capture range divided by its median is 0.74% for total bytes/call and
0.97% for calls/s. These are observed ranges, not confidence intervals.
The largest runner-versus-collector interval discrepancy is 0.0077%.

Three-second idle captures measured 0.180 and 0.190 GB/s before the experiment
and 0.021 GB/s after it. They establish that background traffic was variable,
not absent; they cannot bound every background burst during a run. No idle
subtraction was applied. In particular, small write-traffic differences are
more vulnerable to background activity than the read-dominated totals.

The PMUs were `uncore_imc_free_running_0` and `_1`, with `data_read` and
`data_write` scaled by their sysfs metadata to 64 bytes per count. They measure
controller-wide traffic, not private per-process bytes. Builds, profiling,
large source/model reads and other benchmark captures were kept outside the
measurement runs. Runtime order was alternated between repetitions. The
capacity controls followed the main depth sweep, so the experiment was not
fully randomized across every capacity.

These measurements do not establish mobile bandwidth, exact per-operator
traffic, or numerical parity between backends. Identical forced histories
control the workload; they do not prove identical logits.

## What the measurement means

Keep three quantities separate:

| Quantity | How it is obtained | Interpretation |
| --- | --- | --- |
| Allocated bytes | Tensor storage, allocator counters, RSS/PSS | Memory footprint; not traffic per token. |
| Logical byte estimate | Weight tensors, active KV views, append sizes, graph operations | An accounting model whose caching and reuse assumptions must be stated. |
| Measured DRAM bytes | Memory-controller read/write counters during decode | Traffic through the monitored controllers, including other activity on those controllers. |

For each capture:

```text
read bytes/call  = gated read bytes / measured decode calls
write bytes/call = gated write bytes / measured decode calls
total bytes/call = read bytes/call + write bytes/call
DRAM GB/s       = gated total bytes / gated seconds / 1e9
```

The native benchmark forces a fixed input history. Here, a decode call consumes
one forced input token and produces one complete logits vector. It is the unit
usually called bytes/token, although the next input is not sampled from that
prediction. The first prediction produced from the prompt is excluded.

Measured DRAM bytes can vary with hardware caches, packing, thread count and
backend implementation. A graph byte estimate is not a hardware-independent
prediction of actual DRAM traffic. Nor does whole-decode traffic identify which
operator caused it: weight reads, attention, scratch and host-side work are
all included. Isolating KV traffic requires a controlled comparison or a
narrower measurement interval.

## Native interval and output contract

Setting `LITERT_NATIVE_TRACE_FIFO` enables the marker channel. With the variable
unset, the runner executes normally. The collector supplies a FIFO and receives:

```text
section_begin decode
section_end decode
```

Each measured session emits one pair around all its forced decode calls. The
interval includes input preparation, cache updates, graph execution, logits
reading and finite/argmax checks. It excludes model loading, graph creation,
prefix prefill, the prompt's first prediction, warmup sessions and writing the
result files. The runner rejects diagnostic dumps, memory snapshots and stage
tracing when FIFO capture is enabled. Every case must contain forced inputs.

Per-session JSON records `decode_trace_enabled`, `decode_trace_section`,
`measured_decode_calls` and `decode_trace_elapsed_ms`. Warmup sessions record
zero measured calls. The wrapper checks these fields against the decode pass
records, the requested repetitions, the completed run summary and the number
of completed collector windows. Failed or incomplete captures must not produce
a usable `result_bundle.json`.

FIFO markers are asynchronous: the collector responds when it reads them,
without an acknowledgement to the runner. The runner's recorded interval and
the collector's enabled duration provide a timing cross-check, not proof of
cycle-exact boundaries. Use a multi-second decode interval and inspect their
difference. This protocol is unsuitable for measuring individual microkernels.

## Linux collection

The current collector supports Intel `uncore_imc_free_running_*` PMUs exposed
by Linux sysfs. It reads event encodings and scales from that interface. It
counts all discovered controllers system-wide, not only the benchmark PID.
Run on a quiet host, serialize the runtime comparisons, and measure background
traffic before and after them. Save raw totals; do not silently subtract idle
traffic or assume pinning the benchmark isolates its memory traffic.

Build the native runner using the
[standalone or Bazel instructions](../standalone/README.md), and prepare a
verified matched-bundle export. From the LiteRT root, with sibling checkouts:

```bash
LITERT_ROOT="$PWD"
CONVERTER_ROOT="$(cd ../litert-converter && pwd)"
NATIVE_RUNNER="$LITERT_ROOT/.native-tensor-build/host/bin/gemma4_native_runner"
MATCHED_BUNDLE=/path/to/verified/matched-bundle
CAPTURE=/path/to/new/capture
COLLECTOR=/path/to/build/imc_freerun_probe

c++ -std=c++17 -O2 "$CONVERTER_ROOT/tools/imc_freerun_probe.cc" -o "$COLLECTOR"
uv run --no-project --python 3.13 \
  "$CONVERTER_ROOT/tools/measure_decode_bandwidth.py" \
  --runtime litert_native --collector "$COLLECTOR" \
  --sudo-collector --output-dir "$CAPTURE" -- \
  taskset -c 0,2,4,6 "$NATIVE_RUNNER" \
  --bundle_dir="$MATCHED_BUNDLE" \
  --cases_file="$LITERT_ROOT/tensor/examples/gemma4/native/fixtures/performance_1024_64.tsv" \
  --num_threads=4 --cache_capacity=4096 --prefill_chunk_rows=128 \
  --kv_alignment=32 --preserve_static_int2=true --share_workspace=true \
  --reuse_runtimes=true --warmup_runs=1 --measured_runs=3 \
  --memory_report=false --fixed_attention_extent=false \
  --dump_full_logits=false --dump_cache=false --trace_position=-1
```

Choose CPU affinity for the target machine; `0,2,4,6` is a local example, not
a portable choice. The wrapper supplies a new native output directory if
`--output_dir` is omitted. The collector needs permission to read uncore PMUs;
`--sudo-collector` also runs its benchmark child with elevated privileges.
Use it only in the intended local measurement environment.

Inspect `result_bundle.json` together with `imc.json`, the collector log and
native session JSON. Preserve the executable hashes, source revisions and local
changes, model provenance, fixed histories, runtime flags, affinity, background
measurements and every repetition. Avoid treating one aggregate as a variance
estimate. The wrapper's optional `--predicted-bytes-per-token` accepts an
independently justified estimate; it does not derive one from allocated memory.

## Comparing with LiteRT-LM and interpreting KV improvements

Use the wrapper's `litert_lm` preset with a binary that emits
`LITERT_LM_TRACE_FIFO` decode markers and writes benchmark metrics. Match the
published `.litertlm` bundle to the native export, cache dtype, capacity,
starting context, number of decode model invocations, CPU threads and affinity.
Record any unavoidable difference in token histories, sampling or boundaries.
Do not assume a requested number of generated tokens equals the number of
decode invocations: a prefill graph may produce the first prediction.

For identical forced histories, the local LiteRT-LM comparison adapter also
supports `LITERT_NATIVE_TRACE_FIFO` and the native per-session JSON contract.
Use `--runtime litert_lm_fixed_tokens` with that instrumented adapter. Its
options are `--model_path`, `--cases_file`, `--num_threads`,
`--max_num_tokens`, `--warmup_runs`, `--measured_runs`, `--reuse_runtimes`,
`--dump_full_logits=false`, `--enable_profiling=false`, and
`--enable_ynnpack=false` or `true`. These are adapter options, not flags for
the standard LiteRT-LM command-line binary. The wrapper supplies its output
directory and uses the same forced-call denominator as the native runner.

Build LiteRT-LM with `--define=litert_enable_ynnpack=true` for the enabled
comparison. LiteRT registers YNN before XNN: enabling it selects YNN first,
with XNN available for remaining supported nodes. This is a YNN-enabled CPU
configuration, not a promise that every node runs in YNN.

Verify actual execution separately, with the FIFO environment variable unset
and `--enable_profiling=true --warmup_runs=0 --measured_runs=1` on a short
fixture. Inspect the adapter's `.profile.txt` output for executed
`YNNPackDelegate` rows in decode. With YNN disabled, XNNPACK profiling can
expand the outer delegate into rows such as `Delegate/Fully Connected` and
`Delegate/Sine`, rather than printing a `TfLiteXNNPackDelegate` row.
Accelerator registration and the requested flag alone
do not prove delegate use. An "enable_ynnpack was ignored" warning invalidates
the enabled comparison. Disable profiling again for IMC captures.

A useful experiment varies prompt depth while keeping capacity fixed, then
varies capacity while keeping the same live history. This separates growth
with valid length from growth with allocation size. Compare compact INT2 and
widened weight paths separately if investigating coefficient traffic.

The native bank has 15 unique owners with 30 INT8 K/V buffers. Its payload is
`capacity * 9,216` bytes, and one committed token appends 9,216 payload bytes
across those owners. These are structural quantities. Consumer layers can
read shared owners repeatedly, and cache lines, buffer adapters and packing
can add traffic. Neither number is the measured DRAM cost of attention.

The `fixed_attention_extent` diagnostic also materializes fixed-width K/V
buffers. Comparing it with active views therefore measures that entire path
change, including copies; it does not isolate loop bounds alone. It requires
`kv_alignment=1`, so compare against an active-view control with the same
alignment, separately from the default alignment-32 configuration.

## Android

The marker mechanism is POSIX-compatible and can be tested on Android. The
Intel IMC collector cannot run unchanged against a phone's Arm PMUs. A phone
measurement requires documented controller/interconnect events, permission to
read them, event-to-byte scaling, and a decode-window collector for that PMU.
Generic CPU cache misses are not a substitute for DRAM read/write bytes.

Record event availability and permission failures explicitly. Mobile timing
results remain useful, but multiplying them by a host byte estimate does not
turn them into measured mobile bandwidth.

On September 16, the connected TECNO LJ9 and Pixel 8 allowed a userspace CPU
cycle event, but their shell could not read the exposed Arm uncore event
metadata. `simpleperf` did not expose usable controller events. Both phones
also rejected `mkfifo` with `EACCES` in the tested `/data/local/tmp` and
`/sdcard/Download` locations. The Android runner and marker test compiled,
but the on-device marker tests failed at FIFO creation. No mobile DRAM
measurement was obtained, and no root or SELinux configuration was changed.

## Recorded configuration and reproduction artifacts

The local artifact directory is
[`/data/bt/tmp/native-decode-bandwidth-20260916/`](/data/bt/tmp/native-decode-bandwidth-20260916/).
It contains raw captures, fixture histories, every launch command, summaries,
build/test logs, separate delegate profiles and cache audits. The
[capture harness](/data/bt/tmp/native-decode-bandwidth-20260916/run_host_sweep.py)
records a fresh process per capture and refuses to overwrite existing output.
The [summary script](/data/bt/tmp/native-decode-bandwidth-20260916/summarize.py)
checks histories and recomputes both tables from the raw result bundles.

| Item | Recorded configuration |
| --- | --- |
| LiteRT source | `345852646f25bc3e8d3935fa589243bad8bc5288` plus local native-runner changes |
| LiteRT-LM source | `2d044a3376ebd063752661d8da2704ec5a77c32f` plus local adapter/configuration changes |
| Native XNNPACK | `bf3ee43b63070284f85a4a882f298b6ca5273f01` |
| LM XNNPACK/YNNPACK | `d89ef6669a14db203b3b7935b1b3862cb63fb6df` plus the compiler fix below |
| Native build | Bazel `-c opt`, Clang/LLD 18.1.8, `--cxxopt=-ffp-contract=off`, explicit XNNPACK source override |
| LM adapter build | Bazel `-c opt`, GCC 16, C++20, current local LiteRT source override, YNN enabled at build time |
| Native execution | Compact INT2 preserved, shared workspace, reused runtimes, 128-row prefill chunks, active attention views, KV alignment 32 |
| LM execution | Same binary for both backend settings; profiling disabled during captures; runtime reuse enabled |
| Cache | INT8, matched capacity and forced history; layout/execution remain backend-specific |
| Diagnostics during captures | No full-logit/cache dumps, memory snapshots or stage/operator profiling |

The two builds differ in compiler and dependency revision. These numbers
compare the available validated implementations and cannot attribute the
entire speed gap to graph structure. No dependency revision or compiler
ablation was performed here.

Both paths used the **same published bundle identity**, from
`litert-community/gemma-4-E2B-it-litert-lm` revision
`616f4124e6ff216292f16e7f73ff33b5ba9a4dd4`. LiteRT-LM read its `.litertlm`
file; native read the previously verified export of that file. This experiment
used the older matched fixture supported by this checkout, not the newer
bundle retested in the separate migration worktree, and not Google's original
safetensors. The fixture choice does not change the model pin in that worktree.

```text
.litertlm SHA256:
ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42
Native export manifest SHA256:
3cdd41d803c05a4b29729a9892451122987675867ae1841709b2b480ddd826bf
Measured native executable SHA256:
0bf4e7ac201908b12af2070b63f5dce4553fdcac7e8b77a966d103c86eb59b55
Measured LM adapter executable SHA256:
beb1a79e23e37abc35a867b24de8d5003c25f96f72dddc790bdd1991e64731d5
Measured collector executable SHA256:
72e01ca44a8cf09f96707a64a92515c3edb40f36b51ba18eb14b1576059b30fe
```

YNN initially failed to build because GCC 16 requires `_MM_PERM_ENUM` for
`_mm512_shuffle_epi32`. The saved
[compiler patch](/data/bt/tmp/native-decode-bandwidth-20260916/validation/ynnpack-gcc16-shuffle-enum.patch)
adds explicit casts to eight immediates in four kernel generators without
changing their values. After rebuilding with YNN enabled, separate profiles
confirmed executed `YNNPackDelegate` decode rows, and no ignored-option
warning. Unsupported float-to-integer operations still reported fallback.
The source/build settings and the actual profiles are retained; successful
registration alone was not accepted as evidence of YNN execution.

The [LM build command](/data/bt/tmp/native-decode-bandwidth-20260916/validation/litert-lm-host-build-command.txt)
and `validation/litert-lm-bandwidth-{adapter.cc,decode_trace.h,BUILD}` preserve
the local comparison adapter. It is not a standard upstream LiteRT-LM target.
The [source snapshot manifest](/data/bt/tmp/native-decode-bandwidth-20260916/source-snapshot/manifest.json)
records saved local source, patches and executable copies. Adapt the absolute
paths in the harness when reproducing elsewhere, retain the model identity,
and select a fresh output root.

Executed validation included native Bazel build/marker tests, 12 real-driver
guard cases, five LM adapter guards, native and both LM backend model smokes,
actual cache-capacity audits, and 15 converter collector/parser tests. The
collector compiled with `-Wall -Wextra -Werror`. The Android cross-build passed;
the device restrictions described above prevented its FIFO tests from passing.
Smoke tests checked execution and instrumentation, not full-model quality.

A final review also fixed process-mode JSON reporting a successful child
exit when the child failed. Its regression tests cover exit 0, exit 7 and
SIGTERM. This affects whole-process collection, not the FIFO-gated captures
reported here; the measured collector binary and its source are preserved
separately from the final fixed binary.
