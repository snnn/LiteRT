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

# Gemma4 native runner: decode MLP and fully connected profiling

On the audited Linux desktop, **16.42 ms of the 17.07 ms post-attention
stage time is in fully connected operators**. The three MLP projections account
for **13.86 ms**. Quantization conversions, GELU, and work outside XNN invocation
are much smaller. CPU sampling confirms AVX-VNNI kernels for the large FCs.

This completes the profiling follow-up in
[the native runner improvement report](gemma4_native_runner_improvements.md).
These are diagnostic measurements, not an implemented speedup. Production
runner sources and defaults remain unchanged.

## Configuration and measurement

- Gemma4 E2B matched bundle; prompt 1,024, 64 forced decode inputs, capacity
  8,448, 128-row prefill, KV alignment 32, shared workspace, compact INT2
  weights, reused runtimes.
- Intel Core i9-12900K; four threads restricted to CPUs `0,2,4,6`, the same
  four P cores as the preceding comparison. This is an x86 result, not an
  Android kernel profile.
- One warmup followed by three measured sessions: **192 decode calls per
  operator**, excluding prefill, the first-logit call, and warmup.
- An isolated driver enables `XNN_FLAG_BASIC_PROFILING` only on the 35 decode
  post-attention runtimes. It maps FC source pointers to exact weight names
  and records the pinned XNNPACK runtime's nanosecond timestamps. The public
  profiling API rounds to whole microseconds, which would distort these small
  elementwise operations.
- The driver is relinked against the same archived libraries as the baseline.
  All **92 archive hashes and 172 Tensor source hashes** match the preceding
  experiment's inputs. No dependency or kernel is rebuilt.

The unprofiled baseline's median session mean is **25.171 ms/decode**, with
**16.977 ms** in post stages. The diagnostic executable with profiling disabled
gives **25.222 ms** and **17.038 ms**. With profiling enabled, those medians are
25.308 ms and 17.041 ms. The additive breakdown below uses arithmetic means
over all 192 profiled tokens: 25.360 ms for the whole step and 17.074 ms for
the post stages. No instrumentation overhead is subtracted.

Operator intervals include XNN dispatch and threadpool synchronization, as
well as kernel execution and timestamp overhead. They are **not pure
microkernel compute times**. Sample collection occurs after each stage wall
timer ends; it is included in the diagnostic whole-step timer. Profiling and
logit-dump runs are marked ineligible for ordinary timing benchmarks.

## Where the post-attention time goes

There are 49 invoked XNN operators per post stage, or 1,715 across 35 layers.
Each stage contains six FCs: attention output projection, MLP up/gate/down,
and two per-layer embedding projections. Thus the earlier approximately
17 ms number was broader than the MLP alone.

| Work across all 35 layers | Operators per decode | Mean ms/decode |
| --- | ---: | ---: |
| MLP up projection | 35 | 4.700 |
| MLP gate projection | 35 | 4.666 |
| MLP down projection | 35 | 4.497 |
| Attention output projection | 35 | 1.791 |
| Per-layer embedding gate and projection | 70 | 0.765 |
| FP32 to INT8 conversion | 210 | 0.076 |
| INT8 to FP32 conversion | 210 | 0.091 |
| Approximate GELU, including per-layer embedding gates | 70 | 0.084 |
| Other elementwise, normalization, copy/reshape and transpose work | 1,015 | 0.231 |
| Stage work outside `xnn_invoke_runtime` | — | 0.173 |
| **Total post-attention stages** | **1,715** | **17.074** |

The 210 FCs account for **96.2% of post-stage time**. The 105 MLP FCs account
for 81.2% of post-stage time and approximately 55% of the whole decode step.
Within the other-operator row, multiply/add account for 0.189 ms, mean-square
reductions and reciprocal square roots for 0.029 ms, and copy/reshape/transpose
for 0.012 ms. Rounding can make displayed rows differ slightly from totals.

The stage wrapper interval includes input binding, reshape, output handling,
and setup around invocation. It does not include all host work elsewhere in
the decode step. Threadpool overhead *inside* each FC remains in that FC's
operator interval; it is not part of the 0.173 ms wrapper figure.

## INT2 versus INT4 MLPs

All these FCs use static INT8 input and output quantization. The earlier
15 layers use INT4 MLP weights; the later 20 use INT2 MLP weights and twice
the intermediate width.

| Layer group, zero based | Up/gate weight shape, output × input | Down weight shape | MLP coefficient payload | Three MLP FCs, ms/decode | Mean per layer, ms |
| --- | --- | --- | ---: | ---: | ---: |
| 0–14: INT4 | 6,144 × 1,536 | 1,536 × 6,144 | 202.5 MiB | 5.817 | 0.388 |
| 15–34: INT2 | 12,288 × 1,536 | 1,536 × 12,288 | 270 MiB | 8.046 | 0.402 |

Every MLP matrix has **4.5 MiB of source coefficients**, so each layer has
13.5 MiB across its three matrices. The INT2 group performs twice as many
dot-product terms per layer, but its measured FC time per layer is only about
3.7% greater. This is consistent with weight delivery being important; it
does **not establish that DRAM bandwidth is the bottleneck**. Kernel tiling,
unpacking, cache behavior, and thread scheduling also differ.

The post stages' source coefficient payload totals **561.75 MiB**: 472.5 MiB
MLP, 63 MiB attention output projections, and 26.25 MiB per-layer embedding
projections. These are logical coefficient bytes, not measured memory traffic.
Decode reads XNNPACK's packed weights, whose layout and metadata differ from
these source buffers. Releasing packing sources reduces memory residency;
it does not remove this per-token FC work or the packed coefficients.

## Kernel and threadpool CPU samples

A separate run samples the **unchanged baseline executable**, including its
worker threads, using `perf record -e cpu_core/cycles/u -F 499`. The existing
decode FIFO markers enable and disable sampling only around the three measured
forced-decode sessions. Startup, packing, warmup, and prefill are excluded.

The controller is asynchronous. Marker receipt to perf acknowledgement takes
2.56–4.91 ms in this capture. For attribution, the analyzer retains samples
after each enable acknowledgement and before 10 ms ahead of each end-marker
receipt. This leaves **9,523 samples over 4.832 seconds**; 81 boundary samples
are excluded and perf reports zero lost samples. The leading kernel shares
are nearly unchanged without this trimming.

The following are **period-weighted user CPU-cycle shares across the whole
decode**, not post-stage wall-time percentages. QC4 kernels also execute in
Q/K/V projections, and the head is outside the post-stage profile.

| Sampled symbol or kernel family | CPU-cycle share |
| --- | ---: |
| Static INT8 × INT4, `1x8c8__avxvnni_prfm` | 36.33% |
| Static INT8 × INT2, `1x8c8__avxvnni` | 31.37% |
| Dynamically quantized INT2 with FP32 output, head path | 11.55% |
| pthreadpool `thread_main` | 6.61% |
| Static INT8 × INT8, `1x8c8__avxvnni_prfm` | 4.26% |
| FP32 × INT8, `5x16__avx2_broadcast` | 2.35% |
| Two dynamic INT8 packing kernels combined | 2.57% |
| `pthreadpool_parallelize` | 0.99% |

Annotated `thread_main` samples concentrate in its polling loop. Waiting
workers can overlap other threads' useful computation, so 6.61% CPU cycles
does not mean that removing polling would save 6.61% of latency.

A separate decode-gated `perf stat` capture records 94.34 billion user cycles
and 76.08 billion instructions, or 0.806 instructions/cycle, with both events
scheduled 100% of their counting intervals. These counters alone cannot
distinguish memory stalls, unpacking cost, or spin waiting; no DRAM-byte or
cache-miss measurement is claimed here.

## Implications for the next optimization

1. **Prioritize the M=1 QC2/QC4 FC path.** Benchmark the measured dimensions
   with streamed packed weights, then separate memory/cache behavior,
   low-bit unpacking and dot products, tiling, and thread scheduling. The
   large FCs already use vectorized AVX-VNNI kernels on this desktop.
2. **Use FC-level experiments to evaluate scheduling or gate/up fusion.**
   Fusion does not eliminate either matrix's coefficients or arithmetic.
   Preserve each projection's static input/output scales and rounding;
   dropping INT8 conversions changes the checkpoint's numerical behavior.
3. **Treat elementwise fusion as a smaller opportunity at this shape.**
   Even eliminating every measured FP32/INT8 conversion and GELU interval
   would remove only about 0.252 ms, approximately 1% of decode latency.
   This is an upper bound on those intervals, not a forecast for a fused op.
4. **Keep attention/KV packing as a separate target.** The sampled dynamic
   packing is execution work in attention, not evidence that the static MLP
   matrices are being repacked every token. It does not explain the 13.86 ms
   spent in MLP FCs.

For scale, a hypothetical **20% reduction in MLP FC time** would save about
2.77 ms, approximately 11% of the current whole step if other costs stayed
constant. No such speedup has been demonstrated by this profiling work.

A subsequent [thread-count control and expanded opportunity list](gemma4_native_runner_improvements.md#follow-up-thread-count-and-phase-specific-settings)
tests two, four, and eight P cores. Doubling from four to eight cuts decode
time by 6.4% and prompt processing through first logits by 40.1%, with
bit-identical full-logit validation. That is a resource/configuration change;
the four-thread operator attribution above remains the baseline profile.

## Validation and artifacts

All measured input histories and argmax predictions match across baseline,
profiling enabled/disabled, and correctness captures. The profiler's separate
correctness run matches the prior baseline bit for bit for **65 full-vocabulary
vectors: 17,039,360 FP32 values, or 68,157,440 bytes**. All operator timestamps
are ordered and fit within their corresponding stage wall intervals.

The [experiment scripts and reproduction steps](../experiments/native_mlp_profile/README.md)
and [compact results](../experiments/native_mlp_profile/results.json) are
stored beside the diagnostic patch. Raw artifacts are local to this
desktop:

- [Final operator capture and analysis](/data/bt/tmp/gemma4-native-mlp-profile-20260923/v2/analysis.json)
- [Baseline build input verification](/data/bt/tmp/gemma4-native-mlp-profile-20260923/v2/input-validation.json)
- [Decode CPU sampling analysis](/data/bt/tmp/gemma4-native-mlp-profile-20260923/perf-record-v2/analysis.json)
- [Annotated worker polling loop](/data/bt/tmp/gemma4-native-mlp-profile-20260923/perf-record-v2/thread-main.txt)
- [Decode PMU counts](/data/bt/tmp/gemma4-native-mlp-profile-20260923/perf-stat/stat.csv)

The finalized diagnostic binary SHA-256 is
`4205377d8406e39203ddab3eaeb2d898c98ae6bc7b6854be5ab8fffa1c445d94`;
the unchanged baseline is
`e4354623abffa225548516bbba76a89f91847bdf8a3667b72e18dd6cab45792f`.
This report covers one desktop, one context length, and the prior baseline's
packing-source lifetimes. It is not a profile of the source-release or reshape
optimization variants, and the timings should not be transferred to phones.
