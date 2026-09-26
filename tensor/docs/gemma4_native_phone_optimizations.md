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

# Gemma4 native optimization experiments on TECNO LJ9 and Pixel 8

The desktop's packing-source release also works on both connected phones:
**final-decode RSS falls by 372 MiB on TECNO and 366 MiB on Pixel**, with about
253 MiB less process peak RSS. For speed, **four fast cores remain the best
tested TECNO configuration; five fast cores improve Pixel decode by 12.1%**
relative to its four-core configuration. These are isolated experiments;
production sources and defaults have not changed.

The phone profiles make two further targets concrete: the ARM INT2 FC kernels,
and repeated packing of dynamic K/V operands. MTP and speculative multi-token
verification are **out of scope**. All experiments use the existing checkpoint
and ordinary single-token decode.

## Configuration and controls

Measurements were collected on 2026-09-23 using the same matched Gemma4 E2B
bundle as the [desktop experiments](gemma4_native_runner_improvements.md):
1,024 prompt tokens, 64 forced decode inputs, capacity 8,448, prefill chunk
128, KV alignment 32, shared workspace, preserved compact INT2, and reused
runtimes. Each process discards one warmup session. Every one of the 1,394
coefficient files was SHA-256 checked on both devices before the campaign.

The baseline, lifetime probe, and profilers were relinked against the same
archived Android libraries. Dependencies and kernels were not rebuilt. The
NDK compiler is 27.3.13750724, targeting Android API 24. Exact commands,
archive/source hashes, executable hashes, device fingerprints, and captured
outputs are retained in the [raw artifacts][raw]. This is a native-runner
comparison; it does not rerun or change the previous LiteRT-LM measurements.

| Device | Two-thread mask | Four-thread mask | Additional configurations |
| --- | --- | --- | --- |
| TECNO LJ9, Android 15 | CPUs 6–7, `c0` | CPUs 4–7, `f0` | All eight CPUs, `ff` |
| Pixel 8, Android 17 | CPUs 7–8, `180` | CPUs 5–8, `1e0` | Five fast CPUs 4–8, `1f0`; all nine, `1ff` |

Each process starts only after two eligible HAL readings five seconds apart:
CPU at most 45 °C, skin at most 34 °C, SoC at most 39 °C, and any exposed CPU
cooling states zero. At least 15 seconds separates processes. The controller
records temperatures and current policy frequencies during execution. It
does not fix clocks or change governors, thermal limits, or device security.

This is **not a thermally invariant experiment**. Sampled CPU sensors reached
93.5 °C on TECNO and 103 °C on Pixel during the complete campaign. Pixel's
three exposed CPU cooling states stayed zero in the samples; TECNO exposed
none. That does not establish absence of DVFS, unobserved throttling, or
background interference. Pixel's four-thread timings vary noticeably between
blocks. Treat the figures as observations of these workloads and devices,
not universal optima or energy-efficiency measurements.

## Thread count and affinity

Each configuration runs in two process blocks, with the order reversed in
the second block. Each process measures three sessions. Values below are
medians of six session means; a session mean averages 64 decode steps.
Profiling, memory snapshots, and full-logit dumps are disabled for these runs.

| Device / threads | Decode, ms/token | Post stages, ms/token | Attention, ms/token | Prompt through first logits, ms |
| --- | ---: | ---: | ---: | ---: |
| TECNO: 2 fast | 86.086 | 51.125 | 17.569 | 7,737.8 |
| TECNO: 4 fast | **40.912** | 24.235 | 7.143 | **3,352.1** |
| TECNO: all 8 | 45.976 | 27.527 | 7.998 | 3,785.7 |
| Pixel: 2 fast | 67.567 | 40.220 | 12.594 | 7,698.3 |
| Pixel: 4 fast | 48.920 | 28.926 | 8.896 | 5,246.0 |
| Pixel: 5 fast | **43.007** | 25.323 | 7.340 | **3,770.1** |
| Pixel: all 9 | 44.298 | 26.089 | 7.459 | 5,038.7 |

Pixel's five-core decode reductions are **9.8% and 14.4% in the two blocks**;
the pooled median reduction is 12.1%. Six-session decode ranges are
46.33–51.18 ms for four cores and 42.00–43.45 ms for five. TECNO's four-core
range is 40.30–41.16 ms. Adding little cores is slower than using all the fast
cores on both phones. Two-core results are substantially worse; this experiment
does not isolate the kernel, scheduling, or clock contribution to that loss.

Use four fast cores on this TECNO and consider five on this Pixel when that
CPU budget is acceptable. This is a resource/configuration change, not a
same-resource kernel speedup. The desktop's eight-P-core recommendation does
not transfer to either phone. Component medians are independent, and prompt
timing includes KV allocation/reset and the final prompt token's forward pass;
it excludes model loading and compilation.

## Memory: the same 366 MiB of sources can be freed

These separate diagnostic runs use four fast cores, one warmup, and one
measured session. The probe releases sources after both signatures have packed
their weights and the cache has been hard-finalized. Shape caching is disabled
in this memory comparison.

| Device / configuration | Process peak RSS, MiB | Final-decode RSS, MiB | Final-decode anonymous, MiB |
| --- | ---: | ---: | ---: |
| TECNO baseline | 2,064.72 | 2,055.93 | 1,453.79 |
| TECNO source release | 1,811.08 | 1,683.62 | 1,081.77 |
| **TECNO reduction** | **253.64** | **372.31** | **372.02** |
| Pixel baseline | 2,057.79 | 2,049.11 | 1,448.61 |
| Pixel source release | 1,804.37 | 1,683.29 | 1,082.37 |
| **Pixel reduction** | **253.42** | **365.82** | **366.25** |

Both probes verify expiration of all 60 compact MLP source-buffer owners:
270 MiB of coefficients, plus 960 bytes of padding. They also release the
96 MiB head conversion vector. Corresponding retained-allocation counters
become zero. The 366 MiB identified payload is the explanation shared by both
devices; TECNO's additional RSS difference includes allocator/residency
effects and is not another identified six-MiB model buffer.

Packed weights remain **751.875 MiB**, external stage outputs **48.035 MiB**,
and logical KV **74.25 MiB**. One shared workspace remains: **8.750 MiB on
TECNO and 9.125 MiB on Pixel**, unchanged by source release. Summing the
workspace aliases reported by 151 stage runtimes would count this memory
repeatedly. Final bundle-file RSS changes by only 0.24 MiB on TECNO and zero
on Pixel; process swap is zero for both configurations on both phones.

Current RSS comes from the process's `smaps_rollup`; peak comes from its
`getrusage` high-water counter. Peak falls less because compilation still
temporarily retains the sources. No `madvise`, allocator trim, or packed-kernel
change is involved. This is separate from LiteRT-LM's already-measured
[`MADV_PAGEOUT` benefit](gemma4_linux_xnnpack_memory.md#follow-up-effect-of-madv_pageout).

## Post-attention FC profile

Four-thread diagnostic runs record 192 samples for each of 1,715 operators
across 35 post stages. They use XNNPACK's nanosecond timestamps. Intervals
include dispatch and threadpool synchronization; they are not pure arithmetic
time. Full-logit dumps are disabled in these profile captures.

| Work, summed across 35 layers | TECNO mean ms/decode | Pixel mean ms/decode |
| --- | ---: | ---: |
| Three MLP FCs | 19.382 | 24.088 |
| Attention output FCs | 2.219 | 2.507 |
| Per-layer embedding gate/projection FCs | 0.994 | 1.135 |
| **All 210 FCs** | **22.595** | **27.729** |
| FP32/INT8 conversions | 0.485 | 0.401 |
| GELU | 0.264 | 0.384 |
| Remaining XNN operators | 0.567 | 0.668 |
| Stage wrapper outside invocation | 0.597 | 0.697 |
| **All post stages** | **24.508** | **29.879** |

FCs account for **92.2% and 92.8%** of post-stage time. Removing only the
visible conversion or GELU operators therefore has a limited upside; any
fusion proposal needs to show how it also improves the FC path or its data
movement.

| Three MLP FCs per layer | TECNO mean ms | Pixel mean ms |
| --- | ---: | ---: |
| Layers 0–14, INT4, intermediate width 6,144 | 0.475 | 0.553 |
| Layers 15–34, INT2, intermediate width 12,288 | 0.613 | 0.789 |
| INT2 group's additional time per layer | **29.0%** | **42.6%** |

Each group has 13.5 MiB of **source coefficient payload per layer**, while
INT2 performs twice as many dot-product terms. Equal source bytes do not mean
equal packed layout or measured memory traffic. The corresponding desktop
time difference was only 3.7%. The phone gap makes INT2 unpacking, instruction
scheduling, tiling, and weight delivery worth investigating; it does not prove
which mechanism limits throughput.

## Attention and vocabulary head

A separate diagnostic enables profiling on the 35 attention stages and the
head. It measures one 64-token session and also writes full logits for
correctness. These are operator attributions, not uninstrumented benchmark
results, and should not be added to independently measured medians above.

| Work across all attention layers | TECNO mean ms/decode | Pixel mean ms/decode |
| --- | ---: | ---: |
| QK matrix multiplies, including dynamic packing | 2.733 | 3.912 |
| Probability × V matrix multiplies, including dynamic packing | 3.216 | 3.727 |
| QS8-to-QC8 conversions for K/V | 0.698 | 0.985 |
| Mask addition | 0.074 | 0.139 |
| Softmax | 0.119 | 0.181 |
| Stage work outside invocation | 0.289 | 0.309 |
| **Attention total** | **7.130** | **9.252** |
| **Vocabulary-head FC alone** | **3.917** | **4.144** |
| Complete head stage | 4.072 | 4.349 |

Attention is about 17–18% of the unprofiled four-thread decode time on these
phones, versus about 8% on the desktop at this prompt length. Online softmax
could reduce intermediate traffic, but simply speeding up the current softmax
kernel targets only the 0.12–0.18 ms row. Most attention time belongs to BMM
and its operand preparation.

## CPU samples distinguish packing from matrix computation

An additional isolated profiler saves each measured invocation's absolute
`CLOCK_MONOTONIC` start and its operator durations. `simpleperf` records
`cpu-cycles:u` at 499 Hz using the same clock. Offline analysis retains only
samples inside the measured post/attention/head operator intervals. Startup,
warmup, prefill, and all time outside those invokes are excluded. This also
excludes projection-stage work; it is not a whole-decode CPU distribution.

There are **13,451 retained samples on TECNO and 16,551 on Pixel**, with zero
reported lost samples. TECNO produced 83 kernel-flagged sample IPs despite
`:u`, including four inside the selected intervals; all 83 are excluded.
Samples are weighted by recorded event period and include worker threads.

| Operator interval / identified kernel | TECNO sampled cycle share within interval | Pixel sampled cycle share within interval |
| --- | ---: | ---: |
| INT4 MLP: `qs8_qc4w` 1×16 NEON-dot assembly GEMM | 94.3% | 92.6% |
| INT2 MLP: `qs8_qc2w` 1×8 NEON-dot GEMM | 95.3% | 88.9% |
| QK BMM: scalar GOI weight packer | **24.3%** | **24.0%** |
| Probability × V BMM: scalar GIO weight packer | **31.8%** | **19.5%** |
| Head: dynamic-activation `qd8_f32_qc2w` 1×8 NEON-dot GEMM | 97.7% | 94.8% |

The attention compute kernel is
`xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128`.
Packing hits are specifically
`xnn_x8_packw_gemm_goi_ukernel_x8__scalar_u2` and
`xnn_x8_packw_gemm_gio_ukernel_x8__scalar`. The pinned
[XNNPACK configuration][gemm-config] selects these routines. The
[native attention graph](../examples/gemma4/native/driver.cc) folds eight
query heads into the row dimension: decode attention has M=8, whereas the
MLP and head FCs have M=1.

These are **CPU-cycle shares, not wall-time fractions or attainable speedups**.
The BMM groups contain 1,068–1,501 samples each; short operator intervals,
sampling boundaries, differing core speeds, and thread synchronization limit
precision. Nevertheless, substantial samples inside named scalar packing
functions establish a concrete target that basic BMM timers could not isolate.

## Implementation priorities from these results

1. **Support packing-source lifetimes.** Both phones validate the 366 MiB
   release. Add explicit last-compilation-consumer ownership and a defined
   rebuild/reload path. Direct INT2 import remains a separate startup/peak
   opportunity. Neither requires a new mathematical op.
2. **Tune device thread policies.** Keep four fast TECNO cores and evaluate
   five fast Pixel cores under the application's CPU/energy budget. Preserve
   separate prefill/decode tuning as an option. Affinity and pool scheduling
   are runner/backend concerns.
3. **Optimize ARM INT2 FCs, including the head.** Benchmark the actual M=1
   dimensions and existing quantization against improved tile widths,
   unrolling, and instruction scheduling. Current INT4 uses a wider assembly
   tile. The INT2 MLP and dynamically quantized head are distinct kernel
   families and both need validation. A new FC op name alone changes none of
   this execution work.
4. **Reduce repeated KV preparation.** First benchmark vectorized packers
   with the current layout, then owner-shared and incremental packed KV.
   There are 35 consuming attention layers and 15 KV owners. Cache validity
   must include owner, append/version state, active range, scales, and kernel
   layout; local-window movement and session resets need explicit handling.
   Measure the additional persistent packed storage as well as time saved.
5. **Evaluate tiled attention on longer contexts.** It can combine with KV
   packing reuse and chunked prefill. Preserve the current INT8 KV semantics
   and validate numeric differences from online reduction order. The present
   phone experiment does not implement or measure a tiled-attention speedup.

The earlier bounded-local-KV estimate (roughly 28–29 MiB total versus 74.25
MiB) and external-output reuse opportunity (48.035 MiB currently owned) remain
useful memory projects. Their relevant baseline allocations are confirmed on
both phones; their proposed savings are **not implemented or measured** here.
Shape caching was numerically checked together with source release on both
phones, but its small desktop speed gain was not separately benchmarked on
the phones. Prefill chunk-size tuning also remains future work.

## Correctness and artifacts

On each phone, compare against that phone's rebuilt four-thread baseline:
the post profiler, combined source-release/shape-cache probe, attention/head
profiler, and CPU-sampling profiler reproduce **all 65 complete vocabulary
vectors bit for bit**. Pixel's five-thread baseline does too. Each comparison
covers 17,039,360 FP32 values, or 68,157,440 bytes. Every measured run also
matches input histories and all argmax predictions. Cross-phone bit identity
is not assumed.

Phone correctness covers this 1,024+64 workload. It does not repeat the
desktop's ten-context, full-KV-byte validation or establish other models,
longer phone contexts, alternative quantization, or future optimized kernels.
The memory probe remains model-specific and one-way; this is not a generic
production API implementation.

- [Validated machine-readable results](../experiments/native_mlp_profile/phone_results.json)
- [Experiment scripts and reproduction](../experiments/native_mlp_profile/README.md#phone-follow-up)
- [Raw commands, logs, telemetry, profiles, and dumps][raw]
- [Desktop improvement plan](gemma4_native_runner_improvements.md)

[raw]: /data/bt/tmp/gemma4-native-phones-20260923
[gemm-config]: /home/chasun/src/LiteRT/.native-tensor-build/android/_deps/litert_native_xnnpack-src/src/configs/gemm-config.c:2016
