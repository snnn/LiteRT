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

# Why native Gemma4 decode is faster on phones

The large mobile speedup is reproducible. The strongest explanation for the
gap against LiteRT-LM's XNNPACK path is **attention work over unused cache
capacity**. The native runner restricts attention to the live prefix or local
window. The published graph's XNNPACK decomposition still runs global matrix
operations across the configured capacity, with additional full-width
conversion, masking and softmax work. That costs instructions and cache
traffic even when it does not cause proportionate DRAM traffic.

On TECNO, raising capacity from 2,048 to 8,448 with the same 1,024-token
prompt adds **24.32 ms per decode call to LM-XNNPACK**. Native remains near
41 ms. Separate operator profiles localize about 17.09 ms of the increase to
global-attention matrix operations; conversion, masking, softmax and fill
explain most of the remaining profiled increase. That LM capacity penalty is
about 71% of the observed 8,448-capacity latency gap. This is a diagnostic
control: reducing the maximum supported session length is not a substitute
for making execution respect the live context.

YNNPACK avoids most of that matrix-operation growth and materially narrows
the gap, but it retains other work and has a separate numerical qualification
issue. Cache layout alone, head folding, and exclusive access to INT2 kernels
do not explain these results.

The follow-up [SDPA graph experiment](ynnpack_sdpa_graph_performance.md)
rewrites the official `.litertlm` decode graph while preserving its QAT
coefficients and cache contract. It compares the original and rewritten
bundles in the same patched YNN runtime. The
[YNN assessment](#what-better-graph-authoring-could-give-ynnpack) and
[export handoff](#planned-litert-converter-work) below retain the original
rationale, contracts and acceptance criteria; the follow-up report records
what has actually been implemented and measured.

A subsequent local [INT8 SDPA extension](ynnpack_int8_sdpa.md) addresses one
delegate-admission limitation identified below. It reuses YNN's existing
mixed-type matrix lowering. Its 10 new INT8 tests and 9 existing attention
tests pass on Linux, TECNO and Pixel 8. The performance results in this report
use the preserved binaries and **do not include that extension**.

This September 16, 2026 investigation follows the
[host DRAM measurement report](native_decode_memory_traffic.md). The complete
[local experiment directory][campaign] preserves commands, raw results,
source audits, binaries and analysis scripts. It is separate from the
[September 15 mobile report][historical-report]; historical captures were
preserved.

## What was different in the earlier comparisons

The earlier phone headline used **capacity 8,448**. The recent host headline
used **capacity 4,096**. Both used the same 1,024-token prompt and 64 forced
decode inputs, but they were not the same capacity experiment. At the larger
capacity, the published graph has substantially more unused attention width.

At host capacity 4,096, native used 806.01 MB/call versus 834.42 for LM-XNNPACK
and 829.15 for LM-YNNPACK. Its throughput advantages were 23.7% and 11.2%.
Those DRAM measurements remain valid. They count traffic reaching DRAM,
including traffic caused by packing or fallback operations, but do not count
arithmetic instructions or cache hits, or separately attribute packing,
fallback and scheduling costs. They also do not measure either phone's
memory controller.

Capacity mismatch is only part of the explanation. At the same capacity of
4,096, native's advantage over LM-XNNPACK is 1.24× on the host and 1.40× on
TECNO. The relative cost of the remaining work also differs by platform;
the isolated attention measurements below test that mechanism directly.

The ARM FP32/QC8 attention kernel loads packed INT8 data, sign-extends it,
converts it to FP32 and executes floating-point multiply-accumulate
instructions. Processing masked positions still consumes this work. A small
difference in whole-model DRAM bytes therefore does not imply a small timing
difference. See the [source and graph audit][source-audit] for exact paths,
hashes and shape derivations.

## Reproducing the gap with preserved binaries

The main sweep contains **54 successful fresh-process captures**: two phones,
three backends, three capacities and three repetitions. Each process runs one
excluded warmup and one measured session with 64 forced decode calls.
Capacity and backend order rotate between rounds. All runs use four CPU
threads, the same prompt and forced inputs, the same model coefficients,
INT8 KV, and the retained September mobile binaries. Native uses its
bundle-matched exported weights; both LM modes read the original `.litertlm`
bundle. No operator profiling or logits dumping is enabled in this sweep.

![Decode latency versus allocated capacity, with observed ranges][capacity-plot]

Calls/s is the reciprocal of mean measured per-call latency. The prompt's
first prediction is excluded. These forced calls produce full logits without
sampling the next input, so the workload remains identical between backends.

### TECNO LJ9

Each cell is the median of three processes; brackets give the observed range.

| Allocated capacity | Native calls/s | LM-XNNPACK calls/s | LM-YNNPACK calls/s |
| ---: | ---: | ---: | ---: |
| 2,048 | 24.14 [23.84–24.55] | 19.65 [18.81–20.41] | 20.40 [19.90–21.16] |
| 4,096 | 24.02 [23.59–24.86] | 17.20 [16.33–17.55] | 19.39 [18.45–19.57] |
| 8,448 | 24.37 [23.91–24.43] | 13.29 [12.55–13.50] | 19.56 [19.03–19.65] |

At capacity 8,448, native is **1.83× LM-XNNPACK** and **1.25× LM-YNNPACK**.
The earlier measurements were 23.86, 13.26 and 19.02 calls/s respectively:
the large gap did not disappear. At capacity 2,048 the ratios shrink to
1.23× and 1.18×.

The measured latency change is especially informative:

| Runtime | Capacity 2,048, ms/call | Capacity 8,448, ms/call | Increase |
| --- | ---: | ---: | ---: |
| Native | 41.421 | 41.042 | −0.379 |
| LM-XNNPACK | 50.903 | 75.223 | +24.320 |
| LM-YNNPACK | 49.031 | 51.125 | +2.094 |

Native's built-in attention-stage timer is also almost flat: 7.120, 7.158,
and 7.133 ms across the three capacities. Its append timer is about 0.03 ms.
These timers cover their named native stages; they are not interchangeable
with LiteRT per-operator profiling scopes.

### Pixel 8 and thermal control

All original captures are retained rather than selecting only the faster runs.

| Allocated capacity | Native calls/s | LM-XNNPACK calls/s | LM-YNNPACK calls/s |
| ---: | ---: | ---: | ---: |
| 2,048 | 20.25 [20.16–20.29] | 15.71 [14.66–16.45] | 16.25 [15.08–17.09] |
| 4,096 | 21.87 [21.64–22.16] | 15.51 [14.63–15.62] | 16.11 [13.65–16.99] |
| 8,448 | 20.97 [18.89–21.34] | 9.90 [9.23–12.10] | 15.17 [15.12–15.46] |

The two slower LM-XNNPACK capacity-8,448 processes had nonzero CPU cooling
device states. Android's aggregate `Thermal Status` still reported zero.
The first process ran at 12.10 calls/s with CPU cooling states zero before
and after; the other two ran at 9.23 and 9.90 with active cooling controls.
Their slower decode persists across all four quarters of the 64-call phase.

Consequently, the 2.12× ratio of these aggregate medians mixes graph cost and
device state. It is not a clean estimate of an architectural speedup. Current
HAL readings, rather than stale cached temperatures, expose this confound.
Nominal maximum-frequency settings alone do not rule it out. The
[telemetry audit][telemetry] preserves the readings and their limitations.

A separate confirmation applied the same start gate to all three backends:
two readings 30 seconds apart with all CPU cooling states zero, CPU
temperature ≤45°C, skin ≤34°C and SoC ≤39°C. Each backend then ran one
warmup and one measured session at capacity 8,448:

| Runtime | Calls/s | ms/call |
| --- | ---: | ---: |
| Native | 18.92 | 52.867 |
| LM-XNNPACK | 12.15 | 82.279 |
| LM-YNNPACK | 14.65 | 68.238 |

All endpoint CPU cooling states were zero. XNN recovers close to its earlier
12.09 calls/s; native remains 1.56× faster than XNN and 1.29× faster than YNN.
Native and YNN did not improve simply from starting cooler. These are
**single-process confirmations**, not new medians, and a start gate plus
endpoint snapshots does not establish fixed frequencies throughout decode.
Thermal controls contributed to the concern about the original slow XNN
runs, but do not explain all variance. The [separate cooled results][cooled]
are not pooled into the original sweep.

## Localizing the extra time

Separate diagnostic runs enable profiling **only for the 64 forced calls**,
after a complete warmup and prompt processing. The parser checks exact event
counts: 70 matrix operations × 64 calls and 35 softmax operations × 64 calls
for XNNPACK, or 106 delegated partitions × 64 for YNNPACK. Composite parents
and duplicate delegate totals are not added to their children.

The following TECNO differences compare the same instrumented backend at
capacities 2,048 and 8,448:

| Profile category | Capacity 2,048, ms/call | Capacity 8,448, ms/call | Increase, ms/call |
| --- | ---: | ---: | ---: |
| Global QK and PV matrix operations | 5.837 | 22.930 | +17.093 |
| Local QK and PV matrix operations | 3.570 | 3.597 | +0.027 |
| Fully connected operations | 36.804 | 36.408 | −0.396 |
| Softmax | 0.315 | 1.680 | +1.365 |
| SELECT fallback | 0.646 | 2.092 | +1.446 |

QS8/QC8 adapters add another 3.381 ms and FILL adds 1.035 ms. Selected leaf
time rises by 24.465 ms, close to the independently measured **unprofiled**
increase of 24.320 ms. Global attention is the largest growing component;
local matrix and weight-matrix times remain nearly unchanged.

Profiling raises absolute latency substantially: inclusive LM-XNNPACK times
are 92.034 and 117.702 ms. Therefore these profiles locate the increase;
their categories are not an exact additive decomposition of the unprofiled
native-versus-LM gap. Inclusive minus selected time includes profiler effects
and unrecorded work, and must not be labeled measured interpreter overhead.

YNNPACK's inclusive profile grows only from 51.477 to 54.378 ms. SELECT grows
from 0.701 to 2.120 ms; partition/fallback time accounts for the remaining
roughly 1.48 ms. Its profiler exposes partitions rather than comparable
internal matrix/softmax leaf events. Pixel also shows global attention as the
largest growing XNN category, but FC time rises 14% and local matrix time 17%,
consistent with broader device-state variation. TECNO is the cleaner
localization. See [all eight validated profiles][profiles].

## What the graphs actually execute

E2B has eight query heads and one KV head. Of its 35 attention layers, 28 are
local with head dimension 256, and seven are global with dimension 512.
For this 64-call interval:

- Native global widths are 1,056 or 1,088 after rounding to 32; the mean is
  1,072. Its local mean width is 543, with masking preserving the 512-position
  causal window.
- LM-XNNPACK's decomposition restricts local matrix operations to 512, but
  global matrix operations span capacity. Local QK still fills/updates a
  full-capacity result before the public softmax.
- YNNPACK's direct runtime-BMM path uses the live prefix, averaging 1,056.5,
  for both global and local layers. It does not crop the local lower bound.
  The public softmax/reduction domain still has the capacity-sized axis.

| Capacity | Native QK+PV, million MAC/call | LM-XNNPACK | LM-YNNPACK | LM/native public softmax extent ratio |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 123.75 | 176.16 | 181.75 | 3.16× |
| 4,096 | 123.75 | 293.60 | 181.75 | 6.31× |
| 8,448 | 123.75 | 543.16 | 181.75 | 13.02× |

These are source-derived arithmetic spans and graph domains, **not measured
DRAM bytes, exact machine-loop counts, or predicted latency ratios**. YNN's
mixed-type BMM also dynamically quantizes activations, so a MAC does not have
the same cost or numerical behavior in both backends. Packing, conversion,
kernel padding, FC work and scheduling are excluded from this table.

The capacity-dependent operation count explains why comparing only allocated
KV bytes or whole-model DRAM bytes missed a major part of the mobile result.
The more general layout and addressing discussion is in the
[KV cache reference](kv_cache_layout.md).

## Isolating width from layout

A context-only XNNPACK attention probe changes width, V storage and query
folding independently. For the clean width control, the same 1,088 valid
positions are supplied to widths 1,088 and 8,448; the extra positions are
masked. The probe repeatedly runs QK, mask addition, softmax and PV with INT8
KV and FP32 activations, without full-model weight streaming or deliberate
cache eviction.

For head dimension 512 and folded queries with token-major V:

| Platform | Width 1,088, ms | Width 8,448, ms | Increase |
| --- | ---: | ---: | ---: |
| Linux x86_64 | 0.105 | 0.881 | 8.36× |
| TECNO | 0.469 | 4.277 | 9.12× |
| Pixel 8 | 0.544 | 5.402 | 9.93× |

All values are medians of three pass means. The width ratio is 7.76×.
Large width effects persist across both layouts and both tested dimensions.
The extra masked work is expensive even with repeated input reuse. Its
absolute cost is much larger on the phones, consistent with attention
contributing a larger fraction of their full-model runtime. This does not
identify a unique hardware bottleneck or predict a full-model speedup by
multiplying microbenchmark times by layer counts.

At **the same width**, transposed V takes 2.5–12.3% less time than token-major
V across the ten folded TECNO cells. Pixel layout results are mixed and noisy;
the host favors token-major. Native's token-major storage is valuable because
it permits contiguous live-prefix/window views and efficient appends. The
mobile benefit is the work that this layout and execution strategy permit
avoiding, rather than a universally faster token-major matrix kernel.

Unfolding queries makes the probe slower, but both real graphs already fold
them. All 360 records passed independent double-precision reference,
layout-equivalence, immutable-input and fused-runtime checks; maximum
reference error was 4.77e-7 against a 2e-5 threshold. Pixel has substantial
pass variability, and ascending width order is not a fully randomized
experiment. [All medians, ranges and provenance][attention-probe] are retained.

## Decode-only CPU samples, including YNNPACK

Six separate TECNO captures sample user-mode CPU cycles only while executing
the 64 forced calls. The target stops after warmup/prefill, resumes after the
recorder signals readiness, then stops again before final output writes.
All captures stay within the recorder's duration. Across 18,384 samples,
none were lost and all leaf symbols resolved.

| Selected exclusive self-symbol share | LM-XNNPACK | LM-YNNPACK |
| --- | ---: | ---: |
| All matrix/dot kernels | 64.46% | 70.28% |
| Explicit packing kernels | 7.76% | 0.00% |
| Transpose kernels | 0.00% | 3.03% |
| Named C copy/move/fill routines | 6.22% | 0.13% |
| Pool dispatch/synchronization symbols | 17.14% | 9.11% |

These are medians of exclusive event-weighted shares, not wall-clock fractions
or exact cycles per token. Zero means no classified leaf samples, not proof
that no equivalent work occurs inside another kernel. User-mode sampling
excludes kernel execution and off-CPU waiting. XNN frame-pointer call chains
have substantial errors, so caller/child attribution is deliberately avoided.

The largest XNN symbol is the FP32/QC8 attention GEMM at 22.39%; scalar packing
is 7.76%. The largest YNN symbols are its INT2 dot kernel at 37.92% and INT4
dot kernel at 29.19%. YNN's INT8 dot kernels contribute smaller shares;
`exp_subtract` is 2.40% and reference SELECT is 1.03%.

Instruction-address inspection resolves two XNN categories more precisely.
Over 99% of `thread_main` self weight falls in the worker active-state
polling loop, including ARM `YIELD` and acquire loads; that region accounts
for a median 15.61% of total sampled user-cycle weight. Over 99% of the scalar
packer's self weight falls in its byte-load/store interleave loop. These are
observed polling and packing instructions, not an assumed interpreter cost.
They do not establish recoverable wall time: worker polling can overlap
useful work, and native was not sampled for comparison. The
[instruction audit][instructions] records exact addresses and weighted counts.

This corrects an earlier inference: the old 15-second capture mixed prefill
and decode and attributed 14.83% to YNN transpose and 4.02% to SELECT.
The new decode-only medians are **3.03% and 1.03%**. Those earlier percentages
must not be used to explain decode. Most sampled YNN decode work is in its
low-bit matrix kernels. Its remaining gap to native cannot be assigned to
transpose alone, or to interpreter overhead, from these measurements.

The [sampling report][cpu-samples] contains all six repetitions, ranges,
symbol tables, lifecycle checks and limitations. Native was not sampled in
this phase, so these data are not a matched native-versus-LM kernel ablation.

## Compiler control

The historical LM Android build uses `-Oz` for general C++ code, while many
optimized kernel targets already append `-O2`. Native used a different NDK
and build configuration. To test one concrete concern, the same new LM
adapter was built with the historical flags and with explicit final
`--copt=-O2 --cxxopt=-O2`; source and dependency revisions were unchanged.
Actual compiler argument samples verify the effective final optimization
level. Both diagnostic flags and operator profiling were off during timing.

Twelve TECNO captures form three adjacent pairs per backend, with compiler
and backend order reversed between rounds. They use capacity 8,448 and the
same workload as the main sweep:

| Backend | Historical flags, median calls/s | Explicit O2, median calls/s | Median paired speedup | Paired range |
| --- | ---: | ---: | ---: | ---: |
| LM-XNNPACK | 12.79 | 12.90 | 1.011× | 1.009–1.017× |
| LM-YNNPACK | 19.02 | 19.29 | 1.021× | 1.014–1.048× |

Each compiler pair preserves all 65 top-token predictions. That is a smoke
check, not a full-logit comparison. The small gains do not explain the large
native advantage. Three pairs cannot precisely resolve small effects under
phone DVFS, and this is an optimization-level control rather than a complete
NDK/API/dependency match. These new binaries' absolute timings are kept
separate from the preserved-binary capacity sweep. See the
[compiler results][compiler-control] and [build manifests][profile-adapter].
The [independent compiler audit][compiler-audit] checks raw histories, binary
hashes, the build-flag difference and capture separation. There was no fixed
thermal gate in this control, so the small measured gains should not be
treated as universal compiler speedups.

## Findings that narrow the explanation

- **Query folding is shared.** Both graphs already fold the eight queries
  into `[1,1,8,D]`. Shared KV packing has batch size one; source does not
  support a claim that LM makes eight independent KV copies.
- **Compact INT2 support is shared.** Both XNN paths retain it, and YNN
  supports it too. Native reports 60 compact INT2 operators.
- **FP32/QC8 attention fusion is shared by the two XNN paths.** The LM-XNN
  profile and source show it as well as the native graph.
- **The inspected XNN kernel code is identical.** Six relevant XNN core/ARM
  kernel files are byte-identical between the retained native and LM
  revisions. An earlier wording, “a newer XNN kernel explains everything,”
  named an unsupported hypothesis. It was not a finding that an upgrade
  caused the speedup. The evidence instead points to different amounts of
  work given to those kernels. This limited comparison does not establish
  equality of every source file, selected kernel, dependency or binary.
- **Interpreter overhead has not been isolated.** Neither profile
  subtraction nor whole-model counters assign the residual gap to that
  category. Worker polling samples also cannot be counted as recoverable
  interpreter wall time.

## Numerical scope

All performance repetitions use the same forced histories. Native and
same-phone LM-XNNPACK argmax histories remain stable across these capacities.
This is a workload and regression check, not a new full-logit qualification.

The prior YNN correctness gate did **not** pass the tight full-logit threshold:
all 15 selected top IDs matched, but zero of 15 vectors met both cosine
similarity ≥0.999 and RMSE ≤0.05; the maximum RMSE was 2.708 and minimum cosine
0.9612. Across the larger recorded set, 178/195 argmax positions matched XNN.
YNN's activation quantization changes arithmetic, but previous controls did
not establish it as the sole cause of this drift. Its performance here is an
experimental configuration, not acceptance as a numerically interchangeable
replacement. See the [historical report][historical-report].

## What better graph authoring could give YNNPACK

Better graph authoring is a credible way to recover more of native's
advantage while continuing to use LiteRT-LM. The opportunity is to express
the useful attention region consistently and let YNN lower it efficiently.
YNN's newer implementation already solves much of the large-capacity matrix
problem seen in LM-XNNPACK, so the expected opportunities differ between the
two backends. This assessment concerns the inspected revisions; it is not a
claim that every future YNN release has the same limitations.

### How much room do the measurements establish?

At capacity 8,448 on TECNO, the measured medians are 51.125 ms/call for YNN
and 41.042 for native: a difference of **10.083 ms**. Matching native would
require approximately a **19.7% reduction in YNN latency**, corresponding to
native's **24.6% throughput advantage**. Calling this a smaller gap refers to
the comparison with LM-XNNPACK; it is still a material mobile performance
difference.

The 10.083 ms is a difference between complete configurations, not an
identified budget of removable graph overhead. Decreasing YNN capacity from
8,448 to 2,048 removes only 2.094 ms/call in the current sweep, versus 24.320
for LM-XNNPACK. This demonstrates lower capacity sensitivity in YNN. It is
neither an estimate of all avoidable attention work nor an upper bound on
future graph improvements: the local-prefix work remains at both capacities.

YNN's INT2 and INT4 weight kernels account individually for about 38% and 29%
of sampled user cycles. That makes attention optimization only one part of
the remaining problem. Those sampled shares cannot be substituted for
wall-time fractions in an Amdahl-law speedup calculation. There is no measured
prediction yet that reauthoring will close the gap or make YNN faster than
native.

### Preserve what YNN already does well

The existing runtime-BMM lowering exploits a runtime upper bound, using
internal active extents with capacity-shaped external storage. In the
audited execution, the matrix work follows the live prefix rather than all
8,448 positions. YNN also executes genuine packed INT2/INT4 weight kernels
and INT8 matrix kernels for its mixed-type attention path. These capabilities
should survive a new export.

The graph should therefore expose information YNN is currently missing,
without forcing it to reproduce every operation and storage choice in the
native XNN implementation. Native provides a working reference for the
semantics and amount of useful work. Its exact kernel sequence is not a
required implementation for another backend.

### Prioritized graph opportunities

| Opportunity | Current evidence | Intended change | What still needs verification |
| --- | --- | --- | --- |
| Local window lower bound | YNN's direct runtime-BMM path processes the zero-based live prefix for local layers. | Carry the local begin as well as the live end through attention. | Whether the selected graph form and YNN lowering support a nonzero begin without materializing large copies or falling back. |
| Active score, mask and softmax domain | The public runtime-BMM graph retains capacity-sized score/reduction domains; profiles show capacity-dependent SELECT and other work. | Keep QK, mask application, softmax and PV consistent on the useful region. | Actual compiled reduction bounds, output materialization and partitioning; smaller declared shapes alone do not prove less physical work. |
| Attention semantics across operation boundaries | Separately lowered runtime-BMM operations preserve full-shaped interfaces between stages. | Use a representation that allows the backend to recognize the complete attention region and reuse intermediates efficiently. | Operator dtype/layout support, mask semantics, numerical behavior and fallback implementation. A composite name does not guarantee fusion. |
| Cache addressing and append | Native can bind contiguous token-major live windows; the published V cache has a capacity stride. | Preserve efficient reads and owner-based appends while selecting a layout the backend can execute well. | End-to-end benefit including packing, copies, scratch and cache updates. The equal-width mobile probe does not justify changing V layout by itself. |

Local-window bounds are particularly interesting at longer contexts. For a
decode token at zero-based position `t`, let `end = t + 1` and window size
`W = 512`. Global attention uses `[0, end)`; local attention uses
`[max(0, end - W), end)`. The current YNN local matrix path uses `[0, end)`
even when the earlier positions are masked. Once the history exceeds 512,
that prefix continues growing while the useful local window stays bounded.
This suggests increasing opportunity with context length, but is not a
measured speedup for a newly authored graph.

For the current prompt-1,024 experiment, the mean YNN prefix is 1,056.5,
whereas native's aligned local extent averages 543. The arithmetic table
above therefore gives 181.75 million attention MACs for YNN versus 123.75
million for native. Their different activation arithmetic and whole-model
weight costs prevent translating that ratio into a throughput prediction.

### Graph changes and backend support must meet

The measured baseline's YNN SDPA visitor is a useful example of tighter attention
lowering: it can restrict K before QK, slice a mask to the logits region,
and restrict V to the corresponding prefix. However, the
[inspected visitor][ynn-attention-source] admits FP32, FP16 or BF16 Q/K/V and
output; **that baseline does not admit this bundle's INT8 K/V**. Its supported prefix
bound also does not by itself implement an arbitrary local lower bound.
Its dot/softmax/dot construction is not evidence of online/Flash attention.

Consequently, replacing the runtime-BMM sequence with
`odml.sdpa_transposed` is not an established converter-only fix for this
model. Adding dequantization merely to make the visitor accept the graph
could restore the traffic and materialization costs we are trying to avoid.
An INT8-compatible attention lowering, or a supported bounded sequence of
operations, may be necessary. Availability must be checked against the
specific runtime revision chosen for the next experiment.

The subsequent [local delegate patch](ynnpack_int8_sdpa.md) adds FP32 Q/output
with per-tensor affine INT8 K/V to both SDPA layouts, without inserting
full-cache dequantization. It dynamically quantizes Q and the softmax
probabilities using the existing runtime-BMM machinery. This removes the
specific dtype-admission obstacle for that contract; it does not add a local
lower bound or establish full-model numerical equivalence or a speedup.
The published model still uses its original runtime-BMM graph, so merely
rebuilding its runtime with this patch does not switch it to the new SDPA path.

Similarly, the [runtime-BMM lowering][ynn-bmm-source] uses `is_src` to
distinguish the two matrix stages, but does not consume a local-window lower
bound there. Serializing an additional attribute does not make the existing
consumer use it. The graph, delegate and fallback must agree on its meaning.
The [KV cache reference](kv_cache_layout.md#backend-specific-findings-and-revision-boundaries)
documents these layout and parameter contracts in more detail.

For the published transposed V buffer `[B,Hkv,D,C]`, an element's sequence
stride within each channel is one, but adjacent channels remain `C` elements
apart. A live prefix of length `L < C` cannot be reinterpreted as a dense
`[B,Hkv,D,L]` tensor by changing shape metadata alone. A valid strided internal
view, backend packing, explicit copy, or compatible persistent layout is
required. In contrast, token-major E2B storage with one KV head permits a
contiguous token interval. Multi-KV-head models need their own addressing
analysis; this result is not automatically an E4B or Qwen solution.

### Expectations to carry into the next experiment

The leading hypothesis is that explicit local bounds and a consistent
active score domain can improve YNN further. The large LM-XNN capacity
penalty provides a separate, stronger demonstration of what avoiding unused
work can accomplish; it is not a forecast of a similar YNN gain. Transpose
and SELECT alone are not demonstrated explanations for the entire remaining
gap, and reducing worker polling has an unmeasured latency tradeoff.

A successful export should preserve YNN's current low-bit and prefix
advantages, demonstrate less executed work, and improve unprofiled mobile
latency while meeting the numerical criteria. It need not reproduce native's
layout or require the whole remaining gap to disappear to be useful.

## Planned litert-converter work

**Status: saved plan only.** No converter code is changed, no new `.litertlm`
file is generated, and none of the tests in this section has been run as
part of this documentation update. The next implementation should produce
a separately named bundle and retain the measured original as a baseline.

### Establish a controlled export before optimizing

1. Record the original bundle, checkpoint revision, converter/runtime Git
   revisions and local patches. Retain model and binary hashes, tokenizer and
   metadata, signatures, capacity configuration and quantization parameters.
   The original bundle hash is recorded below.
2. Locate and inspect the existing Gemma4 authoring/export route when converter
   work begins. Establish how it represents global/local attention, runtime
   bounds, INT8 KV scales, low-bit weights and shared KV owners. The exact
   converter entry point and export command remain to be determined; this
   document does not invent a command or assume a ready-made export option.
3. Produce an unchanged-graph control export if the route supports it. Compare
   its coefficients, packed codes, scales, zero points, signatures and outputs
   with the original bundle before attributing differences to graph changes.
   If regeneration from the official checkpoint changes coefficient precision
   or quantization, report that separately; native's current reference uses
   coefficients matched to the original bundle.
4. Check the chosen YNN revision's INT8 operator and bound support with a small
   representative attention graph before committing to a full-model graph
   form. If support requires backend changes, record that dependency rather
   than treating a failed delegate admission as a converter bug.

Keep INT8 KV, published quantization parameters, compact INT2/INT4 weights and
the same capacity for the primary comparison. Switching to floating-point KV,
changing activation precision, changing model coefficients or reducing maximum
context creates a different experiment and must be labeled accordingly.

### Use separate variants to identify the benefit

| Proposed variant | Difference from its control | Main question |
| --- | --- | --- |
| Control export | Re-export through the selected converter without the intended attention optimization. | Does the export preserve the intended model and runtime behavior? |
| Bounded scores | Carry the active global prefix consistently through QK, masks, softmax and PV. | Can supported lowering remove capacity-sized score and fallback work? |
| Local window | Add the local begin/end contract while preserving global attention semantics. | Does YNN stop processing the masked old prefix in local layers? |
| Combined graph | Combine individually validated changes. | Do the gains compose without losing delegation, low-bit kernels or numerical quality? |

The exact ordering depends on what the selected graph representation can
express. These are experimental distinctions, not a claim that separate
converter switches already exist. A cache-layout change or new INT8 SDPA
lowering should have its own control because it changes more than bounds.
Avoid combining a runtime upgrade, new coefficients, different KV dtype and
new attention graph in a single unexplained benchmark delta.

### Semantic contracts that must survive export

| Contract | Required behavior |
| --- | --- |
| Position and causality | Preserve absolute query/RoPE positions. Use an explicit exclusive live end and a correctly defined local begin; include the current appended token. |
| Prefill rows | Apply each row's causal/local limits. A multi-row prefill chunk needs the union of required KV positions plus per-row masking; applying one decode row's window to every row is incorrect. |
| Capacity and live shape | Keep maximum session capacity separate from useful attention extents and physical strides. Retain correct behavior at the final legal token. |
| Parameter ABI | Match the actual producer and each delegate/fallback consumer. The reviewed LM producer writes `{start,end,end}`; YNN uses element 1 for a multi-element bound. The existing `start` is not automatically a local-window begin. Define any extension explicitly. |
| Quantized values | Preserve INT8 KV codes, scale/zero point interpretation and axes, and compact weight quantization. Check whether changing bounds changes dynamic activation quantization statistics or rounding. |
| Shared state | Preserve KV ownership and sharing between layers. Append only for owners, make the new token visible to consumers, and preserve reset/reuse behavior. Do not duplicate full caches to simplify the graph. |
| Composite and fallback | Keep operand layouts, output shapes, attributes and the decomposition semantically consistent. Reusing a composite name does not authorize an incompatible layout contract. |
| External interfaces | Preserve required signature names, inputs/outputs and bundle assets so the normal LiteRT-LM executor can run the new bundle. Record intentional interface changes explicitly. |

### Validate correctness before interpreting speed

Use fixed token IDs first so tokenizer or chat-template differences do not
change the workload. Compare full logits and, where necessary, per-layer
intermediates and INT8 cache updates. A matching first token or readable
generated text is insufficient.

There are two distinct comparisons. The new graph versus the old graph under
the **same backend** isolates an authoring regression. The new YNN result
versus the qualified XNN/native reference addresses replacement correctness.
Matching old YNN alone does not resolve its previous full-logit failure. Keep
the existing cosine/RMSE gate visible; do not silently relax it to approve an
optimization. Arithmetic-changing lowerings need explicit numerical review
rather than an unsupported claim of bitwise equivalence.

Include early positions, the 512-token window transition, 32-token alignment
boundaries, prefill chunk/signature transitions, deeper context and the final
legal capacity position. Test more than one prompt and forced continuation,
global and local layers, and fresh sessions versus reset/reused sessions.
Track exact token positions because the prompt's first prediction is not one
of the subsequent forced decode calls. Run the relevant checks on the Linux
host and the connected Android phones; YNN/XNN kernels can differ by ISA.

On a failure, compare the unchanged control export first, then locate the
first diverging Q/K/V, cache update, score, probability or layer output. This
separates export/coefficient differences, bounds/addressing mistakes and
backend arithmetic effects before a performance claim is made.

### Verify executed work, then measure the new bundle

A useful TFLite graph on paper is not enough. Inspect the actual delegate
partitions and compiled behavior: active QK/PV spans, softmax reduction domain,
mask/SELECT/FILL work, INT8 KV handling, packed low-bit weight kernels, copies
and workspace. Keep fallback diagnostics. An attention composite alone proves
neither fusion nor online/Flash attention; a smaller logical extent alone
does not prove a smaller physical loop.

For the mobile comparison, run the original and new bundles through the same
LiteRT-LM binary with YNN enabled. Run with YNN disabled as a separate XNNPACK
backend control where the graph is supported, and retain native as a reference.
If a backend patch is required, first run the original bundle on that patched
runtime too, so the graph's effect can be distinguished from the runtime's.

Start with the established 1,024-token prompt and 64 forced calls at capacity
8,448. Then use prompts 128, 1,024 and 4,096 at that same capacity, and repeat
the capacity 2,048/4,096/8,448 control at prompt 1,024. Keep every prompt plus
continuation within the valid capacity. An 8K prompt is not needed for this
initial comparison. Test TECNO and Pixel when connected; availability of
another phone should be checked when the experiment begins.

Use the same four-thread affinity, model coefficients, warmup/reset policy
and decoding histories, alternate runtime/bundle order, retain every capture,
and report repeated-run medians and ranges. Apply the same thermal start
policy to every variant and record current HAL CPU cooling controls; endpoint
readings do not guarantee constant in-run frequencies. Separate unprofiled
latency from warmed decode-only operator/CPU attribution. Report prefill,
startup and memory footprint separately so decode gains do not hide material
costs elsewhere. Phone DRAM byte counts remain unavailable unless hardware
counter access is independently established.

The follow-up should leave a reproducible export command, a separately named
`.litertlm` file and hash, source/build provenance, a graph/operator comparison,
full-logit results, mobile timing distributions and a clear account of any
remaining fallback or numerical differences. Improvement is established by
that evidence, not by the age of YNNPACK or a smaller serialized graph.

## Artifacts and reproduction

The main controller is [run_capacity_matrix.py][matrix-script], with
[summary and validation][matrix-summary]. It reuses the preserved mobile
binaries and remote model assets; it does not rebuild them. Phone affinity
masks are `f0` for TECNO and `1e0` for Pixel. The Samsung phone was disconnected
and was not retested.

| Artifact | SHA256 |
| --- | --- |
| Preserved native Android binary | `fbe9897ce1e6bf05ca00536077dbad255b57204d4bf9d5ce666bb38064df6f74` |
| Preserved LM Android binary, both backends | `d9c55e566852bb48f15211779cfde765a2d4c9cdb307b28356c0376c1e8e0ea5` |
| `.litertlm` model | `ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42` |
| Fixed prompt/64-input fixture | `01009f2da16a6c32e7288323029d019b49e1507022fd711f471c8001824b4eef` |

The separate [profile adapter][profile-adapter] adds opt-in decode-only event
collection and SIGSTOP boundaries for CPU sampling. Its flags default off.
Both Android builds succeeded; seven argument guards were exercised on each
phone. Diagnostic captures are explicitly marked ineligible for benchmark
timing. Source copies, effective compiler arguments and build manifests are
preserved beside the binaries.

These results do not provide mobile DRAM read/write counts: the necessary
memory-controller counters remain inaccessible on these phones. They also
do not isolate every remaining millisecond. Uncontrolled DVFS, different
toolchains/dependency revisions, profiling overhead and synthetic cache
conditions are material limits to attribution.

[campaign]: /data/bt/tmp/gemma4-mobile-decode-attribution-20260916
[historical-report]: /data/bt/os/llama.cpp/tmp_models/gemma4_qat_q4_0_perf_graph_compare_20260914/ynnpack-20260915T070241Z/PERFORMANCE_REPORT.md
[source-audit]: /data/bt/tmp/gemma4-mobile-decode-attribution-20260916/source-audit/README.md
[telemetry]: /data/bt/tmp/gemma4-mobile-decode-attribution-20260916/history-audit/telemetry-audit.md
[profiles]: /data/bt/tmp/gemma4-mobile-decode-attribution-20260916/decode-profile-analysis/README.md
[matrix-script]: /data/bt/tmp/gemma4-mobile-decode-attribution-20260916/run_capacity_matrix.py
[matrix-summary]: /data/bt/tmp/gemma4-mobile-decode-attribution-20260916/capacity-summary.md
[profile-adapter]: /data/bt/tmp/gemma4-mobile-decode-attribution-20260916/profile-adapter/README.md
[capacity-plot]: /data/bt/tmp/gemma4-mobile-decode-attribution-20260916/mobile-capacity-decode.png
[attention-probe]: /data/bt/tmp/gemma4-mobile-decode-attribution-20260916/attention-probe/README.md
[cpu-samples]: /data/bt/tmp/gemma4-mobile-decode-attribution-20260916/decode-sample-analysis/README.md
[cooled]: /data/bt/tmp/gemma4-mobile-decode-attribution-20260916/cooled-pixel/summary.md
[compiler-control]: /data/bt/tmp/gemma4-mobile-decode-attribution-20260916/compiler-control/summary.md
[instructions]: /data/bt/tmp/gemma4-mobile-decode-attribution-20260916/source-audit/decode-instructions/README.md
[compiler-audit]: /data/bt/tmp/gemma4-mobile-decode-attribution-20260916/compiler-control/AUDIT.md
[ynn-attention-source]: /data/bt/tmp/gemma4-pr9918/LiteRT/tflite/delegates/ynnpack/attention.cc
[ynn-bmm-source]: /data/bt/tmp/gemma4-pr9918/LiteRT/tflite/delegates/ynnpack/dot.cc
