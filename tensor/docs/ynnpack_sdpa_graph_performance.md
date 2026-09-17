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

# Gemma4 mobile YNNPACK: exposing decode attention as SDPA

This September 16, 2026 experiment tests the graph-authoring hypothesis from
[the native mobile decode analysis](native_mobile_decode_analysis.md): expose
an entire attention region to YNNPACK while preserving Google's exact QAT
weights, INT8 cache contract, and LiteRT-LM runtime behavior.

The candidate is a surgical rewrite of the official mobile CPU `.litertlm`
bundle, **not a fresh conversion from safetensors**. Both variants execute in
the same newly built LiteRT-LM binary with the local
[INT8-KV SDPA extension](ynnpack_int8_sdpa.md). The A/B comparison holds the
runtime constant; a separate original-graph control using the same bundle
writer checks that the benefit extends beyond serialization layout.

## What the September 15 campaign already established

The existing [three-backend performance report][historical-performance] is the
baseline for this work. At prompt length 1,024 and capacity 8,448, it measured:

| Phone | Original YNN tokens/s | Native tokens/s | Native throughput advantage |
| --- | ---: | ---: | ---: |
| TECNO | 19.02 | 23.86 | 25.4% |
| Pixel 8 | 15.09 | 22.14 | 46.7% |
| Samsung SM-S937U1 | 26.47 | 37.96 | 43.4% |

Native was fastest in all nine phone/prompt combinations in that campaign.
Its [YNN execution audit][historical-execution] already established three
distinct facts: YNN QK/PV matrix work can follow the live prefix; the exposed
score, mask and softmax retain capacity-wide work; and local layers use the
whole prefix instead of a sliding 512-token lower bound. The present rewrite
targets the second issue. It neither discovers live-prefix support nor solves
the third issue.

The separate [capacity control][historical-capacity] is consistent with that
distinction. At prompt 1,024, reducing TECNO's capacity from 8,448 to 2,048
improved LiteRT-LM XNN decode from 13.26 to 18.68 tokens/s, while YNN changed
from 19.02 to 18.71. These were separate captures rather than randomized
capacity switches; the small YNN regression is not evidence that extra
capacity helps. They do show why the large XNN capacity penalty cannot simply
be assigned to YNN as well.

The old [sampled kernel report][historical-kernels] also identified YNN
transpose, exponentiation and reference SELECT hotspots. Its sampling windows
mixed prefill and decode, and incomplete stacks did not identify the
transpose's tensor role. Those samples motivate investigation; they cannot
assign percentages of the remaining decode gap or prove that KV packing is
its dominant cause.

Subsequent [decode-only sampling](native_mobile_decode_analysis.md#decode-only-cpu-samples-including-ynnpack)
already refined that result: YNN transpose and reference SELECT accounted for
median sampled user-cycle shares of 3.03% and 1.03%, versus 14.83% and 4.02%
in the older mixed-phase capture. Roughly 70% of the decode-only samples were
in matrix kernels. These are event shares, not wall-time fractions, and that
phase had no matched native sampling. Neither the old transpose percentage
nor an assumed interpreter tax explains the current native gap.

The old measurements are retained separately. The new original-versus-SDPA
tests measure the rewrite's incremental effect with one runtime; the fresh
native-versus-SDPA tests measure the remaining complete-runner gap. Comparing
ratios across the two days does not isolate the amount of gap closed because
the LM build, measurement protocol and device conditions differ.

The recorded XNNPACK/YNNPACK dependency pin is unchanged between those two
LM campaigns (`d89ef6669a14db203b3b7935b1b3862cb63fb6df`). Nine archived
XNN/YNN source-file hashes and Slinky's pipeline hash match the current build
sources. Both use NDK 28.2 and API 31. The LiteRT source override and adapter
target changed, as did session/process repetition. This is further reason
not to describe the new results as a demonstrated kernel-upgrade benefit.
The [provenance comparison][historical-provenance] lists the checked files,
hashes and limits of the preserved manifests.

## What changed

Each of the 35 decode layers originally exposes:

```text
runtime_bmm(Q,K) → SELECT_V2(mask,scores,negative) → SOFTMAX → runtime_bmm(P,V)
```

The rewrite replaces those four operations with one `odml.sdpa_transposed`
composite, passing Q, K, V, the Boolean mask, and the existing runtime parameter
tensor. The original region remains as its fallback decomposition. All QAT
coefficient bytes, tensor qparams, cache updates, local/global masks, prefill
graphs, signatures, tokenizers, and other bundle sections are preserved.

This is region fusion and intermediate-shape propagation. It does not
implement online/Flash attention, trim the lower bound of local windows, change
KV layout or precision, or modify weight matrix kernels.

The [converter export notes][converter-notes] describe the command, preservation
checks, and the separate official-safetensors cache-scale reader. That reader
implements an explicit E2B-only workaround for the observed 16× global-scale
discrepancy and verifies all 70 attention-input scales against the bundle. Its
use produces exactly the same candidate bundle bytes; it is not a new
quantization scheme used to obtain a speedup.

## Benchmark method

- TECNO LJ9 (`144512553T000099`, CPU affinity `f0`) and Pixel 8
  (`3A111FDJH005A0`, affinity `1e0`), plus the newly connected Samsung S25
  (`R5CY51BZC1W`, reported model `SM-S937U1`), four threads each. The Samsung
  uses affinity `f0` (cores 4–7).
- Prompt lengths 128, 1,024, and 4,096; maximum context capacity **8,448 for
  every run**, followed by the same 64 teacher-forced decode inputs.
- One excluded warmup and one measured session per fresh process, in-memory
  weight cache, reused executors, explicit KV reset before prefill.
- Three processes per variant at prompt 1,024, in original/candidate,
  candidate/original, original/candidate order. Two processes per variant at
  128 and 4,096, in original/candidate/candidate/original order.
- No operator profiling, CPU sampling, or full-logit dumping in timing runs.
  Decode excludes prefill's first prediction and includes output access/argmax
  under the same driver for both variants.
- Separate before/after thermal, cooling and CPU-frequency snapshots; the
  same per-phone cooldown policy applies to both variants. The phones run
  in parallel, with one benchmark process at a time on each phone.

Median results summarize independent process means. Reported tokens/s is
`1000 / median(process mean decode milliseconds)`, including the two-process
supplementary cells. Ranges are observed process means, not confidence
intervals. Endpoint telemetry does not establish
constant in-run frequencies; retain every capture instead of selecting the
fastest process. Prefill and peak RSS are reported separately in the raw
summary. The prefill graph is unchanged. However, the driver's prefill metric
also includes an empty-input `DecodeLogits` call for the pending last prompt
prediction, which can enter the modified decode graph once. It is primarily
a variability control, not a strictly untouched execution interval or evidence
of a prefill graph optimization.

Samsung started at 15% battery with USB power. The initial power-policy capture,
taken during the first process's cooldown, recorded Battery Saver disabled.
Measured-process endpoints ranged from 14% to 17%, all with USB power.
The final policy capture also recorded Battery Saver disabled, at 17%.
Its saved telemetry shows
why endpoint thermal status alone is insufficient: one native run ended with
CPU 6's frequency cap at 2.6496 GHz despite a hardware maximum of 4.4736 GHz,
while another ended at 4.2816 GHz. Both reported thermal status zero and began
under the same temperature gate. These snapshots cannot reconstruct clocks
throughout inference, but they are a concrete reason to retain the process
ranges and avoid attributing all variance to graph or kernel changes.
The completed original/SDPA primary captures ended with CPU 4/6 caps of
2.2272/2.2464 GHz, as did all three fresh SDPA comparison captures. Native
ended at higher caps. Thus Samsung's native comparison includes potentially
different DVFS behavior after each runner's prompt processing; it is not an
equal-frequency kernel ablation. The [endpoint frequency audit][frequency-audit]
retains the per-process observations. No clock, cooling or power settings were
changed to improve a result.

## Results

The primary 1,024-token graph A/B comparison is complete on all three phones:

| Phone | Original ms/decode [range] | SDPA ms/decode [range] | Original → SDPA tokens/s | Throughput gain |
| --- | --- | --- | --- | ---: |
| TECNO | 50.130 [49.843–50.763] | 46.307 [46.206–46.601] | 19.95 → 21.59 | 8.3% |
| Pixel 8 | 65.505 [60.692–65.855] | 56.729 [53.305–58.814] | 15.27 → 17.63 | 15.5% |
| Samsung SM-S937U1 | 43.361 [43.259–44.215] | 37.926 [36.167–38.273] | 23.06 → 26.37 | 14.3% |

Each primary entry uses three independent processes. The supplementary
128- and 4,096-token cells use two processes per variant:

| Phone | Prompt | Original ms/decode [range] | SDPA ms/decode [range] | Original → SDPA tokens/s | Throughput change |
| --- | ---: | --- | --- | --- | ---: |
| TECNO | 128 | 43.775 [43.429–44.122] | 38.571 [38.556–38.585] | 22.84 → 25.93 | +13.5% |
| TECNO | 4,096 | 70.933 [68.427–73.438] | 68.465 [68.047–68.883] | 14.10 → 14.61 | +3.6% |
| Pixel 8 | 128 | 54.997 [54.037–55.957] | 45.978 [45.804–46.152] | 18.18 → 21.75 | +19.6% |
| Pixel 8 | 4,096 | 88.244 [87.003–89.485] | 84.971 [76.806–93.136] | 11.33 → 11.77 | +3.9% |
| Samsung SM-S937U1 | 128 | 22.837 [22.502–23.171] | 18.993 [18.940–19.046] | 43.79 → 52.65 | +20.2% |
| Samsung SM-S937U1 | 4,096 | 58.514 [56.394–60.633] | 63.836 [63.411–64.262] | 17.09 → 15.67 | -8.3% |

![Original and rewritten YNN decode medians and observed process ranges](/data/bt/tmp/gemma4-ynn-sdpa-export-20260916/perf/mobile-sdpa-decode.png)

At 128 tokens, all three phones have non-overlapping observed ranges favoring
SDPA. At 4,096 tokens, TECNO and Pixel ranges overlap; their nominal 3–4%
throughput gains are inconclusive. Samsung instead shows an **8.3% throughput
regression** with non-overlapping ranges in these four processes.

Samsung's two 4,096-token SDPA captures ended at thermal status 1 with SKIN
40.3/40.0°C and CPU 4/6 frequency caps of 1.9968/1.9584 GHz. The two original
captures ended at status 0, SKIN 39.6/39.7°C and caps of 2.2272/2.2464 GHz.
All four passed the same initial temperature gate. The regression is retained;
endpoint telemetry does not establish how much comes from graph execution,
post-prefill clock behavior, or other device variation. The 1,024-token
diagnostic profiles below do not isolate the 4,096-token cause. Long prefill
heats the phones before decode, and thermal status zero at an endpoint does
not imply stable clocks throughout the interval.

The [combined summary][comparison] and [individual captures][comparison-json]
retain all 42 processes, their counts, ranges, prefill times and per-quarter
decode means. No slower or warmer capture was discarded.

A separate TECNO serialization control compares the unchanged graph repacked
by the same writer against SDPA, using original/SDPA/SDPA/original order and
two fresh processes each at prompt 1,024. The original graph takes a median
50.903 ms/decode (range 50.618–51.188), versus 45.456 ms (45.294–45.618) for
SDPA: **12.0% higher throughput**. This supports a graph-rewrite benefit beyond
file layout. These captures are not pooled with the main A/B results.

## Why this graph can be faster

Original YNNPACK already crops the matrix work to the live prefix. The rewrite
does **not** convert full-capacity matrix multiplication to live-prefix matrix
multiplication; that was primarily a limitation of the XNNPACK decomposition
in the earlier comparison.

The old YNN QK path must return a public, capacity-shaped score tensor.
`slice_like` with `YNN_NODE_FLAG_KEEP_SHAPE` crops the producer's useful input
extent but retains a padded output copy. A TFLite `SELECT_V2` then consumes and
produces capacity-shaped scores, and softmax retains that reduction width.
The subsequent PV operation crops its inputs again. These boundaries create
work outside the useful attention prefix even though QK/PV arithmetic is
already bounded.

The SDPA path carries the live extent through scores, sliced mask, softmax,
and PV inside one delegate region. It removes 35 capacity-shaped score
selection operations and their intermediate graph boundaries. The remaining
35 `SELECT_V2` operations are small output guards, retained for all-masked-row
semantics. Delegate partitions fall from **106 to 71**.

At capacity 8,448, the removed 35 score-selection outputs collectively contain
`35 × 8 × 8448` FP32 elements, or **9.46 MB of logical output stores per decode
call**. This is a shape-based calculation, not measured DRAM traffic: caches,
vectorization, branch behavior, packing and compiler scheduling determine
physical traffic. At a 1,024-token prompt, the original softmax width is about
eight times the mean useful prefix during the 64 forced calls.

As the live length approaches capacity, replacing `C` with `L` saves less
softmax work. Meanwhile, the unused older positions in local layers continue
growing because their lower bound is unchanged. These mechanisms make a
smaller benefit at long histories plausible. TECNO/Pixel's overlapping
4,096-token ranges and Samsung's different endpoint frequency caps do not
isolate those mechanisms or fully explain Samsung's observed regression.

The large low-bit weight matrix operations remain unchanged. The same YNN
mixed-matmul helper still dynamically quantizes Q and probabilities and uses
INT8 K/V. This explains why the expected benefit is an incremental whole-model
gain, rather than elimination of the entire historical native-versus-LM gap.

See the [source audit][source-audit] for actual delegate and YNN source paths,
including the distinction between logical slicing and physical copies. Full
capacity masks and cache backing storage remain; some conversion and guard
work still crosses delegate boundaries. Local attention still processes the
prefix before applying its mask, rather than using a physical lower bound.

## Remaining gap to the native runner

A separate fresh comparison is recorded in [native-comparison.md][native-comparison].
It alternates native and rewritten YNN in native/SDPA, SDPA/native,
native/SDPA order, at the same 1,024-token prompt, 8,448-token capacity,
64 forced inputs, thread count and per-phone affinity. The native binary is
the preserved reference, SHA256
`fbe9897ce1e6bf05ca00536077dbad255b57204d4bf9d5ce666bb38064df6f74`.
Its exported-weight manifest identifies the same `ab7838...` official source
bundle. Both configurations use INT8 KV and preserve compact INT2 weights.

Fresh 1,024-token comparisons are complete, three processes per runner:

| Phone | Native ms/decode [range] | SDPA ms/decode [range] | Native / SDPA tokens/s | Native throughput advantage |
| --- | --- | --- | --- | ---: |
| TECNO | 39.681 [39.449–39.961] | 45.004 [44.856–47.541] | 25.20 / 22.22 | 13.4% |
| Pixel 8 | 51.041 [50.095–51.903] | 57.805 [57.527–58.680] | 19.59 / 17.30 | 13.3% |
| Samsung SM-S937U1 | 24.071 [22.851–27.168] | 36.587 [36.567–39.404] | 41.54 / 27.33 | 52.0% |

![Fresh native versus YNN SDPA decode medians and observed process ranges](/data/bt/tmp/gemma4-ynn-sdpa-export-20260916/perf/native-vs-sdpa-decode.png)

Native is faster in each phone's observed range. The graph A/B and native
comparison are separate batches, so their SDPA samples are not pooled.
Samsung has the largest residual gap and greater native variability, alongside
the different endpoint frequency caps described above. Its 52.0% advantage
is an observed complete-runner result, not a 52.0% equal-clock kernel advantage.
Native's own attention timer takes about 7.06 ms per decode on TECNO,
9.20 ms on Pixel, and 3.70 ms on Samsung; these are not YNN attention timings
and cannot be subtracted from YNN partition totals to assign the residual gap.

This comparison measures complete configurations, rather than graph structure
alone: the kernels, layouts, runtime code and build configurations differ.
Native uses 128-row prefill chunks; LiteRT-LM can use the model's 1,024-row
prefill signature. Decode timers exclude prompt processing, but post-prefill
temperature/frequency state can still differ. The earlier compiler control
found only a small effect from changing LM's general optimization level; that
does not establish equality of the two toolchains or isolate every residual
millisecond.

Native's token-major K/V views also differ from the bundle's capacity-strided
transposed V. The earlier equal-width mobile probe did not establish that
token-major V is intrinsically faster: TECNO often favored transposed V at
equal width. Its demonstrated value here is enabling compact live/window
views and efficient appends. Layout alone is not an established explanation
for the remaining gap.

For the attention domains specifically, let `C` be capacity, `L` the live
prefix length, and `W` the valid local window (at most 512 tokens). This is
the source-level distinction, before kernel tiling and packing:

| Work domain | Original YNN | YNN SDPA | Native |
| --- | --- | --- | --- |
| Global QK/PV | `L` | `L` | Aligned `L` |
| Global softmax | `C` | `L` | Aligned `L` |
| Local QK/PV | `L` | `L` | Aligned `W` |
| Local softmax | `C` | `L` | Aligned `W` |

Native aligns views to 32 positions and retains masking, so its steady local
extent is typically 512–544 rather than exactly 512. Cache allocation remains
`C` for all runners. The table describes work domains, not measured DRAM reads.

The remaining structural differences and attribution limits are:

1. **Local-window bounds.** Native restricts local K/V reads and arithmetic
   to the useful window with its alignment padding. YNN SDPA still processes
   the complete live prefix for all 28 local layers. This becomes more costly
   as the history grows beyond the 512-token local window. Global attention
   must still see the full live prefix. The prior source-derived count for
   this 1,024 + 64 workload is 181.75 million QK/PV MACs per call for YNN,
   versus 123.75 million for native, including native's alignment padding.
   SDPA does not remove that difference. These are logical arithmetic spans,
   not physical instruction counts or a predicted latency ratio; activation
   precision and kernels differ, and most weight-matrix work is outside
   attention.
2. **Different attention kernels and activation arithmetic.** Native uses its
   FP32/QC8 attention route; YNN dynamically quantizes Q and probabilities for
   INT8 matrix multiplication. Reduction, quantization, packing and matrix
   kernel costs differ. Source inspection alone does not determine the net
   latency of this tradeoff.
3. **Remaining runtime boundaries and scheduling.** Seventy-one YNN partitions,
   35 small output guards and other fallback operations remain. Their count
   is not a measured interpreter-overhead budget. Native component timers and
   YNN partition profiles have different scopes and must not be subtracted
   as if they were equivalent stage measurements.

The old native-versus-LM gap cannot be attributed simply to a newer XNN kernel,
exclusive native INT2 support, or eight redundant KV copies. Those hypotheses
were narrowed or rejected in the [earlier analysis](native_mobile_decode_analysis.md).

An implementation lead for a later experiment is YNN's existing static slice:
negative `begin` offsets are relative to the input's symbolic extent and are
clamped. After the existing live-prefix crop, an end-relative slice could
represent a bounded local window. That would require aligning the K, V and
mask starts, verifying exact token/window semantics and decode bounds, and
handling prefill's per-row windows separately. It has **not** been implemented
or measured in this experiment.

## Diagnostic profiles and correctness scope

Profiles are collected separately after warmup, covering exactly the 64
forced decode calls. They are excluded from benchmark throughput. YNN profile
events represent whole partitions, not individual internal kernels; changes
in partition totals cannot be attributed entirely to one matrix operation.

The host and all three phone diagnostics confirm 106 → 71 YNN partitions
and 70 → 35 TFLite `SELECT_V2` nodes per call. The mobile profiles show:

| Phone | CPU SELECT ms/call, original → SDPA | YNN partitions ms/call, original → SDPA |
| --- | --- | --- |
| TECNO | 2.101 → 0.138 | 48.045 → 43.952 |
| Pixel 8 | 1.758 → 0.116 | 60.407 → 55.778 |
| Samsung SM-S937U1 | 1.306 → 0.043 | 27.955 → 29.118 |

These directly support removal of the large CPU score selections, retaining
the small guards. Partition changes combine internal compute and scheduling;
they are not an isolated softmax measurement. The [profile summary][profiles]
holds all eight diagnostic captures and display-rounding bounds.

Samsung's diagnostic pair does not reproduce its unprofiled throughput gain:
inclusive decode is 30.569 → 30.703 ms/call. The original profile ended with
CPU 4/6 caps at 3.5328/4.4736 GHz and SKIN 35.3°C; the SDPA profile ended at
2.7456/3.072 GHz and SKIN 36.5°C. Both differ from the primary unprofiled
cohort's endpoint caps. The SELECT reduction and partition counts are clear,
but these diagnostic absolute times cannot replace the unprofiled samples
or establish the graph's equal-frequency latency effect.

Original versus candidate produced bit-identical complete 262,144-element
logits on six checked outputs at capacity 2,048 under YNN on the host, TECNO,
Pixel 8, and Samsung. The host/TECNO/Pixel checks preceded benchmarking;
Samsung's identical short check passed after its timing and profile captures.
Host XNN fallback and serialization-control comparisons were also bit-identical.
A debugger counted 35 calls to the actual SDPA lowering routine.
The converter's 91 focused tests passed. These are short smoke cases, not a
full model-quality or long-context qualification.

All 65 saved argmax positions agree between original and SDPA within each
phone/prompt cell across the 42 unprofiled processes. This is supplementary
longer-history evidence, not a comparison of their complete logit vectors.

An existing YNN-versus-XNN numerical discrepancy remains independent of the
rewrite. The original and candidate YNN outputs agree in the smoke cases, but
that does not establish equivalence with XNN or the native runner. Performance
benefits therefore do not imply readiness to replace the accepted backend.

## Artifacts and reproduction

The experiment directory is
`/data/bt/tmp/gemma4-ynn-sdpa-export-20260916/`.

| Artifact | SHA256 |
| --- | --- |
| Original `.litertlm` | `ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42` |
| Rewritten `.litertlm` | `abb1609d675dd6067e204e3bf5d63f77852d5c89a8a7605b52315b30f27f146b` |
| Same Android binary for both variants | `03cef58ba5f35c42b7d786f3c226c25fd3146936609efde5795c6650c8ee3fc8` |

`runtime/` preserves build commands and source provenance. It builds
`//local_decode_profile:adapter` from the retained LiteRT-LM checkout, overriding
LiteRT to the current local source tree containing INT8 SDPA support.
`runtime-validation/` holds full-logit comparisons. `perf/tecno/`,
`perf/pixel8/`, and `perf/samsung_s25/` hold controllers, exact commands, asset
hashes, telemetry and all individual captures; their `profiles/` subdirectories
contain diagnostics.
`perf/analyze.py --require-complete` validates histories/settings and produces
the combined timing summary. `perf/final-validation.json` records the final
capture counts, Samsung full-vector checks, and documentation/link validation.
No performance number in the older reports has
been overwritten or pooled into this experiment.

[converter-notes]: /home/chasun/src/experimental/users/changmingsun/litert-converter/docs/models/gemma4_ynn_sdpa.md
[comparison]: /data/bt/tmp/gemma4-ynn-sdpa-export-20260916/perf/comparison.md
[comparison-json]: /data/bt/tmp/gemma4-ynn-sdpa-export-20260916/perf/comparison.json
[source-audit]: /data/bt/tmp/gemma4-ynn-sdpa-export-20260916/perf/source-audit.md
[profiles]: /data/bt/tmp/gemma4-ynn-sdpa-export-20260916/perf/profile-summary.md
[native-comparison]: /data/bt/tmp/gemma4-ynn-sdpa-export-20260916/perf/native-comparison.md
[historical-performance]: /data/bt/os/llama.cpp/tmp_models/gemma4_qat_q4_0_perf_graph_compare_20260914/ynnpack-20260915T070241Z/PERFORMANCE_REPORT.md
[historical-execution]: /data/bt/os/llama.cpp/tmp_models/gemma4_qat_q4_0_perf_graph_compare_20260914/ynnpack-20260915T070241Z/YNN_EXECUTION_ANALYSIS.md
[historical-capacity]: /data/bt/os/llama.cpp/tmp_models/gemma4_qat_q4_0_perf_graph_compare_20260914/ynnpack-20260915T070241Z/CAPACITY_CONTROL.md
[historical-kernels]: /data/bt/os/llama.cpp/tmp_models/gemma4_qat_q4_0_perf_graph_compare_20260914/ynnpack-20260915T070241Z/SAMPLED_KERNEL_ANALYSIS.md
[frequency-audit]: /data/bt/tmp/gemma4-ynn-sdpa-export-20260916/perf/samsung_s25/endpoint-frequency-audit.json
[historical-provenance]: /data/bt/tmp/gemma4-ynn-sdpa-export-20260916/perf/historical-provenance.md
