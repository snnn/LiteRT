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

# Improving the Gemma4 native runner

## Decision and scope

Start with **the lifetime of weights used only during packing**. The experiment
below releases 366 MiB of identified buffers without changing a single logit or
KV-cache byte. At a 1,024-token prompt and capacity 8,448, final-decode RSS falls
by about 367 MiB and process peak by 255 MiB. This is an additional improvement
over the native configuration that already shares scratch and preserves INT2.

The second experiment skips stage reshapes when external input shapes have not
changed. It preserves every tested output and reduces decode latency by about
0.6–1.0% in the paired desktop measurements. This is a small improvement;
the completed [post-attention profile](gemma4_native_mlp_profile.md) identifies
MLP FCs at 13.86 ms per decode as the next target for a larger speed gain.

These are Linux x86-64 experiments from 2026-09-23. The patch remains isolated
under [native_runner_lifetimes](../experiments/native_runner_lifetimes/README.md);
production runner sources and defaults have not been changed. The prior
[Linux native/LM-XNNPACK comparison](gemma4_linux_xnnpack_memory.md) supplies the
matched assets and baseline configuration. The broader
[LiteRT attention/runtime proposal](prefill_attention_runtime_proposal.md)
and [allocation-owner audit](gemma4_memory_allocation_audit.md) explain the
earlier comparisons, including the separate LM-YNN phone measurements.

## Plan and the experiments performed

The optimized runner already retains just 8.75 MiB of shared XNNPACK scratch on
this desktop for this workload. Reducing that allocation again cannot explain
or deliver another large memory reduction. Inspection identified larger owners:

| Retained allocation at last decode | Baseline MiB | Initial action |
| --- | ---: | --- |
| Packed weights | 751.875 | Keep: kernels read these during inference |
| Compact INT2 MLP source weights | 270 | Test release after every consumer has packed |
| Head's INT2 conversion buffer | 96 | Test release after packing the head |
| External stage outputs | 48.035 | Later: reuse storage using cross-stage lifetimes |
| Persistent INT8 KV | 74.25 | Later: bound local-attention history |
| Shared XNNPACK workspace | 8.750 | Keep current sharing; revisit for longer prefill |

The controlled experiment has four configurations: unchanged rebuilt baseline,
source release only, shape caching only, and both together. All use the same
packed-weight kernels, bundle, threads, prompt/decode inputs, chunk size, and
cache capacity. A separate short comparison checks the rebuilt baseline
against the previously audited executable.

### Release sources after packing

The runner currently keeps the [original compact MLP weights](../examples/gemma4/native/model/helpers/static_int2_fully_connected.h)
and the [head's conversion vector](../examples/gemma4/native/model/helpers/bundle_matched_ops.h)
alongside XNNPACK's packed copies. Inspection of the pinned
XNNPACK `fully-connected.c` confirms that static FC weights are consumed when
the operator is created; setup uses its packed kernel. Execution constants
used by other operators have a different lifetime and must remain live.

The probe waits until both prefill and decode signatures have compiled and the
shared weights cache is hard-finalized. It then:

1. Checks that every matched runtime value is a borrowed static QC2 kernel
   consumed only as the weight input of a fully connected operator.
2. Removes the corresponding source references from the compiled stage graphs
   and original tensor handles, and clears stale runtime/subgraph source-data
   pointers while retaining shapes and quantization metadata.
3. Releases the head's verified 96 MiB conversion vector and checks that weak
   references to all 60 compact source buffers have expired.

The 60 buffers contain 283,115,520 payload bytes plus 960 bytes of allocation
padding. The head source contains 100,663,296 bytes. No allocator trim, page
discard request, or kernel/cache change is used. Buffer ownership, rather than
only a reduction in bookkeeping counters, is verified.

This transition is deliberately one-way: another compilation would need the
sources reloaded. The probe requires retained runtimes and compact INT2. A
production version should explicitly mark packing-only buffers and release
them after their last compilation consumer, with a defined reload/rebuild path.
It should not infer this property merely from a tensor being constant.

### Skip unchanged stage shapes

The [common runner](../runners/common_nnpack/runner.cc) currently calls external-input reshape and runtime reshape
on every invocation. Most projection/post/head shapes are fixed, and attention
extents often stay fixed for multiple decode steps because they are aligned to
32 rows.

The probe caches each external input shape and reshapes the runtime only after
a shape change, including its first invocation. It always binds current input
and output pointers and invokes setup. This distinction matters when the local
KV view advances but retains the same shape. The pinned XNNPACK workspace
allocator already repairs all attached runtimes' pointers/setup when its shared
arena moves.

This is a narrow experiment: the common runner still queries output shapes,
reserves buffers, constructs binding/lock vectors, and creates input wrappers.
It does not measure an implementation that removes all wrapper allocations.

## Memory results

Workload: 1,024 prompt tokens, 64 forced decode inputs, capacity 8,448,
prefill chunk 128, KV alignment 32, four threads pinned to CPUs `0,2,4,6`, one
warmup and one measured session with the same compiled runtimes.

| Configuration | Process peak RSS, MiB | Final-decode RSS, MiB | Final-decode anonymous memory, MiB |
| --- | ---: | ---: | ---: |
| Optimized baseline | 2,051.29 | 2,043.45 | 1,438.35 |
| Release packing sources | 1,795.90 | 1,676.29 | 1,071.34 |
| Cache stage shapes | 2,051.18 | 2,030.68 | 1,438.36 |
| Both | 1,796.01 | 1,676.33 | 1,071.35 |

Source release saves **255.39 MiB of process peak (12.45%)** and
**367.15 MiB of final-decode RSS (17.97%)** in this capture. The directly
identified released payload is **366 MiB**; the small difference in RSS also
includes allocator/residency effects. Immediately before/after release in that
process, RSS changes from 1,796.19 to 1,429.97 MiB.

Peak falls by less than the retained-memory saving because all source weights
still coexist during compilation. Moving release earlier could lower startup
peak further, but requires last-consumer accounting across signatures and
correct weight-cache key lifetimes; this experiment does not establish that
additional saving.

Packed-weight capacity remains exactly 788,398,080 bytes, shared workspace
9,175,072 bytes, stage outputs 50,368,512 bytes, and logical KV 77,856,768 bytes.
The source-release configuration's compact source and graph-constant counters
both become zero. Source file mappings other than these compact buffers remain
owned. These results do not claim to eliminate all duplicate model storage.

The shape-only configuration has essentially unchanged anonymous memory and
allocation counters. Its lower current RSS comes from file-backed residency;
it is not evidence of a new allocation saving. No measured process swapped.
Peak uses child `wait4` accounting, while current RSS uses `smaps_rollup`; these
are different OS counters and should not be combined into an exact partition.

## Correctness

All three probe configurations are **bit-identical** to the rebuilt baseline
for 129 complete vocabulary-logit vectors (33,816,576 FP32 values per
configuration) and every dumped owner's committed INT8 K/V. Each configuration
compares 3,999 files totaling 1,186,434,048 bytes. Input histories and argmaxes
also match for warmup and measured sessions. The runner rejects non-finite
logits before accepting a pass.

The ten prompt lengths are 8, 128, 1,024, 4,096, 8,192, 127, 129, 511, 512,
and 513, in that order. This exercises partial chunks, alignment boundaries,
local-window boundaries, workspace growth, and long-to-short resets in a reused
runtime. The 1,024-token case includes 64 forced tokens; the short case includes
33, crossing a KV-alignment boundary. The archived original executable also
matches the rebuilt baseline and all variants in the separate 34-pass short
test, including all logit and cache bytes.

In the full validation workload, shape caching performs 2,474 runtime reshapes
and skips 35,692: **93.52% fewer calls**. This count includes prefill, decode,
warmup, and measured sessions; it is not a decode-only statistic.

## Latency results

Memory reports and logit/cache dumps are disabled for these runs. Each process
warms up each prompt length, then measures three sessions of prefill plus 64
forced tokens. There are three process blocks, with configuration order
reversed in the middle block: nine measured sessions per configuration and
prompt length. All timing runs' input histories and predictions match.

The table gives the median of three process medians, in milliseconds. Within a
session, decode is the mean of its 64 forced-token steps. The prefill interval
ends at first logits and includes creating/zeroing the session KV bank; it
excludes model loading and compilation.

| Configuration | 128 prefill | 128 decode/token | 1,024 prefill | 1,024 decode/token | 4,096 prefill | 4,096 decode/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 151.92 | 23.79 | 1,040.30 | 25.23 | 4,612.41 | 27.98 |
| Release sources | 139.17 | 23.79 | 1,029.66 | 25.23 | 4,610.20 | 27.95 |
| Cache shapes | 151.31 | 23.56 | 1,045.07 | 24.99 | 4,610.76 | 27.75 |
| Both | 138.40 | 23.59 | 1,023.12 | 25.04 | 4,597.50 | 27.82 |

Changes below use the median of the three **paired within-block percentage
changes**, rather than dividing the independently aggregated table entries:

- Shape caching lowers decode latency by **0.95%, 0.97%, and 0.62%** at prompts
  128, 1,024, and 4,096 respectively. All nine block/context pairs improve;
  individual improvements range from 0.58% to 1.26%. Prefill changes are small
  and inconsistent. Removing 93.52% of reshape calls therefore removes only a
  small fraction of this runner's total work.
- Source release lowers first-logit time by **8.39% at 128 tokens** and **1.33%
  at 1,024**. At 4,096 the paired median is 0.05%, and individual blocks change
  sign. Decode changes also change sign and are within about 0.8%; there is no
  consistent decode-speed benefit from source release alone.
- Combining both gives paired median first-logit reductions of **8.75%, 1.74%,
  and 0.31%**, and decode reductions of **0.61%, 0.77%, and 0.41%**. The small
  decode effects are not additive in these measurements.

The source-release short-prompt effect occurs principally **outside stage
`Run()` calls**: at 128 tokens, that interval falls from 21.31 to 8.21 ms.
It includes KV allocation/zeroing, embedding lookup, mask construction, binding,
and logits checking. This is consistent with a host-memory/allocator effect;
the exact mechanism within that interval has not been isolated. It does not
establish faster GEMM kernels, and a serving path that already reuses its KV
allocation might see a different first-logit effect.

The existing stage timers also locate the larger decode costs:

| Prompt length | Whole decode step, ms | Post-attention stages, ms | Projection stages, ms | Attention stages, ms |
| --- | ---: | ---: | ---: | ---: |
| 128 | 23.79 | 16.93 | 2.37 | 0.70 |
| 1,024 | 25.23 | 17.04 | 2.39 | 1.98 |
| 4,096 | 27.98 | 17.20 | 2.42 | 4.48 |

Other work includes preprocessing, the output head, and host operations.
Component medians are independently aggregated and are not an exact partition
of the whole-step median. Post-attention stages contain the MLPs; these timers
do not isolate their FC kernels from other operations within the stage.
The subsequent [operator and kernel profile](gemma4_native_mlp_profile.md)
does: 16.42 ms of a measured 17.07 ms post-stage interval is in FC operators,
including 13.86 ms in the three MLP projections. Quantize/dequantize conversions
total 0.17 ms and GELU 0.08 ms; the large FCs use AVX-VNNI on this desktop.
At prompt 1,024, attention stage calls account for about 7.8% of decode time;
even removing that entire measured interval would save only about 7.8% of step
latency. At prompt 4,096 the share rises to 16.0%. This supports profiling
post-attention FC/weight traffic for decode speed while evaluating tiled
attention for longer-context workloads.

These are small-sample observations on one desktop. They are not confidence
intervals, Android results, or predictions of universal speedups.

## Further improvements, in order

These are priorities for subsequent implementation/measurement, not additional
measured gains from the experiments above.

1. **Make packing-only ownership explicit.** Turn the successful release probe
   into a supported lifetime contract. Then audit other already-packed FC
   sources, starting with the head's original widened INT4 input. Its exported
   payload is 192 MiB, but that is not a promise of 192 MiB additional RSS
   reduction: check mapping owners, resident pages, and embedding aliases.
   Releasing before the final compilation needs a separate peak-memory test.
   Also import original INT2 coefficients directly. The current
   [compact-weight loader](../examples/gemma4/native/model/helpers/static_int2_fully_connected.h)
   reconstructs them from widened INT4 data. Avoiding that representation
   round trip targets model loading, temporary storage, and peak residency;
   it does not reduce the execution weights' packed size. Preserve provenance,
   scales, and the exact coefficient layout.

2. **Optimize the FC path identified by the completed profile.**
   The [follow-up measurements](gemma4_native_mlp_profile.md) attribute 96.2%
   of post-stage time to FC operators. Next isolate packed-weight delivery,
   low-bit unpacking/dot products, and thread scheduling at the measured M=1
   dimensions. The current measurements do not establish which of those
   limits throughput. Source release does not
   reduce the 751.875 MiB packed-weight allocation or any matrix's packed
   coefficients.
   Small runner-overhead savings and attention-only improvements cannot explain
   a large reduction in the whole decode step at the tested short contexts.

3. **Remove repeated host preparation.** Reuse per-token embedding/position
   buffers and stage binding storage. Cache causal/local masks per distinct
   chunk/context extent instead of constructing equivalent masks for each
   layer. Measure allocations and time outside `xnn_invoke_runtime`, with
   output-pointer changes and input lifetimes covered by tests. Also measure
   retaining/resetting the KV bank between sessions: the current first-logit
   timer includes allocating and zeroing its full 74.25 MiB capacity. Shape
   caching alone does not remove these costs.

4. **Share packed dynamic KV among layers that share an owner.** There are 35
   attention layers but 15 KV owners. Layers using the same K/V still have
   different queries, so attention outputs cannot be reused. However, the
   current dynamic BMM path packs its K/V operands again for each invocation.
   A cache keyed by KV owner, version, range, layout, and kernel format could
   avoid repeated packing. Incremental updates could then pack only appended
   rows. Measure the packing fraction first and account for the new persistent
   packed-KV storage; a faster path might consume more memory.

5. **Use a bounded local KV ring.** At capacity 8,448, the 12 local owners use
   49.5 MiB even though their attention window is 512. Keeping global owners at
   full capacity and local owners at 512 rows would give an ideal 27.75 MiB
   total KV allocation. Chunked prefill needs a larger live union of old and
   new rows plus alignment; roughly 28–29 MiB total is a more useful initial
   target for chunk 128. This is a storage estimate, not a tested saving.
   Preserve absolute RoPE positions, wrapped views, pending writes, rollback,
   and any full-history diagnostic contract.

6. **Reuse external outputs between stages.** Shared XNNPACK scratch does not
   include the 48.035 MiB of separately owned stage outputs. A caller-level
   lifetime plan can reuse hidden/query/attention buffers across serial layers,
   while retaining residuals and per-layer embeddings until their consumers
   finish. Do not assume all 48 MiB can disappear simultaneously.

7. **Add tiled attention where attention measurements justify it.** Online
   softmax avoids materializing the full score/probability matrix and reduces
   memory traffic at long contexts. It composes with chunked prefill and active
   KV extents; it does not replace batching the projection/MLP work. For this
   1,024-token workload, all shared scratch is already only 8.75 MiB, so expect
   the main motivation to be attention speed and longer-context scaling, not
   another hundreds-of-MiB saving here. Validate INT8 KV scales, causal/local
   masks, FP32 accumulation/softmax, and full logits before claiming parity.

8. **Tune prefill shape classes per device.** Chunk 128 is a measured
   configuration, not a universal optimum. Dynamic shapes can reduce the
   number of authored signatures, while chunk size still controls GEMM reuse,
   attention working set, and temporary memory. Compare a few batch sizes
   without retaining a separate large output/workspace allocation for each.

9. **Include the output head in FC optimization work.** The vocabulary head
   is outside the 13.86 ms MLP interval. Its dynamically quantized INT2 kernel
   accounts for about 11.55% of sampled user cycles in the whole decode.
   Measure its wall time separately and tune its distinct matrix shape and
   dynamic-activation path. This is a CPU-cycle share, not an 11.55% latency
   saving. Fusing selection with the head would not eliminate the need to
   compute all vocabulary scores for exact general inference.

10. **Explore small-batch speculative verification for weight reuse.**
    A draft sequence lets the target process several candidate positions in
    one batch, potentially amortizing FC weight reads over accepted tokens.
    The correct speculative sampling algorithm can preserve the target
    distribution; draft cost and acceptance rate determine whether it helps
    ([Leviathan et al., 2023](https://proceedings.mlr.press/v202/leviathan23a.html)).
    This runner would need a full verification graph for every candidate's
    logits, plus KV rollback/partial commit. Its existing cache-only prefill
    stops at the last KV producer and cannot serve as that verification graph
    unchanged. Batched numerical behavior also needs validation. This is a
    larger, unmeasured architectural experiment, not a demonstrated speedup.
    Batching independent sessions is another way to reuse weights, but its
    goal is aggregate throughput rather than one session's next-token latency.

Enabling existing `share_workspace` and `preserve_static_int2` settings by
default is also valuable for users of this runner, but its benefit was already
measured in the previous comparison. It must not be counted again as a gain
over the optimized baseline used here.

### Follow-up: thread count and phase-specific settings

A subsequent control tested two, four, and eight P cores on the same
i9-12900K, using the unchanged baseline executable and the same 1,024-token
prompt, 64 forced decode inputs, capacity 8,448, chunk 128, and KV alignment
32. It ran two process blocks in reversed order: `4,2,8` then `8,2,4`.
Each process discarded one warmup and measured three sessions. CPU affinity
used one logical CPU per P core: the first two/four/eight entries in
`0,2,4,6,8,10,12,14`.

| P cores / threads | Decode, ms/token | Post-attention, ms/token | Attention, ms/token | Prompt through first logits, ms |
| --- | ---: | ---: | ---: | ---: |
| 2 | 28.284 | 18.413 | 3.181 | 1,877.1 |
| 4 | 25.177 | 16.997 | 1.978 | 1,043.7 |
| 8 | 23.567 | 16.196 | 1.453 | 625.0 |

Values are medians of six session means per configuration; component medians
are independent and should not be added as an exact partition. Prompt timing
includes KV allocation/reset work and the final prompt token's full forward
pass; it excludes model loading and compilation.

Eight cores reduce the pooled median decode time by **6.4%** and prompt
through first logits by **40.1%** compared with four cores. The decode
reductions using each block's medians are 6.5% and 6.8%. This uses twice as
many P cores; it is not a same-resource kernel improvement. The post stages
improve only about 4.7%. This demonstrates diminishing returns for decode,
without by itself proving a DRAM-bandwidth bottleneck.

The different scaling makes **separate prefill and decode thread settings**
worth exposing. The current implementation gives both signatures one shared
threadpool. A future version could borrow separate pools, while retaining
serialized execution and shared weights/workspace. Per-operator thread
thresholds are a further backend experiment, especially for small FCs;
reducing polling alone is not a guaranteed wall-time gain. This test does
not measure energy consumption or establish optimal settings on phones.

All 18 measured sessions match their token histories and 65 argmax predictions.
Separate two- and eight-thread validation runs each reproduce all 65
full-vocabulary vectors bit for bit against the previous four-thread baseline:
17,039,360 FP32 values per configuration. Diagnostic dump timings are excluded.

Artifacts and reproduction:
[thread sweep and validator](../experiments/native_mlp_profile/thread_sweep.py),
[saved results](../experiments/native_mlp_profile/thread_results.json), and
[raw commands/captures](/data/bt/tmp/gemma4-native-thread-sweep-20260923/analysis.json).
Production settings remain unchanged.

## Implications for LiteRT and the tensor runner

### LiteRT-LM already has a related source-weight reclamation path

For the CPU/XNNPACK cache configuration used in the Linux comparison, the
[weight-cache provider](../../tflite/delegates/xnnpack/weight_cache.cc)
calls `MarkMemoryNotNeeded` on registered original kernel/bias buffers after
inserting their packed representations. On Linux and Android,
[that helper](../../tflite/delegates/xnnpack/mmap_handle.cc) aligns the range
to whole pages and calls `madvise(..., MADV_PAGEOUT)`. The same call sites and
advice value were verified in the exact archived LM executable used for the
measurements; see the [binary verification record][lm-pageout-check].

This requests reclamation while keeping the model mapping and addresses valid.
Later accesses can bring pages back into RAM. The cache call site ignores the
helper's success/failure result, and this source/binary check does not measure
which individual requests succeeded or how many bytes stayed nonresident.
Residual model-mapping RSS therefore does not establish that LM never attempts
to reclaim original weights.

A subsequent [matched pageout control](gemma4_linux_xnnpack_memory.md#follow-up-effect-of-madv_pageout)
does measure the effect: all 277 actual requests succeed and suppressing only
pageout increases final-decode RSS by about **739 MiB**, with essentially no
anonymous-memory change. The effect is the same for 1,024- and 128-row prefill.
The earlier LM measurements already benefited from this mechanism.

The native experiment's **366 MiB is not a matching pair of extra heap buffers
in LM**. The original bundle already stores the MLP and head coefficients as
INT2; the delegate can pass their mapped data directly to XNNPACK. Native's
export/loader path creates the compact MLP heap buffers and head conversion
vector that this experiment frees. The measured gain fixes those native-owned
copies; it is not evidence that LM lacks original-weight reclamation or would
gain the same 366 MiB from the same patch. Both paths retain their execution
weights in packed form.

### What still needs an API contract

The successful memory experiment requires **no new mathematical operator**.
It needs an ownership/lifetime distinction between compiler inputs and buffers
read during execution. An interpreter arena cannot reclaim allocations owned
by a graph wrapper or delegate merely because tensor execution lifetimes have
ended. Backend packing must report when original coefficients can be released,
and the graph/model loader must be able to relinquish them without losing
constants required by fallback kernels, embeddings, or later compilation.

Similarly, avoiding unchanged reshapes is a runner scheduling change. Dynamic
shape support supplies the ability to reshape; it does not require reshaping
every invocation or automatically choose a good prefill batch size.

Packed-KV reuse, local-ring storage, and tiled attention need stronger backend
contracts, possibly exposed through a stateful/fused attention operation. The
gap is therefore a combination of graph semantics, persistent state/layout,
buffer ownership, scheduling, and kernels. New op names alone do not deliver
these gains.

## Evidence and limits

Raw artifacts: `/data/bt/tmp/gemma4-native-next-20260923`.
The validated summaries and executable identities are also preserved in
[results.json](../experiments/native_runner_lifetimes/results.json).

- `inputs.json` records source/archive/patch identities and compiler version;
  each build has `commands.json` with exact compile/link commands and binary hash.
- `smoke-analysis.json`, `correctness-analysis.json`, `memory-analysis.json`,
  and `timing-analysis.json` are generated only after their assertions pass.
- Each process has its exact command, executable/fixture/manifest hashes,
  completion/resource status, process log, and original runner output.
- The memory captures contain phase snapshots and allocation counters. Timing
  captures disable memory diagnostics and all logit/cache dumps.

The matched bundle's 1,394 payload checks were completed in the preceding Linux
audit. These probes reuse those immutable assets and record their manifest
identity. Both rebuilt executables link the same preserved XNNPACK/Tensor API
archives; no kernel or global build-option comparison is hidden in the result.
This is an isolated source-level experiment, not a claim that current Bazel
targets were rebuilt or that Android has been validated.

The one-way release patch uses E2B-specific knowledge and is not ready as a
generic API. The unchanged-shape cache is valid for these stages' input-shape
dependencies; a generic runner must also handle any shape-affecting state not
represented by external input dimensions. Tests cover chunk 128/alignment 32,
the preserved Linux kernels, and this model. They do not establish behavior for
other models, threading modes, or delegate versions.

[lm-pageout-check]: /data/bt/tmp/gemma4-native-next-20260923/lm-original-weight-pageout-check.json
