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

# Gemma4 native versus LiteRT-LM: allocation ownership audit

**September 23, 2026.** The large memory difference in the native versus
LM-YNN-SDPA comparison has a measured explanation: LiteRT-LM selected a
1,024-row prefill graph that materializes attention intermediates across the
full 8,448-position cache capacity. Its **687.25 MiB TFLite prefill arena stays
allocated during decode**. Native used 128-row chunks and active KV extents.

Using the same LiteRT-LM executable and model weights, forcing the existing
128-row prefill signature reduced that arena to **85.91 MiB**. It reduced
peak process RSS by **706.05 MiB** in an unmodified-binary control pair.
Allocation counters establish the cause more precisely than RSS alone.

The earlier native optimization that shared 151 private XNNPACK workspaces
does not explain this LM comparison. In the measured YNN path, temporary heap
storage is released after each invocation. TFLite's arena also already reuses
storage between tensors with non-overlapping lifetimes.

## Scope and reproducibility

This audit revisits the [saved Pixel 8 native/SDPA comparison][historical],
whose historical median peaks were 2,054.9 MiB native and 2,891.4 MiB LM-YNN-SDPA.
That 836.5 MiB historical difference is kept separate from the new runs.

The later [Linux XNNPACK comparison](gemma4_linux_xnnpack_memory.md) measures
the original bundle at the same prompt length and capacity on the desktop.
Its backend and platform differ from the YNN phone measurements in this audit.

New captures use Pixel 8 `3A111FDJH005A0`, four CPU threads, affinity `1e0`,
capacity 8,448, the same 1,024-token prompt and 64 forced continuations, one
warmup session and one measured session per process. Six processes completed:
one native capture, two LM-YNN inventories at 1,024 rows, one at 128 rows, and
two unmodified LM executable controls. These are memory diagnostics, not
latency benchmarks; no speed claim is made from their timings.

The saved LM executable has SHA256
`03cef58ba5f35c42b7d786f3c226c25fd3146936609efde5795c6650c8ee3fc8`.
Relinking the preserved objects reproduced it **byte-for-byte** before adding
diagnostic wrappers. The observation build recompiles the copied driver and
observer translation units only. It preserves all original execution-library
objects, model-execution settings, kernels, and thread pools. The
[build record][build] includes commands and hashes of the preserved inputs.

The observer records:

- Creation/destruction and identity of compiled models and XNN/YNN runtimes.
- Actual delegate identities in each interpreter's execution plans.
- `Subgraph::GetMemoryAllocInfo` arena sizes, plus large allocated tensor shapes
  and addresses. Tensor sizes are not blindly summed across reused storage.
- Unique address ranges for YNN static buffers and externally bound buffers;
  shared aliases count once.
- YNN delegated-runtime heap allocation/free callbacks, preserving the original
  callbacks and allocation policy. This measures their inference heap scratch,
  not all process allocations, stack scratch, or compiler temporaries.
- Current RSS/PSS, anonymous/file mapping groups, swap, and lifetime peak RSS.

For the 128-row control, a copy of the bundle renames `prefill_1024` to
`ignored_1024` in its signature and subgraph names. This makes the existing
static scheduler choose `prefill_128`. The [byte-diff validation][model-control]
proves that only 14 bytes across those two names changed: all tensor data,
operators, subgraphs, and section offsets remain identical. The otherwise
unused 1,024-row and verification graphs are still available for delegation,
so the experiment does not conflate smaller chunks with graph pruning.

All 1,394 native manifest/payload checksums were revalidated. Both warmup and
measured sessions preserve the exact token histories. All 65 recorded argmax
IDs match across the LM diagnostic, unmodified, 1,024-row, and 128-row cases.
Native differs at the same three pass indices, 29, 40, and 63, as in the saved
comparison. Full-logit parity was not rerun for this allocation audit.

## Actual runtime and delegate owners

| Path / owner | What was observed |
| --- | --- |
| Native `LiveRuntime` | 151 direct XNNPACK stage runtimes, one shared workspace, one packed-weight cache, and one persistent worker pool. It does not use TFLite's arena. |
| LM main transformer `CompiledModel` | One interpreter containing 1,011 subgraphs; one **YNNPACK delegate instance**; 1,654 YNN runtime objects across its delegated partitions. |
| LM token embedding `CompiledModel` | One additional interpreter and one XNNPACK delegate/runtime. |
| LM per-layer embedding `CompiledModel` | One additional interpreter and one XNNPACK delegate/runtime. |
| LM total during inference | **Three live compiled models and three active delegate instances:** one YNN and two XNN. Two additional empty compiled models were created and destroyed during setup. |

The main graph has 71 YNN partitions in decode, 66 in each prefill signature,
and 109 in verification; additional runtimes belong to composite fallback
subgraphs. Only **137 distinct YNN runtime objects** are invoked in this
workload: 66 from the selected prefill path and 71 from decode. The other
1,517 remain uninvoked. This count is not a count of independent retained
scratch arenas.

The main graph's remaining nodes execute through TFLite CPU kernels. The two
auxiliary embedding models use default embedding compilation options, which
do not propagate the main executor's YNN option. Their XNN workspaces are
separate because they belong to separate delegates, but together retain only
**6,176 bytes**. They are not a material source of this memory gap.

At executor release the observer sees zero live compiled models, YNN runtimes,
and XNN runtimes. Model resources and their mapped source data still outlive
the executor at that snapshot.

## Allocation ledger

All sizes below are MiB, with `1 MiB = 1,048,576 bytes`. Retained entries are
from after the last decode of the second session. Peaks are labeled explicitly.
The LM columns use the same observer binary.

| Allocation / metric | Native, 128 rows | LM-YNN, 1,024 rows | LM-YNN, 128 rows |
| --- | ---: | ---: | ---: |
| Persistent INT8 KV payload | 74.250 | 74.250 | 74.250 |
| Selected prefill TFLite arena | N/A | 687.252 | 85.907 |
| Decode TFLite arena | N/A | 0.474 | 0.474 |
| All TFLite arenas, including embedding models | N/A | 687.800 | 86.455 |
| Bound external buffers, excluding KV | N/A | 50.348 | 7.251 |
| Native owning stage-output buffers | 48.035 | N/A | N/A |
| Native shared XNN workspace | 9.125 | N/A | N/A |
| Auxiliary XNN workspaces, combined | N/A | 0.006 | 0.006 |
| YNN delegated heap scratch, peak | N/A | 68.000 | 8.492 |
| YNN delegated heap scratch, retained after invocation | N/A | 0 | 0 |
| Native packed-weight capacity | 751.875 | N/A | N/A |
| YNN unique static-buffer address ranges | N/A | 754.176 | 754.176 |

The weight rows identify different backend representations, not interchangeable
allocation categories. Of YNN's static ranges, 753.104 MiB is anonymous and
1.072 MiB is file-backed. Naively adding static sizes across all 1,654 runtimes
would give **2,032.640 MiB**; deduplicating their address ranges gives
**754.176 MiB**. Actual sharing is substantial. The backend's
[constant cache][ynn-constants] is consistent with this observation.

Native separately reports 96 MiB of graph constant capacity, 2.992 MiB of
graph dequantized capacity, and 270 MiB of compact source storage for the 60
INT2 MLP matrices. Its exported source buffers total 2,149.718 MiB of storage
capacity, much of which is file-backed and not resident. These numbers must
not be added to RSS as though every byte were resident or every category were
independent. The published LM bundle already stores those 60 MLP matrices as
INT2; their earlier widening was a native-adapter issue.

## Why the prefill arena is large

The prior SDPA rewrite changed **decode only**. Its
[experiment report](ynnpack_sdpa_graph_performance.md) explicitly leaves
prefill unchanged. A filename containing `sdpa` does not imply fused SDPA in
every model signature.

The observed 1,024-row prefill still exposes tensors such as:

```text
QK score output:       [1, 1, 8192, 8448] FP32 = 264 MiB
Equivalent head axes: [1, 8, 1024, 8448] FP32 = 264 MiB
Masked score output:  same logical score size
```

Here 8,192 folds eight heads and 1,024 query rows. The tensor inventory also
contains expanded masks and intermediate mask conversions. For 128 query
rows, the corresponding score tensor is 33 MiB. Both still use capacity
8,448 for the history axis, even when useful history is shorter.

YNN's `runtime_bmm` lowering can limit useful matrix-multiply work internally
while its TFLite-visible output retains the full-capacity shape. Subsequent
masking and softmax boundaries keep those large tensors visible to the arena.
This distinction is already described in the
[attention proposal](prefill_attention_runtime_proposal.md#3-dynamic-shapes-runtime-bmm-and-ynnpack).

The arena is already reusing addresses across layers. It does not allocate a
separate 264 MiB buffer for every logical score tensor. Its 687.25 MiB capacity
is the retained allocation for the selected graph's planned lifetimes.
Scratch reuse cannot eliminate values while graph consumers still need them.

The prefill arena remains 687.25 MiB after first logits, later decode calls,
and executor reset. The decode arena itself is only 0.474 MiB. After the
128-row control, the unused 1,024-row arena stays at zero and the 128-row arena
remains at 85.91 MiB.

## Controlled attribution of the chunk-size effect

Changing only prefill selection produces these allocation reductions:

| Quantity | 1,024 rows minus 128 rows |
| --- | ---: |
| Retained TFLite arena capacity | 601.345 MiB |
| Retained external-buffer address ranges | 43.097 MiB |
| Combined retained-capacity difference | **644.443 MiB** |
| Observed anonymous current-RSS difference at last decode | **644.441 MiB** |
| YNN inference heap-scratch peak difference | 59.508 MiB |
| Sum of those three allocation-capacity differences | 703.950 MiB |

The close agreement of anonymous RSS with the retained-capacity difference
is strong evidence for the attribution. KV, static-buffer ranges, delegate
counts, runtime counts, and weights are unchanged.

An independent pair using the **unmodified saved LM executable** reports:

| Selected prefill | Lifetime peak RSS |
| --- | ---: |
| 1,024 rows | 3,051.414 MiB |
| 128 rows | 2,345.359 MiB |
| Difference | **706.055 MiB** |

This is consistent with the roughly 704 MiB allocation reduction. Allocation
capacities and process RSS remain different measurements; the arithmetic is
supporting evidence, not a rule for converting one to the other. These are
single-process controls, not a new estimate of the historical median gap.

## Residual difference after matching chunk sizes

The new native capture peaks at 2,058.297 MiB. Matching prefill rows therefore
does not make LM and native memory identical. At the matched last-decode phase,
the diagnostic snapshots show:

| Current memory | Native, 128 rows | LM-YNN, 128 rows | LM minus native |
| --- | ---: | ---: | ---: |
| RSS | 2,049.434 | 2,329.664 | +280.230 MiB |
| Anonymous | 1,448.629 | 1,219.504 | -229.125 MiB |
| Non-anonymous remainder, primarily file-backed | 600.805 | 1,110.160 | +509.355 MiB |

The model-file mapping groups account for 596.043 MiB RSS in native and
1,078.805 MiB in LM. Source representation, mapped model resources, loaded
code, and other retained allocations must remain in the accounting. This
residual cannot be described as 280 MiB of extra private YNN scratch: the
measured retained YNN inference scratch is zero, and LM uses less anonymous
memory in this comparison. The mapping groups do not identify each resident
page with an individual weight tensor; that finer attribution is not claimed.

Two 1,024-row LM diagnostic processes have identical arena/static capacities
but current RSS of 2,980.676 and 2,893.430 MiB. Their anonymous values differ
by only 0.402 MiB; the bundle-mapping RSS differs by 86.375 MiB. Thus mapped-page
residency materially affects the reported process total. All these snapshots
report zero swap. The audit does not assign an exact causal decomposition to
the older 836.5 MiB peak difference by subtracting measurements from different
processes or dates.

The older September 13 audit is a different comparison again: its saved
1,024-token LM run used **capacity 2,048 and two CPU threads**, with the older
XNNPACK-based executable. Its current-RSS results cannot substitute for the
four-thread, capacity-8,448 YNN comparison audited here.

## Implications for LiteRT changes

1. **Expose and exercise prefill policy.** The measured LM stack can already
   chunk prefill; its static scheduler chose the larger available signature.
   Selecting the existing smaller signature produced a large memory saving
   without adding an operator or changing kernels.
2. **Apply bounded/fused attention to prefill as well as decode.** Keep full
   capacity allocation separate from live attention extent, and avoid exposing
   capacity-shaped score/mask intermediates at delegate boundaries. Existing
   SDPA semantics and lowering are a starting point; online attention can
   reduce internal intermediates further.
3. **Address prefill allocation lifetime during decode.** Releasing or safely
   reusing an idle prefill arena can reduce retained decode memory. Releasing
   it after prefill alone cannot lower the peak that already occurred during
   prefill. External prefill buffers have a separate owner and lifetime.
4. **Audit unused graph preparation separately.** Preparing 1,654 runtimes
   when only 137 are invoked is a concrete initialization/metadata target.
   Constants already share backing storage and uninvoked prefill/verification
   arenas remain unallocated, so do not count each runtime as another private
   scratch arena or another full weight copy.

The principal measured gap is the chosen prefill shape, the exposed attention
intermediates, and retained graph/buffer lifetimes. A new `shared_scratch`
operator would not address these findings.

## Evidence and implementation references

- [Machine-readable summary and validation][summary].
- [Observation source][observer], [build/relink recipe][build-script], and
  [device capture controller][capture-script]. Diagnostic sources and commands
  are local artifacts; production runtime source was not changed for this audit.
- [Native phase snapshots][native-memory], [1,024-row LM inventory][lm-inventory],
  and [128-row LM inventory][small-inventory].
- [Native asset checksum validation][native-validation].
- [Native ownership](../examples/gemma4/native/stage_runner.h) and
  [stage construction](../examples/gemma4/native/driver.cc).
- [TFLite arena accounting](../../tflite/core/subgraph.cc),
  [XNN delegate workspace sharing](../../tflite/delegates/xnnpack/xnnpack_delegate.cc),
  and [YNN delegate runtime creation](../../tflite/delegates/ynnpack/ynnpack_delegate.cc).
- The archived YNN [invoke implementation][ynn-runtime] releases its Slinky
  pool after each invocation. Its [constant cache][ynn-constants] shares
  prepared constants across subgraphs. These source references are from the
  dependency checkout used by the preserved executable.

[historical]: /data/bt/tmp/gemma4-ynn-sdpa-export-20260916/perf/pixel8/REPORT.md
[summary]: /data/bt/tmp/gemma4-allocation-audit-20260923/summary.json
[build]: /data/bt/tmp/gemma4-allocation-audit-20260923/build.json
[model-control]: /data/bt/tmp/gemma4-allocation-audit-20260923/prefill128-control.json
[observer]: /data/bt/tmp/gemma4-allocation-audit-20260923/allocation_audit.cc
[build-script]: /data/bt/tmp/gemma4-allocation-audit-20260923/prepare_build.py
[capture-script]: /data/bt/tmp/gemma4-allocation-audit-20260923/run_capture.py
[native-memory]: /data/bt/tmp/gemma4-allocation-audit-20260923/native-r0/memory.jsonl
[lm-inventory]: /data/bt/tmp/gemma4-allocation-audit-20260923/ynn-r1/allocations.jsonl
[small-inventory]: /data/bt/tmp/gemma4-allocation-audit-20260923/ynn128-r0/allocations.jsonl
[native-validation]: /data/bt/tmp/gemma4-allocation-audit-20260923/native-payload-validation.json
[ynn-runtime]: /data/bt/tmp/gemma4-pr9918/LiteRT-LM/.bazel-output/external/XNNPACK/ynnpack/subgraph/runtime.cc
[ynn-constants]: /data/bt/tmp/gemma4-pr9918/LiteRT-LM/.bazel-output/external/XNNPACK/ynnpack/subgraph/subgraph.cc
