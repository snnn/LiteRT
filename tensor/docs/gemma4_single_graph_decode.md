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

# Can native Gemma4 decode run without host intervention between layers?

**Yes, for the tested decode path.** An isolated variant of the current native
runner executes all 35 layers and the vocabulary head in **one XNNPACK runtime
invocation per token**, replacing 107 staged invocations. It reproduces all
tested full logits and committed KV bytes exactly. At a 1,024-token prompt,
median decode is **25.52 ms versus 25.12 ms** for staged execution on the
audited desktop: similar performance, with the joined version 1.6% slower by
this aggregate. This is evidence that per-layer host intervention is not
required for this workload, not an optimization-speedup claim.

The original staged runner alone could not establish that result. This new
experiment supplies the missing control. It uses ordinary XNNPACK primitives,
the same archived kernels, and no new SDPA operation. Production sources and
defaults are unchanged. MTP is out of scope.

## What moved into the graph

The staged runner alternates graph calls and C++ work:

```text
projection → return to C++ → append owner's new K/V
           → bind active cache views → attention → post-attention → next layer
```

The joined variant retains graph edges across preprocessing, projections,
attention, post-attention, and the head. For each of the 15 KV owners, it
constructs an active cache tensor using:

```text
past KV view ──────────────┐
new KV from projection ───┼─ concatenate → attention → remaining graph
alignment padding ────────┘
           │
           └─ new KV is also a graph output for the final host commit
```

All consumers of a shared owner reference that owner's concatenated K/V
tensors. Graph dependencies ensure each current token's new K/V is available
before its attention consumers. C++ binds all old views, positions, and masks
before the call. After the entire graph finishes, it copies only the 15 owners'
new rows into the persistent bank and commits the new length.

Thus there is **no C++ intervention between layers**. There is still host-side
input preparation, including loops over input bindings/masks, and a cache
commit after each complete invocation. The experiment does not claim to remove
all model-specific C++ or put persistent in-place cache mutation inside XNNPACK.
Moving the commit after the graph avoids depending on an unexpressed alias
between an external cache input and an intermediate output.

This is a functional dataflow construction using `Concatenation`, not a new
cache-update kernel, custom callback, or a list of hidden sub-runtime calls.
It preserves static INT2 FCs, INT8 KV scales, FP32 attention activations,
folded query heads, absolute positions, local-window masking, and alignment 32.

## Scope and validation

Only the decode signature changes. **Prefill remains staged and unchanged**,
including chunk 128 and its cutoff after the final distinct KV producer.
Each first-logit pass uses the joined graph for the last prompt token; every
subsequent forced input uses the same joined graph. The resident configuration
has 44 prefill runtimes plus one decode runtime, versus 44 plus 107 before.

The final executable supports both modes through
`--joined_decode_graph=false|true`. It was relinked against the same 92 archived
libraries used by the preceding desktop experiments; all 92 hashes match.
The unchanged mode also reproduces all 3,999 dump files from the previous
archived baseline's correctness capture.

The main new correctness run tests prompt lengths, in order:
**1, 8, 128, 1,024, 4,096, 8,192, 127, 129, 511, 512, 513**. It includes:

- A BOS-only prompt, so the first invocation has empty past-KV inputs.
- 33 forced inputs after the one- and eight-token prompts and 64 after 1,024.
- Empty alignment padding when the end is exactly divisible by 32.
- Local-window and chunk boundaries, growing active extents, and long-to-short
  transitions using a reused runtime and a fresh session KV bank.

Across those cases, **163 full-vocabulary logit vectors** (42,729,472 FP32
values) and **4,890 committed KV files** are bit-identical. There are 5,053
compared files totaling 1,227,569,152 bytes. All measured histories and argmaxes
also match. A preceding smoke capture independently matches 99 vectors and
2,970 KV files; it is not added to the main validation count.

The diagnostic runtime audit checks all **70 attention BMMs** after invocation:
their FP32/QC8W operator family, FP32/INT8/FP32 operand/output types, folded
query dimensions, local/global active extents, and head dimensions match the
expected staged workload. Graph construction checks 277 FCs, and the existing
static-INT2 audit finds all 60 MLP operators. Invocation counters at call sites
report one decode runtime call in joined mode and 107 in staged mode.

## Timing and memory

Host: Intel Core i9-12900K, four threads pinned to distinct P cores
`0,2,4,6`. Workload: prompt 1,024, 64 forced decode inputs, capacity 8,448,
chunk 128, alignment 32, compact static INT2, shared workspace, and reused
runtimes. Source-weight release and the separate shape-cache probe are not
enabled in either mode.

Timing uses two fresh-process blocks with reversed order:
`staged, joined` then `joined, staged`. Each process discards one warmup and
measures three sessions. Logit/cache dumps and memory reporting are disabled.
The table uses the median of six session means; each session averages 64
forced decode calls. Neither compilation nor model loading is timed as decode.

| Metric | Staged | Joined |
| --- | ---: | ---: |
| Runtime calls per decode token | 107 | 1 |
| Median decode, ms/token | 25.123 | 25.524 |
| Session-mean decode range, ms | 25.081–25.888 | 25.445–25.607 |
| Block 0 median decode, ms | 25.558 | 25.454 |
| Block 1 median decode, ms | 25.106 | 25.599 |
| Prompt through first logits, ms | 1,035.47 | 1,034.19 |

Joined decode is 0.4% faster in block 0 and 2.0% slower in block 1. The pooled
median is 1.6% slower. These observations support comparable performance at
this context and CPU setting; they do not establish statistical equivalence,
a speedup, or the cause of the small difference. Timing at longer contexts or
on phones has not been measured for this variant.

The graph construction adds 30 INT8 K/V concatenations. At this decode
workload, their logical output payload averages **6.32 MiB per token**.
That is a tensor-size calculation, not measured DRAM traffic. Eliminating
host boundaries therefore introduces an explicit history-copy/storage cost;
the result does not rely on claiming the joined implementation is copy-free.
Future alias-aware updates or stateful views could address that cost, but
are not part of this experiment.

Separate memory captures use one warmup and one measured session:

| Metric, MiB except runtime count | Staged | Joined |
| --- | ---: | ---: |
| Resident runtime count, prefill plus decode | 151 | 45 |
| Process peak RSS | 2,051.08 | 2,052.12 |
| Final-decode RSS | 2,043.21 | 2,050.51 |
| Final-decode anonymous memory | 1,438.45 | 1,445.75 |
| Unique shared workspace | 8.750 | 10.673 |
| Owned external outputs | 48.035 | 47.134 |
| Packed weights | 751.875 | 751.875 |
| Logical persistent KV | 74.250 | 74.250 |

Process swap is zero. The joined graph grows the shared workspace by 1.92 MiB
and reduces owned external outputs by 0.90 MiB. Those counters do not fully
partition the 7.30 MiB RSS difference. Current RSS uses `smaps_rollup`, while
peak uses the process's `getrusage` high-water counter. The previously identified
366 MiB of packing-source buffers remain retained in both modes, so their
separate release benefit must not be attributed to this graph change.

## What this establishes for LiteRT

This is a constructive demonstration that **the current model's decode
dependencies can be expressed in one XNNPACK primitive graph with active
cache extents and comparable observed performance**. The native runner does
not fundamentally need a host return after every projection, attention, or MLP.

The subsequent [LiteRT mapping experiment](gemma4_litert_joined_decode.md)
serializes this joined graph, obtains one XNNPACK delegate partition with an
identical kernel-family inventory, and matches all full-logit/KV comparisons.
The LiteRT Interpreter path measures 1.11% slower than its matched native
control at the same desktop workload, with no new SDPA op or delegate changes.
These experiments do not establish the removal of staged prefill, a fully
graph-owned persistent KV bank, integration into every higher-level runner,
or equal performance on every device/context.

## Reproduction and artifacts

- [Experiment scripts and graph changes](../experiments/native_single_graph_decode/README.md)
- [Validated results](../experiments/native_single_graph_decode/results.json)
- [Raw captures and final copied-source patch](/data/bt/tmp/gemma4-native-single-graph-20260923)
- [Attention/runtime proposal](prefill_attention_runtime_proposal.md)

The reported binary is `build-v3/runner`, SHA-256
`1ab13ff13d6b624f09f339f33b862967d692c3235e18d1a4afc734b4be2cf934`.
`build-v3/commands.json` records exact compile/link commands and source hashes;
`build-v3/driver.patch` contains the complete isolated change. Per-capture
`command.json` and `status.json` identify diagnostic runs; their timings do
not enter the benchmark table.
