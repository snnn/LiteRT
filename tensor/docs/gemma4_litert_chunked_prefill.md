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

# Gemma4 chunked prefill through native XNNPACK and LiteRT

Chunked prefill now runs as one connected graph per chunk in the local
[experiment](../experiments/litert_chunked_prefill/README.md). Both native
XNNPACK and LiteRT's Interpreter/XNNPACK path preserve the existing runner's
Gemma4 KV-sharing optimization. LiteRT uses existing primitive operators and
its existing delegate; this work adds no SDPA op or XNNPACK/delegate changes.
MTP is excluded.

## Desktop results

| Prompt tokens | Staged prefill | Joined native | Joined LiteRT |
| ---: | ---: | ---: | ---: |
| 128 | 110.72 ms | 107.03 ms | 106.61 ms |
| 1,024 | 1,003.95 ms | 967.85 ms | 967.11 ms |
| 4,096 | 4,562.93 ms | 4,427.09 ms | 4,427.77 ms |
| 8,192 | 10,550.26 ms | 10,088.11 ms | 10,101.32 ms |

These are medians of six measured prefix runs, with the last prompt token
handled by the full decode signature afterward. At 1,024 tokens, first-logit
latency is 1,055.48 ms (staged), 998.17 ms (native), 997.71 ms (litert).

| Memory metric (MiB) | Staged prefill | Joined native | Joined LiteRT |
| --- | ---: | ---: | ---: |
| Whole-process peak RSS | 2,110.23 | 2,029.86 | 2,114.03 |
| Retained RSS after 8,192-token case | 2,043.84 | 1,977.95 | 2,062.01 |
| Owned stage/Interpreter output payload | 47.13 | 2.13 | 2.13 |
| Unique shared XNNPACK workspace | 52.49 | 52.49 | 52.49 |
| Persistent KV capacity | 74.25 | 74.25 | 74.25 |
| Packed weights | 751.88 | 751.88 | 751.88 |

Native peak RSS falls by **80.36 MiB**. LiteRT peak RSS is
**3.80 MiB higher** than staged prefill, and its retained RSS is
**18.17 MiB higher**. Thus LiteRT reproduces the low activation-buffer
budget and approximately the staged peak, but does not beat the staged runner
in retained RSS. Its remaining premium includes graph/affine metadata and
allocator retention; the counters do not fully attribute its 84.05 MiB
RSS difference from joined native execution.

The 45 MiB output-payload reduction is directly attributable: native staged
prefill retains 46.125 MiB of boundary outputs, whereas joined prefill retains
only 1.125 MiB of new KV. Decode contributes the same 1.008789 MiB in both.
The old per-layer mask allocations are temporary, so they are absent from the
retained-output counter but included in peak RSS. The joined host mask capacity
at the end of the sweep is 4.58 MiB, including both prefill and decode masks.
Scratch is counted once even though two signatures use it. No swap was observed.

**Correctness:** both joined paths are bit-identical to staged execution for
163 full vocabulary vectors and 4,890 committed KV files: 5,053 files,
1,227,569,152 bytes in each comparison. The native and LiteRT kernel-family
inventories match for both signatures.

## Execution

For a 128-row chunk, the graph contains 14 complete transformer layers and
layer 14's K/V projections. Its roots are the 15 owners' 30 new K/V tensors.
Unused later layers and the vocabulary head do not enter the prefill runtime.
The final prompt token runs through the full decode signature to produce the
first vocabulary logits. This is the same useful prefix computation as the
staged baseline: 129 FC operators and 28 attention BMMs per chunk.

The host prepares token embeddings, RoPE positions, masks, and borrowed views
of committed history, invokes the graph once, then commits only valid new KV
rows. It performs no work between transformer layers. Runtime invocations per
prefix chunk fall from 44 to one. A 1,024-token prompt uses eight prefix chunks
plus one full decode invocation for first logits.

LiteRT serializes both `prefill` and `decode` signatures in one actual TFLite
model, reloads it with `InterpreterBuilder`, and delegates both subgraphs to
one XNNPACK delegate. They share the delegate's scratch workspace, a four-thread
pool, and a packed-weight cache. Each signature is exactly one delegated
partition/runtime, with zero fallback operators. Normal non-strict subgraph
input resize, custom allocation, allocation/reshape, and invocation APIs execute
the model. The link-time observer only records and forwards the XNNPACK calls.

The authoring/export adapter supplies the same existing-builtin mappings used
by the [decode experiment](gemma4_litert_joined_decode.md): quantized conversion,
RMSNorm primitives, original compact INT2 weights for decode, and explicit KV
dequantization that XNNPACK fuses into FP32/QC8W BMMs. Constant bindings retain
the original weight pointers for packed-cache reuse. The serialized prefill
subgraph has 1,815 operators, 2,583 tensors, 74 inputs, and 30 outputs. Decode
has 3,830 operators, 5,396 tensors, 99 inputs, and 31 outputs.

The exporter expands per-channel zero points only while serializing, then
restores the compact authoring metadata. This avoids retaining 9,697,112 bytes
(9.25 MiB) of expanded zero points in the authoring graph. The Interpreter and
FlatBuffer keep their required affine metadata. A separate memory probe
observed about 9 MiB lower RSS and peak RSS from this lifetime change.

## Bounded activation and mask storage

Persistent KV backing capacity is 8,448, but attention width follows the
committed history plus the current chunk, with 32-token alignment. Global
attention grows with context. Local attention uses the union of the chunk's
sliding windows and masks each query to its exact visible interval.

The final partial chunk pads query rows and newly projected K/V to 128. Those
extra rows are masked and never committed. Compared with the staged runner's
valid-token attention extent, this may add up to 127 masked positions; it does
not expand early attention to full KV capacity. Runtime shape audits explicitly
use this padded chunk extent.

A local mask and a global mask serve every relevant layer. Keeping the eight
query heads as a BMM batch dimension lets each `[1,1,T,S]` mask broadcast across
heads. Its storage is eight times smaller than explicitly repeating the mask
for every head. Score/probability tensors still scale as `H*T*S`; this work does
not implement online softmax or tiled KV attention.

Connecting stages also makes intermediate activations internal to one XNNPACK
lifetime planner. Only new K/V rows need to escape the prefill invocation.
Serialized prefill and decode share scratch across signatures. These are
allocation-lifetime changes, independent of an SDPA interface.

## Validation and measurements

The final measurements and allocation breakdown are in the adjacent
[results file](../experiments/litert_chunked_prefill/results.json). Timing uses
four threads pinned to desktop P cores `0,2,4,6`, one warmup and three measured
runs per case, in two blocks with reversed variant order. Prompt lengths are
128, 1,024, 4,096, and 8,192; each has two forced decode inputs. Long timing
prompts repeat the 1,024-token fixture so all variants receive identical inputs. Prefix latency
excludes the final full decode used to produce first logits. First-logit latency
is recorded separately, and runtime compilation/model loading are excluded.
Memory captures are separate, use the same case order, and include both warmup
and measured sessions. Retained-memory snapshots therefore include prior
allocation high-water marks. Peak RSS covers the whole process.

The correctness suite uses prompt lengths 1, 8, 128, 1,024, 4,096, 8,192, 127,
129, 511, 512, and 513, in that order. It covers empty history, zero alignment
padding, local-window and chunk boundaries, partial chunks, and long-to-short
runtime reuse. Both joined paths compare full vocabulary logits and every
committed owner K/V byte against staged prefill. During this diagnostic run,
actual attention operand shapes, quantization/kernel families, runtime counts,
and invocation counts are checked.

Native joined and LiteRT joined prefill have the same kernel-family inventory:
100 QS8/QC4W FCs, 29 QS8/QC8W FCs, 28 FP32/QC8W BMMs, and 14 FP32 softmaxes,
plus matching auxiliary primitives. Decode retains all 60 original static INT2
MLP tensors/operators. The staged prefill retains its original worker-spinning
policy; both joined paths use the delegate's `DONT_SPIN_WORKERS` policy.
Consequently, a latency improvement over staged execution is a result for these
runner configurations, not an isolated measurement of graph-call overhead.

## Scope and follow-up

This establishes connected chunked prefill through the LiteRT Interpreter in
addition to native XNNPACK. Production integration into `LitertDynamicRunner`,
the public CompiledModel runner, or LiteRT-LM is separate. It does not establish
phone performance or a globally optimal chunk size.

The remaining costs include active-history K/V concatenation, materialized
attention scores, retained source-weight buffers, and LiteRT's graph/quantization
metadata. No source-release optimization, `MADV_PAGEOUT`, or heap trimming is
used in these comparisons. One could reduce the first two costs with cache
views and tiled attention later; neither is required for the working chunked
prefill demonstrated here.

Artifacts: `/data/bt/tmp/gemma4-litert-chunked-prefill-20260924`. The directory
contains the final `runner-validated` executable, source/binary hashes, build
log, exact commands, full correctness dumps, both timing blocks, and separate
memory snapshots.
See the experiment [README](../experiments/litert_chunked_prefill/README.md)
for build/run commands.

Final capture label: `validated-`. Binary SHA-256:
`ec8da28c1238edc17385a6d7650c4bcea041d5eecabd4f51c9a639b8f5c0d65c`.
