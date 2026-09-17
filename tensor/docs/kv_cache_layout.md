<!--
Copyright 2026 Google LLC.

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

# KV cache layout, attention execution, and validation

This is the consolidated design reference for the local LiteRT tensor API,
litert-converter, and direct-XNNPACK Qwen experiments. It explains the common
semantics, the layouts actually used, and which performance conclusions the
measurements support. It belongs beside the tensor API because the native
runner development now lives here; the converter and backend notes remain
the detailed evidence.

**Reviewed September 16, 2026.** Source inspection uses the local LiteRT checkout
at HEAD `345852646f25bc3e8d3935fa589243bad8bc5288` and converter at HEAD
`71b7f802c0400816040106d8fa9ceb0f68386233`, including local changes.
Those Git IDs alone do not reproduce the local implementations. The Qwen
mobile refresh records clean XNNPACK revisions, source/binary hashes, and raw
results separately. Historical reports, current implementations, and proposed
interfaces are identified below.

Links to `../../../litert-converter`, `../../../qwen3-xnnpack-direct`, and
`../../../LiteRT-LM` assume sibling checkouts under the same `src` directory.
They are local research references, not dependencies of the tensor API.

For a concrete comparison, start with [implementations in this workspace](#implementations-in-this-workspace).
For the design constraints, read [layouts and addressing](#layouts-and-their-address-consequences)
and [prefill versus online attention](#exact-lengths-chunking-and-online-attention).
The [mobile evidence](#what-the-mobile-results-establish) and
[validation guidance](#memory-accounting-and-numerical-validation) explain
which conclusions are measured and how to test changes.

The [mobile decode attribution](native_mobile_decode_analysis.md) follows up
the host DRAM measurements with capacity controls and phase-isolated phone
profiles. It distinguishes the benefit of avoiding unused attention work from
the speed of a particular cache layout at the same attention width.

## Design principles supported by the evidence

1. Separate persistent allocation capacity from the valid history read by
   attention. A mask establishes visibility, but generic matrix multiplication
   still processes the tensor extents it receives.
2. Choose cache storage, append/update behavior, attention operands, and the
   serialized model interface together. Removing one transpose is insufficient
   if it introduces a full-history copy on every token.
3. Preserve the model's KV sharing: between query heads for GQA/MQA, and
   between layers where the model specifies shared KV producers.
4. Count explicit copies, backend packing, dequantization, and runtime
   overhead separately. Borrowing an input pointer does not guarantee that the
   entire attention implementation is free of copies.
5. Exact query length, chunking, active KV length, and online attention are
   independent capabilities. Implementing one does not imply the others.
6. Validate numerical behavior and target-phone performance before changing a
   default. A smaller graph, newer dependency, or lower cache dtype does not
   establish a speedup by itself.

## Vocabulary and model semantics

| Symbol | Meaning |
| --- | --- |
| `B` | Batch size; the experiments here primarily use one. |
| `Hq` / `Hkv` | Number of query heads / distinct key-value heads. |
| `G = Hq / Hkv` | Query heads sharing each KV head. |
| `T` | Query rows executed in an invocation: one for decode, multiple for prefill. |
| `U` | Valid new tokens appended; a padded invocation can have `U < T`. |
| `L` | Valid cached token count, including newly appended tokens when visible. |
| `C` | Allocated cache capacity; normally `L <= C`. |
| `S` | K/V sequence extent presented to an operator; may be `C`, `L`, or an aligned/window interval. |
| `D` | Values per head vector; may differ between local and global layers. |
| `W` | Local attention window length. |
| Owner | A layer that produces persistent K/V consumed by itself or other layers. |

Standard multi-head attention has `Hq = Hkv`. Grouped-query attention (GQA)
shares a KV head among several query heads. Multi-query attention (MQA) is the
single-KV-head case. Every query head still computes its own attention weights:

```text
head h uses KV head floor(h / G)
P_h = softmax(scale * Q_h * K_group^T + mask_h)
O_h = P_h * V_group
```

| Model/configuration | Hq | Hkv | Cache-sharing implication |
| --- | ---: | ---: | --- |
| Direct Qwen3-0.6B | 16 | 8 | Two query heads share each KV head. |
| Gemma4 E2B | 8 | 1 | All eight query heads share one K/V history per owner. |
| Gemma4 E4B configuration | 8 | 2 | Four query heads share each KV head; the optimized E2B native driver is not an E4B implementation. |

Counts come from the [Qwen source][qwen-source] and
[Gemma configuration][gemma-config]. Head sharing does not require physically
repeating K/V up to `Hq` heads. The converter's
[grouped-attention guidance][converter-techniques] folds query groups into
matrix rows and preserves `Hkv` distinct histories. A broadcast node must be
checked after lowering: a logical broadcast and a materialized duplicate have
different memory costs.

For a simple append, define `L` as the post-update valid length:
`L = old_length + U` and `append_position = L - U`. Padded query rows must not
advance logical history. Multiple sequences require per-sequence lengths and
positions; a scalar-bound, batch-one implementation does not establish support
for a ragged batch.

## Three interfaces that must agree

| Interface | What it specifies | What it does not establish alone |
| --- | --- | --- |
| Attention mathematics / public operator | Q/K/V ordering, head grouping, scaling, masks, softmax, output ordering. | Persistent cache ownership or efficient append behavior. |
| Serialized model and executor ABI | State tensor names/metadata, dtype, quantization, shapes, capacity, positions, input/output relationships. | Which bytes a delegated implementation actually copies or reads. |
| Backend storage and execution | Addresses, strides, views, append operations, kernel packing, allocation reuse. | Permission to change the public operator's interpretation or quantization. |

The converter's [SDPA plan][converter-sdpa] explicitly treats these as a joint
design. A layout-specific backend lowering can differ from the serialized
representation, provided the boundary conversion and state semantics remain
correct and their cost is included.

### Existing SDPA and state contracts

For the existing XNNPACK `odml.scaled_dot_product_attention` path, the contract
is Q/output `[B,T,Hq,D]` and K/V `[B,S,Hkv,D]`. The converter must normalize to
that layout when preserving this composite, or use a valid decomposition.
The [XNNPACK delegate][xnn-delegate] contains the corresponding input/output
transposes. Reusing the same name for an incompatible layout is not a valid
optimization.

The converter document's proposed
`odml.scaled_dot_product_attention_head_major` is a proposal, not evidence
that this name is supported. Current YNNPACK also recognizes a separate
`odml.sdpa_transposed` path; its backend-specific contract is discussed below.
Support for one backend's path must not be assumed for another backend.

LiteRT-LM does not require K and V to have identical shapes. Its
[ABI documentation][converter-abi] describes both legacy name-based discovery
and explicit state-buffer metadata. The runtime also distinguishes dynamic
query dimensions from dynamic cache dimensions. In particular, CPU dynamic
prefill can execute actual query rows while keeping state capacity fixed and
decode at `T=1`.

The inspected executor-metadata consumer migration is incomplete. Legacy
cache-name discovery can still constrain compatibility; explicit metadata in
a model does not establish that every deployed runtime accepts arbitrary state
names or layouts.

Operators such as `odml.cache_update` and `odml.runtime_bmm` are part of an
artifact-specific producer/consumer contract. Check their actual attributes,
parameter tensors, and delegate lowering. A composite can be preserved,
decomposed, rejected by a delegate, or executed through fallback; the name
alone does not establish active-length execution.

### Update fragments and persistent tensors can have different layouts

The current converter uses a paired `odml.cache_update` with seven inputs:

```text
src_k, src_v, param_tensor, cache_k, cache_v, start_k, start_v
  -> updated K, updated V
```

Single-cache helpers use the internal `litert.cache_update_single` name.
The [lowering implementation][converter-lowering] distinguishes these paths;
some older experiment notes used the public name more broadly.

The Qwen runtime-BMM authoring path can produce fresh V as `[B,Hkv,U,D]`
while persistent V is `[B,Hkv,D,C]`. Transposing or adapting only those `U` new
rows is different from transposing `C` or `L` rows of history each step.
See the [attention authoring][converter-attention] and
[cache-update helpers][converter-cache].

Functional reference code can clone a full cache to express an updated value.
The deployed implementation can use DYNAMIC_UPDATE_SLICE, alias input/output
storage, or update a bounded region. Conversely, an apparently compact graph
can materialize additional buffers inside a delegate. Inspect actual binding
and lowering rather than inferring physical copying from the reference clone
or from full-capacity signature shapes.

The current Qwen CLI normally enables float `runtime_bmm`/`param_tensor`
authoring, with explicit exceptions such as the MediaTek NPU target or its
disable flag. Its [eager helper][converter-attention] slices the live K prefix
for QK and the live probability/V region for PV; QK can still expose a padded,
fixed-width output. The converter's [native fallback][converter-core] has shape
and parameter eligibility checks and may dequantize a sliced RHS before BMM.
Thus live-length intent, successful lowering, and fused integer-cache execution
are separate claims. The eager helper and native fallback both select
`param_tensor[1]` for the live bound. The native fallback clamps
it to `[0,C]`, while the eager reference treats nonpositive values as full
capacity. Tests should specify the intended empty/invalid-bound behavior.

## Layouts and their address consequences

Shapes below describe separate K and V arrays unless stated otherwise.
The final dimension is contiguous.

| Persistent layout | Useful property | Constraint to account for |
| --- | --- | --- |
| Head-major `[B,Hkv,C,D]` | One head's token history is contiguous; direct per-head prefix/window views. | Multiple heads' live prefixes have gaps when `L < C`. |
| Token-major `[B,C,Hkv,D]` | All heads of a newly appended token are contiguous; at `B=1` the complete live prefix is contiguous. | Each individual head's history has stride `Hkv*D`; generic dense per-head BMM cannot reinterpret it as `[L,D]`. |
| Transposed within a head `[B,Hkv,D,C]` | Can match a particular dot/GEMM operand order. | Appending one token touches a column; a shortened sequence retains row stride `C` and is not a compact `[D,L]` tensor. |
| Backend-tiled or paged storage | Can match kernel tiles, paging, or sharing requirements. | Needs explicit addressing/update support; these layouts are not implemented by the two direct runners discussed here. |

“Token-major” can describe either ordering across heads or ordering within one
head. Always provide dimensions. The native E2B bank calls each owner's
`[C,D]` array token-major; because `Hkv=1`, this is compatible with the
contiguous per-head principle used by Qwen.

For a head-major cache, the element offset is:

```text
offset(b, h, position, d) = ((b * Hkv + h) * C + position) * D + d
stride_head = C * D
```

Changing only the shape to compact `[B,Hkv,L,D]` would instead imply
`stride_head = L*D`. When `L < C` and multiple heads are present, those
addresses differ. One head's `[begin:end,D]` slice is contiguous; concatenating
all such slices into one compact tensor generally requires gathering, a
stride-aware view, or a different storage layout. Multiple batches can introduce
the same capacity-versus-live-length gap even when `Hkv=1`.

The current XNNPACK external-value binding supplies an ID and pointer; it does
not carry arbitrary external strides for the generic BMM subgraph inputs.
The tensor API's [borrowed-buffer methods][common-runner] preserve this
constraint. `SetInput` can avoid a host-side copy, but it cannot attach missing
stride semantics to a shape.

## Implementations in this workspace

| Path | Persistent cache | Attention reads | Scope |
| --- | --- | --- | --- |
| Direct Qwen3-0.6B | FP32 `[layer,Hkv,C,D]` for both K/V. | Per-KV-head contiguous live prefixes; optional compact all-head packing probe. | Standalone XNNPACK prototype, 28 layers, 8 KV heads. |
| Optimized native Gemma4 E2B | Separate INT8 `[C,D]` K/V per owner. | Borrowed aligned live prefix or local window, presented as `[1,1,extent,D]` for both. | `tensor/examples/gemma4/native`; batch one, one KV head. |
| Earlier bundle-matched E2B adapter / fixed-layout reference | K `[1,1,C,D]`, V `[1,1,D,C]`; INT8. | Fixed-layout graph update/read paths, including earlier two-bank and gathered-window controls. | Retained compatibility/reference helpers; not the optimized active bank. |
| Standard Gemma4 safetensors example | Growing FP32 K/V with logical `[B,Hkv,L,D]` shapes. | Parent example's concatenation/update path. | Separate from the bundle-matched native driver; E2B/E4B model support does not imply identical cache optimization. |
| Converter Qwen default layout | Fixed-capacity K `[B,Hkv,C,D]` and transposed V `[B,Hkv,D,C]`. | Depends on runtime_bmm/delegate/fallback support. | Current Qwen wrapper accepts FP32 KV; preserved-SDPA authoring can select another layout. |

The [mobile-checkpoint guide][mobile-ct], [native guide][native-guide],
[converter wrapper][converter-wrapper], and [Qwen source][qwen-source] identify
the actual entry points. “Native tensor runner” or “LiteRT-LM” alone is not a
complete cache configuration.

### Optimized E2B: owner sharing and stable storage

[ActiveKvBank][native-bank] owns 15 K buffers and 15 V buffers. E2B's first 15
layers produce their own caches; later local layers reuse owner 13 and later
global layers reuse owner 14. The [sharing map][gemma-graph] encodes this model
behavior. Local owner vectors have `D=256`; global owner vectors have `D=512`.

The bank allocates once. Appending copies only newly produced INT8 rows at
`length*D`; it does not transpose or copy the committed history. Only an owner
appends its K/V. Consumers reuse the owner's storage.

The transaction sequence is `BeginAppend -> Append each owner -> Commit`.
An owner's pending rows are visible to attention once that owner has appended.
Commit requires all owners to finish. Abort/reset invalidate logical rows
without promising to clear bytes. Borrowed views must respect these visibility
rules and the bank's lifetime.

The [driver][native-driver] constructs both attention operands using
`MakeInt8KeyCache`, which has shape `[1,1,extent,D]`. The old
`MakeInt8ValueCache` helper still describes transposed V for the reference
path. Likewise, a bundle manifest's transposed V schema describes the imported
artifact; it does not override the active bank's physical storage.

Global attention borrows the valid prefix. Local attention borrows the relevant
window. The default alignment is 32 rows: boundaries round outward, and masks
exclude padded, stale, future, and out-of-window positions. A full 512-token
local decode window can therefore execute over 512 or 544 positions. This is
bounded active work, not necessarily exactly `L` or `W` rows.

The local cache still allocates `C` rows in this implementation. A window
restricts which rows attention can see; it does not automatically reduce the
persistent allocation to `W` rows. A query chunk can also require the union of
several windows, with a separate causal/window mask for each query row.

At capacity 2,048, the 15 owner pairs occupy **18 MiB** of INT8 payload plus
kernel padding. Allocation capacity still affects memory and initialization
cost even when it does not determine attention width.

### Preserve head sharing without repeating K/V

For compact head-major query tensors, grouped attention can fold query heads
into rows:

```text
Q: [B,Hq,T,D] -> [B,Hkv,G*T,D]
K/V: [B,Hkv,L,D]
scores: [B,Hkv,G*T,L]
```

The mask and output reshape must preserve the original head/token ordering.
For E2B, `Hkv=1`, so all eight query heads can share the same contiguous K/V
operand without a multi-head cache gather. This simplification does not
automatically solve the eight-KV-head Qwen layout problem, or E4B's two-KV-head
case.

Removing materialized `repeat_kv` is also different from compacting distinct KV
heads. The first removes duplication of the same information; the second
gathers different histories to satisfy a dense operand layout. Both can affect
traffic, but their semantics and remedies differ.

## Why the packed Qwen probe exists

The ordinary path performs QK, softmax, and PV separately for each of eight
KV heads: 24 attention-runtime invocations per layer. The packed probe gathers
K/V into compact `[Hkv,L,D]` inputs and performs three invocations per layer.
Across 28 layers this reduces these calls from 672 to 84 per decode step,
excluding projections and other work.

The expected benefit is lower invocation overhead and potentially more
parallel work per call. The cost includes the explicit gather, different
internal BMM packing, and changed scheduling/cache behavior.

For the FP32 Qwen configuration at `L=1024`:

```text
K + V payload gathered per layer
  = 2 * Hkv * L * D * sizeof(float)
  = 2 * 8 * 1024 * 128 * 4
  = 8 MiB

Across 28 layers: 224 MiB of copied payload per decode step.
Logical copy accesses: 224 MiB read + 224 MiB written.
```

These are calculated accesses, not measured DRAM traffic. Actual memory-system
traffic depends on caching and internal execution.

A larger model does not automatically improve this tradeoff. Longer context
increases gather cost while the number of saved calls stays fixed. More heads
or larger vectors increase both useful work and copied data. More queries per
shared KV head can increase reuse. Larger unrelated MLPs may hide an attention
penalty as a smaller percentage of total time without reducing the penalty.

The structural target is to batch heads while consuming persistent cache
storage directly, through appropriate views or a cache-aware operation.

## Quantized KV: storage format and executed arithmetic

INT8 storage reduces persistent cache payload relative to FP32. It does not,
by itself, establish how much memory traffic or compute the attention path
requires. Distinguish:

| Executed path | Consequence |
| --- | --- |
| INT8 cache -> materialized FP32 cache -> FP32 BMM | Compact persistent state, but additional conversion buffers and accesses. |
| FP32 queries/probabilities with fused INT8-cache BMM | Avoids the materialized FP32 K/V tensors; internal adapters/packing can remain. |
| Dynamically quantized queries/probabilities with integer matmul | Different activation arithmetic and potentially different numerical behavior; not interchangeable with the preceding path. |

The native E2B [runtime audit][native-audit] checks actual executed operator
objects and edges: Q/probabilities/output remain FP32; K/V remain signed INT8;
the BMM is FP32/QC8W; cache extents match the active interval. QS8-to-QC8
adaptation can include byte copies and scale metadata, so this audit does not
claim copy-free attention.

K and V have separate published scales per owner and zero point zero.
New K/V are quantized before their first attention use. Attending to FP32
current rows and quantizing only when storing them would change the reference
arithmetic.

The old statement that LiteRT-LM's INT8 cache “only saves RAM” described a
particular older execution path. The [native improvement record][native-history]
documents an updated LiteRT-LM baseline that also used fused FP32/INT8-cache
attention. Remaining differences must be attributed to the actual graph,
active extents, updates, and kernels rather than to the label “INT8 KV.”

Checkpoint serialization and cache dtype are separate decisions. Loading
safetensors does not inherently prevent INT8 KV, but matching a calibrated
reference requires its quantization contract and scales. The standard local
Gemma4 safetensors example currently uses FP32 KV. The current converter
[Qwen wrapper][converter-wrapper] rejects non-FP32 KV because the corresponding
validated quantization setup is unavailable there; historical INT8 experiments
are not a supported default recommendation.

## Exact lengths, chunking, and online attention

| Capability | Work/memory it controls | Current status in these experiments |
| --- | --- | --- |
| Actual query rows | Dense projection, norm, MLP, and attention query work. | Qwen can use exact prompt rows. Native E2B reuses fixed-size chunks and pads the final partial chunk. |
| Active KV interval | Amount of historical K/V consumed by attention. | Both direct runners implement it; native E2B defaults to aligned, masked intervals. |
| Chunked prefill | Work and scratch per invocation, while retaining history across calls. | Native E2B reuses 128-row or 1,024-row stages. |
| Online/tiled attention | Avoids storing the complete query-by-history score/probability matrices. | Not implemented by either direct runner. |

These capabilities can be combined. A future chunked implementation can use an
exact final chunk, and an online kernel can consume exact-length queries with
fixed-capacity persistent state.

In Qwen, the batched per-token-attention mode stores live score rows one query
position at a time. It is not online softmax over KV tiles. The full-prefill
mode materializes quadratic score/probability buffers: about 16 MiB at
1,024 tokens for the ordinary per-KV-head-group path, or 128 MiB for all query
heads together. Scaling the ordinary shape to 4,096 tokens gives 256 MiB for
those logical buffers. These figures exclude other scratch and are not RSS.

Native E2B processes all but the final prompt token through cache-only prefill,
then uses decode on the last token to produce the first prediction. Its
prefill graph stops after the last distinct KV producer. For a 512-token prompt
with 128-row stages, 511 prefix tokens use four chunks and the last token uses
decode. This scheduling, cache-owner sharing, and unused-layer elimination
must be held constant when isolating an exact-length experiment.

An online attention implementation would tile query/KV work and retain running
softmax normalization and output accumulators across KV blocks. Naming an op
“SDPA,” putting the graph into one delegate partition, or reducing runtime
calls does not establish that algorithm or its scratch behavior.

The [dynamic-sequence design note][qwen-design] also distinguishes static graph
capture from logical state changes. A captured one-token decode can retain
fixed tensor shapes while positions, masks, cache contents, and valid length
change as data. A backend must still use the valid length to bound work.
The documented ONNX Runtime/GenAI comparison is backend-specific evidence,
not proof that the XNNPACK path has the same implementation.

In the locally reviewed ORT/GenAI revisions (`db036604f1` / `7a7e2969`), GQA
separates capacity-shaped storage from valid length, and GenAI's capture path
uses shared past/present cache storage for one-token decode. Chunked prefill
does not take that capture path. Even so, the reviewed ORT CPU GQA can allocate
probability scratch proportional to the full cache extent while its GEMMs
respect valid lengths. Efficient active reads and bounded scratch must be
verified independently; a fused semantic operator is not sufficient evidence.

## Backend-specific findings and revision boundaries

### XNNPACK

The direct runners lower attention to BMM, softmax, and BMM, using external
views for the applicable cache layout. The September Qwen refresh to
`c057fa51c93d2045f01009f732194500ec22168d` did not add a cache-aware
online-attention implementation to that runner. The public SDPA visitor
continues to author generic operations. Neither a composite name nor the
dependency upgrade eliminates quadratic prefill intermediates automatically.

### YNNPACK

YNNPACK can represent internal views differently from XNNPACK. The documented
`ynn_define_slice_like` mechanism allows a capacity-shaped external tensor to
retain its physical storage while an internal value changes its active
extents. `YNN_NODE_FLAG_KEEP_SHAPE` can preserve an external shape while
constraining the computed region. An ordinary internal slice can alias its
input buffer; external slice outputs require copying. `KEEP_SHAPE` uses a
cropped, zero-padded copy and is not itself a zero-copy view. These mechanisms
require supported lowering and explicit view semantics.

Current LiteRT contains both a runtime-BMM lowering and a separate
[YNNPACK attention visitor][ynn-attention]. The latter recognizes
`odml.scaled_dot_product_attention` and `odml.sdpa_transposed` and can use an
optional runtime-bound operand. It still constructs dot/softmax/dot.
For short query lengths, its algebraic rewrites can avoid some of the cache
transposes measured by the old generic-dot experiment.

The current YNN [model fixtures][ynn-model] and visitor specify:

| YNN semantic name | Q / output | K | V |
| --- | --- | --- | --- |
| `odml.scaled_dot_product_attention` | `[B,T,H,D]` | `[B,S,H,D]` | `[B,S,H,D]` |
| `odml.sdpa_transposed` | `[B,H,T,D]` | `[B,H,S,D]` | `[B,H,D,S]` |

In particular, `sdpa_transposed` has transposed V; it is not the proposed
symmetric head-major K/V contract. The bound is an optional fifth integer
input after the mask, or a fourth integer input when the mask is omitted.
K/V are read inputs, and the only output is attention output: this path does
not append cache state. These matching-head test fixtures do not establish
support for every GQA configuration.

For query length at most 32, the visitor can compute QK through
`(K Q^T)^T` and, for transposed SDPA, PV through `(V P^T)^T`. This avoids some
large cache transposes by changing multiplication order. It is still distinct
from a tiled online-softmax algorithm.

Consequently, the older [YNN layout benchmark][ynn-experiment] is useful
evidence that operand layout and lowering matter, but its desktop transpose
penalty is not a universal property of all current YNN attention. The
converter's June 23 rerun already reported a substantially smaller penalty
than its earlier run and explicitly recorded host-load noise. No YNN mobile
benchmark was added during the September Qwen XNNPACK refresh.

Runtime-bound parameter encodings must be matched to their producer and
consumer. The local YNN delegate supports a scalar bound or a multi-element
parameter tensor using element 1. Other documented consumers and fallback
paths can differ in indexing and handling of nonpositive bounds. Do not
assume an old `param_tensor` label alone specifies a portable ABI.

YNN converts these bounds into internal symbolic extents and calls
`ynn_reshape_runtime` on each evaluation with dummy inputs. Fixed external
cache shapes therefore do not imply fully shape-static backend execution.

In the inspected [YNN invocation code][ynn-invoke], a scalar uses element 0;
an array with at least two elements uses element 1. Positive bounds are clamped
to capacity, while nonpositive values retain full extent. The current
[LiteRT-LM parameter writer][lm-params] supplies `{start,end,end}` and documents
the third slot as the MLDrift BMM bound. Equal end values make those particular
slots compatible for this producer, but that is not proof of interchangeable
consumers. This review did not locate the MLDrift kernel implementation.

### Public layout proposals

The converter's proposed fixed head-major SDPA contract and proposed
cache-aware semantics are design directions. A useful cache-aware contract
would specify Q, persistent K/V, valid length, append position if it updates
state, masks/window bounds, quantization, and ownership.

Backend-specific physical layouts can remain private while those semantics
stay stable. A paged or ring-buffer implementation would additionally need
explicit addressing and wraparound rules; the current native bank is an
append-only allocation with window views, not a ring buffer.

## What the mobile results establish

The [September 16 Qwen report][qwen-refresh] is the current direct-XNNPACK
phone evidence: TECNO LJ9 and Pixel 8, four threads, fixed affinity per device,
identical coefficient archive, 29 cases and three measured repeats per phone,
plus a focused five-pair TECNO revision check. All 184 measured records were
audited. No Linux performance run was used for this refresh.

| Observation | TECNO | Pixel 8 | Interpretation |
| --- | --- | --- | --- |
| Capacity 128/2,048/4,096 at the same 64-token prompt | Decode within 1% of the 2,048 case | Within 2% | Supports active-history execution independently of allocation size. |
| Packed decode at 1,024 tokens versus per-head | 72.8% higher latency | 77.3% higher latency | Reducing calls did not compensate for the changed memory/execution costs. |
| Full-prefill attention versus batched per-token attention at 1,024 | 2.22x prefill throughput | 2.05x | Supports batching at these lengths, without proving exact-versus-padded superiority. |
| Fused MLP prefill latency reduction across tested lengths | 19–36% | 0–2% | Fusion rankings depend on the device/settings. |

The refresh itself did not produce a broad speedup. The focused TECNO
1,024-token decode check changed from 41.677 to 43.397 ms/token: +4.1% by
ratio of medians, or +2.5% by median paired change. Ranges overlapped and
latest was slower in four of five pairs. The initial main sweep's +10.1%
was not a stable magnitude. The affected attention phases were located;
no particular upstream commit was identified as the cause.

Combined QKV did not consistently improve total latency on these phones.
Some optional modes produced different greedy histories, which the report
flags. They retain comparable fixed-prompt prefill inputs but are not strict
same-history decode comparisons. These Qwen FP32-KV numbers must not be
presented as Gemma4 INT8-KV measurements or as each phone's peak capability.

Historical native Gemma4 comparisons establish the value of its combined
active-cache, prefill, quantization, and memory work under their recorded
conditions. They do not isolate cache layout as the sole cause of every
reported speedup.

## Memory accounting and numerical validation

### Keep footprint, logical accesses, and DRAM traffic separate

For uniform K/V types and dimensions, persistent payload is:

```text
K + V bytes = 2 * B * C * element_bytes * sum_over_unique_owners(Hkv * D)
```

Use separate K/V terms when types or dimensions differ. Count shared owners
once for allocation. Multiple consumers can still read the same owner during
one token's execution, so memory sharing does not eliminate their attention
work.

The converter's [bytes-per-token summary][converter-bytes] distinguishes
allocated session-contract accounting from a valid-length-aware estimate.
Neither tensor shapes nor allocator/RSS bytes directly measure traffic.
Aliased inputs/outputs, active views, cache reuse, materialized conversions,
and backend packing change the relationship.

The [bandwidth measurement plan][converter-bandwidth] describes direct
memory-controller counters and decode-window gating. Its early “not yet
measured for LiteRT-LM” status is historical: the later bytes summary contains
LiteRT-LM direct-counter results. Do not merge those historical host results
with the current phone measurements. Multiplying a graph byte estimate by
tokens/second yields an estimate of effective bandwidth, not a hardware
measurement.

The native Gemma4 runner now exposes a forced-decode FIFO interval for the same
collector. See [native decode memory traffic](native_decode_memory_traffic.md)
for the marker contract, reproducible commands, and the distinction between
host IMC counters and Android counter availability. Its September 16 matched
Gemma4 captures include verified YNN execution. At a fixed 1,024-token prompt,
increasing capacity from 2,048 to 8,192 raised LM-XNNPACK's measured total
traffic by 7.7% and reduced its throughput by 23.1%. Native traffic and
throughput were effectively unchanged; LM-YNNPACK traffic rose only 0.7%,
but throughput still fell 4.1%. This supports a capacity-sensitive cost in
the measured XNN-only configuration, without establishing that every YNN
path is capacity-independent. A whole-decode capture includes all work in
that interval and cannot attribute bytes to a particular attention operator.

The historical Qwen4B investigation also revised its early cache-copy diagnosis:
later captures no longer showed a large memmove hotspot and shifted attention
to weights and packing. Likewise, preserved SDPA once had few serialized
transpose nodes but many executed delegate transpose calls. Use the latest
artifact-specific observations in the [investigation][converter-qwen-history],
not its earliest conclusion or a graph-node count, to identify current costs.

### Changes that can affect correctness

Changing shapes, chunking, alignment, or fusion can change floating-point
reduction order. Static activation quantization can amplify small differences
near thresholds. This does not prove a different quantizer rounding rule.
Preserve RoPE positions, query/head ordering, scale application, causal/local
masks, quantization placement, and shared-cache ownership.

Prompt-end closeness is one regression check. In the Qwen refresh, old/latest
prompt-end logits were bit-identical at six tested lengths per phone, and
their measured greedy histories matched. Optional authoring modes could still
diverge after several generated tokens. Compare full logits under identical
forced continuation tokens to isolate numerical differences. Greedy divergence
alone neither establishes a quality failure nor establishes acceptable parity.

Memory contracts also need boundary cases. The refresh found an existing
137-token softmax crash because an external input lacked XNNPACK's required
tail storage. Adding `XNN_EXTRA_BYTES`, 16 bytes on the tested ARM64 builds,
fixed it without changing the logical 137-row shape. Allocation padding and
padded query/cache extents are separate requirements.

For a cache or attention change, validate:

1. Append offsets and integer codes, owner sharing, reset/abort behavior,
   capacity limits, and borrowed-view lifetimes.
2. Odd lengths, alignment boundaries, partial final chunks, window boundaries,
   and masks over deliberately nonzero stale/padded rows.
3. Actual executed operand shapes and dtypes, including fusion, adaptation,
   activation quantization, and fallback behavior.
4. Full logits under the same tokens, followed by appropriate model-quality
   checks when numerical behavior changes.
5. Phone prefill/decode timing separately, with artifact hashes, capacity,
   live length, thread/affinity settings, repeats, timing boundaries, and
   numerical-history differences recorded.

## Decisions still requiring controlled experiments

- Compare whole-prompt exact rows, padded buckets, and chunks with an exact
  final length while holding model arithmetic and attention implementation
  constant. Record scratch and reshape/setup cost as well as latency.
- Evaluate a stride-aware or cache-aware all-head implementation without
  per-token compaction; a packed baseline cannot answer that question.
- Evaluate online attention separately from query-shape changes, preserving
  quantization and mask semantics and checking actual scratch behavior.
- Validate multiple KV heads and batches before generalizing the E2B active
  bank. E4B support needs its own cache addressing and sharing checks.
- Recheck newer YNN attention lowering on target devices if it becomes a
  candidate backend. Historical desktop dot-layout timings are insufficient.

This consolidation adds documentation and source reconciliation. It does not
claim that these follow-up implementations or benchmarks have been performed.

## Evidence and implementation map

| Reference | Role |
| --- | --- |
| [Converter SDPA improvement plan][converter-sdpa] | Public layout contract, joint cache/update design, historical direct-XNN/YNN probes, proposed head-major op. |
| [Converter performance techniques][converter-techniques] | GQA without materialized KV repetition; separation of frontend, converter, and backend responsibilities. |
| [LiteRT-LM format and ABI][converter-abi] | Cache/state discovery, unequal K/V shapes, dynamic query versus fixed-state execution. |
| [Converter Qwen wrapper][converter-wrapper] / [CLI][converter-cli] | Current cache axes, FP32 restriction, and runtime-BMM export policy. |
| [Qwen4B decode investigation][converter-qwen-history] | Historical copy/update/packing investigations; later sections qualify earlier bottleneck claims. |
| [Bytes-per-token summary][converter-bytes] / [bandwidth plan][converter-bandwidth] | Contract/live-byte estimates and physical-traffic measurement boundaries. |
| [Qwen dynamic-sequence note][qwen-design] / [refresh][qwen-refresh] | Shape versus capacity semantics, online-attention gap, verified phone measurements. |
| [Native Gemma4 guide][native-guide] / [mobile CT guide][mobile-ct] | Separate optimized bundle-matched and standard safetensors execution paths. |
| [Active bank][native-bank] / [driver][native-driver] / [runtime audit][native-audit] | Actual persistent storage, stage bindings, and executed dtype/extent checks. |
| [XNNPACK delegate][xnn-delegate] / [YNNPACK attention][ynn-attention] / [YNNPACK runtime BMM][ynn-dot] | Backend-specific attention lowering; support must be checked at the deployed revision. |
| [Preserved YNN layout experiment][ynn-experiment] | Historical microbenchmark source/provenance, distinct from the native XNNPACK runner. |

[converter-sdpa]: ../../../litert-converter/docs/SDPA_IMPROVEMENT_PLAN.md
[converter-techniques]: ../../../litert-converter/docs/LLM_PERFORMANCE_TECHNIQUES.md
[converter-abi]: ../../../litert-converter/docs/litertlm_format_and_abi.md
[converter-wrapper]: ../../../litert-converter/aten_to_tflite/hf_export_wrappers.py
[converter-cli]: ../../../litert-converter/mlir_backend/tools/compile_hf_prefill_decode.py
[converter-lowering]: ../../../litert-converter/mlir_backend/importer/final_lowering_handlers.py
[converter-attention]: ../../../litert-converter/aten_to_tflite/litert_lm_attention.py
[converter-cache]: ../../../litert-converter/aten_to_tflite/litert_lm_export_cache.py
[converter-core]: ../../../litert-converter/cpp_ext/src/optimizer_core_ir_bridge.cpp
[converter-qwen-history]: ../../../litert-converter/docs/QWEN3_4B_DECODE_PERF_NOTES.md
[converter-bytes]: ../../../litert-converter/docs/LLM_DECODE_BYTES_PER_TOKEN_SUMMARY.md
[converter-bandwidth]: ../../../litert-converter/docs/LLM_DECODE_BANDWIDTH_MEASUREMENT_PLAN.md
[qwen-source]: ../../../qwen3-xnnpack-direct/src/qwen3_xnn_generate.cc
[qwen-design]: ../../../qwen3-xnnpack-direct/docs/dynamic_sequence_attention_design.md
[qwen-refresh]: ../../../qwen3-xnnpack-direct/docs/xnnpack_refresh_20260916.md
[native-guide]: ../examples/gemma4/native/README.md
[mobile-ct]: ../examples/gemma4/MOBILE_CT.md
[native-bank]: ../examples/gemma4/native/active_kv_bank.h
[native-driver]: ../examples/gemma4/native/driver.cc
[native-audit]: ../examples/gemma4/native/active_runtime_audit.h
[native-history]: ../experiments/history/2026-09-15-native-runner-improvements.md
[gemma-config]: ../examples/gemma4/gemma4_config.h
[gemma-graph]: ../examples/gemma4/native/model/gemma4_graph.h
[common-runner]: ../runners/common_nnpack/runner.h
[xnn-delegate]: ../../tflite/delegates/xnnpack/xnnpack_delegate.cc
[ynn-attention]: ../../tflite/delegates/ynnpack/attention.cc
[ynn-model]: ../../tflite/delegates/ynnpack/attention_model.cc
[ynn-invoke]: ../../tflite/delegates/ynnpack/ynnpack_delegate.cc
[ynn-dot]: ../../tflite/delegates/ynnpack/dot.cc
[ynn-experiment]: ../experiments/ynnpack_attention_layout/README.md
[lm-params]: ../../../LiteRT-LM/runtime/executor/litert_compiled_model_executor_utils.cc
