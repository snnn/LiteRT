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

# Prefill and attention performance: proposed LiteRT changes

**Status: design discussion, September 23, 2026.** This document consolidates
the reasoning behind native Gemma4 prefill, dynamic shapes, tiled attention,
and a proposal for bringing comparable execution to the Tensor API's LiteRT
runner and LiteRT-LM. Current behavior refers to the inspected local checkouts,
including local changes. Proposed interfaces below are not implemented APIs
or claims of upstream support. No new benchmarks were run for this document.

Comparable performance is feasible, but introducing an operator name is only
one part of the work. The graph must expose the necessary semantics, the
delegate must execute only useful work with efficient kernels, and the runner
must preserve state and reuse resources across invocations.

The recommended direction is to extend existing attention and cache mechanisms,
complete dynamic-shape and buffer support in the Tensor API runner, and add
online tiled attention behind the same attention contract. Chunk size remains
an execution policy. Gemma4's elimination of unnecessary prefix computation
remains a separate graph-authoring optimization.

Related evidence and implementation references:

- [KV cache layout and execution](kv_cache_layout.md).
- [Native mobile decode attribution](native_mobile_decode_analysis.md).
- [YNNPACK SDPA graph experiment](ynnpack_sdpa_graph_performance.md).
- [Local INT8-KV SDPA extension](ynnpack_int8_sdpa.md).
- [Native Gemma4 runner](../examples/gemma4/native/README.md).
- [Existing nonstandard operator contracts](../../docs/nonstandard_ops.md).

Links into `../../../LiteRT-LM` assume a sibling checkout under the same
`src` directory. They are source references, not Tensor API dependencies.

## 1. Separate the optimizations

| Capability | What it changes | What it does not establish |
| --- | --- | --- |
| Batched prefill | Runs several prompt tokens together, enabling matrix multiplication across token rows. | That the whole prompt must run in one invocation. |
| Exact query length | Avoids executing padded query rows in projections, attention, norms, and MLPs. | That attention avoids unused cache positions. |
| Active KV bounds | Restricts attention to the live history or useful local interval. | That scores and probabilities are computed without large intermediates. |
| Chunked prefill | Limits tokens processed per model invocation and bounds temporary activations. | That smaller chunks always reduce total latency. |
| Online tiled attention | Accumulates attention across KV tiles without retaining full score/probability rows. | That cache storage or useful global-attention arithmetic disappears. |
| KV-sharing prefix elimination | Omits earlier-token computations that cannot affect required caches or final logits. | That a particular shape, chunk size, or backend is required. |

Here, batching refers to token rows from one prompt, not necessarily multiple
independent requests. An exact-size invocation over 1,000 prefix tokens is
batched prefill. Executing `256 + 256 + 256 + 232` is also batched prefill,
with an outer chunking policy.

Use the following dimensions when comparing implementations:

| Symbol | Meaning |
| --- | --- |
| `N` | Total prefix tokens considered in a prefill comparison. |
| `T` | Query rows executed by one invocation. |
| `U` | Valid new tokens; `U <= T` when rows are padded. |
| `C` | Persistent cache allocation capacity. |
| `L` | Valid history end after the new tokens are appended. |
| `S` | History extent actually processed by attention. |
| `W` | Local window length, including the current token. |
| `Hq`, `Hkv` | Query heads and distinct KV heads. |
| `Dk`, `Dv` | Key/query and value head dimensions. |
| `Bq`, `Bk` | Query and KV tile sizes inside an attention implementation. |

Ignoring batch/head axes, `Q[T,Dk] @ K[Dk,S]` produces scores `[T,S]`.
Changing `T`, changing `S`, and changing how the matrix is evaluated are
different optimizations. Persistent allocation can remain at `C` throughout.

## 2. What the native Gemma4 runner does

The local E2B native runner builds reusable prefill stages with either 128 or
1,024 query rows, and a one-row decode path. For a 300-token prompt with
128-row prefill stages, its execution is:

```text
prompt tokens   1..128 -> prefill, 128 valid rows
prompt tokens 129..256 -> prefill, 128 valid rows
prompt tokens 257..299 -> prefill,  43 valid rows + 85 padded rows
prompt token       300 -> full decode path -> first prediction
```

Chunks execute in order. Positions are absolute, and the persistent KV bank
retains earlier chunks. Each real query sees its causal history, further
restricted by the local window where applicable. Only valid rows are appended;
padded rows do not advance the logical history.

The [driver](../examples/gemma4/native/driver.cc) already reshapes K, V, and
mask inputs to active history intervals. Query rows remain fixed by the chosen
prefill signature. Active extents are aligned and masked; exact useful length
and physical kernel extent need not be identical.

### Why KV sharing permits a shorter prefix graph

In this E2B configuration, there are 35 layers but only 15 distinct KV-producing
layers, numbered 0 through 14. Later local layers reuse layer 13's KV; later
global layers reuse layer 14's KV. This is inter-layer sharing, separate from
the eight query heads sharing one KV head.

For the earlier prompt tokens, the runner needs the caches that future queries
will read. It therefore executes full layers 0 through 13 and only the needed
K/V-producing part of layer 14. It skips layer 14's remaining attention/MLP
work and layers 15 through 34 for those prefix tokens. The last prompt token
runs through all 35 layers and the output head.

```text
earlier prompt tokens -> layers 0..13 -> layer 14 K/V -> persistent caches
                                                              |
last prompt token --------------------> all 35 layers <--------+
                                             |
                                           logits
```

The omitted upper-layer states of earlier tokens have no consumer needed for
these caches or final-token logits. This is dependency-based elimination,
not an approximation. It applies to inference requesting the final prediction;
requests for every prompt token's logits or upper-layer states need a different
output graph. Other model variants require their own dependency analysis.

The avoided work includes large MLPs, projections, norms, and attention. With
the E2B dimensions `1536 -> 6144` and three MLP matrices, skipping 20 complete
layers for 1,023 prefix tokens avoids approximately
`20 * 1023 * 3 * 1536 * 6144 = 579,254,353,920` logical MLP MACs alone.
This is an illustrative operation count, not a measured latency reduction.

See the [configuration](../examples/gemma4/gemma4_config.h),
[sharing map](../examples/gemma4/native/model/gemma4_graph.h), and driver.
The inspected LiteRT-LM baseline already has cache-only prefill behavior;
do not attribute the entire native-versus-LM gap to this optimization.

### How the 151 stage runtimes are counted

In this native E2B implementation, each fully executed transformer layer is
split into three authored XNNPACK subgraphs. Each subgraph has its own prepared
`StageRunner` runtime:

1. Projection: produce Q and, for a KV owner, new K/V, including the associated
   normalization, position, and quantization work.
2. Attention: read the active KV interval and compute attention output.
3. Post-attention: output projection, residuals, MLP, and per-layer embedding
   processing.

The driver appends new owner K/V to the persistent bank between projection and
attention. That append is C++ cache-manager work, not a fourth XNNPACK runtime.
These boundaries let the driver bind live cache views and reshape the
attention stage independently.

Prefill and decode have separate reusable stage sets. For the 35-layer E2B
configuration, the [construction loop](../examples/gemma4/native/driver.cc)
creates:

| Path | Runtime count | Total |
| --- | --- | ---: |
| Prefill | 1 preprocessing + 14 full layers times 3 stages + layer 14's K/V-only projection | 44 |
| Decode | 1 preprocessing + 35 full layers times 3 stages + 1 logits tail | 107 |
| Both resident stage sets | 44 + 107 | 151 |

Layer indices are zero-based. The runtime objects are reused for successive
chunks and tokens. Stage invocations execute in dependency order and can share
one scratch workspace even though their graphs and runtime objects are
distinct. This stage decomposition is an implementation choice; Gemma4 does
not inherently require three runtime objects per layer.

### Why batching and chunking can help

Compared with token-at-a-time prompt processing, batching offers matrix kernels
more query rows over which to reuse weights and amortize dispatch. Once useful
tokens are already batched, dividing them into smaller chunks does not reduce
their useful dense projection or MLP arithmetic.

Chunking can still remove padding from coarse signature selection. For example,
511 prefix tokens require 513 padded rows in one 1,024-row invocation but only
one padded row across four 128-row invocations. Exact dynamic query shapes
would remove that remaining padding too. This comparison is about those shape
choices, not proof that LiteRT-LM always chooses the larger signature.

Chunking also avoids attention against keys in future chunks. For global
attention starting from empty history, assume `N` is divisible by chunk size
`T`, and each chunk uses a dense rectangle ending at its current history end:

```text
one dense full-prefix invocation: N^2 score entries per head
chunked dense invocations:        T * (T + 2T + ... + N)
                               = (N^2 + N*T) / 2
useful causal entries:            N * (N + 1) / 2
```

Within-chunk future entries are still masked after matrix multiplication.
For `N=1024, T=128`, the dense rectangles contain 589,824 entries rather than
1,048,576: about 44% fewer. This illustration excludes final-token handling,
padding, alignment, and previous conversation history. It is not a model-wide
44% speedup.

For local attention, the union of the windows for a chunk has at most
`W + T - 1` positions before history/capacity clipping and alignment. Per-query
masks still enforce the individual windows.

Smaller chunks incur more invocations and can require repeated weight reads,
packing, or less efficient matrix kernels. Chunk size is a tradeoff among
latency, temporary memory, and scheduling opportunities. In multi-request
serving, chunks can let ongoing decodes progress between prefill work; this
is a demonstrated use in [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal).
That scheduling benefit is not automatically present in a single-request
native benchmark, and the prompt must still be processed before its answer.

## 3. Dynamic shapes, runtime BMM, and YNNPACK

Dynamic shapes remove the fixed-row restriction when supported throughout
authoring, serialization, shape propagation, delegate execution, and buffers.
One dynamic graph can run the entire prefix or variable-size chunks, including
an exact final chunk. The allocation for KV state can remain fixed while its
valid bounds change as runtime data.

The `runtime_bmm` connection is useful but narrower: a runtime bound can make
matrix work follow the live history rather than cache capacity. It does not
necessarily change query length, every intermediate shape, or the attention
algorithm.

The locally inspected YNN `runtime_bmm` path already bounds QK/PV arithmetic
to the live prefix, but its public score result can retain the capacity shape.
Masking and softmax consequently retain unnecessary work. The local SDPA
rewrite propagates the live extent through scores, mask, softmax, and PV in
one attention region. It does not add online softmax or a local-window lower
bound. See the [graph experiment](ynnpack_sdpa_graph_performance.md#why-this-graph-can-be-faster)
and [delegate implementation](../../tflite/delegates/ynnpack/attention.cc).

YNNPACK's loop fusion through Slinky can improve locality and tile intermediate
work internally. Consequently, a logical tensor shape alone does not prove
the size of an allocated intermediate. Inspect the actual schedule, allocations,
and work bounds. Query-row tiling or ordinary loop fusion also does not by
itself establish online normalization across KV tiles.

The fixed 128/1,024-row limitation is removable. Chunking remains a general
execution policy whose benefit may shrink if the backend performs equivalent
tiling and fusion internally. A full-prefix dynamic invocation may then be
faster, especially when it improves weight reuse and reduces dispatch.

## 4. Comparison with tiled attention and online softmax

Current native attention divides the query rows between model invocations but
expresses ordinary `QK -> mask -> softmax -> PV` within each attention stage.
An online implementation divides the history dimension too, incorporates each
KV tile into an accumulated result, and reuses the tile storage. FlashAttention
combines this approach with fusion and a schedule designed to reduce transfers
between memory levels. See [FlashAttention, section 3.1](https://arxiv.org/html/2205.14135v2#S3.SS1).

For one query, write attention as:

```text
output = sum_j exp(score_j) * value_j / sum_j exp(score_j)
```

The running state consists of a maximum `m`, normalization sum `z`, and weighted
value vector `u`. A numerically stable update for a tile with at least one
visible, finite score is:

```text
m_new = max(m, max(tile_scores))
alpha = exp(m - m_new)
p = exp(tile_scores - m_new)
z = alpha * z + sum(p)
u = alpha * u + p @ V_tile
m = m_new

output = u / z  # after all tiles
```

Initialize `m=-infinity`, `z=0`, and `u=0`; masked scores contribute zero weight.
Skip fully masked tiles, avoiding the undefined `-infinity - -infinity` case.
Define the final output separately for a query with no visible keys. The
normalizer spans all tiles, preserving mathematical attention; independently
normalizing each tile and averaging its output would change the result.
See the [online normalizer paper](https://arxiv.org/abs/1805.02867) and
[FlashAttention's attention accumulation](https://arxiv.org/html/2205.14135v2#S3.SS1).

Illustrative FP32 score storage for eight query heads and 4,096 history tokens:

| Execution | Logical score working set | Bytes |
| --- | --- | --- |
| Full 4,096-query dense prefill | `8 * 4096 * 4096` | 512 MiB |
| Current 128-query dense chunk | `8 * 128 * 4096` | 16 MiB |
| Online 32-query by 128-key tiles, eight heads concurrently | `8 * 32 * 128` | 128 KiB |

These calculated figures describe score storage if materialized. They exclude
Q/K/V, output accumulators, masks, packing, additional live tiles, and other
scratch. Scheduling fewer heads concurrently changes the live footprint.
The 128-fold difference between the last two rows is neither a speedup claim
nor a reduction in total process memory. The persistent cache is still needed.

Online evaluation reduces intermediate memory and traffic; it does not make
exact global attention's useful arithmetic subquadratic. This distinction is
also established by [Self-attention Does Not Need O(n^2) Memory](https://arxiv.org/abs/2112.05682).
Skipping causal or out-of-window pairs requires a schedule that understands
those bounds; online normalization alone does not provide that skipping.

| Property | Outer chunked prefill | Online tiled attention |
| --- | --- | --- |
| Scope | All model operations in an invocation. | Attention operations. |
| Score working set per head | `T * S` for dense attention. | Approximately `Bq * Bk`, plus accumulators. |
| MLP temporary activations | Bounded by the outer chunk. | Need separate tiling or outer chunking. |
| Scheduling/cancellation boundaries | Between chunks. | Need separate runtime support. |
| Useful dense weight work | Same for the same valid rows and retained layers. | Does not remove projection/MLP work. |
| Compatibility | Can contain online attention. | Can run within any chosen outer chunk size. |

On CPU, tile selection, vector kernels, quantization, packing, threading, and
cache reuse determine the latency tradeoff. Long global histories offer a
larger opportunity than local layers whose history is already bounded. Small
tiles or repeated packing can outweigh saved score traffic. A GPU algorithm's
published speedup is not a prediction for these CPU implementations.

## 5. Existing support and the remaining gaps

| Area | Inspected local support | Work needed for the proposed path |
| --- | --- | --- |
| Tensor graph export | StableHLO composites and decompositions can be serialized. | Add typed attention/cache authoring helpers and consistent contracts. |
| LiteRT compiled model | `ResizeInputTensor` exists; buffer requirements may change after resizing. | Use it correctly through the runner and supported delegates. |
| Tensor API LiteRT runner | Initial signature buffers, shared bindings, and feedback are available. `LitertDynamicRunner` dynamically loads models but does not expose resizing. | Dynamic-dimension authoring/serialization, shape propagation, resize/rebinding, and resource reuse. |
| LiteRT-LM | Dynamic CPU prefill and configurable chunking already exist. | Export compatible models and connect their bounds/state semantics to delegate execution. |
| Local YNN SDPA | Live-prefix propagation; the local extension admits a specific FP32-query/INT8-KV contract. | Local interval bounds, consistent attributes, numerical qualification, and an online implementation. |
| KV state | Existing cache-update/update-slice mechanisms and feedback/state metadata. | Confirm bounded updates, aliasing, sharing across signatures, and reset/rollback behavior. |
| Gemma4 prefix elimination | Native dependency-based cutoff; cache-only prefill also exists in the inspected LM baseline. | Preserve required graph roots and shared-cache ownership in exported models. |

Sources: [Tensor composite serialization tests](../backends/tflite/tflite_flatbuffer_conversion_test.cc),
[compiled-model API](../../litert/cc/litert_compiled_model.h),
[runner behavior](../runners/litert/README.md),
[LiteRT-LM CPU settings](../../../LiteRT-LM/runtime/executor/llm_executor_settings.h),
[LM prefill execution](../../../LiteRT-LM/runtime/executor/llm_litert_compiled_model_executor.cc),
and [state metadata](../../../LiteRT-LM/runtime/proto/executor_metadata.proto).

Successful delegation is not sufficient evidence: an op can be delegated and
still traverse full capacity, materialize a mask, repack history, or use slower
kernels. Conversely, partition count alone is not a measurement of interpreter
overhead. Each suspected source of cost needs an isolated measurement.

## 6. Proposed attention contract

Prefer a versioned extension or consistent subset of the existing SDPA
composites before introducing another core opcode. Start with the single-
sequence case needed by the native runner; ragged batches, paged caches, and
ring buffers need explicit extensions rather than implicit assumptions.

An illustrative interface is:

```text
Attention(
    Q, K_cache, V_cache,
    query_start, kv_begin, kv_end,
    causal, window_size,
    scale, softcap, optional_mask
) -> O
```

This is a semantic sketch. It does not prescribe input-slot numbers or an
existing callable API. The initial logical shapes could be:

```text
Q:       [1, Hq, T, Dk]
K_cache: [1, Hkv, C, Dk]
V_cache: [1, Hkv, C, Dv]
O:       [1, Hq, T, Dv]
```

The V shape is a logical description, not a requirement to transpose existing
transposed-V bundles on every invocation. Define supported layout metadata and
backend packing agreements; do not treat arbitrary strided views as already
supported by every TensorBuffer/backend combination.

Required semantics:

1. **Runtime bounds.** `kv_begin` is inclusive, `kv_end` is exclusive, and both
   refer to absolute positions in the initial linear-cache contract. Validate
   `0 <= kv_begin <= kv_end <= C`. Equal bounds mean empty; `kv_end=0` does
   not inherit a backend's legacy "zero means full capacity" behavior.
2. **Query positions.** Row `r` has position `query_start + r`. With causal
   attention it can see `p <= query_start + r`. With window length `W`, it
   additionally requires `p >= query_start + r - W + 1`. These conditions
   intersect the supplied KV interval and optional mask for each row. Omit
   the window for global attention; a supplied window must be positive.
3. **Query validity.** Prefer exactly `T` valid query rows. If padded queries
   are supported, expose their valid count separately from `kv_end`; define
   output padding and prevent invalid rows from updating the cache.
4. **Head mapping.** Require `Hq % Hkv == 0`; query head `h` reads KV head
   `floor(h / (Hq/Hkv))`. Physical head duplication is not required.
5. **Scores and masks.** The proposed Q is unscaled: begin with
   `score = scale * dot(Q, K)`, then apply optional positive softcap, masking,
   and normalization. Adapt legacy prescaled Q explicitly to avoid double
   scaling. A proposed canonical rule is Boolean `true=visible`, floating
   masks additive, and zero output for an empty row. The fallback must
   implement the same rule. Existing contracts with other finite sentinels
   or empty-row behavior require explicit adaptation/versioning.
6. **Quantization.** Specify supported K/V types, scales, zero points, and
   accumulation expectations. Distinct K/V quantization is allowed only where
   implemented. Do not silently reinterpret signed bytes or change activation
   precision when selecting a fast path.
7. **Observable output.** Return attention outputs, keeping scores and
   probabilities internal. Tile sizes and running accumulators normally belong
   to the backend, so improving its algorithm does not require model re-export.

Structured causal/window metadata should avoid requiring a dense mask for
those common cases. An arbitrary additive mask remains an optional capability
whose storage and processing costs must be accounted for.

The initial lowering can use compact matmul, softmax, and matmul. A later
lowering can perform online tiled attention under the same contract. The
decomposition supplies a correctness reference and fallback; supported
delegates must preserve the composite until their attention lowering sees it.
Rebuilding a runtime alone does not rewrite old exported BMM regions.

### Resolve existing contract differences before sharing artifacts

The [operator inventory](../../docs/nonstandard_ops.md) records backend-specific
differences in runtime parameter indices, scale handling, mask semantics,
supported layouts, and softcap attributes. In the inspected local CPU SDPA
code, the cap attribute is `logit_cap`; the documented transposed GPU/converter
contract uses `softcap`. Runtime length interpretation also differs by backend.

Choose a canonical version, define adapters for legacy artifacts, and test the
fallback and each backend against it. Do not change a parameter's interpretation
or assume support transfers between similarly named composites.

### Numerical behavior of an online INT8-KV path

Mathematical equivalence of online softmax does not promise bitwise equality
after changing reduction order, alignment, or quantization. The native path
uses FP32 queries/probabilities with INT8 KV through FP32/QCINT8 kernels. The
local YNN path dynamically quantizes query and probability reduction rows for
INT8 matrix multiplication.

Quantizing each probability tile independently changes its quantization domain
relative to quantizing a complete row. An online replacement must define and
validate its numerical contract; it cannot assume that the existing YNN
probability quantization carries over unchanged. Preserve published model
weights and KV scales, and distinguish algorithmic changes from precision
changes in both correctness and performance comparisons.

## 7. Proposed cache and runner changes

Keep cache update and attention separately expressible:

```text
K_next, V_next = CacheUpdate(K_prev, V_prev, K_new, V_new,
                             write_start, valid_new_rows)
O = Attention(Q, K_next, V_next, ...)
```

This is also a proposed semantic sketch. Existing `odml.cache_update` or
update-slice mechanisms may suffice if their lowering satisfies the desired
behavior. A functional graph can describe updated state while a memory planner
aliases storage when dependencies permit it.

For Gemma4, only an owner writes its cache; multiple subsequent layers read it.
Preserve that sharing across layers, prefill chunks, and decode signatures.
Do not require every attention invocation to append K/V. Backend fusion of
append and attention remains possible when the ownership and ordering allow it.

The execution requirements are:

- Update only valid new rows. Padded rows must not become visible state.
- Reuse persistent storage and avoid copying unchanged capacity on append.
- Bound conversion/packing to useful data, or maintain compatible persistent
  packing. A borrowed pointer alone does not prove the absence of packing.
- Keep storage capacity, valid history, absolute positions, and any aligned
  physical view distinct.
- Preserve live state and shared-cache bindings through query resizes.
- Define initialization, reset, cancellation, and rollback visibility.
  Resetting a logical length need not zero memory, but stale entries must not
  influence subsequent output or become visible after a failed append.
- Share packed weights, thread pools, and reusable scratch where supported.
  Scratch growth must not invalidate outputs or state still used elsewhere.

For the Tensor API's LiteRT runner, add an explicit resize-and-bind path that
propagates dynamic dimensions, re-queries buffer requirements after resizing,
and only reallocates buffers when necessary. Existing feedback swaps buffer
handles between invocations; it does not prove that a delegated cache update
performs an in-place bounded write. Check both layers.

For LiteRT-LM, connect the same exported contract to its existing chunk policy
and state metadata. Retain cache-only prefix signatures and a full final-token
path. The Gemma4 cutoff is established by graph dependencies and required
outputs; it does not require a Gemma4-specific inference opcode.

### Scratch reuse and allocation ownership

Shared scratch means reusing temporary storage after its previous contents are
no longer needed. TFLite already supports this within its arena: graph edges
identify tensor lifetimes, and the planner can assign overlapping storage to
tensors whose lifetimes do not overlap. Operator temporaries registered with
the arena are also included. No explicit buffer-reuse operator is required.
See [ArenaPlanner](../../tflite/arena_planner.cc) and the
[TensorFlow memory-arena explanation](https://blog.tensorflow.org/2023/08/simpleperf-case-study-fast.html).

The native change concerned a different allocation boundary:

| Storage | Who plans or owns it? | Scope of reuse |
| --- | --- | --- |
| TFLite arena-managed intermediate tensors and registered temporaries | TFLite's arena planner | Non-overlapping lifetimes visible to that planner. |
| XNNPACK internal tensors and operator workspaces | XNNPACK's memory planner | Non-overlapping lifetimes within an XNNPACK runtime. Delegate-private allocations are not automatically managed by the TFLite arena. |
| Scratch retained by separate native stage runtimes | Caller-owned shared XNNPACK workspace | Serialized stage invocations borrow the same allocation. Each stage still has its own internal allocation plan. |

Our earlier native runner retained 151 private XNNPACK workspaces. Each runtime
could reuse storage internally, while all those private arenas remained
allocated between invocations. Attaching the sequential stages to a shared
workspace changed retained scratch from approximately the sum of their
individual requirements to the largest requirement, subject to alignment and
retained capacity. For example, separate 4, 6, and 8 MiB workspaces would retain
18 MiB; sharing sufficient storage for these sequential invocations would need
about 8 MiB. This is illustrative, not a measurement.

The native [StageRunner](../examples/gemma4/native/stage_runner.h) passes a
shared workspace to `xnn_create_runtime_v4`. Stage preparation and complete
invocations are serialized. Persistent KV, weights, and outputs needed by
later stages stay outside this scratch; those live values cannot be overwritten
when the next stage begins.

The current local [TFLite XNNPACK delegate](../../tflite/delegates/xnnpack/xnnpack_delegate.cc)
already implements the same cross-runtime mechanism within one delegate
instance: it creates one workspace, passes `delegate.workspace()` to its
XNNPACK subgraph runtimes, and locks a workspace mutex during their preparation
and invocation. The native improvement therefore does not establish a missing
TFLite graph capability or a failure of ordinary arena lifetime planning.
It also does not establish sharing across separate delegate instances,
interpreters, or compiled models, or describe YNNPACK's allocation policy.
Historical benchmark binaries require their own allocation audit.

For the runtime proposal, first identify the allocation owners and existing
sharing scope. Additional sharing, if needed, belongs in workspace ownership,
lifetime, and execution scheduling support. A new `shared_scratch` graph
operator is not required for this optimization. Arena reuse also cannot remove
a large tensor while consumers still need it; reducing that live requirement
requires a change such as chunking, bounded attention, or online attention.

## 8. What the existing measurements establish

The saved September 16 graph experiment rewrote official Gemma4 decode
attention regions into SDPA while keeping the compared variants in the same
patched LiteRT-LM/YNN runtime. At prompt length 1,024 and cache capacity 8,448:

| Phone | Original decode tokens/s | SDPA decode tokens/s | Throughput gain |
| --- | ---: | ---: | ---: |
| TECNO LJ9 | 19.95 | 21.59 | 8.3% |
| Pixel 8 | 15.27 | 17.63 | 15.5% |
| Samsung SM-S937U1 | 23.06 | 26.37 | 14.3% |

See the [recorded methodology and ranges](ynnpack_sdpa_graph_performance.md#results).
These are decode results from a specific local experiment, not measurements
of the proposed online kernel or proof that chunked prefill caused the gain.
The report also retains inconclusive long-context gains and a Samsung
long-context regression; the improvement is not universal.

That rewrite reduced unnecessary intermediate work but did not reach the
native runner. Native restricts local attention to its aligned window; the
tested YNN SDPA path still reads the full live prefix for local layers. Native
and YNN also use different attention arithmetic/kernels, and runtime boundaries
remain. The separate native comparison includes device frequency and thermal
differences; it is not an equal-clock kernel ablation.

The earlier [capacity control](native_mobile_decode_analysis.md) identifies
unused-capacity attention as a major measured cost in the LM-XNN path. YNN
already avoids much of that matrix work. Do not transfer the XNN attribution
to YNN, treat all remaining time as interpreter overhead, or assume native has
exclusive INT2 support. The saved evidence does not isolate every remaining
millisecond.

### Memory reductions need a named baseline and measurement phase

LiteRT-LM supporting chunked prefill does not imply that it uses the same
chunk size, attention extent, or allocation lifetimes as native. Its dynamic
CPU chunk-size setting applies to dynamically exported models and defaults
to no outer chunking. Static models use a separate signature-selection policy.
The saved 1,024-token LM diagnostic invoked its cache-only prefix graph once;
the native runner instead reused 128-row stages. See the
[current executor](../../../LiteRT-LM/runtime/executor/llm_litert_compiled_model_executor.cc)
for current scheduling and the
[historical diagnostic report](/data/bt/os/llama.cpp/tmp_models/gemma4_qat_q4_0_perf_graph_compare_20260914/ynnpack-20260915T070241Z/PERFORMANCE_REPORT.md)
for that measured run. Current source and historical binary behavior must not
be treated as interchangeable evidence.

Chunking limits query rows. It does not by itself bound history columns or
release other retained allocations. For an illustrative original capacity-wide
attention tensor, `8 * 1024 * 8448 * 4` bytes is 264 MiB of FP32 scores. A final
native chunk at history 1,024 has `8 * 128 * 1024 * 4` bytes, or 4 MiB. These
logical sizes combine two independent choices: query chunk size and history
extent. They are not measured RSS, and the capacity-wide example must not be
assigned to a YNN SDPA lowering that already carries live extents internally.

The largest isolated native before/after memory improvement in the archived
review was **726–797 MiB less current RSS** after the last decode of the second
session. Both native versions already used 128-row chunked prefill. Two other
changes account for the observed direction and approximate scale:

- Preserving the original INT2 MLP coefficients reduced their source payload
  from 540 to 270 MiB. Recorded packed-weight capacity fell separately from
  about 1,021.9 to 751.9 MiB. This removed widening introduced in the earlier
  native adapter; compact INT2 support is not exclusive to native.
- Sharing scratch across 151 sequential native stage runtimes replaced
  197–268 MiB of private workspace capacity with one roughly 8.8–9.1 MiB arena.
  This is a measured native before/after change, not evidence that LiteRT-LM
  owns 151 such private arenas. The current XNNPACK delegate already shares
  scratch within each delegate instance; see
  [allocation ownership](#scratch-reuse-and-allocation-ownership).

Allocation capacities and source bytes cannot simply be added to RSS, but the
individual toggles also produced large RSS reductions in that experiment. See
the [archived memory-change review](../experiments/history/2026-09-15-native-runner-improvements.md#staged-execution-and-memory-reductions)
and [phase observations](/data/bt/tmp/gemma4-memory-20260913/analysis/results/MEMORY.md).

For Pixel 8 in that older phase audit, with both runners still owning KV:

| Prompt | Earlier native current RSS | Compact/shared native current RSS | LiteRT-LM CPU current RSS |
| --- | ---: | ---: | ---: |
| 128 | 2,637.5 MiB | 1,897.8 MiB | 1,516.9 MiB |
| 1,024 | 2,785.3 MiB | 1,988.8 MiB | 2,004.7 MiB |

Thus the 1,024-token native before/after reduction was 796.5 MiB, while its
advantage over that LiteRT-LM configuration was only 15.9 MiB of current RSS.
At 128 tokens, the optimized native runner used 380.9 MiB more. Comparing a
native post-session snapshot after KV release with an LM snapshot retaining
KV would introduce another ownership mismatch.

A later, separate Pixel 8 native-versus-YNN-SDPA comparison at prompt 1,024 and
capacity 8,448 reported median **process-lifetime peak RSS** of 2,054.9 MiB
for native and 2,891.4 MiB for LM-YNN-SDPA, a real 836.5 MiB difference in that
metric. The [summary](/data/bt/tmp/gemma4-ynn-sdpa-export-20260916/perf/pixel8/native-comparison/summary.json)
reports peaks rather than a matched allocation breakdown. It does not isolate
how much comes from chunk size, prefill temporaries, weight packing, backend
allocation lifetimes, or initialization. Do not combine this peak comparison
with the earlier current-RSS audit to manufacture a causal breakdown.

The memory proposal therefore needs separate accounting for source/packed
weights, persistent KV, live temporary tensors, retained workspace, and runtime
allocations, with identical phase boundaries. Equal chunk sizes are one
control in that experiment; they are not sufficient to equalize memory use.

## 9. Implementation order and validation

Deliver the work in independently reviewable stages:

| Stage | Deliverable | Evidence required |
| --- | --- | --- |
| 1. Contract and authoring | Typed SDPA/cache helpers, versioned semantics, faithful decomposition, correct prefix roots. | Serialized models and independent semantic tests. |
| 2. Runner and state | Dynamic query rows, resize/rebinding, shared persistent KV, reusable weights/resources. | Correct results across changing shapes, chunks, and sessions without unintended state copies. |
| 3. Bounded attention | Live prefix and local interval lowering with ordinary attention kernels. | Work extents follow useful bounds; isolated prefill/decode comparisons. |
| 4. Online attention | Backend tiling and running normalization behind the same SDPA interface. | Tile-bounded intermediates, qualified numerics, and target-device measurements. |
| 5. Policy tuning | Choose whole-prefix or chunked execution and suitable chunk sizes per workload. | End-to-end latency and memory tradeoffs with the other stages held constant. |

The first performance milestone is to reproduce native's useful work through
LiteRT, before attributing a gain to online attention. Neither the proposal
nor the presence of SDPA alone establishes parity with native.

### Correctness coverage

Use an independent attention reference and full-model checks appropriate to
each stage:

- Query lengths around chunk/tile boundaries, exact and partial final chunks,
  and changing lengths on one reused runner.
- Nonzero query offsets, causal boundaries, local windows, empty rows, and
  history/capacity limits. Local prefill needs per-row windows, not just the
  decode window applied to every row.
- Poisoned inactive/stale cache entries to detect unintended reads, while
  separating NaN behavior from finite quantized-cache tests.
- GQA/MQA head mapping and multiple layers consuming one cache owner.
- Published KV scales, distinct K/V quantization, supported zero points, and
  signed INT8 endpoints including `-128`.
- Cache append equivalence, state visibility after cancellation/rollback,
  reset, and prefill-to-decode transitions.
- Explicit evidence that the intended delegate path ran; separately validate
  its decomposition fallback.
- Logit/error distributions and representative generation or evaluation
  checks. Argmax agreement alone is insufficient to establish numerical parity.

### Performance comparisons

Compare native, Tensor API through LiteRT, and LiteRT-LM using matching weights,
quantization, context capacity, useful query/history spans, CPU affinity,
threads, and retained prefix computation. Record revisions, local changes,
build flags, and binary/model hashes. Keep cold initialization and warm reusable
execution separate; include shape-change costs in the relevant end-to-end run.

Use isolated experiments for exact rows versus padding, whole-prefix versus
chunked prefill, capacity versus live prefix, live prefix versus local window,
and ordinary versus online attention. Do not change precision and kernel
schedule in the same ablation without a separate numerical/performance control.

Measure prefill/first-prediction latency, subsequent decode latency, peak
temporary memory, persistent KV storage, allocations, and packing/copy costs.
Logical tensor bytes, allocator peak, RSS, and measured DRAM traffic answer
different questions. Collect profiling separately from timing, repeat and
alternate variants on phones, and retain thermal/frequency observations and
unfavorable results.

Acceptance should demonstrate that fixed useful input work no longer grows
with unused capacity, local attention stops growing with obsolete history,
cache updates scale with valid new rows, and online score scratch follows tile
sizes rather than `T*S`. Qualify these checks for documented alignment,
allocation thresholds, and backend layouts. Choose defaults only after the
target-device latency/memory results support them.

## 10. Decisions still requiring implementation evidence

- The smallest interoperable attention version and supported layout/quantization
  subset across the intended CPU/GPU backends.
- Whether a direct FP32/INT8 path or the existing dynamically quantized path
  gives the best qualified CPU performance for each phase and head dimension.
- Which state-layout and buffer-binding changes are needed to avoid full-cache
  work without introducing new copies at another boundary.
- How much YNN scheduling already achieves automatically, and which online
  reduction/kernel support must be added explicitly.
- The best outer chunk size after bounded and online attention are available;
  this may include the entire prefix.
- The remaining runtime overhead after useful work, precision, and resources
  have been matched. Existing partition counts cannot answer this alone.
