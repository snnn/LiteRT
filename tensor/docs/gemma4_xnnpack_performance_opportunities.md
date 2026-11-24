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

# Improving Gemma4 CPU performance in the LiteRT Tensor API

This report proposes improvements to the upstream Gemma4 Tensor API example
and its XNNPACK backend. The strongest opportunities are to eliminate unused
prefill computation, make local attention process only its active window, and
reduce KV-cache traffic and retained memory. These should take priority over
changing graph partitioning alone.

The review uses `upstream/main` at
[`1360198bb80111e7078c5ccb53017f5c07bab29b`][baseline], inspected on
November 23, 2026. Findings describe that source snapshot. Memory figures below
are calculations from tensor shapes; they are not measured process memory or
promised speedups. The proposed changes require matched benchmarks.

## Recommended priorities

| Priority | Change | Primary benefit | Implementation scope |
| --- | --- | --- | --- |
| First | Measure the complete token loop and warm operator execution | Reliable attribution and regression detection | Example timing and runner profiling |
| First | Compute only the final prompt position's logits for generation | Lower prefill latency and output memory | Gemma4 graph and output contract |
| Next | Bound local attention to its 512-token window | Less attention work at long contexts | Cache views, graph construction, backend support |
| Next | Prefill only the dependencies of the KV cache; add chunking | Less prompt computation and lower temporary memory | Gemma4 execution schedule and attention semantics |
| Next | Use persistent cache layouts and evaluate smaller cache types | Less cache movement and storage | Runtime, backend interfaces, numerical validation |
| Next | Share scratch resources and retire unused prefill allocations | Lower peak and steady-state memory | Runner ownership and session lifecycle |
| Profile first | Remove redundant conversions and setup; evaluate fused attention | Lower remaining operator and host overhead | XNNPACK lowering and runner implementation |

## 1. Compute only the logits needed for generation

`BuildGemma4Graph` applies the final normalization, vocabulary projection, and
logit soft cap to every prompt position. `ExecutePrefillPass` then selects only
the last position for the first generated token. Earlier rows are unused by
this generation flow. [Graph output construction][lm-head],
[prefill output selection][prefill-logits].

Add an explicit output mode for generation that selects the final hidden-state
row **before** the final normalization and LM head. Retain the existing all-token
mode for scoring and evaluation. Selecting a row after the LM head would keep
the expensive projection work.

For batch size 1 and a 1,024-token prompt, the configured 262,144-token
vocabulary produces the following logical FP32 logits allocation:

| Output | Shape | Size |
| --- | --- | ---: |
| Every prompt position | `[1, 1024, 262144]` | 1 GiB |
| Final position only | `[1, 1, 262144]` | 1 MiB |

This reduces the projection's output rows by 1,024 times; it does not imply a
1,024-fold reduction in total prefill latency. The transformer still processes
the prompt. Both supported variants use this vocabulary size.
[Model configuration][config].

Validate the selected logits and all cache outputs against the existing path,
including prompt length 1. Record total prefill latency, LM-head time, and peak
memory separately.

## 2. Make local attention visit only its active window

The example configures a 512-token sliding window, but decode creates masks of
length `cache_len + 1` and masks the older positions. Its cache path concatenates
the previous K/V with the current token, then supplies those tensors to dense
QK and probability-times-V matrix multiplications. The mask does not shorten
those matrix dimensions. [Decode masks][decode-prep],
[cache concatenation][cache-path], [attention decomposition][attention].

For a local layer at total context length `T`, supply only
`min(T, 512)` keys and values, including the current token. Global layers must
continue to see the full valid prefix. Ensure layers sharing K/V consume the
same bounded view from their cache owner.

Start with a bounded attention view; then evaluate a ring buffer that also
bounds local-cache storage. Keep absolute token positions for RoPE and preserve
chronological attention across wraparound. For chunked prefill, each query has
a different window: taking the last 512 tokens of an entire chunk is not a
correct substitute for per-query bounds.

At `T = 8192`, reducing local attention width to 512 removes a factor of 16
from the sequence dimension of its QK/PV work. That is an operation-level
estimate; global attention, projections, FFNs, and other work remain.

The shared `TransformerBlock` already exposes parameterized cache updates and
local-cache extraction. Gemma4 currently supplies neither cache parameters nor
a local size, and the XNNPACK transformer mixins do not implement those cache
operations. An implementation needs suitable lowering or a host-managed view;
this is not just a configuration change. [Gemma4 cache wiring][cache-wiring],
[cache operation path][cache-path], [XNNPACK transformer mixins][xnn-mixins].

Validate around lengths 511, 512, and 513, multiple wraparounds, and shared
local/global owners. Confirm the executed matrix dimensions and copy traffic,
not just the mask values.

## 3. Use KV-only prefill, then process the final prompt token

The example builds one full-prompt prefill graph and executes every transformer
layer. Gemma4's upper layers reuse K/V from earlier layers: E2B has 15 cache
owners followed by 20 sharing layers; E4B has 24 owners followed by 18 sharing
layers. For ordinary next-token generation, earlier prompt positions in the
sharing layers do not contribute new persistent state.
[Prefill graph construction][build-graphs], [sharing rule][sharing],
[model configuration][config].

A generation-specific schedule can exploit this dependency structure:

1. Process prompt tokens `0` through `S - 2` through the computation needed to
   produce every owner layer's K/V. Omit the upper sharing layers and LM head.
2. In the final owner layer, stop after the required K/V projection,
   normalization, and RoPE; its attention output and FFN are unnecessary for
   these prefix tokens if no later cache depends on them.
3. Process prompt token `S - 1` through the full network using that prefix
   cache, producing the first next-token logits and completing the prompt KV.

This is a proposal derived from the causal graph, not a measured speedup.
Determine the required computation from cache dependencies rather than
hard-coding a layer count. A one-token prompt skips the prefix stage. Preserve
the all-token execution mode for callers needing all hidden states or logits.

Add chunking to this KV-only stage to bound intermediate attention and FFN
storage and reuse a small set of compiled shapes. Benchmark chunk sizes, since
smaller chunks also reduce matrix batching efficiency. Chunking alone does not
remove global attention's total quadratic prefill work.

Correct chunking requires prior-prefix attention, absolute RoPE offsets, causal
masks, and a final partial-chunk contract. The current parameterized-cache
`TransformerBlock` path explicitly ignores prior cache contents when
`seq_len > 1`; extending that path is necessary before using it for chunked
prefill. [Cache semantics][cache-contract].

Validate owner-cache tensors and final logits under fixed token histories,
especially at chunk boundaries. Floating-point kernel selection may differ
between batched prefill and single-token execution, so establish numerical
tolerances rather than assuming bitwise equality.

## 4. Reduce KV-cache movement and storage

There are two distinct movement costs to address. The graph concatenates the
entire live prefix with each new K/V row. Separately, the host append routine
shifts existing history when there is more than one KV head, because its dense
head stride grows with the sequence length. That second cost applies to E4B's
two KV heads; E2B's one-head append already copies only the new row.
[Cache concatenation][cache-path], [host append implementation][cache-append],
[model configuration][config].

Use persistent storage with stable head strides and direct writes at the token
offset. Expose the valid prefix or local window to attention through an
appropriate backend view/cache operation. Avoid replacing the host shift with
an equally large repack elsewhere. Where layouts match, bind prefill outputs
directly into the persistent cache to remove the initial copy as well.

The existing `SetInput` span overload binds memory without copying it. A timer
label such as “upload” should therefore not be interpreted as a full-history
transfer. Measure actual host moves, graph concatenations, and any backend
packing separately. [Input binding contract][input-binding],
[prefill-to-decode cache initialization][cache-init].

Cache precision is a separate opportunity. The example allocates FP32 host
caches at the full requested generation capacity for every owner, including
local owners. For E2B, batch size 1, and an illustrative capacity of 8,448:

```text
K/V elements = 2 × 8448 × (12 local owners × 256 + 3 global owners × 512)
```

| Cache representation | Raw K/V storage |
| --- | ---: |
| Current FP32, full capacity for all owners | 297 MiB |
| FP16, same layout and capacity | 148.5 MiB |
| INT8, same layout and capacity | 74.25 MiB |
| FP32, local capacity 512 and global capacity 8,448 | 111 MiB |

These figures exclude scales, padding, graph temporaries, and weights. The
current allocation policy is visible in [cache initialization][cache-init].
Window bounding and precision reduction are independent experiments.

Evaluate FP16 and INT8 only with a compatible attention path and explicit
quality criteria. INT8 needs a scale/zero-point contract shared by each cache
owner and its consumers. Expanding the whole cache to FP32 on every token can
offset the bandwidth benefit. Report conversion cost, full token latency, and
model-quality results alongside storage savings.

## 5. Share scratch resources and reduce retained allocations

The example creates separate prefill and decode runtimes, configures a thread
pool for each, and already shares a packed-weight cache when enabled. The
runner also exposes `SetWorkspace`, which the example does not use.
[Runtime construction][compile-runners], [runner resource APIs][resources].

Give a session explicit ownership of a shared XNNPACK workspace for serial
prefill/decode execution. Keep persistent KV and externally retained outputs
outside reusable scratch storage. Verify XNNPACK's workspace setup and lifetime
requirements when switching runtimes; concurrent sessions need isolation or
synchronization. Measure both peak memory and memory retained during decode.

For a single generation request, allow prefill-only runtimes and buffers to be
released after cache handoff. For repeated requests or chunked prefill, make
retention an explicit reuse policy so memory savings do not cause recompilation
on every request. A borrowed/shared thread-pool API could also avoid duplicate
pools; tune prefill and decode thread counts independently before choosing a
default.

Weight loading deserves a separate memory audit. The registered BF16 fallback
materializes an FP32 copy of each BF16 tensor. Preserve compact embedding
storage and convert only requested rows where possible; release other converted
source buffers only after confirming that packing and graph consumers no longer
reference them. Quantized embedding lookup already decodes requested rows, so
extend that principle to supported floating-point storage formats.
[Fallback registration][weight-loading], [BF16 conversion][bf16-fallback],
[embedding lookup][embedding-lookup].

Preserve the existing packed-weight sharing and the release of temporary FP16
packing-scale buffers after runtime creation. These are already implemented.
[Runtime preparation][runtime-create].

## 6. Target remaining overhead with operator profiles

The following candidates should be driven by profiles after the larger work
reductions above:

| Candidate | Current evidence | Suggested experiment |
| --- | --- | --- |
| Reuse activation quantization | Each call to `DynamicallyQuantizeInput` emits a new conversion for a quantized FC input. | Inspect the optimized XNNPACK graph for duplicate conversions of the same Q/K/V or gate/up input. If retained, reuse a conversion keyed by input and quantization/layout contract. |
| Consolidate per-layer input projection | Gemma4 issues a separate FC from the same embedding input for each layer. The weight hook rejects quantized projection slices. | Compare one combined projection followed by views and per-layer normalization; alternatively implement scale/zero-point-aware weight slicing. Preserve the normalization axis and measure temporary-memory growth. |
| Stabilize shapes and reuse host buffers | Cache and mask dimensions grow each token, and two mask vectors are allocated in each decode call. | Reuse mask buffers; evaluate bounded local shapes and modest global length buckets. Account for padded attention work before claiming a gain. |
| Fuse or tile attention | Attention is expressed as QK, optional soft cap, masking, softmax, and PV. | Evaluate a lowering with tiled/online softmax that reduces score/probability materialization while preserving causal, local, and grouped-query semantics. |

Sources: [dynamic conversion helper][dynamic-quant],
[FC lowering][fc-lowering], [per-layer projection][per-layer-projection],
[weight slicing limitation][projection-slicing], [decode preparation][decode-prep],
[attention decomposition][attention].

The runner already skips shape propagation and runtime reshape when shapes are
unchanged. Growing global attention still changes shapes in the combined graph,
even after local attention reaches a fixed window. Bucketing therefore needs a
valid-length/masking contract, initialized padding, and measurements of the
reshape-versus-extra-compute tradeoff. [Existing shape reuse][shape-reuse].

Grouped-query attention already folds query groups into matrix rows rather than
explicitly repeating K/V heads. Preserve that behavior.
[Grouped-query implementation][attention]. For conversion reuse and attention
fusion, inspect the final optimized graph: source-level operation counts do not
prove how many kernels execute.

## Measurement and acceptance criteria

Improve measurement before ranking smaller optimizations. The current decode
timer is scoped to `ExecuteDecodeStep`, while `UpdateKvCache` runs afterwards.
Report the complete loop time as well as phase timings, so cache improvements
appear in the headline metric. Also, `LRT_XNN_PROFILE=1` logs only each runtime's
first invocation; add selectable warm iterations and context lengths for
operator attribution. Use separate profiling and uninstrumented timing runs.
[Decode timer][decode-prep], [generation loop][decode-loop],
[profiling implementation][profiling].

Use a repeatable benchmark matrix:

- Test E2B and E4B with fixed checkpoint, quantization metadata, token IDs,
  context capacity, and generated-token count. Include prompts of 128, 1,024,
  4,096, and 8,192 tokens where supported, plus single-token correctness cases.
- Feed the same continuation tokens when comparing decode implementations so
  small numerical differences cannot change the workload. Use at least 64
  decode steps for sustained measurements.
- Report load/compile/packing time, warm request time to first token, complete
  decode milliseconds per token, and context-dependent latency. Specify whether
  tokenization and sampling are included. Measure cold cache creation separately
  from reuse of a packed-weight cache.
- Hold device, build flags, dependency revisions, thread count, affinity, and
  thermal conditions constant. Alternate baseline/candidate runs; report repeat
  counts, median, and dispersion rather than a single best result.
- Measure peak and post-prefill memory, separating anonymous allocations from
  file-backed weights where practical. Include workspace, cache, and temporary
  sizes; raw tensor savings do not translate directly into RSS savings.
- Check logits and cache contents across short/long requests, chunk boundaries,
  window boundaries, wraparound, and session reset. Precision changes also need
  perplexity or task-quality evaluation; matching a few greedy tokens is
  insufficient.

Land independently measurable changes. Start with complete timing and the
last-position LM head, then bounded local attention and persistent cache
handling. Develop KV-only/chunked prefill against that cache contract. Evaluate
precision, resource retention, and profile-driven operator changes separately,
so their benefits and regressions can be attributed.

[baseline]: https://github.com/google-ai-edge/LiteRT/commit/1360198bb80111e7078c5ccb53017f5c07bab29b
[config]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/gemma4/gemma4_config.h#L95-L148
[lm-head]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/gemma4/gemma4_graph.h#L438-L477
[prefill-logits]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/gemma4/gemma4_runtime.cc#L342-L360
[decode-prep]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/gemma4/gemma4_runtime.cc#L364-L416
[cache-path]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/ops/transformer/transformer_block.h#L380-L459
[attention]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/ops/transformer/transformer_ops.h#L289-L399
[cache-wiring]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/gemma4/gemma4_graph.h#L355-L371
[xnn-mixins]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/ops/transformer/transformer_ops_xnnpack.h#L32-L64
[build-graphs]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/gemma4/gemma4_runtime.h#L156-L177
[sharing]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/gemma4/gemma4_graph.h#L89-L118
[cache-contract]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/ops/transformer/transformer_block.h#L765-L790
[cache-append]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/gemma4/gemma4_runtime.cc#L144-L162
[input-binding]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/runners/common_nnpack/runner.h#L75-L109
[cache-init]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/gemma4/gemma4_runtime.cc#L475-L526
[compile-runners]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/gemma4/xnnpack_main.cc#L109-L153
[resources]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/runners/xnnpack/runner.h#L89-L112
[weight-loading]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/gemma4/gemma4_runtime.cc#L189-L213
[bf16-fallback]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/gemma4/gemma4_weights.cc#L188-L203
[embedding-lookup]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/gemma4/helpers/quantized_embedding.cc#L137-L185
[runtime-create]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/runners/xnnpack/runner.cc#L116-L131
[dynamic-quant]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/backends/xnnpack/arithmetic.cc#L55-L69
[fc-lowering]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/backends/xnnpack/arithmetic.cc#L697-L777
[per-layer-projection]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/gemma4/gemma4_graph.h#L374-L408
[projection-slicing]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/gemma4/gemma4_weights.cc#L136-L185
[shape-reuse]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/runners/common_nnpack/runner.cc#L159-L274
[decode-loop]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/examples/gemma4/gemma4_runtime.h#L360-L393
[profiling]: https://github.com/google-ai-edge/LiteRT/blob/1360198bb80111e7078c5ccb53017f5c07bab29b/tensor/runners/xnnpack/runner.cc#L44-L111
