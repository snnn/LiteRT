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

# Estimating memory use for the Gemma4 E2B CPU runner

A large language model (LLM) runner is the program that executes the model.
It needs memory for model weights, the conversation history, and temporary
calculations. Those parts grow for different reasons. Understanding them
separately helps explain benchmark results and identify useful optimizations.

For our Gemma4 E2B configuration, a useful starting point is **862.125 MiB for
packed weights, one copy of the embedding scales, and a KV capacity of 8,448
tokens**. This is a subtotal. Embedding pages, temporary buffers, and runtime
overhead must still be added.

The current implementation also retains a second copy of the embedding scales,
adding 36 MiB and bringing that subtotal to **898.125 MiB**. This distinction
between an ideal budget and the implementation is central to using the model
correctly.

**This does not demonstrate a complete runner below 1 GB.** The recorded
phone peaks were 1,134.6–1,168.8 MiB, or approximately 1.19–1.23 GB in decimal
units. The published model package occupies approximately 2.59 GB on disk.
It includes large embedding tables accessed by row, plus audio and vision
components unused by a text-only request. File size is therefore different
from the resident memory required by that request.

The [model-size and resident-memory analysis](gemma4_e2b_model_size_vs_memory.md)
breaks down the published container, reports a new memory diagnostic, and
explains why embedding layout and token diversity affect the result.

## 1. A few terms before the math

| Term | Meaning |
| --- | --- |
| Token | A unit of text processed by the model. A token can be a word, part of a word, or punctuation. Token count is not word count. |
| Weights | The learned numerical parameters of the model. They stay the same across conversations. |
| Packed-weight cache | A saved copy of weights arranged for efficient execution by the CPU. Loading this cache avoids rebuilding that arrangement. |
| KV cache | Saved information about previously processed tokens. Attention uses it to refer to earlier text without recomputing that information. It changes as the conversation grows. |
| Prefill | Processing the input prompt before generating the first answer token. |
| Decode | Processing one new token at a time during generation. |
| Chunked prefill | Processing the prompt in groups, such as 128 tokens per group, to limit temporary memory while retaining efficient batched computation. |
| Embedding | A numerical representation looked up for a token. This model uses both a main token embedding and additional per-layer embeddings. |
| Scratch space | Temporary memory that operators can reuse once previous calculations no longer need its contents. |
| Resident memory, or RSS | The memory pages currently present in physical RAM for the process. A page is the unit in which the operating system manages memory. |

The packed-weight cache and the KV cache are unrelated caches. Having the
first one does not remove the memory required by the second one.

All calculations use **KiB and MiB**: one KiB is 1,024 bytes, and one MiB is
1,048,576 bytes. INT8 values occupy one byte, packed INT4 values occupy half
a byte, and FP32 values occupy four bytes.

## 2. What “ideal” means here

The baseline assumes one active request, our current model representation,
INT8 KV storage, and a previously created packed-weight cache. It assumes
the CPU can keep the execution weights resident for fast inference. It also
assumes unnecessary source-weight copies have been released and temporary
buffers are reused when their lifetimes allow it.

This is an efficient execution budget, not the smallest theoretical RAM usage
at any performance cost. Continually reloading weights from storage could
reduce residency, but would change the performance problem.

“The cache already exists” also does not mean “all cache pages are already
resident in this process.” The file can be ready on disk while its pages still
need to be brought into memory. The budget includes the packed weights needed
during execution.

The numerical constants describe the audited Gemma4 E2B CPU configuration.
Changing weight precision, kernels, or model variant can change them. The
structure of the memory model applies more broadly.

The packed-weight figure comes from an earlier allocation audit. The KV,
embedding, and attention-matrix sizes are calculated from model dimensions.
Savings from proposed changes are payload estimates, not new benchmark results.

## 3. The overall memory function

Use these inputs:

| Symbol | Meaning | Example |
| --- | --- | ---: |
| `S` | Number of processed tokens in the active context at a particular step | 1,024 |
| `C` | Maximum number of tokens for which KV storage is provisioned | 8,448 |
| `B` | Number of query rows in a prefill chunk | 128 |
| `P` | Input prompt length, before subsequent generation | 1,024 |
| `U` | Number of distinct token IDs whose embedding rows have been accessed and retained | Depends on the text |

The active context must fit the configured capacity: `S <= C`. Temporary
attention shapes can include extra padded positions; the persistent bank's
capacity formula still uses `C`.

The planning equation is:

```text
Runner memory ≈ packed weights
              + embedding data and scales
              + persistent KV storage
              + temporary working memory
              + runtime overhead
```

Each allocation must appear once. A shared buffer is not counted again for
every operator that uses it.

For an exact peak, count all live memory at the same execution point, then
take the largest total. Adding the separate peak of every category can
overestimate memory if those peaks occur at different times. The formulas
below provide a planning budget and exact component payloads, not an exact
prediction of the operating system's peak measurement.

## 4. Packed weights: a mostly fixed cost

The audited packed-weight payload is:

```text
Packed weights = 751.875 MiB
```

This cost is largely independent of prompt length and KV capacity. The
packed-weight cache avoids packing again; it does not make these weights
disappear from RAM during inference.

For the ideal budget, count this payload once. Do not also count redundant
original weights or conversion buffers that execution no longer needs.
Small remaining constants and cache metadata belong in runtime overhead.

## 5. KV storage: capacity and sequence length are different

Gemma4 E2B has 35 computation stages, called transformer layers. Some layers
reuse another layer's KV information. Consequently, there are only
**15 independent KV owners**. An owner means one distinct set of keys and
values that needs storage.

| Owner type | Owners | Values per token, per key or value | Storage per value |
| --- | ---: | ---: | ---: |
| Local attention | 12 | 256 | 1 byte |
| Global attention | 3 | 512 | 1 byte |

Local attention uses a recent window of 512 tokens. Global attention can
refer to the full active history. Both keys and values must be stored, which
explains the factor of two below.

### Current policy: reserve the full capacity for every owner

```text
KV bytes per capacity token
    = 2 × (12 × 256 + 3 × 512)
    = 9,216 bytes
    = 9 KiB

K_current(C) = 9 × C / 1,024 MiB
```

At capacity 8,448:

```text
K_current(8,448) = 74.25 MiB
```

The current runner allocates and initializes this full bank. A short prompt
still pays the capacity cost. Using dynamic shapes for attention restricts
the calculations to the active history, but does not shrink the bank itself.

### Possible improvement: bound local storage to its window

A circular buffer can overwrite old local KV once it is no longer needed.
Global KV still needs storage for the full supported history.

```text
K_windowed(C), in bytes
    = 6,144 × min(C, 512) + 3,072 × C
```

For capacities of at least 512 tokens:

```text
K_windowed(C) = 3 + 3 × C / 1,024 MiB
```

At capacity 8,448, this gives **27.75 MiB**, saving **46.5 MiB** compared
with the current bank. This is a potential storage optimization, not the
behavior of the current runner.

If global storage also grows only as tokens arrive, its persistent live
payload can instead follow the active length:

```text
K_growing(S), in bytes
    = 6,144 × min(S, 512) + 3,072 × S
```

| Active length `S` | Capacity `C` | Current full-capacity bank | Windowed local bank, global capacity reserved | Windowed local bank, global storage grown to `S` |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 2,048 | 18 MiB | 9 MiB | 6 MiB |
| 1,024 | 8,448 | 74.25 MiB | 27.75 MiB | 6 MiB |
| 8,192 | 8,448 | 74.25 MiB | 27.75 MiB | 27 MiB |
| 8,192 | 32,768 | 288 MiB | 99 MiB | 27 MiB |

These are persistent payloads, excluding small allocation padding. A chunk
can need recent history plus its new tokens simultaneously, and a circular
buffer can require temporary handling when it wraps. Those costs belong in
working memory. The storage arithmetic alone does not prove equal speed.

## 6. Embeddings: why the text itself affects resident memory

The model's vocabulary has 262,144 entries. The two INT4 embedding tables
contain 1,536 main embedding values and `35 × 256` per-layer values for each
entry.

```text
All embedding codes
    = 262,144 × (1,536 + 35 × 256) / 2 bytes
    = 1,312 MiB
```

This is the runner's exported representation. The published package stores
the main token table in INT2; the export widens those codes losslessly to
INT4 for lookup. It also combines the 35 per-layer tables into a layout
that places all of one token's entries together. The exported raw-table
sizes therefore differ from the published file-section sizes, which also
include quantization metadata and graph descriptions.

These tables are mapped from files. The runner looks up rows for the token
IDs it encounters; it does not have to read every vocabulary entry.
Therefore, adding the entire 1,312 MiB to resident memory would generally
overstate the required working set for one request.

The complete raw embedding rows for one distinct token occupy:

```text
Embedding bytes per distinct token
    = (1,536 + 35 × 256) / 2
    = 5,248 bytes
    = 5.125 KiB
```

For 1,024 distinct token IDs, the raw row payload is 5.125 MiB. This is not
an RSS prediction: the operating system loads whole pages, may read ahead,
and may retain pages accessed by earlier requests. Nearby rows can share
pages. Repeated tokens reuse the same rows.

Embedding scales are additional numbers used to interpret the compressed
values. One complete set occupies:

```text
One set of embedding scales
    = 262,144 × (1 + 35) × 4 bytes
    = 36 MiB
```

An ideal design within the current loading approach needs one copy. The
current implementation retains the tensor's scale arrays and a separate
copy in the embedding lookup wrapper: **72 MiB of scale payload in total**.
Sharing those arrays would remove 36 MiB of duplicate payload. This has
been identified from ownership; it is not a newly measured RSS saving.
Another possible design would map scales and access only the needed pages.

Thus, neither prompt length nor model-file size alone determines embedding
residency. A precise resident-memory estimate also needs the token IDs,
page layout, and page-retention policy. Our repeated 1,024+64-token phone
fixture contains only 280 distinct token IDs, plus one extra ID used for
padding. Repeating that fixture does not test the resident-memory growth
of a process handling increasingly diverse text.

## 7. Working memory: why chunking matters

Attention computes how strongly each new token relates to earlier tokens.
In ordinary attention, one important temporary array contains a score for
each query token and each history position, for each attention head.

This configuration computes eight parallel attention views, called query
heads, and uses four-byte FP32 scores. For `Q` query rows and an attention
extent of `T` tokens:

```text
One attention matrix
    = 4 × 8 × Q × T bytes
    = Q × T / 32,768 MiB
```

For prefill, `Q = B`; for decode, `Q = 1`. The eight query heads share
one set of keys and values per owner. The score arrays therefore have a
factor of eight that the persistent KV bank does not have.

For planning, `T` is approximately the active context rounded up to the next
multiple of 32. More precisely, our runner uses the prior-history length
plus the configured query rows, then rounds that total. A partially filled
last chunk still has the configured number of rows, so its temporary shape
can exceed its valid token count.

| Global attention extent `T` | One prefill matrix, `B = 128` | One decode matrix |
| ---: | ---: | ---: |
| 1,024 | 4 MiB | 0.03125 MiB |
| 4,096 | 16 MiB | 0.125 MiB |
| 8,192 | 32 MiB | 0.25 MiB |

An ideal implementation can overwrite the score array with its normalized
probabilities. If both arrays coexist, count two matrices instead of one.
The table is not the full attention-workspace budget: masks, query vectors,
KV concatenation, and backend packing can require additional memory.

Without chunking, an 8,192-token prefill would have 8,192 query rows. One
full attention matrix would then occupy **2,048 MiB**. With 128-token
chunks, the largest corresponding matrix is **32 MiB**. The runner reuses
that storage across chunks; it does not keep one matrix for every chunk.

Local attention has a smaller extent. Across a chunk, the union of its
required positions can cover approximately `512 + B - 1` tokens before
alignment, rather than just 512. Global attention continues to grow with
the full active history.

### Other temporary buffers

The feed-forward part of a layer transforms each token's numerical
representation using learned matrices. It also needs intermediate storage.

| Example payload at chunk size 128 | Size | Interpretation |
| --- | ---: | --- |
| Two FP32 prefill feed-forward intermediates of width 6,144 | 6 MiB | Example of retaining both expanded token representations; extra operator intermediates depend on implementation. |
| Main and all 35 per-layer host embedding outputs | 5.125 MiB | Expanded values for the current chunk, distinct from the compressed embedding tables. |
| New INT8 KV for all 15 owners | 1.125 MiB | New rows produced by a chunk before they are committed to the persistent bank. |
| One FP32 vocabulary-score vector | 1 MiB | Scores for the next token; this runner does not produce one such vector for every prefill row. |

Other live values include residuals, projected per-layer inputs, masks, and
kernel workspace. The listed sizes must not simply be added and presented
as an exact peak. Some buffers overlap in lifetime; others can share storage.

The mathematical definition for ideal temporary reuse is:

```text
A_peak = maximum, over execution steps, of
         the sum of all unique temporary buffers still needed at that step
```

Do not multiply one layer's scratch by all 35 layers. Layers execute in
sequence and can reuse memory, except where earlier results remain needed
by later work. A large connected graph can also permit this reuse if the
runtime correctly tracks those lifetimes.

For a whole request, keep prompt length separate from generated length:

```text
Ideal shared working-memory budget
    = max(peak working memory during prefill of the prompt,
          peak working memory during subsequent decode)
```

A long decode run does not retroactively make prefill use that longer
context. Conversely, a small decode score matrix does not guarantee that
decode's total workspace is smaller: KV copies and packing still count.
The current implementation can retain additional buffers separately for
the two phases; perfect reuse is an assumption of the ideal budget.

Tiled attention with online softmax processes the attention scores in small
pieces and updates the result as it goes. It can avoid storing the entire
query-by-history score matrix. That can further reduce working memory,
provided masks and KV handling also avoid large intermediates. It does not
eliminate packed weights or persistent KV storage, and chunked prefill can
still be used together with it.

## 8. A usable budget and worked examples

Let `E_pages` mean resident embedding-code pages, `A_peak` mean the complete
temporary-memory budget, and `R` mean remaining runtime overhead. All three
are measured in MiB below. `E_pages` depends on which token IDs were accessed
and which pages remain resident; the count `U` alone does not determine it.

For the ideal one-copy embedding-scale policy and the current full-capacity
KV bank:

```text
M_ideal ≈ 751.875 + 36 + 9 × C / 1,024 + E_pages + A_peak + R
        ≈ 787.875 + 9 × C / 1,024 + E_pages + A_peak + R MiB

A_peak(P, S, B) = max(A_prefill(P, B), A_decode(S))
```

Here `A_prefill` and `A_decode` represent the complete working-memory peaks
within those phases, including the relevant masks, copies, and packing.
They are not just the single score-matrix formula. This is how sequence
length and chunk size enter the overall function, while capacity enters
through the separate KV term.

For the current two-copy scale ownership, add another 36 MiB. This adjusts
the identified fixed payload; it does not claim all other allocations are
already ideal.

| Active context | KV capacity | Ideal packed weights + scales + KV subtotal | Same subtotal with current scale duplication | One prefill attention matrix, 128 rows |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 2,048 | 805.875 MiB | 841.875 MiB | 4 MiB |
| 1,024 | 8,448 | 862.125 MiB | 898.125 MiB | 4 MiB |
| 8,192 | 8,448 | 862.125 MiB | 898.125 MiB | 32 MiB |
| 8,192 | 32,768 | 1,075.875 MiB | 1,111.875 MiB | 32 MiB |

The last column is one part of `A_peak`, not the amount needed to complete
the subtotal into a process-memory prediction. These examples assume a
prefill chunk reaching the listed aligned context.

For the common capacity of 8,448, the ideal planning function is:

```text
M_ideal ≈ 862.125 + E_pages + A_peak + R MiB
```

With window-bounded local KV, the same ideal subtotal would fall to
**815.625 MiB**. With global storage also grown only to an active length of
1,024, it would be **793.875 MiB**. Both alternatives require implementation
changes and still need embedding pages, working memory, and overhead.

These figures identify known costs and possible savings. They do not
establish a complete minimum-memory target such as “the runner should use
exactly 862 MiB.” In particular, `R` has not been fully derived and should
not be assumed negligible.

## 9. Relating the math to a phone measurement

Three different quantities are often called “memory use”:

| Quantity | What it tells us |
| --- | --- |
| Allocated payload | The bytes reserved for buffers and data structures. This is what the exact component formulas calculate. |
| Resident memory, or RSS | Pages currently present in physical RAM for the process, including resident mapped files and runtime allocations. |
| Peak RSS | The highest resident-memory level recorded during the process lifetime. Setup and earlier requests can contribute. |

A mapped model can have a large address range without all its pages being
resident. Conversely, runtime metadata, allocator retention, read-ahead,
and temporary setup copies can make RSS larger than a minimal live-payload
budget. Shared file pages also require care when adding memory across
multiple processes.

`madvise(..., MADV_PAGEOUT)` asks the operating system to reclaim eligible
pages. For file-backed weights that are no longer needed, this can reduce
current RSS while keeping the mapping and addresses valid. Later access
can bring the pages back. This is a residency change, not a reduction in
the mathematical size of the weights or KV bank.

Reclaiming pages does not erase a peak already recorded. Reclaiming them
before a later allocation can, however, prevent a higher peak. The earlier
LiteRT-LM measurements already benefited from their pageout path; that
saving must not be subtracted from those measurements a second time.

The production runner's earlier phone peaks of roughly 1.1 GiB therefore
cannot be compared directly with the 862.125 MiB subtotal to assign all
remaining memory to runtime inefficiency. Embedding residency, duplicated
scales, temporary buffers, metadata, and setup history all need accounting.
Likewise, a lower RSS than a naive sum of file sizes does not prove better
scratch planning.

## 10. What this model lets us predict

| Change, keeping other conditions fixed | Expected effect |
| --- | --- |
| Increase current KV capacity by 1,024 tokens | Add 9 MiB of persistent KV payload, even for a short prompt. |
| Increase active context without changing capacity | Leave the current bank size unchanged; increase global attention work and potentially workspace and embedding residency. |
| Halve prefill chunk size at the same attention extent | Halve one score matrix and many per-chunk payloads; fixed weights and KV capacity stay unchanged. Speed must be measured separately. |
| Use a 512-token circular buffer for local KV at capacity 8,448 | Save 46.5 MiB of persistent KV payload before accounting for any new temporary handling. |
| Share the currently duplicated embedding scales | Remove 36 MiB of duplicate payload; actual RSS and peak changes need measurement. |
| Use a pre-existing packed-weight cache and avoid retaining packing sources | Remove avoidable setup work and source duplication while preserving the packed execution weights. |

The same budget categories apply to a native XNNPACK runner and a
LiteRT-based runner. Their mathematical work does not by itself require
different weight or KV payloads. Differences in packing, buffer ownership,
temporary lifetimes, graph representation, and page residency determine
how closely each implementation approaches the budget.

To use this model for a product memory target, first fix the capacity,
prompt/decode workload, chunk size, and cache policy. Calculate the exact
persistent terms, then determine the remaining embedding, workspace, and
runtime terms under those conditions. Compare like-for-like current RSS
or like-for-like process peaks; keep those two metrics separate.
