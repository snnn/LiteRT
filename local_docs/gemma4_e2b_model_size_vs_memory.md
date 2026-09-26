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

# Why a 2.59 GB Gemma4 E2B package can run with substantially less resident memory

The model file contains all data needed for the model's vocabulary and its
supported input types. A text request uses only part of that data. In
particular, Gemma4 E2B has very large embedding tables, but each token reads
only a small row from each table.

That explains how a file larger than 2 GB can run with substantially less
resident memory. It does **not** establish that our complete runner uses
less than 1 GB. Our recorded phone peaks were **1,134.6–1,168.8 MiB**, or
approximately **1.19–1.23 GB**. The earlier 862.125 MiB figure was an ideal
subtotal, before embedding pages, working memory, and runtime overhead.

This analysis combines the current published container's header, the
benchmarked model's tensor metadata, the saved phone workload, and new
Linux memory diagnostics. The numerical file sizes and allocation sizes
are kept separate throughout. No new inference-speed claim is made.

## 1. What is actually inside the file?

The inspected published package is **2,588,147,712 bytes**: 2.588 GB in
decimal units, or **2,468.25 MiB**.

| File sections | Size on disk | Role in a text-only request |
| --- | ---: | --- |
| Main language-model graph | 780.37 MiB | Supplies the core calculations; the CPU uses a packed representation of its dense weights. |
| Per-layer embedding section | 1,225.01 MiB | Supplies selected vocabulary rows as tokens are processed. |
| Main token embedding section | 99.00 MiB | Supplies the selected token's input representation. |
| Audio and vision sections | 316.99 MiB | Unused by our text-only token runner. |
| Other bundled graphs, tokenizer, metadata, and alignment | 46.88 MiB | Usage depends on the execution path; our benchmark supplies token IDs directly. |
| **Total** | **2,468.25 MiB** | **A storage total, not an RSS requirement.** |

The two embedding sections account for **53.6% of the file**. Their size is
the main reason a calculation based only on dense execution weights can
look surprisingly small compared with the download.

The section sizes above come from the currently published file. The deeper
tensor and runtime accounting uses the locally audited benchmark artifact;
its main language-model and embedding section sizes match the values above.
This does not assert byte-for-byte identity of different published files.

The publisher describes the same deployment approach: mixed low-bit text
weights, mapped embedding tables, and audio/vision components loaded as
needed. See the [LiteRT-LM model description](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm).

## 2. “E2B” describes effective parameters, not two billion equally active values

Google describes E2B as approximately **2.3 billion effective parameters**
and **5.1 billion including embeddings**. The additional per-layer
embedding parameters are large lookup tables. See the
[Gemma4 model description](https://huggingface.co/google/gemma-4-E2B-it).

The relevant distinction is how the data is accessed:

```text
Dense computation: use a large weight matrix to transform a token's state.
Embedding lookup: select the row associated with that token's ID.
```

For the per-layer tables:

```text
Stored values = 262,144 vocabulary entries × 35 layers × 256 values
              = 2,348,810,240 values

Values selected for one token = 35 × 256 = 8,960 values
```

Each token selects one vocabulary row out of 262,144. The model retains the
full vocabulary on storage, while the current request needs only its selected
rows in RAM. This is part of the model's architecture; no learned values
have to be discarded to get this behavior.

The vocabulary output projection still evaluates scores for every possible
next token. Its dense weights are included in the packed-weight budget.
Sparse input-embedding lookup does not mean the runner evaluates only a
small subset of possible output tokens.

## 3. Why the dense weights fit in about 752 MiB

The audited core graph's distinct fully connected weight payloads are:

| Representation | Original coefficient payload |
| --- | ---: |
| INT2: four values per byte | 366.000 MiB |
| INT4: two values per byte | 335.250 MiB |
| INT8: one value per byte | 39.375 MiB |
| **Total** | **740.625 MiB** |

These matrices contain approximately 2.279 billion coefficient values.
Their different precisions explain why their payload is much smaller than
two billion four-byte floating-point values.

The measured packed-weight payload is **751.875 MiB**, about **11.25 MiB
larger** than the original coefficient payload. Packing arranges data for
efficient kernels and includes additional execution information. It does
not explain a 2.59 GB to 0.75 GB compression ratio: the latter number covers
the dense execution weights, not the entire package.

A warm persistent cache lets the process use those packed weights without
also retaining unnecessary source and conversion copies. Memory mapping
alone would not help if the runner read and retained every source-weight
page as well.

## 4. Embedding file bytes, metadata, and lookup bytes

The published embedding sections have a particularly clear breakdown:

| Component | Main token embeddings | Per-layer embeddings |
| --- | ---: | ---: |
| Quantized codes | 96 MiB, INT2 | 1,120 MiB, INT4 |
| Scaling values | 1 MiB | 35 MiB |
| Serialized zero-point arrays | 2 MiB | 70 MiB |
| Graph and other small metadata | About 0.003 MiB | About 0.012 MiB |
| **Section total** | **99.003 MiB** | **1,225.012 MiB** |

Scales and zero points describe how to interpret the compressed numbers.
These are serialized sizes, not a claim that every runtime must retain
exactly that amount of metadata in physical RAM.

Our runner's preparation changes the storage layout while preserving the
weight values. The main token codes are widened from INT2 to INT4, making
that exported code file 192 MiB. The per-layer codes remain 1,120 MiB, but
are reorganized so all entries for one token are adjacent. Repeated
zero-point metadata can be represented compactly in this loading path.

The complete exported embedding-code files therefore total **1,312 MiB**.
That explains the apparent difference between the previous memory guide's
table size and the published file's embedding-section sizes.

The current runner also retains two copies of its embedding scales, for
**72 MiB** of scale payload. One shared copy would be 36 MiB. These scale
copies are separate from the mapped code pages described next.

## 5. Data layout changes how many pages one lookup touches

The published per-layer representation consists of 35 separate tables.
One token needs 128 code bytes from each table, for a total of 4,480 bytes.
Those fragments are far apart in the file.

The runner's exported representation groups the same 4,480 bytes together:

```text
Published layout: token t → 128 bytes in each of 35 separate tables
Runner layout:    token t → one contiguous 4,480-byte row
```

Operating systems bring memory into RAM in pages, rather than individual
128-byte fragments. Consequently, the original layout can touch many more
pages even though the useful code bytes are identical. Both our earlier
native runner and the production LiteRT runner use the reorganized layout.
This difference from the published embedding layout does not explain the
memory gap between those two custom runners.

The saved 1,024-token prompt contains **280 distinct token IDs**. All 64
forced decode IDs are already in that set. Padding introduces one extra
ID, so the complete diagnostic lookup set has **281 IDs**.

For that set, the raw per-layer code payload is only **1.201 MiB**. Counting
the actual pages that intersect those rows gives:

| Page size | Published separate-table layout | Runner grouped-by-token layout |
| --- | ---: | ---: |
| 4 KiB | 30.922 MiB | 2.207 MiB |
| 16 KiB | 86.719 MiB | 5.266 MiB |

These are calculated code-page footprints, excluding scales, graph metadata,
read-ahead, and unrelated retained pages. The page count follows from the
file offsets and the exact token IDs. It is not a measurement of a whole
LiteRT-LM process.

An isolated Linux lookup experiment used fresh processes, default memory
mapping, and 4 KiB pages to read the same rows in each layout. The resulting
mapped residency was **214.99 MiB** for the separate tables and **15.89 MiB**
for the grouped layout. The code bytes read had identical SHA-256 hashes.

The measured values exceed the minimal page footprints because of the
operating system's page-loading behavior, including read-ahead and mapping
nearby cached pages. File-cache state was not forcibly reset. These values
illustrate a real layout effect under this machine's conditions; they do not
establish a 199 MiB saving for every LiteRT-LM workload or phone.

## 6. What was resident during an actual runner invocation?

A new Linux diagnostic reused the existing production executable and its
packed-weight cache. It processed the same 1,024+64-token fixture with
capacity 8,448, chunk size 128, and four CPU threads. The run used forced
input IDs, so its input history was fixed. Timings from this instrumented
run are not performance measurements.

A late-execution memory-map snapshot totaled approximately **1,106.69 MiB**:

| Resident category | MiB | How it was established |
| --- | ---: | --- |
| Packed-weight cache | 751.89 | Resident pages of the mapped cache file, including small file overhead. |
| Per-layer embedding codes | 15.57 | Resident pages of the 1,120 MiB code mapping. |
| Main token embedding codes | 11.09 | Resident pages of the 192 MiB code mapping. |
| Exported model file | 31.39 | Resident pages of the approximately 759.63 MiB model mapping. |
| Persistent KV bank | 74.25 | Known initialized payload within anonymous memory. |
| Two copies of embedding scales | 72.00 | Known copied payload within anonymous memory. |
| Remaining anonymous/special memory | 140.76 | Residual after subtracting the two known payloads; includes working buffers, runtime structures, allocator retention, and stacks. |
| Other mapped files | 9.73 | Executable, libraries, and other file mappings. |
| **Total** | **1,106.69** | **Approximately 1.16 GB in decimal units.** |

The three anonymous/special rows together account for 287.01 MiB. The KV
and scale split uses verified allocation sizes; it is not a complete
allocation-stack attribution of the remaining 140.76 MiB. No swap was
reported. Displayed values are independently rounded. Small differences
between a sampled map total and the process's reported RSS peak are expected
from the different sampling methods.

The important observation is that the full packed-weight mapping is
resident, while only **26.66 MiB of the 1,312 MiB exported embedding-code
tables** is resident. The runner's low residency is not explained by
swapping out its actively used dense weights in this diagnostic.

The original model file remains mapped, but most original coefficient pages
are unnecessary after loading the packed representation. Cache hits and
source-page reclamation determine how much of that mapping stays resident.

## 7. Why this is not a universal memory ceiling

The same token fixture was repeated during the phone campaign. Repetition
reuses the same embedding rows. It does not test what happens when one
process serves many conversations with new vocabulary, code identifiers,
or different languages.

Embedding residency depends on:

- Which token IDs have been accessed over the process lifetime.
- How the tables group those IDs into pages.
- Page size, read-ahead, cached file pages, and memory pressure.
- Whether earlier pages are retained or reclaimed.

Current sequence length and maximum KV capacity are therefore insufficient
to predict RSS. Resetting a conversation's KV length also does not evict
its embedding pages or reset the process's earlier peak.

As a counterfactual, keeping every exported embedding-code page resident
while leaving the other measured categories unchanged would raise the
diagnostic total from about **1,107 MiB to 2,392 MiB**. This is arithmetic,
not a forced-preload experiment. It demonstrates how strongly the total
depends on the resident portion of the tables.

Prefill chunking and a local KV window address different terms. Chunking
reduces temporary attention and feed-forward buffers. A windowed KV bank
reduces persistent history storage. Neither change removes the embedding
files from the model package.

## 8. How to use these findings

For product planning, specify the workload and memory metric alongside the
number. “About 1.1 GiB peak RSS for the recorded text fixture with a warm
packed-weight cache” describes our phone evidence. “An E2B model always
fits in less than 1 GB” does not.

The model-size difference is primarily explained by selective embedding
access, unused input-modality components, and using one packed copy of the
dense execution weights. Layout makes selective access cheaper in pages;
buffer reuse and chunking reduce additional runtime costs.

One decimal GB is approximately **953.67 MiB**. Even if removing the extra
36 MiB of scales and saving 46.5 MiB through windowed local KV reduced this
diagnostic's RSS by exactly those payload amounts, the result would be
about **1,024.19 MiB**, or **1.074 GB**. Those two identified opportunities
alone would not demonstrate a sub-1-GB process. Actual peak savings would
need to be measured after implementing them.

Useful next measurements would hold prompt length and KV capacity fixed
while varying token diversity and the number of distinct conversations
served by one process. A separate comparison of embedding layouts should
use the same model coefficients, input IDs, page policy, and backend. That
would measure how much of the layout effect carries over to full LiteRT-LM
inference without attributing unrelated runtime differences to it.
