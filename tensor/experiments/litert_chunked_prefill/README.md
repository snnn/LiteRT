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

# Joined chunked prefill experiment

The Gemma4 E2B KV-producing prefix runs as one connected graph per chunk,
using native XNNPACK or LiteRT's Interpreter and XNNPACK delegate. The default
chunk has 128 rows. Both paths retain the cache-only prefix optimization:
14 complete transformer layers followed by the final owner's K/V projections.
The final prompt token uses the complete decode graph to produce logits.
MTP and tiled online-softmax attention are outside this experiment.

| Mode | Prefill | Decode | XNNPACK calls per prefix chunk |
| --- | --- | --- | ---: |
| `staged` | Existing native stages | Joined native | 44 |
| `native` | Joined native | Joined native | 1 |
| `litert` | Joined LiteRT signature | Joined LiteRT signature | 1 |

LiteRT exports a real model with `prefill` and `decode` signatures, loads it
with `InterpreterBuilder`, and applies one existing XNNPACK delegate to both
subgraphs. Both signatures must delegate completely. They share one scratch
workspace, one thread pool, and one packed-weight cache. Only the outer chunk
loop, input binding, and new-KV commit run on the host. There is no per-layer
host execution or new attention operator.

Two masks serve all layers: one local, one global. Prefill broadcasts each
`[1, 1, T, S]` mask over the eight query heads. Attention still materializes its
score/probability tensors; it is not FlashAttention. `S` follows active history
plus the current chunk, rounded to 32-token alignment. Local layers use the
union of their per-query windows, with the precise window enforced by masking.
The 8,448-token persistent KV capacity does not determine early attention width.

A partial final chunk is padded to `T` rows. Its padded K/V participate only in
masked graph storage and are never committed to the persistent cache. This can
add up to `T-1` masked positions versus the staged runner's valid-token extent.
The correctness sweep includes short tails and verifies the full committed
cache and vocabulary logits. Empty history and zero-length alignment padding
remain genuine zero-dimensional inputs.

Measured desktop results and allocation accounting are in the
[report](../../docs/gemma4_litert_chunked_prefill.md) and [results.json](results.json).

## Build and reproduce

From the repository root:

```sh
mkdir -p .bazelisk-cache .cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  //tensor/experiments/litert_chunked_prefill:runner
python3 tensor/experiments/litert_chunked_prefill/run.py --mode correctness --label validated-
python3 tensor/experiments/litert_chunked_prefill/run.py --mode timing --label validated-
python3 tensor/experiments/litert_chunked_prefill/run.py --mode memory --label validated-
python3 tensor/experiments/litert_chunked_prefill/analyze.py \
  --label validated- --save tensor/experiments/litert_chunked_prefill/results.json
```

`run.py` accepts `--binary`, `--out`, `--bundle`, `--fixture`, `--variants`,
`--cpus`, and `--chunk-rows`. Artifact directories must be new. It defaults to
the matched bundle and correctness fixtures from the earlier local experiments;
override the paths when reproducing elsewhere. Timing uses one warmup and three
measured runs per case, in two blocks with reversed variant order. Memory and
byte-comparison captures are separate from timing.

The runner permits power-of-two chunks from 32 through 1,024. To invoke the
LiteRT path directly, set `--joined_prefill_graph=true`,
`--litert_prefill_graph=true`, `--joined_decode_graph=true`, and
`--litert_decode_graph=true`, along with the bundle, case, and output paths.
The benchmark script also enables shared workspace and compact static INT2
weights, fixes four threads, and pins the desktop's four P cores.

## Implementation and limits

- `driver.cc` authors the connected prefix, binds active inputs, commits valid
  KV rows, and audits actual attention kernel types and extents.
- `model/helpers/attention.h` connects new K/V, active-history concatenation,
  two BMMs, causal mask, and softmax using existing primitives.
- `mapped_runner.h` exports existing LiteRT builtins and groups both signatures
  under one interpreter/delegate, and restores compact authoring quantization
  metadata after export. Its observer wrappers forward XNNPACK API
  arguments unchanged and reject fallback or extra runtime invocations.
- `source_identity.json` records the unmodified parent experiment's inputs.
  These authoring helpers are a local overlay and are deliberately included
  before the native bundle loader's copies.

This is a runnable Tensor API authoring and LiteRT Interpreter experiment.
Integration into the production `LitertDynamicRunner`, public CompiledModel
runner, or LiteRT-LM remains separate. No delegate or XNNPACK source changes
are introduced here. Model-source buffers remain retained, and the experiment
does not call `MADV_PAGEOUT` or trim the heap. Active-history concatenation still
copies K/V; cache views and tiled attention remain possible follow-up work.
