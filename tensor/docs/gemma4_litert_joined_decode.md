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

# Gemma4 joined decode through LiteRT

**September 23, 2026, Linux CPU experiment.** The
[joined native decode graph](gemma4_single_graph_decode.md) can run through
LiteRT's TFLite Interpreter and its existing XNNPACK delegate with comparable
observed performance. The mapped model uses existing primitive operators,
requires no SDPA operation, and required no delegate changes.

| Result | Native joined XNNPACK | LiteRT Interpreter + XNNPACK |
| --- | ---: | ---: |
| Decode, median of six session means | 25.610 ms/token | 25.894 ms/token |
| XNNPACK invocations per decode | 1 | 1 |
| CPU fallback operators | 0 | 0 |
| Final-decode RSS | 2,050.66 MiB | 2,107.41 MiB |
| Full logits and KV comparison | Reference | Bit-identical |

The measured latency difference is **1.11%**. This is evidence that a faithful
primitive mapping can reproduce the native computation through the LiteRT
engine. It is not a claim of statistical equivalence, globally optimal
attention, or a completed integration into `LitertDynamicRunner`/LiteRT-LM.
Prefill still uses the existing native stages. MTP is excluded.

The subsequent [chunked prefill experiment](gemma4_litert_chunked_prefill.md)
also connects the KV-producing prefix into one graph per chunk. It places
prefill and decode in two signatures of one LiteRT model with shared scratch.

## Mapping and execution contract

Both variants are modes of one Bazel-built binary. They author the same joined
decode dependency graph, including the 30 INT8 K/V history concatenations from
the previous experiment. The host binds the active history and masks before
each invocation and commits the 15 owners' new K/V rows afterward. No host
code runs between decode layers.

The LiteRT mode serializes a real `.tflite` file, reloads it using
`FlatBufferModel` and `InterpreterBuilder`, and delegates it using the normal
XNNPACK delegate. Its input tensors are resized using
`Interpreter::ResizeInputTensor` and bound using
`SetCustomAllocationForTensor`; `AllocateTensors` propagates shapes before
`Invoke`. Backing KV capacity remains 8,448 while attention shapes track the
active aligned local/global ranges. Empty history and empty alignment padding
are represented by zero dimensions.

The experiment's [export adapter](../experiments/litert_joined_decode/mapped_runner.h)
supplies these authoring rules:

- Quantized casts serialize as `QUANTIZE`/`DEQUANTIZE`. Numeric `CAST` alone
  would not preserve the native scales.
- Native RMSNorm helpers expand to the same square, mean, epsilon addition,
  reciprocal square root, and multiplication primitives as native lowering.
- Static INT2 FC helpers serialize as standard `FULLY_CONNECTED` with compact
  INT2 weights and the original scales. The dynamic head's original two-bit
  codes are losslessly compacted from their INT4 transport container.
- Per-channel metadata repeats the broadcast zero point for each scale,
  because the Interpreter requires matching vector lengths.
- KV dequantization remains explicit before FP32 BMM. XNNPACK fuses this to
  the same FP32/QC8W attention kernels as the native graph. Direct hybrid
  FP32/INT8 BMM lowering would choose a different activation-quantization path.

These rules change serialization, not the arithmetic precision or weights.
The exported graph has 3,830 LiteRT operators, 5,429 tensors, 132 inputs, and
31 outputs (logits plus 30 new K/V rows).

Caller-owned constant bindings preserve the original weight pointers so
native prefill and delegated decode hit the same packed-weight cache.
`ExternalCpuBackendContext` shares one four-thread pthreadpool. The native
control uses `XNN_FLAG_DONT_SPIN_WORKERS` for joined decode to match the
delegate's existing policy. Prefill's policy remains unchanged.

Link-time observer wrappers forward XNNPACK creation and invocation arguments
unchanged. They verify one runtime, one invocation per token, the shared pool
and weight cache, and the expected flags. The adapter rejects partial
delegation. It does not replace the delegate with a direct native invocation.

## Correctness and backend audit

The final binary passes 11 cases with prompt lengths 1, 8, 128, 1,024, 4,096,
8,192, 127, 129, 511, 512, and 513, in that order. The sweep covers empty history,
zero padding, alignment/window boundaries, and reuse after long-to-short
session changes. It compares:

- 163 full FP32 vocabulary vectors: 42,729,472 values.
- 4,890 complete committed owner K/V files.
- 5,053 files totaling 1,227,569,152 bytes, all bit-identical.

During this diagnostic run, the actual 70 attention BMM operand types and
shapes are checked against the expected active range on every decode call.
All 60 static INT2 FC operators remain present. The complete optimized
operator-family inventories match, including:

| XNNPACK operator family | Count in each variant |
| --- | ---: |
| FP32/QC8W attention BMM | 70 |
| Static INT8 activation / INT2 FC | 60 |
| Static INT8 activation / INT4 FC | 145 |
| Static INT8 activation / INT8 FC | 71 |
| Dynamically quantized activation / INT2 head | 1 |
| FP32 softmax | 35 |
| Mean squared reduction | 276 |

The complete inventories and comparison details are in
[results.json](../experiments/litert_joined_decode/results.json).
The earlier native-control build also matched the archived joined native
reference across all 5,053 files, before matching worker policy for the final
binary and repeating native/LiteRT correctness.

## Timing and memory

The i9-12900K run pins four threads to CPUs `0,2,4,6`, uses the existing
1,024-token prompt and 64 forced decode inputs, cache capacity 8,448, prefill
chunk 128, and KV alignment 32. Two blocks reverse variant order. Each process
discards one warmup session and measures three sessions. Compilation, export,
model loading, diagnostic dumps, and memory collection are outside decode
timing. Each session mean averages the 64 forced decode calls.

| Timing | Native joined | LiteRT joined |
| --- | ---: | ---: |
| Block 0 median decode | 25.571 ms | 25.849 ms |
| Block 1 median decode | 25.678 ms | 25.932 ms |
| Session-mean decode range | 25.531–25.697 ms | 25.842–25.976 ms |
| Pooled median decode | 25.610 ms | 25.894 ms |
| Prompt through first logits | 1,035.81 ms | 1,020.96 ms |

LiteRT is 1.09% and 0.99% slower in the individual blocks. The 0.284 ms pooled
difference is not an isolated measurement of Interpreter overhead; it also
includes allocation/layout effects and timing variation. Prefill computation
is unchanged, so the first-logit difference does not establish a new prefill
optimization. Long-context timing and phone timing have not been measured for
this mapping.

Separate memory runs use one warmup and one measured session:

| Memory / resources | Native joined | LiteRT joined |
| --- | ---: | ---: |
| Process peak RSS (`getrusage`) | 2,052.20 MiB | 2,106.84 MiB |
| Final-decode RSS (`smaps_rollup`) | 2,050.66 MiB | 2,107.41 MiB |
| Final-decode anonymous memory | 1,445.93 MiB | 1,485.97 MiB |
| Process swap | 0 | 0 |
| Packed weights | 751.875 MiB | 751.875 MiB |
| Packed-cache entries / hits / misses | 277 / 129 / 277 | 277 / 129 / 277 |
| Persistent KV payload | 74.250 MiB | 74.250 MiB |
| Unique XNNPACK workspace total | 10.673 MiB | 19.423 MiB |
| Externally owned output payload total | 47.134 MiB | 47.134 MiB |

The LiteRT prototype retains separate workspaces: 8.750 MiB for native prefill
and 10.673 MiB for delegated decode. Native mode shares one 10.673 MiB workspace
across both phases. This accounts for 8.750 MiB of the 56.758 MiB RSS increase.
The remaining difference is not fully partitioned into allocation owners.
Additional graph metadata, quantization arrays, and allocator retention are
present; the saved FlatBuffer has 14.377 MiB before appended weight storage,
and its scale/zero-point arrays logically occupy 13.883 MiB. The Interpreter's
affine arrays logically occupy 9.255 MiB. These sizes overlap different file
and heap accounting categories and must not be added as an exact RSS ledger.

The Interpreter's allocated read/write tensor payload is just 1.009 MiB,
matching logits plus new K/V outputs. It does not retain a separate full set
of the delegated graph's attention/MLP intermediates in its arena. Those
intermediates live in XNNPACK's workspace. Thus the observed memory difference
does not demonstrate that LiteRT cannot reuse intermediate scratch.

The 270 MiB compact MLP sources and 96 MiB compact head source remain retained
in both variants. The native `graph_constant_capacity_bytes` counter records
the head, while the mapped mode retains it as an exported source buffer;
that counter becoming zero is not a 96 MiB saving. No `MADV_PAGEOUT` or
packing-source release optimization is used. Peak RSS and sampled smaps RSS
use different kernel accounting and should not be treated as an exact shared
measurement scale. Export/setup memory is included in process peak.

## Remaining integration work

This proves the Interpreter/delegate part of the LiteRT path. The experiment
uses its own adapter, not `LitertDynamicRunner`, `CompiledModelRunner`, or
LiteRT-LM's model executor. Moving this behavior into those runners needs
faithful export, active shape propagation, borrowed buffer bindings, reusable
output storage, and resource ownership. It does not inherently need new
mathematical operators.

`CompiledModel` already exposes strict and non-strict resize APIs and can
detect input-buffer layout changes. A follow-up regression reproduced rejection
of zero dimensions before execution. The local changes now allow empty tensors
through resize/signature validation, host tensor-buffer validation, and packed
or strided size calculation. Negative dimensions remain invalid; an empty
dimension cannot hide a later negative dimension or cause a product overflow.

The new public-API tests exercise empty → nonempty → empty → nonempty execution
with strict resize, non-strict resize, and automatic resizing from buffer
layouts. All 53 tests across the public `CompiledModel`, tensor-buffer C API,
and tensor-size utility targets pass. The separate internal runtime test target
could not be analyzed because its `ml_drift` repository has no download URL in
this checkout. This follow-up was made after the frozen Interpreter benchmark;
it is not a full-Gemma4 `CompiledModel` performance result.

The Tensor API wrapper must still expose the necessary resizing/binding
behavior instead of copying every raw input into buffers allocated at model
creation. The remaining integration is not blocked by the blanket
empty-dimension checks removed here.

The existing concatenations still copy active history. Eliminating that work
is a separate cache append/view/aliasing problem shared by native and LiteRT
graphs. Tiled attention with online softmax is another independent backend
optimization. Neither is supplied by replacing graph orchestration alone.
YNNPACK and joined prefill have not been measured in this experiment.

## Reproduction

See the [experiment README](../experiments/litert_joined_decode/README.md).
Raw captures, serialized models, build identity, and the frozen executable are
in `/data/bt/tmp/gemma4-litert-single-graph-20260923`.
The tested binary SHA-256 is
`4bf8095daedbc491f141f38d5e0e630180d6d57b44c349ae8ac17f01a5c7d8e9`.
