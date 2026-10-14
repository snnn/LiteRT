# Non-Standard LiteRT Operators: Converter Reference

This is a source-audited catalog of named non-standard operators that have a
LiteRT runtime kernel, delegate implementation, or compiler-plugin lowering.
It describes the **serialized TFLite contract** for a converter that writes
FlatBuffers directly, including `litert-converter`. Using the
`STABLEHLO_COMPOSITE` opcode does not require a StableHLO conversion pipeline.

**Converter contracts and backend readiness are separate.** The converter can
emit the intended contract while a backend implementation catches up. Preserve
the model's semantics and record the required runtime support; do not change
the serialized math to accommodate a backend bug. For execution today, provide
an equivalent supported decomposition or select a compatible backend. A custom
op has no automatic decomposition, and a custom-op stub cannot execute it.

**Prefer one serialized graph for CPU and GPU where their contracts overlap.**
Backend-specific packing and kernel selection should stay inside the backend.
For a shared artifact, check the same operands and attributes against every
intended consumer; recognition of the same op name does not establish matching
semantics or active-length performance. Document required runtime configuration
and unresolved compatibility gaps explicitly.

## Scope and source revisions

Refreshed on 2026-09-30 after reading the recent commit messages and checking
the corresponding emitters, parsers, kernels, and tests:

| Repository | Reference | Commit |
| --- | --- | --- |
| `~/src/LiteRT` | Fetched `upstream/main` | `6c6af0163c346459fe6f3024b0bb5c3a02f4a834` |
| `~/src/litert-torch` | Local `HEAD` | `78ff964262f8a5f9c536789d68a4ec29cf39061a` |
| `~/src/ml-drift` | Local `HEAD` | `17ec4e6aa3f0103020bdaacc1303b1f2ad270c87` |

LiteRT and torch evidence comes from those Git trees, not local changes. The
separate MLDrift checkout also qualifies; source links below pin the audited
revisions. Torch emitters help establish intended semantics, but do not prove
backend support. An entry needs a concrete execution path. This catalog is
bounded by these sources; it cannot enumerate externally supplied custom
libraries or every operation in the open-ended Select TF Ops family.

Excluded from the runtime catalog are builtin-only frontend wrappers, compiler
test fixtures, internal GPU kernel names without a TFLite parser, and names
recognized only by a converter. In particular, `odml.detector` and
`odml.quantize_and_dequantize` have custom legalization entries but no matching
runtime consumer in the checked sources. Do not emit them as runtime custom
ops. Their occurrence in a converter allowlist does not establish support.
Legacy GenAI CPU custom kernels are intentionally outside this catalog.

### Changes relevant to converter authors

| Area | Converter action |
| --- | --- |
| Transposed SDPA causality | Emit boolean `is_causal` explicitly. Set it true only for a proven purely causal mask; eligible GPU paths may discard that mask. Missing means false. |
| Shared SDPA graph | Keep Q/output rank 4 and GQA heads on the head axis at the composite boundary; put decode flattening outside. Use `softcap` as the canonical cap attribute and keep backend gaps documented below. |
| Torch export controls | `use_sdpa_composite=True` now enables prefill and decode and implies `apply_gpu_composites=True`. The old `use_sdpa_composite_for_prefill` flag is deprecated. |
| MoE | Emit `activation="gelu_tanh"` for the torch reference math; pass per-expert output scales exactly once. Both MLDrift and XNNPACK now accept blockwise int8 expert scales. |
| YNNPACK expert composite | `odml.moe_experts` now admits 7/10 inputs, 16-bit floating types, and quantized expert weights. It remains a separate contract from custom `moe`. |
| Gated delta | Preserve distinct Q/K and V head counts and the new padding sentinels; the runtime now handles head sharing without converter-side repetition. |
| GPU kernels and tests | Upstream fixes earlier SDPA mask/bias/GQA problems and adds OpenCL FlashDecode. These improve backend support without requiring new op names. |

The top torch commit explains why `is_causal` matters: older exports omitted
it, so the GPU retained a large global mask even when runtime parameters could
replace it. Text generation alone is not proof of causality. The emitter now
requires a text-only task, a known causal model key, a global layer, and a
causal attention module. Multimodal, sliding-window, noncausal, and unrecognized
exports retain their masks. A direct FlatBuffer converter should establish
the same semantic guarantee from its model rather than blindly copy an
allowlist. See the detailed contracts below for runtime conditions.

Sources: [torch causal attribute and background](https://github.com/google-ai-edge/litert-torch/commit/78ff964262f8a5f9c536789d68a4ec29cf39061a),
[export configuration](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/generative/export_hf/core/exportable_module_config.py).

## Direct serialization and execution requirements

| Field | Preserved composite | Runtime custom op |
| --- | --- | --- |
| `OperatorCode.builtin_code` | `STABLEHLO_COMPOSITE` | `CUSTOM` |
| Name | `StableHLOCompositeOptions.name` | `OperatorCode.custom_code`, exact case-sensitive string |
| Options location | `Operator.builtin_options_2`, union type `StableHLOCompositeOptions` | `Operator.custom_options` byte vector |
| Options encoding | `composite_attributes` bytes and `composite_attributes_format=FLEXBUFFERS` | Per-op format; usually FlexBuffers, sometimes a native C struct |
| Decomposition | `decomposition_subgraph_index` identifies a real matching subgraph | None attached to this operator |
| Versions | `OperatorCode.version` and composite `version` are separate fields | `OperatorCode.version` must match resolver registration |
| Execution | Named backend implementation, or an executable decomposition | Actual custom registration or a delegate that claims the node |

For `litert-converter`, construct the typed `CoreStableHLOCompositeOptions`
with name, version, decomposition index, attribute bytes, and format. The
current binding defaults the composite version to 1 and the decomposition
index to -1: **replace the latter with a valid subgraph index**. Set
`CoreOp.builtin` before setting its typed options, because the builtin setter
resets the options. Custom ops use `CoreOp.custom_code`, `custom_options`, and
`custom_options_format`. These fields map directly to the table above; no
textual MLIR or torch HLFB wrapper is needed. Version 1 is the current direct
emitter convention, not a claim that every backend accepts arbitrary versions.

A decomposition must have the same ordered inputs and outputs, matching types
and shapes, and reproduce all attributes, masking, quantization, and state
updates. Use supported builtin operations inside it. Test it with delegation
disabled as well as on the selected backend. A decomposition makes fallback
possible; it does not make an otherwise unsupported *name* a useful native ABI.

For FlexBuffers, preserve scalar types: boolean flags, integer dimensions and
axes, floating epsilon/scales, strings for enums, and the specific typed vectors
where required. JSON bytes are not FlexBuffers. Missing keys sometimes read as
zero rather than receiving a default. Tensor quantization metadata and explicit
scale operands/attributes are different mechanisms; supply each that the
consumer reads. Native-struct options require the matching runtime's layout;
the schema currently offers no separate raw-struct format enumerator.

“Runtime input” below means a nonconstant operand as counted by that parser.
“Constant” means stored weight data, not a model input whose example value
happens to be fixed. `-1` optional tensor indices are valid only where a parser
explicitly accepts them. Model tensor names do not control dispatch.

### Runtime setup

| Support label | What the application must provide |
| --- | --- |
| Standard CPU resolver | A build containing that custom registration in `BuiltinOpResolver`; a reduced resolver may omit it. |
| Explicit CPU registration | Link the kernel and register the exact custom code/version. It is not enabled merely by selecting CPU. |
| Delegate | Enable the named delegate, provide any required resolver registration, and confirm that it claims the node. A fallback stub will fail if the delegate declines it. |
| Compiler plugin | Compile the op/partition with the named plugin and use its accelerator runtime. Plugin recognition is not a generic CPU kernel. |

The LiteRT compiled-model runtime installs accelerator stubs for a limited
list, including `moe` and the legacy convolution/pooling names. It also accepts
application-supplied custom registrations. It does **not** automatically
register every custom alias listed here. A standalone TFLite interpreter must
likewise arrange its own resolver and delegates. MLDrift's legacy
`GraphFloat32` and newer IR paths differ; an alias supported by one is not
automatically supported by the other.

Sources: [FlatBuffer schema](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/converter/schema/schema.fbs),
[composite fallback](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/stablehlo_composite.cc),
[standard registrations](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/core/kernels/register.cc),
[custom registrations and accelerator stubs](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/runtime/compiled_model.cc),
[MLDrift legacy factory](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/custom_parsers.cc),
[MLDrift IR factory](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/ir/custom_parsers.cc).

## Catalog

The tables identify concrete consumers, not a promise of support on every
device or every build. See each linked section for the admitted contract.
`C` means `STABLEHLO_COMPOSITE`; `U` means `CUSTOM`.

| Exact name | Encoding | Concrete consumer / setup |
| --- | --- | --- |
| [`odml.scaled_dot_product_attention`](#odmlscaled_dot_product_attention) | C; U for XNNPACK/YNNPACK | XNNPACK, YNNPACK, MLDrift composite |
| [`odml.sdpa_transposed`](#odmlsdpa_transposed) | C; U for YNNPACK | MLDrift, YNNPACK |
| [`odml.rms_norm`](#odmlrms_norm) | C | MLDrift; Qualcomm, MediaTek, NVIDIA and Samsung plugin implementations |
| [`odml.group_norm`](#odmlgroup_norm) | C | MLDrift, Qualcomm; MediaTek plugin for `num_groups=1` |
| [`odml.l2_norm`](#odmll2_norm) | C | Qualcomm and MediaTek plugins |
| [`odml.runtime_bmm`](#odmlruntime_bmm) | C; U for YNNPACK | MLDrift, YNNPACK, NVIDIA plugin |
| [`odml.cache_update`](#odmlcache_update) | C | MLDrift, NVIDIA plugin |
| [`odml.rope`](#odmlrope) | C or U | MLDrift |
| [`odml.qkv_norm_rope`](#odmlqkv_norm_rope) | C or U | MLDrift |
| [`odml.swiglu`](#odmlswiglu) | C or U | MLDrift |
| [`odml.short_conv_step`](#odmlshort_conv_step) | C or U | MLDrift |
| [`odml.moe_experts`](#odmlmoe_experts) | C | YNNPACK |
| [`moe`](#moe) | U | MLDrift; XNNPACK MoE delegate kernel |
| [`gated_delta_update`, `custom_call.gated_delta_update`](#gated_delta_update-and-custom_callgated_delta_update) | U | Explicit gated-delta-net CPU registerer; MLDrift handles unprefixed name |
| [`gdn_tril_inv`, `custom_call.gdn_tril_inv`](#gdn_tril_inv-and-custom_callgdn_tril_inv) | U | Explicit gated-delta-net CPU registerer |
| [`custom_call.GroupNorm`, `custom_call.LayerNorm`, `custom_call.RmsNorm`](#legacy-gpu-normalization-custom-ops) | U | MLDrift legacy parser |
| [`custom_call.rotary_positional_embedding`](#positional-embedding-custom-ops) | U or C | MLDrift rotary parser/kernel |
| [`custom_call.absolute_positional_embedding`](#positional-embedding-custom-ops) | U | MLDrift |
| [`custom_call.PixelShuffle`, `custom_call.pixel_shuffle`](#pixel-shuffle) | U | MLDrift legacy / IR spellings, respectively |
| [`Convolution2DTransposeBias`](#convolution2dtransposebias) | U | XNNPACK, GPU/MLDrift |
| [`MaxPoolingWithArgmax2D`](#pooling-with-indices-and-unpooling) | U | XNNPACK, GPU/MLDrift |
| [`MaxPoolWithArgmax`](#pooling-with-indices-and-unpooling) | U | Explicit perception CPU registerer |
| [`MaxUnpooling2D`, `custom_call.MaxUnpooling2D`](#pooling-with-indices-and-unpooling) | U | GPU/MLDrift; XNNPACK and explicit perception CPU kernel for unprefixed name, with different index conventions |
| [`Resampler`](#resampler-and-denseimagewarp) | U | GPU/MLDrift |
| [`DenseImageWarp`](#resampler-and-denseimagewarp) | U | Explicit perception CPU kernel |
| [`TFLite_Detection_PostProcess`](#tflite_detection_postprocess) | U | Standard CPU resolver |
| [`AudioSpectrogram`, `Mfcc`](#audiospectrogram-and-mfcc) | U | Standard CPU resolver |
| [`NumericVerify`](#numericverify) | U | Standard CPU resolver; diagnostic |
| [`aeq.hadamard_rotation`](#aeqhadamard_rotation) | U | Standard CPU resolver |
| [`ParseExample`, `ParseExampleV2`](#parseexample-and-parseexamplev2) | U | Explicit parse-example CPU registerer |
| [`atan2`, `Sign`, `RandomStandardNormal`, `RandomUniform`, `RandomUniformInt`, `Multinomial`, `Roll`, `Irfft2d` / `IRFFT2D`, `AveragePool3D`, `MaxPool3D`, `Table`, `BroadcastGradientArgs`](#other-explicit-cpu-custom-registrations) | U | Explicit CPU registrations; often superseded by builtins |

Compiler partition markers, `DISPATCH_OP`, and Select TF Ops are covered
[separately](#compiler-partitions-and-select-tf-ops), because they have
partition- or external-library-defined signatures.

Notation: `B`=batch, `T`=query/update tokens, `S`=KV length/capacity,
`Nq`/`Nkv`=query/KV heads, `H`=head dimension, `D`/`C`=channels,
`G`=normalization groups. MoE additionally uses `E`=experts, `A`=selected experts,
`F`=expert hidden dimension. Operand order is the serialized order.

## Attention, normalization, and cache composites

### Active cache length and attention work

**Converter policy: favor `odml.sdpa_transposed` over
`odml.scaled_dot_product_attention` for fixed-capacity KV-cache attention when
designing the converter's cache-aware attention contract.** The primary
reason is to avoid computing over unused cache capacity by supplying the live
active length. This preference is especially important for MLDrift, whose
standard SDPA implementation has no active-length operand. The converter must
emit the required layouts, scaling, and runtime parameters; choosing the name
alone does not provide the saving.

For a fixed-capacity KV cache, distinguish its allocated capacity S from the
number of valid tokens L. **Masking positions `[L,S)` does not by itself avoid
computing their attention scores or reading their values.** Kernel fusion and
layout optimizations are separate from limiting work to the active prefix.

| Backend | `odml.scaled_dot_product_attention` | `odml.sdpa_transposed` |
| --- | --- | --- |
| MLDrift GPU | Q, K, V, optional mask; no active-length operand. Work follows the supplied K/V extent. | Supports an explicit runtime parameter tensor; element 2 bounds the main matmul/softmax reductions and fused kernels. Some unfused GPU stages still process full capacity; see below. |
| XNNPACK CPU | No active-length operand consumed; work follows the supplied K/V extent. | No handler in the checked implementation. |
| YNNPACK CPU | Supports an optional runtime parameter operand and slices K/V to the active prefix. | Supports the same active-prefix mechanism. |

XNNPACK needs the transposed composite's executable decomposition or a future
native implementation. On YNNPACK, both names can honor active length, so this
particular advantage does not distinguish them; choose the intended model
layout and document the available execution path.

In YNNPACK, a fourth integer operand supplies parameters when no mask is
present; with a mask in slot 3, parameters occupy slot 4. It reads length from
element 1 (element 0 for a one-element tensor). Positive lengths are capped at
capacity; nonpositive lengths leave full capacity selected. This extension is
not a portable standard-SDPA signature: appending the operand does not enable
active-length handling in MLDrift or XNNPACK.

YNNPACK's CPU lowering limits **both matmuls and softmax**: it slices K before
the first matmul, computes scores and softmax over that shortened sequence
dimension, and slices V before the second matmul. At each invocation it reads
the live parameter and reshapes the delegated graph. This is active-prefix
computation, not just a mask applied to full-capacity scores, and it applies
to both SDPA names.

Selecting CPU alone does not enable that path. In the checked LiteRT runtime,
YNNPACK is opt-in: build with `--define litert_enable_ynnpack=true`, enable
`LrtSetCpuOptionsEnableYNNPack`, and use delegate kernel mode. It then handles
supported CPU nodes before XNNPACK. Otherwise, a preserved transposed composite
can execute its decomposition, whose active-length behavior depends on the
actual subgraph. The checked torch emitter's decomposition retains parameters
through a zero-valued dependency and computes full-extent matmuls; it does not
inherit YNNPACK's active-prefix optimization.

For MLDrift cache-aware decode, choose the supported `odml.sdpa_transposed`
contract and supply the live length. Its fused kernels use a parameter length
only when `0<L<=S`; otherwise they fall back to full capacity. Without that
operand, the transposed name alone does not establish active-prefix execution.
Alternatively, supply genuinely shortened K/V tensors through a supported
dynamic-shape/slicing path. Confirm that the selected backend executes that
shortened shape; a fixed-size tensor with a prefix mask is insufficient.

For example, L=512 in an S=8192 cache means approximately 16 times less
sequence-dependent matmul work when the kernel honors L. This is not an
end-to-end speedup estimate. Benchmark fixed capacity with several active
lengths, and verify delegation as well as numerical results, to confirm the
exported model actually benefits.

Sources: [MLDrift standard parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/tflite/model_builder.cc),
[MLDrift transposed kernels](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.cc),
[XNNPACK attention visitor](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/xnnpack/xnnpack_delegate.cc),
[YNNPACK input decoding and slicing](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/ynnpack/attention.cc),
[YNNPACK live length handling](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/ynnpack/ynnpack_delegate.cc),
[CPU backend selection](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/c/options/litert_cpu_options.h),
[YNNPACK accelerator setup](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/runtime/accelerators/ynnpack/ynnpack_accelerator.cc),
[torch fallback matmul](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/generative/export_hf/experimental/composites/runtime_batched_matmul.py).

#### Sharing an SDPA graph between CPU and GPU

Use the following intended converter contract for new shared models. Native
consumers currently include YNNPACK CPU and MLDrift GPU; XNNPACK can execute
supported builtins in the decomposition but has no transposed-SDPA handler.
Backend implementation gaps do not require a different semantic contract.

| Concern | Intended serialized representation | Current backend status |
| --- | --- | --- |
| Layout/output | Q/output `[B,Nq,T,H]`, K `[B,Nkv,S,H]`, V `[B,Nkv,H,S]`; flatten decode output with a following reshape. | Torch now marks the rank-4 output before flattening. MLDrift can fuse the following decode reshape internally; YNNPACK requires rank 4 at the composite boundary. |
| GQA | Require `Nq % Nkv == 0`; query head h uses KV head `h // (Nq/Nkv)`. Keep heads separate from tokens at the boundary. | Torch packs query heads only inside the fallback decomposition to avoid copying K/V. MLDrift's multi-operation path now handles this mapping. YNNPACK uses 5D broadcasting for transposed prefill and folding for decode. |
| Scale | Pre-scale Q once; omit `scale` or use floating `1.0`. | MLDrift expects pre-scaled Q; YNNPACK's transposed default is 1. |
| Runtime bounds | Supply live `INT32 [1,1,1,7]` parameters, an accurate query offset, and a mask covering the semantic active prefix. | YNNPACK reads element 1; MLDrift reads element 2 and some kernels require aligned work bounds. See the alignment caveat below. |
| Logit cap | Emit floating `softcap=C` for positive C and reproduce it in the decomposition. | MLDrift reads `softcap`, with `logit_cap` as a fallback alias. YNNPACK still reads only `logit_cap`; canonical `softcap` support remains backend work. |
| Causality | Emit boolean `is_causal`, true only when runtime bounds/positions fully reproduce the causal mask. Keep the explicit mask in the serialized graph/decomposition. | MLDrift may prune a BOOL mask on eligible Apple paths. YNNPACK follows the supplied mask and does not read `is_causal`. |
| State | Keep logical cache shapes and explicit fallback state updates. | Native GPU cache packing, initialization, and aliasing remain runtime responsibilities. `from_cache_update` describes actual packed cache origin, not a generic optimization hint. |

The exact active length L and an aligned GPU work bound are not interchangeable.
The current ATS cases use a 32-aligned L in both elements 1 and 2. For unaligned
lengths, an aligned element 2 can include padding that the explicit mask must
exclude. Fused kernels also read element 2 as `active_tokens`; mask pruning
therefore requires checking that their causal bounds and valid-query handling
still reproduce the mask, including partial chunks. Do not infer this from
`is_causal=true` alone or treat aligned ATS coverage as proof for exact lengths.
The intended contract remains the model's exact valid-token semantics; any
alignment discrepancy belongs in the backend implementation.

Validate cache allocation, initialization/reset, state feedback, and masking
on both runtimes. GPU packed state is not automatically a portable buffer for
switching to CPU midway through a session. Source support for a dtype, GQA, or
layout is not an end-to-end performance or numerical validation result.

Sources: [torch composite boundary and fallback](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/generative/export_hf/experimental/composites/sdpa.py),
[YNNPACK attention](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/ynnpack/attention.cc),
[GPU parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/ir/sdpa_transposed_parser.cc),
[GPU kernels](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.cc),
[decode reshape fusion](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/fuse_sdpa_transposed_reshape.cc),
[aligned ATS inputs](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/test/generators/sdpa_transposed.h),
[GPU bound alignment](https://github.com/google-ai-edge/ml-drift/blob/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87/ml_drift/common/task/gpu_operation.h).

#### Correctness status and backend follow-up

Several GPU findings in the September 17 audit have been fixed upstream.
Keep the regression cases, but do not describe those old implementations as
current behavior:

| Area | Status at the checked revisions |
| --- | --- |
| Fused decode masks and softcap | The fast loop now applies masks and softcap as well as the tail. The later unified Metal/OpenCL implementation retains explicit mask handling. |
| Fused prefill additive bias | Additive bias is now multiplied by `1/ln(2)` to match the base-2 softmax logits. |
| Fused prefill causality | `is_causal` defaults false; causal bounds are conditional on the attribute. A noncausal mask is no longer overridden merely because the fused prefill kernel was selected. |
| GPU GQA fallback | Query heads are grouped for matmuls and restored for mask application/output, matching the contiguous head mapping. |
| YNNPACK cap attribute | Still reads only `logit_cap`. A converter emitting canonical `softcap` must use its decomposition or a backend with the attribute implemented to preserve capped results today. |
| YNNPACK BOOL masks | Still converts false to an additive `-10000` penalty. Torch/GPU decomposition semantics replace the logit with `-10000`; extreme logits distinguish these functions. |

Useful converter/backend regression cases are: a masked prefix spanning the
fused decode fast loop; finite additive prefill biases; noncausal prefill with
future keys allowed; `softcap=1.0` with saturating logits; BOOL replacement with
an excluded large positive logit; GQA prefill with different masks per query;
and exact/unaligned active lengths with partial chunks. Include all-masked
rows explicitly if the model permits them: fused kernels use large-negative
sentinels and special handling that should not be assumed identical to a
finite-fill decomposition in every numerical corner case.

These remaining backend gaps do not change the recommended `softcap` spelling,
BOOL selection semantics, or logical layout for new converter output. Keep
compatibility workarounds explicit and reproduce the same math in fallback.

Sources: [GPU fixes](https://github.com/google-ai-edge/LiteRT/commit/77a7a2ce9bc88d003273d7bd0e84cf5f4f8123ff),
[GPU GQA fix](https://github.com/google-ai-edge/LiteRT/commit/6d59736316f2e63d215904c152059f8d81aaec95),
[current fused kernels](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.cc),
[current CPU attributes and masking](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/ynnpack/attention.cc).

#### What the current ATS coverage establishes

The `SdpaTransposed` registration still requests 16 grid cases for each
float32/float16, mask, softcap, and parameter combination (256 before filters
or generation failures). Its generator has changed substantially:

| Area | Current coverage and limits |
| --- | --- |
| Layout and GQA | Q is now `[B,Nq,T,H]`; query length is independent of head count. The decomposition groups contiguous query heads correctly. |
| Masks | Actual BOOL masks include holes inside the active prefix; floating masks include finite negative biases and excluded keys. The BOOL reference uses selection rather than an additive penalty. |
| Active cache | Parameters include a query offset and matching lengths in elements 1/2. Where possible, length is below capacity and rounded to a multiple of 32. Unaligned bounds and live changes across calls still need separate tests. |
| Softcap | Cap is now 1.0, and both `softcap` and `logit_cap` are emitted. This improves numerical sensitivity but does not test a `softcap`-only contract on YNNPACK. |
| Fused packed-cache paths | K/V remain ordinary inputs, with no cache-update producer or `from_cache_update=true`. The cases do not establish coverage of packed Flash kernels or causal-mask pruning. No `is_causal` attribute is emitted. |
| CPU suite | Default CPU targets still exclude `SdpaTransposed` and do not enable YNNPACK. They do not establish CPU/GPU equivalence for this composite. |

The device `gpu_ats` registration now includes `SdpaTransposed` explicitly;
the earlier claim that its positive filter excludes these cases is obsolete.
Suite configuration and source tests are evidence of coverage intent, not a
claim that this refresh ran ATS on hardware. Add cache-update-to-attention
models and explicitly enabled YNNPACK execution for portability testing; verify
which native path was selected and compare against independent model math.

Sources: [ATS registration](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/ats/register_sdpa_transposed.cc),
[grid iteration count](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/ats/register_composite_ops.cc),
[current generator/reference](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/test/generators/sdpa_transposed.h),
[suite filters](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/ats/BUILD),
[CPU options](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/ats/configure.cc).

#### MLDrift work still tied to cache capacity

The specialized MLDrift implementation itself selects either a fused kernel
(Apple Metal prefill/decode or OpenCL decode) or a multi-operation GPU graph in
`BuildSdpaTransposedGpuGraph`. Both execute within the native implementation
after delegation. The observations
here concern that native GPU graph, independently of the composite's TFLite
fallback decomposition.

For packed-cache attention (`from_cache_update=true`), the clearest remaining
gap is **boolean mask application in the multi-operation GPU path**. The
generic-path findings below apply only when `from_cache_update=false` and the
specified kernel is selected. Supplying the active bound does not make every
stage of either native path proportional to that bound.

| Stage | When it occurs | Work that still depends on capacity S |
| --- | --- | --- |
| Boolean mask application | Multi-operation path with a boolean mask, including packed-cache attention. | `SelectV2(mask, logits, -10000)` receives no runtime bound and traverses the full score tensor. This kernel is not linkable into QK by the checked elementwise merger. |
| K/V transpose and packing | Generic `BatchedMatMul` path (`from_cache_update=false`) when the direct fully connected implementation is not selected. | The RHS transpose and weight conversion receive no runtime bound. They process full-capacity K/V even though the following matmul receives the bound. |
| Final softmax normalization | Generic path when `Softmax` selects its two-pass implementation: score rows per GPU compute unit >= 256. | The reduction receives the bound, but `SoftmaxElementwise` / `CreateSoftmaxFinal` computes `exp(logit-max)/sum` over the full score tensor without it. |
| Redundant softmax reductions | Generic path when `SelectSoftmax` chooses `Softmax1x1` and the full-capacity output requires multiple workgroups. | Dispatch uses the full number of output slices. Every workgroup performs the active-prefix reduction before checking whether its output slice is active, so groups entirely beyond the bound still repeat that reduction. They do not scan inactive keys, but their redundant work depends on S. |

The two-pass final normalization cannot simply link into the preceding QK
kernel: logits feed both the reduction and normalization, and the reduction
result is normalization's second input. The checked linker requires a sole
consumer connected through input 0. In contrast, softcapping and additive
masking **can** link into QK and inherit its bounded execution; their lack of
a separate parameter operand is not by itself evidence of a full-capacity
pass. Inspect the generated graph before counting them as separate costs.

The packed-cache multi-operation path already uses a bound-aware weights
converter when repacking is needed, then `SoftmaxReduce` with normalization
fused into the value matmul. It therefore avoids the generic transpose/packing
and separate-normalization gaps above, but its boolean `SelectV2` remains.
The selected fused prefill/decode kernels use active-prefix loops and
avoid these separate intermediate passes. Eligible Apple causal paths also
prune the BOOL mask; retained masks in the multi-operation path still incur
the selection pass described above. Selection requirements are listed
under [`odml.sdpa_transposed`](#odmlsdpa_transposed).

Intermediate allocations and some dispatch grids also retain capacity-sized
shapes; matmul bounds can round up to kernel alignment. Active-prefix
arithmetic therefore does not imply exact-L allocation or zero overhead for
inactive workgroups. Mask construction outside the composite likewise does
not acquire a bound merely because SDPA consumes one. These are source-audit
findings; their latency impact requires profiling the selected path.

Sources: [SDPA graph construction](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.cc),
[boolean selection kernel](https://github.com/google-ai-edge/ml-drift/blob/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87/ml_drift/common/kernels/select_v2.cc),
[generic matmul preparation](https://github.com/google-ai-edge/ml-drift/blob/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87/ml_drift/common/gpu_model_builder.cc),
[softmax path selection](https://github.com/google-ai-edge/ml-drift/blob/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87/ml_drift/common/gpu_model_builder.cc),
[unbounded final normalization](https://github.com/google-ai-edge/ml-drift/blob/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87/ml_drift/common/kernels/softmax.cc),
[Softmax1x1 reduction and dispatch](https://github.com/google-ai-edge/ml-drift/blob/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87/ml_drift/common/kernels/softmax1x1.cc),
[softmax selector](https://github.com/google-ai-edge/ml-drift/blob/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87/ml_drift/common/selectors/simple_selectors.cc),
[node linking](https://github.com/google-ai-edge/ml-drift/blob/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87/ml_drift/common/merge_nodes.cc),
[packed-cache weights conversion](https://github.com/google-ai-edge/ml-drift/blob/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87/ml_drift/common/gpu_model_builder.cc),
[runtime-bound alignment](https://github.com/google-ai-edge/ml-drift/blob/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87/ml_drift/common/task/gpu_operation.h).

### `odml.scaled_dot_product_attention`

For fixed-capacity KV-cache attention, follow the
[converter preference for `odml.sdpa_transposed`](#active-cache-length-and-attention-work)
to avoid unnecessary work over unused capacity on backends such as MLDrift.

XNNPACK and YNNPACK recognize composite and same-name custom forms. MLDrift
recognizes the composite. Use an executable builtin decomposition for CPU
fallback. The custom form requires resolver setup and a delegate that claims
the node; it has no attached decomposition.

| Slot | Tensor | Portable float contract |
| --- | --- | --- |
| input 0 | query | `FLOAT32 [B,T,Nq,H]`, sequence-major BTNH |
| input 1 | key | `FLOAT32 [B,S,Nkv,H]` |
| input 2 | value | `FLOAT32 [B,S,Nkv,H]` |
| input 3 | mask | Additive `FLOAT32`, rank 4, broadcastable to `[B,Nq,T,S]`, last dimension S |
| output 0 | result | `FLOAT32 [B,T,Nq,H]` |

Require `Nq % Nkv == 0`, matching K/V shapes and Q/K/V head dimensions.
Delegate paths can accept omitted masks; provide an explicit additive mask
when masking is part of the model's computation.
Causality is represented by the mask, not inferred from the operator name.
There is no dropout or training-state operand.

FlexBuffers float `scale` defaults to `1/sqrt(H)` in XNNPACK/YNNPACK. Positive
float `logit_cap` requests `cap*tanh(logits/cap)` before masking/softmax there.
The legacy MLDrift composite parser reads `scale` but does not read `logit_cap`;
do not select it for capped attention unless the cap is implemented elsewhere.
The intended math is `softmax(cap_if_requested(scale*Q@K^T)+mask)@V`.

YNNPACK currently requires equal Q/K/V head counts for this sequence-major
entry point; its new native GQA handling is admitted for `odml.sdpa_transposed`.
It additionally accepts float16/bfloat16 tensors, boolean masks
(true=keep; false is converted to additive -10000), and an optional runtime
length operand as described under `odml.sdpa_transposed`. Its Q/K/V/output
admission is floating-only and rank 4. Validate these extensions against the
selected delegate rather than assuming they apply to every consumer.

Common PyTorch inputs are head-major `[B,N,T,H]`. Transpose to the serialized
sequence-major contract; tensor names do not change the interpretation.
XNNPACK also transposes internally, so preserving this op alone does not prove
a performance gain. Emit floating attributes and preserve scale application
exactly once in both the native path and decomposition.

Sources: [XNNPACK visitor](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/xnnpack/xnnpack_delegate.cc),
[YNNPACK visitor](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/ynnpack/attention.cc),
[MLDrift composite parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/tflite/model_builder.cc).

### `odml.rms_norm`

Preserved composite handled by MLDrift and vendor compiler plugins. Emit
floating `x[...,D]`, constant floating `gamma[D]`, and one same-shaped output.
The portable source contract is float32; supported quantized forms depend on
the selected plugin's transformations, not just this name.

Required float attribute `epsilon` defines
`y=x*rsqrt(mean(x*x, axis=-1, keepdims=true)+epsilon)*gamma`.
It is added inside the reciprocal square root. MLDrift consumes gamma as a
constant channel scale, not an arbitrary runtime broadcast tensor. Models
using `(1+weight)` must supply that effective gamma; the op does not add 1.
There is no bias input. Normalize the final dimension; multiaxis frontend
RMSNorm needs an explicit compatible reshape or builtin decomposition.

MLDrift's IR admission requires opcode version 1, nonconstant x of rank 2–4,
and constant rank-1 gamma when provided. It accepts float32/float16/bfloat16
tensor types; backend precision/storage still has to support them. That IR
path also supports omitting gamma for unweighted normalization. Use the
two-input form when sharing a model with consumers that require gamma.

The checked NVIDIA native lowering accepts static floating tensors and
optional gamma, but **hardcodes epsilon to `1e-6`** instead of reading the
attribute. Only select it when that matches the model. The compiler control
`LITERT_NVIDIA_TENSORRT_NATIVE_COMPOSITES` can restrict the enabled native
names (comma-separated `rms_norm,cache_update,runtime_bmm`), or use `none` to
inline them; unset enables the supported native paths.

Sources: [MLDrift parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/tflite/model_builder.cc),
[IR conversion](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/tflite/convert/convert_rms_norm.cc),
[IR admission](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/tflite/support/support_rms_norm.cc),
[Qualcomm builder](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/vendors/qualcomm/core/builders/rms_norm_op_builder.cc),
[MediaTek dispatch](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/vendors/mediatek/compiler/create_model.cc),
[NVIDIA builder](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc),
[Samsung builder](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/vendors/samsung/compiler/builders/rms_norm_op_builder.cc).

### `odml.group_norm`

Preserved composite handled by MLDrift and Qualcomm; MediaTek additionally
recognizes the `num_groups=1` specialization. Ordered operands are
floating channel-last `x`, constant floating `gamma[C]`, constant floating
`beta[C]`; output has x's shape and type. Use float32 for the common contract.
Group normalization computes `(x-mean)*rsqrt(variance+epsilon)*gamma+beta`
within each channel group and its spatial dimensions, separately per batch.
Require positive `G=num_groups` and `C % G == 0`.

| Attribute | Required converter behavior |
| --- | --- |
| `epsilon` | Floating scalar; emit explicitly. |
| `num_groups` | Integer G; emit explicitly for GroupNorm. |
| `channel_axis` | Integer channel axis. MLDrift's GroupNorm parser requires the **positive** last-axis index, `rank(x)-1`, when present; `-1` is not accepted by that check. |
| `sub_type` | MLDrift discriminator: integer 0 for GroupNorm, 1 for LayerNorm. Emit explicitly to avoid legacy ambiguity. |
| `_TENSOR_V1_reduction_axes` | Optional tensor-attribute **map**, with integer vector in its `TENSOR_DATA` member. With explicit GroupNorm subtype, MLDrift checks `[1,...,rank-1]`; LayerNorm checks `[rank-1]`. A bare axes vector is not the encoding this parser reads. |

MLDrift accepts rank at most 4, with provided gamma/beta rank 1 and C elements.
Its legacy dispatch without `sub_type` selects **LayerNorm when the axes key is
absent**, GroupNorm when it is present; the latter then expects only `[rank-1]`
for backward compatibility. Do not infer GroupNorm solely from the name.
For subtype 1, normalize the last channel dimension per location; this is not
general multiaxis LayerNorm.

MediaTek's current options-aware path admits `num_groups=1` and lowers to
`NEURON_LAYER_NORMALIZATION`, reading `epsilon` and `channel_axis` (default
last dimension). Its older name-only path still recognizes only RMS/L2 norm.
This is a LayerNorm specialization, not evidence for arbitrary spatial
GroupNorm or G>1; match the reduction axes and decomposition.

Qualcomm has its own group/layer lowering and maps `num_groups==1` to its layer
builder. Check that the chosen reduction dimensions agree with the model and
decomposition before selecting that path. Attributes tolerated by one parser
do not change another backend's normalization axes.

Sources: [MLDrift dispatch and normalization checks](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/tflite/model_builder.cc),
[Qualcomm group builder](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/vendors/qualcomm/core/builders/group_norm_op_builder.cc),
[MediaTek admission](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/vendors/mediatek/compiler/legalizations/common_op_legalization.cc),
[MediaTek LayerNorm lowering](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/vendors/mediatek/compiler/legalizations/layer_norm_op_legalization.cc),
[Qualcomm composite mapping](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/vendors/qualcomm/compiler/qnn_compose_graph.cc).

### `odml.l2_norm`

Preserved composite with one floating input `x[...,D]` and one same-shaped
output. Source test decomposition uses float attributes `epsilon`, integer
`axis=-1`, and `y=x/sqrt(sum(x*x,axis=-1,keepdims=true)+epsilon)`.

Qualcomm's builder consumes epsilon and hardcodes the final axis. MediaTek
maps the composite directly to `NEURON_L2_NORMALIZATION` without forwarding
these composite attributes. Therefore arbitrary axis/epsilon settings are
**not** a cross-backend contract; verify target normalization semantics,
especially at zero and tiny norms. Use float32 unless a target-specific
quantized path is verified. Builtin `L2_NORMALIZATION` has its own contract and
is not automatically interchangeable with every epsilon placement.

Sources: [decomposition](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/test/testdata/l2_norm_composite.mlir),
[Qualcomm builder](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/vendors/qualcomm/core/builders/l2_norm_op_builder.cc),
[MediaTek dispatch](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/vendors/mediatek/compiler/create_model.cc),
[MediaTek attribute handling](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/vendors/mediatek/compiler/legalizations/common_op_legalization.cc).

### `odml.runtime_bmm`

MLDrift implements this preserved composite as a runtime-bounded matmul.
YNNPACK also accepts it as a composite or same-name custom op; NVIDIA has a
compiler implementation. The following is the MLDrift ABI.

| Slot | Tensor | Type / shape |
| --- | --- | --- |
| input 0 | lhs | Floating, normally `FLOAT32 [B0,B1,M,K]` |
| input 1 | rhs | Floating or symmetric `INT8 [B0,B1,N,K]`; logically transposed RHS |
| input 2 | parameters | `INT32 [1,1,1,7]`, runtime operand |
| output 0 | result | Floating `[B0,B1,M,N]` |

All three inputs must be runtime operands; the parser compares the parameter
shape with BHWC `[1,1,1,7]`. Emit the literal rank-4 shape. It flattens B0 and
B1 internally when B0 is not 1. There is no non-transposed-RHS attribute.

| FlexBuffers attribute | MLDrift meaning |
| --- | --- |
| `is_global` | Required boolean; currently validated but not otherwise consumed. |
| `is_src` | Required boolean. True bounds source channels, false bounds destination channels, using parameter element 2. Also selects V versus K scale metadata. |
| `rhs_cache_update` | Optional boolean, default false; true forces cache/external-weight handling. |
| `scale` | Optional float; its presence selects external-weight handling. Required for standalone int8 RHS dequantization. Use a positive finite value. |

The ordinary path computes `lhs@transpose(rhs)`. External-weight handling is
selected by `rhs_cache_update`, a `scale` attribute, or a visible
`odml.cache_update` RHS producer. That path supports float or packed quantized
cache storage. For an int8 cache producer, `is_src=true` uses its `scale_v`
with head-size entries; false uses `scale_k` with cache-size entries. Emit the
corresponding producer scales explicitly: this path dereferences them rather
than relying on the cache writer's defaults. An affine tensor scale alone is
not a substitute for this metadata.

| Parameter index | Consumer contract |
| --- | --- |
| 0 | Cache-update write offset. |
| 1 | Cache-update active-token end; YNNPACK's runtime-BMM/attention active length. |
| 2 | MLDrift BMM/attention active-channel bound. Its kernel handles channel packing/alignment. |
| 3 | Cache-update ring-buffer update length, when ring mode is enabled. |
| 4..6 | Not read by the implementations described here. |

For a linear cache, element 0 is `start` and element 1 is `start+T`; element 2
supplies the GPU work bound. A 32-aligned `start+T` can use
`[start,start+T,start+T,0,0,0,0]` directly. For other lengths, check the selected
kernel's alignment and retain masking of any padded work, as discussed in the
shared-SDPA section. The executor must fill live values; zero-filled export
examples must not become constants.

YNNPACK reads active length from element **1**, not 2 (a one-element parameter
tensor uses element 0). Nonpositive values leave the full capacity selected;
positive values are capped by capacity. It uses `is_src` for P×V versus Q×Kᵀ
slicing and may infer the mode from dimensions. Its quantization uses its
tensor support/metadata path; do not assume MLDrift's scale-only external
weights contract transfers to it. Set parameters and attributes explicitly
and compare the full result with the builtin fallback.

NVIDIA admits static equal-rank tensors (rank at least 2), floating lhs/output,
and floating or symmetric per-tensor int8 RHS. Its ordinary native lowering
computes a **full-context** matmul and does not use the runtime parameter
operand to bound it. Equivalence requires inactive scores to be masked later,
or inactive value probabilities to be zero. It dequantizes using tensor
metadata. Do not select this path for a standalone BMM that requires the
inactive output region to match a runtime-bounded placeholder exactly.

Sources: [MLDrift parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/runtime_batched_matmul_parser.cc),
[IR parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/ir/runtime_batched_matmul_parser.cc),
[YNNPACK lowering](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/ynnpack/dot.cc),
[live parameter reading](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/ynnpack/ynnpack_delegate.cc),
[NVIDIA implementation](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc),
[torch emitter](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/generative/export_hf/experimental/composites/runtime_batched_matmul.py).

### `odml.cache_update`

MLDrift implements a paired K/V cache writer; NVIDIA also has a composite
compiler implementation. The MLDrift contract accepts exactly 3 or 7 runtime
inputs and two outputs. `litert-converter` uses the seven-input form so a
builtin decomposition can express explicit state updates.

| Slot | Tensor | Contract |
| --- | --- | --- |
| input 0 | new K | Floating, typically `[1,Bkv,T,H]`; width is update length. |
| input 1 | new V | Matching floating values and logical shape. |
| input 2 | parameters | `INT32`; use `[1,1,1,7]` shared with runtime BMM. |
| inputs 3..6 | fallback state | In the torch seven-input carrier: K cache, V cache, K update indices, V update indices. Their shapes/layouts belong to the decomposition. MLDrift ignores these four operands. |
| output 0 | updated K cache | Floating or `INT8`, backend-packed storage. |
| output 1 | updated V cache | Same storage class; different packed matrix layout from K. |

Required integer attributes are `kv_cache_batch_size`, `cache_size=S`, and
`head_size=H`, all positive. Float `scale_k` and `scale_v` default to 1 in the
writer; emit positive finite values explicitly for quantized caches.
Boolean `is_ring_buffer` defaults false.

Linear mode writes token `p=parameters[0]+x` only while `p<S` and
`p<parameters[1]`. Ring mode instead processes `x<parameters[3]` and writes
`p=(parameters[0]+x)%S`. Use nonnegative offsets and an update length within
the supplied slice. A linear dynamic-update-slice fallback is not equivalent
to ring mode; its wraparound, masks, and attention positions must also match.

K is packed as external weights with output size S and input size H; V has
output size H and input size S. They cannot share one assumed physical layout.
With int8 outputs, MLDrift preserves TFLite tensor references but uses unsigned
packed bytes internally and quantizes incoming float values with scale_k/v.
This is backend-owned packing, not an instruction to reinterpret arbitrary
public UINT8 tensors as signed caches.

The shader reads state from the cache destination; ignored fallback inputs do
not initialize or reset it on the native path. Arrange cache allocation,
initialization, aliasing, lifetime, and feedback with the executor. The parser
does not fully check K/V shape compatibility. A single-cache update is not this
ABI; use supported builtin update operations for that case.

NVIDIA's ordinary quantized lowering has a different state contract: it uses
the explicit int8 caches in slots 3/4 and int32 update indices in slots 5/6,
quantizes with output tensor metadata, transposes V by `[0,1,3,2]`, then scatters
into the caches. It requires symmetric per-tensor int8 caches and static
floating update tensors. This path does not read MLDrift's runtime bounds or
ring attributes. NVIDIA additionally has a specialized native float16 prefill
cache path: seven inputs, `[1,1,1,7]` int32 parameters, K `[1,Nkv,S,H]`, V
`[1,Nkv,H,S]`, fresh K/V `[1,Nkv,T,H]` with `1<T<=S`, and explicit boolean
`is_ring_buffer`. The caches must be subgraph inputs, outputs must have no
internal consumers, and existing cache readers must meet its ordering/aliasing
checks. That specialization uses parameter 0 for offset and 3 for valid update
length. Do not assume a node accepted by its generic quantized path has these
prefill semantics.

Sources: [MLDrift parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/add_values_to_cache_parser.cc),
[packing and ring-buffer kernel](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/add_values_to_cache_kernel.cc),
[NVIDIA implementation](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc),
[seven-input decomposition](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/generative/export_hf/experimental/composites/cache_update.py).

### `odml.sdpa_transposed`

This preserved attention composite uses head-major tensors and can consume
MLDrift's packed K/V cache. It is distinct from
`odml.scaled_dot_product_attention`. MLDrift handles the composite; YNNPACK
also recognizes the same-name `CUSTOM` form, which requires registration and
has no attached decomposition.

| Slot | Tensor | Converter contract |
| --- | --- | --- |
| input 0 | `query` | Floating `[B,Nq,T,H]`, **already multiplied by the attention scale**. |
| input 1 | `key` | Floating, logical `[B,Nkv,S,H]` for `k_ts_idx=2`. |
| input 2 | `value` | Floating, logical `[B,Nkv,H,S]` for `v_ts_idx=3`. |
| input 3 | `mask` or `param_tensor` | Four-input form: `INT32` means parameters; otherwise mask. BOOL true=keep, false=replace with `-10000`; floating masks are additive. |
| input 4 | `param_tensor` | Five-input form: mask at slot 3, runtime `INT32 [1,1,1,7]` parameters at slot 4. MLDrift reads active bound at index 2. |
| output 0 | `output` | Floating `[B,Nq,T,H]`. Put decode flattening to `[B,1,Nq*H]` outside the composite. |

MLDrift admits 3–5 runtime inputs and one output. The three-input form is
unmasked attention. For GQA require `Nq % Nkv == 0` and map query head h to
KV head `h // (Nq/Nkv)`. Do not pack heads into T at the composite boundary.

| FlexBuffers attribute | Converter meaning / consumer behavior |
| --- | --- |
| `k_ts_idx=2`, `v_ts_idx=3` | Integer logical time axes used by the torch decomposition. MLDrift does not use these attributes to transpose arbitrary input layouts. |
| `softcap` | Optional positive float C: `C*tanh(logits/C)` before masking. MLDrift reads this first, otherwise `logit_cap`; YNNPACK still reads only `logit_cap`. Use canonical `softcap` for new exports. |
| `is_causal` | Boolean, default false. True promises that explicit mask semantics are fully represented by causal positions and the active cache bound. This is permission to remove the mask on an eligible path. |
| `from_cache_update` | Optional boolean identifying MLDrift cache-produced packed K/V. The parser also infers it from a visible cache-update producer. Do not assert it for ordinary external K/V. |

Set `is_causal=true` only after proving the mask is purely causal. Sliding
windows, bidirectional image spans, arbitrary padding/attention patterns, and
finite additive biases need their explicit mask unless their complete semantics
are otherwise represented. The torch helper defaults false; the top commit's
export path opts in only for known causal text-generation models, global
layers, and modules that permit causality. A missing module `is_causal` uses
true only within that already gated export path. Task/model names alone are
not a general-purpose causality test for a direct converter.

Keep the mask operand and its semantics in the serialized decomposition.
In the checked Apple-build parsers, pruning requires all of `is_causal=true`,
cache origin, runtime parameters, a BOOL mask, and either eligible prefill
(`H % 4 == 0`, `H <= 128`) or single-token decode with `H == 128`.
The intended Metal fused path then uses `key < active_tokens` and
`key <= q_start + query_index`. Missing `is_causal` retains the mask. Older
runtimes can ignore the extra attribute while still using that mask. Do not
remove host mask construction merely because the attribute was emitted:
that requires runtime confirmation that every relevant consumer can skip it.

Kernel selection is broader than mask pruning. Current fused prefill requires
Apple Metal, cache-produced buffer storage, `H % 4 == 0`, and `H <= 128`.
Fused decode now supports Apple Metal **and OpenCL**, with buffer K/V,
`H % 4 == 0`, `S % 4 == 0`, divisible head counts, and device workgroup limits.
Its mask, if present, must be shared across heads, have one row or T rows,
and cover S entries. The parser treats `T>2` or batch greater than 1 as prefill;
other cases can enter decode. The `H==128` restriction now describes the
special wave-SIMD implementation and the narrow mask-pruning guard, not all
fused decode support. Other configurations use the native multi-operation
GPU graph. Quantized K/V remain TODOs in this SDPA parser.

Torch now keeps the composite boundary/output rank 4, packs GQA only inside
its decomposition, and applies the final decode reshape after `mark_outputs`.
MLDrift can fuse that following reshape itself. Runtime parameters are retained
through zero-valued dependencies in the torch fallback matmuls; fallback still
computes full extent and relies on the mask for valid-token semantics.
The torch helper also converts a floating mask to `mask == 0` when parameters
are present. A direct converter must preserve finite additive bias instead of
copying that conversion for arbitrary floating masks.

YNNPACK accepts float32/float16/bfloat16 rank-4 Q/K/V/output and floating or BOOL
masks. Its parameter may be int32 or int64 and supplies active length at index
1; nonpositive selects full capacity. It defaults `scale` to 1.0 for this name,
implements GQA, and still converts false BOOL entries to additive `-10000`.
See the shared-contract section for cap, BOOL, and exact/aligned-bound gaps.

Sources: [torch emitter](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/generative/export_hf/experimental/composites/sdpa.py),
[task/model gating](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/generative/export_hf/core/exportable_module.py),
[layer gating](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/generative/export_hf/core/attention.py),
[legacy GPU parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/sdpa_transposed_parser.cc),
[IR GPU parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/ir/sdpa_transposed_parser.cc),
[GPU kernels and selection](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.cc),
[decode reshape fusion](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/fuse_sdpa_transposed_reshape.cc),
[YNNPACK contract](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/ynnpack/attention.cc),
[YNNPACK live length](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/ynnpack/ynnpack_delegate.cc).

### `odml.rope`

MLDrift recognizes `STABLEHLO_COMPOSITE` and same-name `CUSTOM` encodings. The
torch helper emits a composite. The kernel supports two different signatures:

| Form | Inputs, in order | Outputs, in order | Meaning |
| --- | --- | --- | --- |
| Combined tensor | `x`, `positions` | `rotated_x` | Split the channel dimension into rotary pairs, rotate, and concatenate. |
| Split tensors | `left`, `right`, `positions` | `rotated_left`, `rotated_right` | Rotate corresponding left/right components together. These are **paired components**, not independently rotated full Q and K tensors. |

The GPU kernel uses MLDrift's `[B, height, width, channels]` interpretation:
`width` is the position axis. For attention this normally means `[B,N,T,H]`.
Position reads use width coordinates; use a shape such as `[B,1,T,1]` for an
explicit GPU-facing position tensor. The torch helper accepts `[T]` or `[B,T]`,
but its decomposition selects row 0 from rank-2 positions. Distinct positions
per batch require separate verification. Its docstring also permits `[B,T,N,H]`;
that does not make the GPU kernel infer which axis holds tokens.

Use floating-point activations and position values representable by the GPU
tensor reader. The parser does not establish a complete dtype/rank contract;
in particular, do not infer unrestricted `INT64` GPU support from PyTorch's
position dtype. Outputs have the corresponding input shapes and floating type.

| Attribute | Default / accepted spellings | Meaning |
| --- | --- | --- |
| `min_timescale` | float `1.0` | Minimum rotary timescale. |
| `max_timescale` | float `10000.0`; aliases `base`, `rope_theta`, `theta`, in that priority after the canonical key | Maximum rotary timescale. |
| `proportion` | float `1.0`; alias `partial_rotary_factor` | Fraction of rotary channels. |
| `kernel_type` | integer `0` | `0`: planar 1-D along width; `1`: interleaved 2-D along width and height. |

The split kernel computes `out_l=l*cos(angle) - r*sin(angle)` and
`out_r=r*cos(angle) + l*sin(angle)`. For split component index i in width d,
`angle=position/(min_timescale*(max_timescale/min_timescale)^(i/d))`.
The combined planar form uses `2*i/H` for a pair within an H-channel tensor.
Partial rotation tests this fraction against `proportion`; it does not
automatically recompute frequencies using a smaller rotary dimension. Use
positive timescales and a proportion in `[0,1]`.
The implementation processes vectors of four
channels and tests the last lane against `proportion`; use rotary dimensions
compatible with that packing and validate partial rotation at the boundary.
The split form does not use `kernel_type` to select the combined form's 2-D
pairing. Keep the emitted decomposition's pairing and timescale formula exact.

Sources: [emitter](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/generative/export_hf/experimental/composites/rope.py),
[parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/rope_parser.cc),
[kernel](https://github.com/google-ai-edge/ml-drift/blob/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87/ml_drift/common/kernels/rope.cc),
[attribute defaults](https://github.com/google-ai-edge/ml-drift/blob/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87/ml_drift/common/operations.h).

### `odml.qkv_norm_rope`

Composite or same-name custom op recognized by MLDrift. It fuses QKV splitting,
Q/K RMS normalization, layout conversion, and RoPE. V is only reshaped and
transposed.

| Slot | Tensor | Shape / type |
| --- | --- | --- |
| input 0 | `qkv` | Floating `[B,T,(Nq+2*Nkv)*H]`, concatenated **Q, K, V**. |
| input 1 | `position` | Torch: `[T]` or `[B,T]`; GPU must see a readable position for every token. |
| input 2 | `q_weight` | Floating RMSNorm gamma, normally `[H]`. |
| input 3 | `k_weight` | Floating RMSNorm gamma, normally `[H]`. |
| output 0 | `query` | Floating `[B,Nq,T,H]`. |
| output 1 | `key` | Floating `[B,Nkv,T,H]`. |
| output 2 | `value` | Floating `[B,Nkv,T,H]`. |

Emit integer `num_heads=Nq`, `num_kv_heads=Nkv`, `head_dim=H`; float
`min_timescale`, `max_timescale`, `proportion`, and `epsilon`. The torch emitter
uses `1.0`, configurable `base` (default `1000000.0`), `1.0`, and `1e-6`,
respectively. It computes `x*rsqrt(mean(x*x,-1)+epsilon)*weight` before RoPE.
It does not add 1 to gamma; models using an offset scale must supply the
effective gamma themselves.

The parser checks four inputs and three outputs. The GPU builder resolves
head counts and head dimension from output shapes and rejects conflicting
positive attributes. Arity acceptance is not full shape validation. Use the
emitter's floating tensor contract and verify batch handling, head packing,
rotary pairing, and positions against its decomposition. As with the standalone
torch RoPE helper, rank-2 positions use row 0 in the decomposition.

Sources: [emitter](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/generative/export_hf/experimental/composites/qkv_norm_rope.py),
[parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/ir/qkv_norm_rope_parser.cc),
[kernel](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/qkv_norm_rope_kernel.cc).

### `odml.swiglu`

Composite or same-name custom op recognized by MLDrift. It performs the
activation/gating step, not either surrounding linear projection.

| Form | Inputs | Output |
| --- | --- | --- |
| Concatenated | floating `gate_up[...,2*D]`, ordered gate then up | floating `y[...,D]` |
| Separate | floating `gate[...,D]`, `up[...,D]` | floating `y[...,D]` |

`y = (gate * sigmoid(gate)) * up`. Optional integer `gate_size` selects the
split point; a nonpositive/unset kernel value uses half the input channels.
The torch emitter always writes `gate_size` and uses the one-input form.
For portable use, both halves have equal size and identical leading dimensions.
The GPU concatenated path splits at a four-channel slice boundary, so choose
`D` divisible by four; do not assume an arbitrary scalar split is implemented.
The parser counts 1 or 2 **runtime** inputs and exactly one output.

Sources: [emitter](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/generative/export_hf/experimental/composites/swiglu.py),
[parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/ir/swiglu_parser.cc),
[kernel](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/swiglu_kernel.cc).

### `odml.short_conv_step`

Composite or same-name custom op recognized by MLDrift. This is a gated
single-token decode operation, not a general sequence convolution.

| Slot | Tensor | Torch emitter contract |
| --- | --- | --- |
| input 0 | `in_proj_out` | Floating `[B,1,3*D]`, ordered `b`, `c`, `x`. |
| input 1 | `conv_state` | Floating `[B,D,L-1]`, oldest to newest. |
| input 2 | `conv_weight` | Floating `[D,1,L]` or `[D,L]`. |
| input 3 | `conv_bias` | Optional floating `[D]`; omit the operand when absent. |
| output 0 | `y` | Floating `[B,1,D]`. |
| output 1 | `next_state` | Floating `[B,D,L-1]`. |

Emit integer `conv_L_cache=L` (default 3). Form `u=b*x`, concatenate `u` onto
the state, apply the depthwise dot product with the length-L weights, add bias,
and multiply by `c`. The new state contains the last `L-1` values of that
concatenation. There is no implicit SiLU activation.

The parser checks 3 or 4 inputs and 2 outputs. The GPU implementation indexes a
single decode location and handles projection segments in four-channel slices;
use `B=1`, `T=1`, and `D` divisible by four for that path. Its state/weight
layout handling is specialized; verify the exported tensor descriptors with
the kernel, rather than extending the torch helper's batch support to the GPU.
State is explicit: return output 1 as input 1 on the next invocation.

Sources: [emitter](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/generative/export_hf/experimental/composites/short_conv.py),
[parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/short_conv_step_parser.cc),
[kernel](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/short_conv_step_kernel.cc).

### `odml.moe_experts`

YNNPACK recognizes this preserved composite. It now supports 7 or 10 inputs
and one output; it still does not recognize plain custom `moe` through this
visitor. Similar operand order does not make the two names/encodings aliases.

| Tensor | 7-input slot | 10-input slot | Shape |
| --- | --- | --- | --- |
| tokens | 0 | 0 | `[B,T,D]` |
| routing weights | 1 | 1 | `[B,T,A]` |
| expert indices | 2 | 2 | `INT32 [B,T,A]`, IDs in `[0,E)` |
| gate weights | 3 | 3 | Constant `[F,E,1,D]` |
| gate scales | None | 4 | Constant floating scale tensor; per-row form `[F,E,1,1]` |
| up weights | 4 | 5 | Constant `[F,E,1,D]` |
| up scales | None | 6 | Constant floating scale tensor; per-row form `[F,E,1,1]` |
| down weights | 5 | 7 | Constant `[D,E,1,F]` |
| down scales | None | 8 | Constant floating scale tensor; per-row form `[D,E,1,1]` |
| output scale | 6 | 9 | Constant floating tensor, one or E elements |
| output | output 0 | output 0 | `[B,T,D]`, same dtype as tokens |

Tokens, routing weights, output scale, and output can be float32, float16, or
bfloat16. All three weight tensors must have the same dtype: one of those
floating types, or INT8/INT4/INT2. Sub-byte packing must be valid both before
and after the weight transpose: the input and output channel dimensions must
hold whole bytes (multiples of 2 for int4, 4 for int2). Constants and tensor
quantization must also pass YNNPACK's tensor-support checks.

For quantized weights, the 7-input form takes quantization parameters from
tensor metadata; the 10-input form takes explicit floating projection scales
and leaves weight zero points implicit (symmetric). Those scales are
transposed/gathered with the weights and must broadcast to the intended dot
quantization. The explicit scales are not applied to floating weights. Do not
infer that every block-scale layout accepted by custom `moe` is supported here.

E is inferred from weight axis 1 and A from the last index dimension; require
`0<A<=E`. No composite attributes are consumed. The gate uses approximate
GELU, then multiplies the up projection, applies the down projection, weights
by the route and output scale, and sums selected experts. Quantized execution
dynamically quantizes activations and the intermediate to int8 and dequantizes
int32 dot accumulations. Match fallback/reference tolerances to that arithmetic;
float metadata support alone does not establish identical low-bit numerics.

Sources: [YNNPACK implementation](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/ynnpack/moe.cc),
[MoE tests](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/ynnpack/moe_test.cc).

## Expert and other runtime custom ops

### `moe`

Plain `CUSTOM` routed expert block implemented by MLDrift and XNNPACK. It is
distinct from YNNPACK's `odml.moe_experts` composite. Provide routing outside
this op, with `0<A<=E`, expert IDs in `[0,E)`, and the intended normalized route
weights. There are no expert biases.

| Slot | Tensor | Common contract |
| --- | --- | --- |
| input 0 | token states | Floating `[1,T,D]` or `[1,1,T,D]` for MLDrift |
| input 1 | top weights | Floating `[1,1,T,A]` |
| input 2 | top indices | `INT32 [1,1,T,A]` |
| output 0 | result | Floating, same logical shape as input 0 |

Floating mode has **7 inputs**, quantized mode **10**. Remaining operands are
constant weights/scales. Weight layout is `[out_channels,E,1,in_channels]`,
not `[E,out_channels,in_channels]`. Let Gg, Gu, Gd be the scale block counts
for gate/up/down; each is positive and divides that projection's input width.

| Tensor | Float slot / shape | Quantized slot / shape |
| --- | --- | --- |
| gate weight | 3 / `[F,E,1,D]` | 3 / `[F,E,1,D]` |
| gate scale | None | 4 / `[F,E,1,Gg]` |
| up weight | 4 / `[F,E,1,D]` | 5 / `[F,E,1,D]` |
| up scale | None | 6 / `[F,E,1,Gu]` |
| down weight | 5 / `[D,E,1,F]` | 7 / `[D,E,1,F]` |
| down scale | None | 8 / `[D,E,1,Gd]` |
| per-expert output scale | 6 / `[1,1,1,E]` | 9 / `[1,1,1,E]` |

For the torch/Gemma-style contract:

```text
hidden = gelu_tanh(gate_weight[e] @ x[t]) * (up_weight[e] @ x[t])
expert_result = per_expert_scale[e] * (down_weight[e] @ hidden)
y[t] = sum(route_weight[t,r] * expert_result[expert_id[t,r]])
```

Required FlexBuffers keys are integer `num_experts=E`,
`num_active_experts=A`, `model_dim=D`, `hidden_dim=F`, and string `weight_type`.
Emit the complete map, plus `activation="gelu_tanh"` and boolean
`renormalized_top_weights=true` for this contract. MLDrift can infer properties
only when the entire options buffer is absent, not from an incomplete map.

**Activation spelling affects math.** Torch now emits `"gelu_tanh"`, matching
`F.gelu(..., approximate="tanh")`. MLDrift accepts it and legacy `"gelu"`, both
executing the tanh approximation. XNNPACK distinguishes them: `"gelu"` selects
exact GELU, `"gelu_tanh"` the approximation. Thus `"gelu"` was not numerically
portable for the torch reference. SiLU is not part of this custom-op contract.

**Apply per-expert output scale exactly once.** The current torch binding moves
`router.per_expert_scale` into the last custom-op operand and disables router
folding, avoiding an external one-hot/matmul scale lookup. If input route
weights already include that scale, pass ones in the final operand instead.
This scale is independent of the projection quantization scales.

MLDrift supports `weight_type="fp32"` and `"int8"`. Int8 expert scales now accept
both one scale per output/expert row and multiple input-axis blocks. For a
projection of shape `[O,E,1,I]` with G blocks:

```text
row = out_channel * E + expert
block_size = I / G
w_real[out_channel, expert, input_channel] =
    w_int8[out_channel, expert, input_channel] *
    scales[row * G + input_channel // block_size]
```

Use positive finite floating scales. MLDrift accepts float32/float16 scale
operands and symmetric affine, BlockwiseQuantization, or
BlockwiseQuantizationV2 metadata. For blockwise metadata, a negative zero-point
tensor index means zero; otherwise the referenced tensor must be constant
INT8 or INT32 containing only zeros. The explicit scale operands supply the
kernel scales; quantization metadata must agree and is not a replacement.

XNNPACK also supports blockwise int8 scales, but requires float32 activations,
outputs, fp32-mode weights, and all scales. Weights/scales must be read-only
constants (`kTfLiteMmapRo`). It infers block counts from element counts; emit
the canonical shapes above and enforce exact divisibility in the converter
even though some CPU helpers tolerate a ragged final block.

XNNPACK additionally supports `weight_type="int4"` with the ten-input order.
Weights may be native INT4 tensors or INT8 containers holding half as many
packed bytes. Low nibble stores the earlier signed value, high nibble the next.
Use even input widths, with the same row/block scale order and no zero-point
subtraction. MLDrift's custom parser still does not admit this int4 form;
YNNPACK's separate composite support does not extend the custom `moe` ABI.

GPU decode/remapping optimizations and MLDrift's new shared MoE builder helpers
change implementation and packing, not this serialized operand order. The
separate `ml-drift` checkout's latest ringed-weight read-size change concerns
profiling accounting; it introduces no new converter attribute or op name.

Sources: [torch options, math, and scale binding](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/generative/layers/moe.py),
[legacy GPU parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/moe_experts_parser.cc),
[IR GPU parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/ir/moe_experts_parser.cc),
[GPU implementation](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/moe_experts_kernel.cc),
[XNNPACK admission and execution](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/xnnpack/moe_delegate_kernel.cc),
[XNNPACK scale layout](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/xnnpack/moe_block_scale.h),
[shared GPU MoE builder](https://github.com/google-ai-edge/ml-drift/blob/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87/ml_drift/common/gpu_model_builder_moe_util.cc),
[ringed-weight read-size accounting](https://github.com/google-ai-edge/ml-drift/commit/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87).

### `gated_delta_update` and `custom_call.gated_delta_update`

Both names have an explicit TFLite CPU registration. MLDrift's custom factory
recognizes the **unprefixed** `gated_delta_update` name. These are `CUSTOM` ops;
they do not acquire a decomposition merely because their math is decomposable.

| Slot | Tensor | Type / shape |
| --- | --- | --- |
| input 0 | `q` | `FLOAT32 [B,Nk,T,Dk]` |
| input 1 | `k` | `FLOAT32 [B,Nk,T,Dk]` |
| input 2 | `v` | `FLOAT32 [B,Nv,T,Dv]` |
| input 3 | `beta` | `FLOAT32 [B,Nv,T]` (same contiguous element count with trailing singleton also works for the CPU loop) |
| input 4 | `g` | `FLOAT32 [B,Nv,T]`; **log decay**; positive values are padding sentinels |
| input 5 | `state` | `FLOAT32 [B,Nv,Dk,Dv]` |
| output 0 | `output` | `FLOAT32 [B,Nv,T,Dv]` |
| output 1 | `new_state` | `FLOAT32 [B,Nv,Dk,Dv]` |

Exactly six inputs and two outputs. Require `Nk>0` and `Nv % Nk == 0`.
Value head h uses Q/K head `h // (Nv/Nk)`; beta, g, state, and output remain
indexed by Nv. The updated CPU and GPU kernels perform this mapping directly,
so the converter need not repeat Q/K to Nv heads.

For each token/head the updated recurrence is:

```text
decay = exp(g) if g <= 0 else 1
beta_effective = beta if beta > 0 else 0
S = decay * S
delta = beta_effective * (v - k @ S)
S = S + outer(k, delta)
output = q @ S
```

For finite Q/K/V, right padding with `g>0` and `beta<=0` leaves state unchanged.
It does not automatically zero the output (`q@S` is still computed). Preserve
these sentinels and mask padded outputs when required. Older kernels that
unconditionally exponentiate g or use negative beta do not implement this
contract. Normalize/scale Q and K outside the op when required by the model;
state initialization and feedback are explicit.

The checked torch eager reference still uses `exp(g)` and raw beta. Its GQA
mapping is updated, but it is not a reference for positive-g/negative-beta
sentinel cases; validate those against the updated recurrence above.

Optional FlexBuffers integer `mode` defaults to 0 (recurrent); the LiteRT C++
custom-kernel implementation also selects a chunked algorithm with mode 1.
The TFLite registration currently always calls the recurrent implementation,
irrespective of this option. MLDrift parses `mode` but has its own execution
path; do not use the option to infer identical scheduling across backends.
MLDrift requires Dk and Dv to be powers of two, at least 16, and multiples of
four. CPU shape/type admission is incomplete, including the head-ratio checks
in the TFLite wrapper. Validate the full contract in the converter even when a
backend parser does not reject a mismatch. The torch Qwen3.5 path now forwards
`use_fused_gdn` and `gdn_mode` through its static-model patches; these select the
existing custom contract rather than introducing a new op name.

Sources: [CPU registrations](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/experimental/custom_ops/gated_delta_net/gated_delta_update_tflite_op.cc),
[numerical implementation](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/experimental/custom_ops/gated_delta_net/gated_delta_update_impl.cc),
[LiteRT custom-kernel API](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/experimental/custom_ops/gated_delta_net/gated_delta_update_litert_custom_op.cc),
[GPU admission](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/gated_delta_update_parser.cc),
[GPU head mapping and padding](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/delegate/composite/gated_delta_update_kernel.cc),
[torch Qwen3.5 emitter](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/generative/export_hf/model_ext/qwen3_5/gated_delta_rule.py),
[torch export flags](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/generative/export_hf/model_ext/qwen3_5/patch.py).

### `gdn_tril_inv` and `custom_call.gdn_tril_inv`

`CUSTOM`, explicitly registered by the same gated-delta-net TFLite registerer.
One `FLOAT32` input `A[...,C,C]`, one `FLOAT32` output of the same shape; rank
must be at least two and the last two dimensions equal. Computes
`(I + A)^-1` by forward substitution for a **strictly lower-triangular** A.
This is not a general matrix inverse. No custom options are consumed. No
same-name MLDrift parser is registered in the checked factory.

Sources: [registration and shape checks](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/experimental/custom_ops/gated_delta_net/gated_delta_update_tflite_op.cc),
[forward substitution](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/experimental/custom_ops/gated_delta_net/gated_delta_update_impl.cc).

### Legacy GPU normalization custom ops

These names use `CUSTOM` and a FlexBuffers option map. They are handled by
MLDrift's legacy `GraphFloat32` parser. Do not infer that the newer IR parser
recognizes the same spelling merely because a runtime stub exists.

| Exact custom code | Ordered operands → result | Options / restrictions |
| --- | --- | --- |
| `custom_call.GroupNorm` | floating channel-last `x`, optional constant `gamma[C]`, optional constant `beta[C]` → same-shaped floating output | `num_groups` integer and `epsilon` float. With **no option buffer**, defaults are 32 and `1e-6`. A present map reads both keys directly; missing keys do not receive those defaults. C must be divisible by the group count. |
| `custom_call.LayerNorm` | floating channel-last `x`, optional constant `scale[C]`, optional constant `bias[C]` → same-shaped floating output | FlexBuffers `epsilon` float; normalizes the channel dimension. |
| `custom_call.RmsNorm` | floating channel-last `x`, optional constant `scale[C]` → same-shaped floating output | FlexBuffers `epsilon` float. |

Use `kTfLiteOptionalTensor` only in slots the parser explicitly treats as
optional. A provided scale/bias must contain exactly C elements. These parsers
load scale and bias into GPU attributes; a dynamically computed scale tensor
is not equivalent to a constant weight. Prefer preserved norm composites when
their backend contract fits and a CPU decomposition is required.

Source: [legacy normalization parsers](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/tflite/model_builder.cc).

### Positional embedding custom ops

| Exact custom code | Inputs → outputs | Options / layout |
| --- | --- | --- |
| `custom_call.rotary_positional_embedding` | `x,positions → rotated_x`, or `left,right,positions → rotated_left,rotated_right` | Same GPU rotary kernels as `odml.rope`; FlexBuffers `min_timescale`, `max_timescale`, `proportion`, `kernel_type`. Use canonical keys; the legacy path does not implement all aliases of the new composite parser. |
| `custom_call.absolute_positional_embedding` | floating `x[B,H,T,C]`, `positions[B,1,T,1]` → same-shaped floating output | No options consumed. Adds sinusoidal position encoding, with sine channels followed by cosine channels. Timescales are fixed to 1 and 10000; position width must match input width. |

Both names are recognized by MLDrift's legacy and IR routing. The rotary name
is also recognized as a preserved composite. The rotary
three-input form rotates paired components together. For the absolute-position
kernel, use C compatible with two equal four-channel-packed halves (C divisible
by eight); the kernel selects sine/cosine by channel slice. Position data must
be representable by the GPU tensor reader, and the absolute-position kernel
uses the shared position lookup rather than a per-batch position reference.

Sources: [parsers](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/tflite/model_builder.cc),
[IR names](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/tflite/ir_model_builder.cc),
[absolute-position kernel](https://github.com/google-ai-edge/ml-drift/blob/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87/ml_drift/common/kernels/positional_embedding.cc),
[rotary kernels](https://github.com/google-ai-edge/ml-drift/blob/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87/ml_drift/common/kernels/rope.cc).

### Pixel shuffle

`custom_call.PixelShuffle` is the legacy MLDrift spelling;
`custom_call.pixel_shuffle` is the newer IR spelling. They are case-sensitive
names, not automatically interchangeable aliases in each backend.

One floating NHWC input `[B,H,W,C*r*r]`, one floating output `[B,H*r,W*r,C]`.
FlexBuffers integer `block_size=r` must be positive and divide the channel
dimension as `r*r`. The parser selects the GPU depth-to-space operation. Verify
channel ordering against the frontend's pixel-shuffle convention; builtin
`DEPTH_TO_SPACE` may be a better direct export when it expresses the same
ordering.

Sources: [legacy parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/tflite/model_builder.cc),
[IR dispatch](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/tflite/ir_model_builder.cc).

### `Convolution2DTransposeBias`

`CUSTOM`; XNNPACK, the legacy GPU delegate, and MLDrift have handling for this
name. Ordered inputs are floating NHWC activation `[B,H,W,Cin]`, constant
floating filter `[Cout,Kh,Kw,Cin]` (OHWI), and optional constant bias `[Cout]`.
There is one NHWC floating output. **There is no output-shape tensor in slot 0**,
unlike builtin `TRANSPOSE_CONV`.

Custom bytes contain a native `TfLiteTransposeConvParams` structure, including
padding, strides, and activation fields. This is not a FlexBuffers map and is
not a portable cross-version C-struct ABI. Generate it against the runtime's
matching headers; ensure the chosen backend supports the requested activation.
The common delegate path uses float32 input/output and static weights. Use the
builtin transpose-convolution form when its contract suffices.

Sources: [GPU parser](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/tflite/model_builder.cc),
[XNNPACK visitor](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/xnnpack/xnnpack_delegate.cc).

### Pooling with indices and unpooling

All data tensors below use NHWC. The index convention is part of the ABI:
matching shapes alone do not make pooling and unpooling implementations
interchangeable.

| Exact custom code | Ordered inputs → outputs | Options / execution |
| --- | --- | --- |
| `MaxPoolingWithArgmax2D` | floating `x` → pooled values, argmax indices of the same pooled shape | Delegate op, raw native `TfLitePoolParams` bytes. GPU kernels use positions within each pooling window; some legacy models declare the index tensor as float and the parser retypes it internally. Follow the selected delegate's index contract. |
| `MaxPoolWithArgmax` | `FLOAT32 x[B,H,W,C]` → `FLOAT32 values[B,Ho,Wo,C]`, `INT32 indices[B,Ho,Wo,C]` | Explicit perception CPU kernel; FlexBuffers `ksize=[1,Kh,Kw,1]`, `strides=[1,Sh,Sw,1]`, `padding="SAME"/"VALID"`, `include_batch_in_index`. The checked kernel requires `include_batch_in_index=false`. Indices are flattened input HWC offsets, excluding batch. |
| `MaxUnpooling2D` | floating values, integer indices of the same shape → expanded NHWC output | Raw `TfLitePoolParams`. The explicit perception CPU kernel requires float32 values and int32 indices and scatters by flattened output HWC offset. Delegate handling uses its own paired pooling convention; do not silently substitute that CPU fallback for a delegate-local index model. |
| `custom_call.MaxUnpooling2D` | values, indices → expanded output | MLDrift IR spelling; raw `TfLitePoolParams`, consumed by its unpooling converter. Legacy dispatch uses the unprefixed name. |

For the perception unpooling CPU kernel, SAME gives
`Ho=H*Sh, Wo=W*Sw`; VALID gives `Ho=(H-1)*Sh+Kh` and similarly for W.
Output is cleared before scattering. Inputs and indices must have identical
rank-4 shapes, and every flattened index must be in range. Duplicate scatter
indices overwrite earlier values; this is not a summing scatter.

Sources: [CPU pooling](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/perception/max_pool_with_argmax.cc),
[CPU unpooling](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/perception/max_unpooling_2d.cc),
[legacy GPU contracts](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/tflite/model_builder.cc),
[IR unpooling](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/tflite/convert/convert_unpooling2d.cc),
[XNNPACK contracts](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/xnnpack/xnnpack_delegate.cc).

### `Resampler` and `DenseImageWarp`

| Name | Ordered inputs → output | Semantics / support |
| --- | --- | --- |
| `Resampler` | floating image `[B,H,W,C]`, warp `[B,Ho,Wo,2]` → sampled image `[B,Ho,Wo,C]` | GPU/MLDrift custom op. Coordinates contain `(x,y)` **absolute sampling locations**. Bilinear sampling with zero contributions outside the image. No options. MLDrift IR admits float32/float16 and requires both inputs to be nonconstant. |
| `DenseImageWarp` | `FLOAT32 image[B,H,W,C]`, `FLOAT32 flow[B,H,W,2]` → same-shaped float32 image | Explicit perception CPU kernel, no options. Flow contains `(dy,dx)` **displacements**: sample at `(y-dy,x-dx)`, with coordinates clamped to image borders. H and W must be at least 2. |

These names differ in coordinate ordering, absolute-versus-relative coordinates,
and boundary behavior. They cannot be renamed into each other.

Sources: [resampler conversion](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/tflite/convert/convert_resampler.cc),
[resampler admission](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/ml_drift_delegate/tflite/support/support_resampler.cc),
[GPU sampling](https://github.com/google-ai-edge/ml-drift/blob/17ec4e6aa3f0103020bdaacc1303b1f2ad270c87/ml_drift/common/kernels/resampler.cc),
[CPU warp](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/perception/dense_image_warp.cc).

### `TFLite_Detection_PostProcess`

Registered by the standard TFLite builtin resolver despite using opcode
`CUSTOM`. It decodes boxes and performs nonmaximum suppression (NMS).

| Slot | Tensor | Type / shape |
| --- | --- | --- |
| input 0 | `box_encodings` | `FLOAT32` or affine `UINT8`, `[1,N,4 or more]`; first four values are center/size deltas. |
| input 1 | `class_predictions` | `FLOAT32` or affine `UINT8`, `[1,N,C]`; C is `num_classes` or `num_classes+1` with a background class. |
| input 2 | `anchors` | `[N,4]`, center y/x and height/width; `FLOAT32` when box encodings are float32, affine `UINT8` when box encodings are uint8. The quantized decoder reads anchor bytes as uint8. |
| output 0 | `detection_boxes` | `FLOAT32 [1,M,4]`, ordered ymin,xmin,ymax,xmax. |
| output 1 | `detection_classes` | `FLOAT32 [1,M]`, class IDs represented as floats. |
| output 2 | `detection_scores` | `FLOAT32 [1,M]`. |
| output 3 | `num_detections` | `FLOAT32 [1]`; valid output count. |

Here `M=max_detections*max_classes_per_detection`; execution supports batch 1.
The required FlexBuffers keys are integer `max_detections`,
`max_classes_per_detection`, `num_classes`, and float `nms_score_threshold`,
`nms_iou_threshold`, `y_scale`, `x_scale`, `h_scale`, `w_scale`.
Optional integer `detections_per_class` defaults to 100; boolean
`use_regular_nms` defaults to false. Regular NMS and fast NMS select different
class/box suppression algorithms. Supply a positive IoU threshold ≤1 and
nonzero box scales. Preserve the source model's background-class convention
and quantization metadata; this op does not perform classifier calibration.

Source: [kernel and options](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/detection_postprocess.cc).

### `AudioSpectrogram` and `Mfcc`

Both are standard-resolver `CUSTOM` registrations, with float32 arithmetic.

| Name | Ordered inputs → output | Required FlexBuffers options |
| --- | --- | --- |
| `AudioSpectrogram` | `FLOAT32 audio[samples,channels]` → `FLOAT32 [channels,frames,bins]` | Integer `window_size`, integer `stride`, boolean `magnitude_squared`. `frames=max(0,1+floor((samples-window_size)/stride))`; FFT size is the power-of-two size chosen for the window, `bins=fft_size/2+1`. Uses the kernel's windowing, not an arbitrary STFT window input. |
| `Mfcc` | `FLOAT32 squared_spectrogram[channels,frames,bins]`, `INT32 sample_rate` (one element) → `FLOAT32 [channels,frames,dct_coefficient_count]` | `upper_frequency_limit`, `lower_frequency_limit`, `filterbank_channel_count`, `dct_coefficient_count`. The checked reader uses `AsInt64()` for **all four** options, including frequency limits; emit integral values to match it. |

Feed squared magnitudes into `Mfcc`; its implementation takes the magnitude
before the mel filterbank and log/DCT processing. Passing unsquared magnitudes
changes the features. Match sample rate, window/stride, frequency bounds, and
DCT count to model preprocessing.

Sources: [spectrogram](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/audio_spectrogram.cc),
[window/FFT implementation](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/internal/spectrogram.cc),
[MFCC](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/mfcc.cc).

### `NumericVerify`

Standard-resolver diagnostic custom op. Inputs are the quantized/float16 result
and a float32 reference; output is float32 with the first input's shape,
containing `dequantized_result-reference`. Supported first-input types are
`UINT8`, `INT8`, `INT16`, and `FLOAT16`. Shapes and element counts must agree.
FlexBuffers options are float `tolerance` and boolean `log_if_failed`. When
`log_if_failed` is true and `tolerance>=0.1`, an absolute difference exceeding
`tolerance*input.params.scale` causes an invocation error. Otherwise it records
the differences/statistics. This is a debugging boundary, not a replacement
for quantize/dequantize inference math.

Source: [kernel](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/numeric_verify.cc).

### `aeq.hadamard_rotation`

Standard-resolver custom op with one input and one same-shaped output.
Use `FLOAT32 [rows,D]` or `[B,rows,D]`. FlexBuffers options contain integer
`hadamard_size=K` and vector `random_binary_vector`. The implementation computes
a normalized Walsh–Hadamard transform independently over consecutive K-element
blocks (`1/sqrt(K)` scaling). K must be a positive power of two and divide D.

The checked code reads and stores `random_binary_vector` but does **not** apply
it in `Eval`; do not expect random sign flips. Although `Prepare` permits
`INT32`, `Eval` reads/writes float pointers. Float32 is the usable converter
contract. The kernel does not infer/resize the output in `Prepare`; serialize
the correct shape, dtype, and storage size.

Source: [implementation](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/hadamard_rotation.cc).

### `ParseExample` and `ParseExampleV2`

Explicit CPU registrations parse TensorFlow `Example` protobufs in string
tensors. Let Ns be the sparse-feature count and Nd the dense-feature count.

| Name | Ordered input slots |
| --- | --- |
| `ParseExample` | `serialized`, `names`, Ns scalar string sparse keys, Nd scalar string dense keys, Nd dense-default tensors. |
| `ParseExampleV2` | `serialized`, `names`, string-vector `sparse_keys`, string-vector `dense_keys`, `ragged_keys`, then Nd dense-default tensors. |

Outputs are grouped: Ns `INT64` sparse-index tensors `[nnz,2]`, Ns sparse-value
vectors, Ns `INT64` sparse-shape vectors `[2]`, then Nd dense tensors. Sparse
and dense feature values support `FLOAT32`, `INT64`, or `STRING`; defaults and
declared output types must agree. Sparse outputs are dynamic. The checked V2
implementation constructs sparse/dense results only; keep ragged keys empty
rather than assuming TensorFlow's full ragged-output contract is implemented.

Custom options accept a two-element FlexBuffers vector whose second element
contains a serialized TensorFlow `NodeDef`, or the kernel's attribute-map form.
V1 reads `Ndense` and `Nsparse`; V2 reads `num_sparse` and derives dense count
from the dense-key tensor. The NodeDef form also provides `dense_shapes` (and
V2 `Tdense`); the map path relies on predeclared output shapes for missing shape
metadata. Reuse the kernel's converter/test construction for that mode rather
than supplying only feature counts. These kernels cache feature configuration
after first use; keys and default configuration should be fixed model inputs
or constants, not changed between invocations.

Source: [parser kernel and registerer](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/parse_example/parse_example.cc).

### Other explicit CPU custom registrations

These kernels require an explicit resolver entry. Several operations also have
modern builtin equivalents. A matching builtin is generally the direct-export
choice; use these custom forms only for a runtime integration that registers
the exact code. Spellings below come from in-tree registrations or kernel
tests, not from automatic case-insensitive dispatch.

| Custom code | Ordered inputs → output | Types, options, and constraints |
| --- | --- | --- |
| `atan2` | `y`, `x` → elementwise angle | Identical shape/type; float32 or float64. No broadcasting or options. |
| `Sign` | `x` → same-shaped sign | Float32 or float64; no options. |
| `RandomStandardNormal` | integer shape vector → sampled tensor | Shape int32/int64; output float32/float64. No consumed seed options; uses a stateful standard-library engine. |
| `RandomUniform` | integer shape vector → sampled tensor | Shape int32/int64; output float32/float64 in `[0,1)`. No consumed seed options. |
| `RandomUniformInt` | shape, scalar min, scalar max → sampled tensor | Shape int32/int64; integer limits and output int8/int32/int64. The checked custom kernel uses `std::uniform_int_distribution(min,max)`, so its range is **inclusive `[min,max]`**, unlike the usual exclusive upper bound. No consumed seed options. |
| `Multinomial` | logits `[B,C]`, scalar int32 `num_samples` → `[B,num_samples]` | Float32/float64 logits, int32/int64 sampled class IDs; sampling with replacement. No consumed options. |
| `Roll` | `x`, `shift`, `axis` → same-shaped tensor | Shift/axis scalar or rank-1 int32/int64 with equal element counts. Input float32, int8/int16/int32/int64, uint8, bool, or string. No options. |
| `Irfft2d` / `IRFFT2D` | complex64 spectrum, int32 `fft_length[2]` → float32 real tensor | Rank ≥2; transform the last two axes, preserve leading dimensions. Both FFT lengths must be powers of two. The kernel test and test-driver register different spellings; register the exact exported code. No options. |
| `AveragePool3D`, `MaxPool3D` | NDHWC tensor → pooled NDHWC tensor | Float32, int8, or int16 with matching input/output type; quantized scale/zero point must match. FlexBuffers `data_format="NDHWC"`, typed vectors `ksize=[1,Kd,Kh,Kw,1]`, `strides=[1,Sd,Sh,Sw,1]`, `padding="SAME"/"VALID"`. Strides positive; no fused activation option. |
| `Table` | integer input, rank-1 lookup table → same-shaped integer output | All tensors int8 or all int16. Int8 table has 256 entries; int16 uses the kernel's 513-entry interpolated LUT. Int16 input/output zero points must be zero. No options. |
| `BroadcastGradientArgs` | two rank-1 shape vectors → two reduction-axis vectors | Same int32/int64 type throughout; dynamic output lengths. Returns axes needed to undo broadcast in each operand's gradient. No options. |

The random custom kernels do not use the seed attributes of the corresponding
random **builtins**. Do not promise portable, bitwise reproducible RNG output
across standard-library implementations. Header declarations for custom
hash-table factories alone do not establish a custom wire contract; the
checked hash-table runtime operators are builtins.

Sources: [factory declarations](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/custom_ops_register.h),
[atan2](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/atan2_custom.cc),
[sign](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/sign_custom.cc),
[normal RNG](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/random_standard_normal_custom.cc),
[uniform RNG](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/random_uniform_custom.cc),
[categorical RNG](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/multinomial.cc),
[roll](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/roll.cc),
[IRFFT](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/irfft2d.cc),
[3-D pooling](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/pooling3d.cc),
[LUT](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/table.cc),
[broadcast gradient axes](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/kernels/gradient/bcast_grad_args.cc).

## Compiler partitions and Select TF Ops

`odml.npu_call` and `odml.cpu_call` are preserved compiler partition markers,
not fixed-arity numerical operators. Ordered operands/results and the matching
decomposition describe the outlined partition. LiteRT compiler processing
selects `odml.npu_call` for accelerator compilation and shields
`odml.cpu_call` from plugin selection before late inlining. Emit them only as
part of that compilation workflow.

`DISPATCH_OP` is the compiler/runtime custom boundary for a compiled partition.
Its inputs/outputs follow the partition; options identify generated executable
information, accompanied by bytecode attachments/build metadata. Use LiteRT's
compiler and serializer to produce it. Writing only the custom code does not
produce an executable accelerator model.

`Flex<OpName>` (for example `FlexAddV2`) is an open custom-op family supported
by Select TF Ops/Flex, requiring that delegate and its TensorFlow kernels.
Options are a FlexBuffers vector containing the TensorFlow op name and
serialized `NodeDef`. Signatures, attributes, and availability belong to the
selected TensorFlow kernel set. For a direct converter targeting LiteRT without
TensorFlow, emit supported LiteRT builtins instead; the Flex prefix does not
provide a universal fallback.

Sources: [partition processing](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/compiler/plugin/compiler_plugin.cc),
[compiler integration](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/COMPILER_PLUGIN.md),
[dispatch name and metadata](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/litert/core/build_stamp.h),
[Flex integration](https://github.com/google-ai-edge/LiteRT/blob/6c6af0163c346459fe6f3024b0bb5c3a02f4a834/tflite/delegates/flex/delegate.cc).

## Builtin fallbacks affected by recent torch changes

Recent frontend changes also improve decompositions without adding runtime
custom names. These are useful when choosing builtin fallbacks:

| Source operation | Current lowering pattern |
| --- | --- |
| Static contiguous `aten.as_strided` | Flatten, optional static slice for storage offset, reshape. Check row-major strides while ignoring size-1 axes; other views retain the gather path. |
| Static slice with step > 1 | `STRIDED_SLICE`; the entire shape and bounds must be concrete. Avoid an unnecessary `GATHER_ND` for a simple strided slice. |
| PReLU | `relu(x) - weight*relu(-x)` avoids the old comparison/SELECT decomposition. Preserve the source's numerical behavior when adopting the rewrite. |
| Pixel shuffle/unshuffle | Interleave one spatial axis at a time using reshape/transpose with intermediates of rank at most 4. This does not require either legacy pixel-shuffle custom spelling. |
| Qwen3.5 prefill convolution state | `GATHER` along axis 2 with INT32 indices replaces the floating one-hot selector/matmul. This is state preparation outside `gated_delta_update`. |

Sources: [contiguous-view lowering](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/backend/lowerings/_jax_lowerings/torchax_ops.py),
[static slices](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/backend/lowerings/_jax_lowerings/lowerings.py),
[PReLU and shuffle decompositions](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/backend/lowerings/_decomp_registry.py),
[Qwen3.5 state update](https://github.com/google-ai-edge/litert-torch/blob/78ff964262f8a5f9c536789d68a4ec29cf39061a/litert_torch/generative/export_hf/model_ext/qwen3_5/gated_delta_rule.py).

## Converter validation

Record the intended backend and runtime registration alongside the exact
serialized contract. Before marking an export ready for execution, validate:

1. The final FlatBuffer's exact name, encoding/version, ordered operand/result
   counts, types, shapes, constant buffers, and option bytes.
2. Native execution on the selected backend, including confirmation that the
   node was delegated/compiled. Successful loading or CPU decomposition alone
   does not demonstrate native support.
3. Agreement with source-model math and, for composites, the builtin fallback.
   Cover masks, broadcast dimensions, scale/epsilon values, and quantized
   boundaries relevant to the op.
4. Stateful behavior over multiple calls: nonzero offsets, partial prefills,
   cache reuse/reset, capacity boundaries, and ring wraparound where enabled.

Record backend work separately from the intended export contract. For an
execution-ready artifact, a custom node needs its registration/delegate; a
composite can use its equivalent builtins when native support is unavailable.
Do not silently change attributes or model semantics to satisfy an old parser.
An invented name or converter-only allowlist does not make an op executable.
Native support on one backend does not establish that another implements the
same contract; keep unimplemented targets visible as backend follow-up work.

## Maintaining the inventory

Refresh the revisions, enumerate registrations and name-dispatch sites, then
trace each candidate to the actual parser and execution implementation.
Inspect emitters/tests for intent and examples, not as substitutes for a
runtime consumer. Keep MLDrift implementations even when a name is absent from
the two official main branches, provided a concrete LiteRT integration exists.
The separate checkout's default custom factory returns unsupported; an
internal kernel with similar math alone is not an exportable operator ABI.

```bash
git -C ~/src/LiteRT fetch upstream main
git -C ~/src/litert-torch fetch origin main

# Runtime registrations and delegate dispatch, including names split over lines.
git -C ~/src/LiteRT grep -n -E 'AddCustom|custom_name|custom_code|odml\.' upstream/main -- \
  litert/runtime litert/experimental litert/vendors ml_drift_delegate tflite/delegates tflite/kernels tflite/core/kernels tflite/experimental

# Producer intent; a hit here alone does not establish execution support.
git -C ~/src/litert-torch grep -n -E 'odml\.|custom_call\.' origin/main -- litert_torch

# Check all three sources before removing a name; inspect the resulting code.
op_name=odml.runtime_bmm
git -C ~/src/LiteRT grep -a -l -F "$op_name" upstream/main --
git -C ~/src/litert-torch grep -a -l -F "$op_name" origin/main --
rg --hidden -l -F -g '!.git' "$op_name" /data/home/chasun/src/ml-drift
```

The contracts above are source-audited at the recorded revisions. They are not
a claim that every shape/type combination has been executed on every backend.
