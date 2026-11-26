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

Refreshed on 2026-11-25 against the upstream revisions below. The audit
compared registrations and name dispatch with the November 5 inventory and
traced changed contracts through their emitters, parsers, kernels, and tests.

| Git repository URL | Audited commit |
| --- | --- |
| <https://github.com/google-ai-edge/LiteRT.git> | `9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d` |
| <https://github.com/google-ai-edge/litert-torch.git> | `7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f` |
| <https://github.com/google-ai-edge/ml-drift.git> | `57dc62c9ee54396f242b122f8ce384248647fe14` |

Runtime and emitter evidence comes from those Git trees, excluding local
commits and uncommitted changes. Source links below pin the audited revisions.
Torch emitters help establish intended semantics, but do not prove backend
support. An entry needs a concrete execution path. This catalog is bounded by
these sources; it cannot enumerate externally supplied custom libraries or
every operation in the open-ended Select TF Ops family.

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
| Sparse attention | The QSA indexer returns block IDs and two updated states. Its sparse attention consumer takes head-major Q and returns token-major output. Emit the scale and a BOOL visibility mask for shared XNNPACK, MLDrift, and NVIDIA models. |
| Factorized feed-forward layers | MLDrift implements `odml.blocked_ffw` and the alias `odml.block_ffw`. NVIDIA implements `odml.factorized_mlp` with six constant INT2 weights and six activation scales. |
| Dense attention and cache updates | MLDrift supports packed INT4 K/V and symmetric INT8 queries. YNNPACK accepts quantized INT4/INT8 Q/K/V and additive masks, and can return INT8. NVIDIA adds quantized cache paths and a parameters-only float-cache SDPA form. |
| Recurrence and activation | The gated-delta block can use a sigmoid output gate. Custom `moe` accepts SiLU on MLDrift, XNNPACK, and NVIDIA. MLDrift short convolution accepts up to eight taps. NVIDIA accepts up to sixteen with a different state and weight layout. |
| Normalization and RoPE | The Arm ML Extensions for Vulkan plugin (`arm_vulkan_ml`) checks composite versions for RMS, group, and L2 norm. NVIDIA RMSNorm reads the serialized epsilon. MLDrift IR supports constant RoPE positions and constant Q/K normalization weights. |
| Export and coverage | Externalizing quantized MoE weights is opt-in for XNNPACK bundles. YNNPACK ATS now permits transposed SDPA and the previously filtered runtime-BMM cases. |

In litert-torch, `use_sdpa_composite=True` enables `apply_gpu_composites`,
which fills unset GPU optimization flags including vision SDPA. RoPE also
defaults on when `externalize_rope=False`. Fused ring-cache
attention requires a separate opt-in. [Torch export configuration](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/core/exportable_module_config.py).

## Direct serialization and execution requirements

| Field | Preserved composite | Runtime custom op |
| --- | --- | --- |
| `OperatorCode.builtin_code` | `STABLEHLO_COMPOSITE` | `CUSTOM` |
| Name | `StableHLOCompositeOptions.name` | `OperatorCode.custom_code`, exact case-sensitive string |
| Options location | `Operator.builtin_options_2`, union type `StableHLOCompositeOptions` | Inline `Operator.custom_options`, or an appended payload located by `large_custom_options_offset` and `large_custom_options_size` |
| Options encoding | `composite_attributes` bytes and `composite_attributes_format=FLEXBUFFERS` | Per-op format; usually FlexBuffers, sometimes a native C struct |
| Decomposition | `decomposition_subgraph_index` identifies a real matching subgraph | None attached to this operator |
| Versions | `OperatorCode.version` and composite `version` are separate fields | `OperatorCode.version` must match resolver registration |
| Execution | Named backend implementation, or an executable decomposition | Actual custom registration or a delegate that claims the node |

For `litert-converter`, set `CoreOp.builtin` before assigning typed options.
The builtin setter resets the options. For a preserved composite, populate
`CoreStableHLOCompositeOptions` with the name, version, valid decomposition
subgraph index, attribute bytes, and format. Select the version for the
intended consumer. Arm Vulkan ML requires version 0 for RMSNorm and L2Norm
and version 1 for GroupNorm. Custom ops use `CoreOp.custom_code`,
`custom_options`, and `custom_options_format`.

The LiteRT loader reads inline custom options first. When that vector is absent,
it accepts a file-relative large-options offset greater than 1 and checks that
the full byte range fits the allocation. The weight externalizer preserves
appended option payloads when rebuilding a model. [Model loader](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/core/model/model_load.cc) ·
[Weight externalizer](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/tools/externalize_tflite_flatbuffer.py).

A decomposition must have the same ordered inputs and outputs, matching types
and shapes, and reproduce all attributes, masking, quantization, and state
updates. Prefer supported builtin operations inside it. If it contains custom
ops, record and provide those registrations too: the current torch
`odml.gated_delta_block` decomposition is one such case. Test it with delegation
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

For dynamically loaded runtimes, `LiteRtSetOpResolverOption` accepts C lookup
callbacks. The caller owns their user data and returned `TfLiteOperator`
objects for the lifetime of the compiled models. Minimal static CPU builds
reject this callback API. [LiteRT C options](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/c/litert_options.h).

`LiteRtAddCustomOpKernelOption` records the requested version, but the audited
compiled-model path calls `MutableOpResolver::AddCustom(name, registration)`
with its default version of **1**. A model declaring custom opcode version 2
therefore needs a resolver entry for version 2 even if the application passed
2 to this option. [C option implementation](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/c/litert_options.cc#L114-L128) ·
[Compiled-model registration](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/runtime/compiled_model.cc#L563-L571) ·
[Resolver default](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/mutable_op_resolver.h#L82-L87).

Sources: [FlatBuffer schema](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/converter/schema/schema.fbs),
[composite fallback](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/stablehlo_composite.cc),
[standard registrations](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/core/kernels/register.cc),
[custom registrations and accelerator stubs](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/runtime/compiled_model.cc),
[MLDrift legacy factory](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/custom_parsers.cc),
[MLDrift IR factory](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/ir/custom_parsers.cc).

## Catalog

The tables identify concrete consumers, not a promise of support on every
device or every build. See each linked section for the admitted contract.
`C` means `STABLEHLO_COMPOSITE`; `U` means `CUSTOM`.

| Exact name | Encoding | Concrete consumer / setup |
| --- | --- | --- |
| [`odml.scaled_dot_product_attention`](#odmlscaled_dot_product_attention) | C; U for XNNPACK/YNNPACK | XNNPACK, YNNPACK, MLDrift composite, Google Tensor plugin |
| [`odml.sdpa_transposed`](#odmlsdpa_transposed) | C; U for YNNPACK | MLDrift, YNNPACK; restricted NVIDIA plugin path |
| [`odml.fused_sdpa_cache_update`](#odmlfused_sdpa_cache_update) | C | MLDrift and NVIDIA plugin |
| [`odml.qsa_indexer_update`](#odmlqsa_indexer_update) | C, also U on MLDrift | XNNPACK, MLDrift, NVIDIA plugin |
| [`odml.sparse_sdpa_transposed`](#odmlsparse_sdpa_transposed) | C, also U on MLDrift | XNNPACK, MLDrift, NVIDIA plugin |
| [`odml.rms_norm`](#odmlrms_norm) | C | MLDrift and the Qualcomm, MediaTek, NVIDIA, Samsung, Arm Vulkan ML, and Google Tensor plugins |
| [`odml.group_norm`](#odmlgroup_norm) | C | MLDrift, Qualcomm, Arm Vulkan ML, Google Tensor, and MediaTek for `num_groups=1` |
| [`odml.l2_norm`](#odmll2_norm) | C | Qualcomm, MediaTek, and Arm Vulkan ML plugins |
| [`odml.runtime_bmm`](#odmlruntime_bmm) | C; U for YNNPACK | MLDrift, YNNPACK, NVIDIA plugin |
| [`odml.cache_update`](#odmlcache_update) | C | MLDrift, NVIDIA plugin |
| [`odml.rope`](#odmlrope) | C or U on MLDrift | MLDrift and NVIDIA quantized projection fusion (C) |
| [`odml.qkv_norm_rope`](#odmlqkv_norm_rope) | C or U for MLDrift | MLDrift; NVIDIA plugin for full QKV composite |
| [`odml.swiglu`](#odmlswiglu) | C or U on MLDrift | MLDrift and NVIDIA concatenated form (C) |
| [`odml.short_conv_step`](#odmlshort_conv_step) | C or U on MLDrift | MLDrift and NVIDIA ungated channels-first state form (C) |
| [`odml.gated_delta_block`](#odmlgated_delta_block) | C or U | MLDrift; composite fallback needs recurrence registration |
| [`odml.blocked_ffw`, `odml.block_ffw`](#odmlblocked_ffw-and-odmlblock_ffw) | C or U | MLDrift |
| [`odml.factorized_mlp`](#odmlfactorized_mlp) | C | NVIDIA plugin |
| [`odml.moe_experts`](#odmlmoe_experts) | C | YNNPACK |
| [`moe`](#moe) | U | MLDrift; XNNPACK MoE delegate kernel; restricted NVIDIA plugin path |
| [`gated_delta_update`, `custom_call.gated_delta_update`](#gated_delta_update-and-custom_callgated_delta_update) | U | Explicit gated-delta-net CPU registerer and NVIDIA plugin for both names, MLDrift for unprefixed name |
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

Emit `scale` explicitly. XNNPACK and YNNPACK default to `1/sqrt(H)`, while
the MLDrift fused kernel defaults to 1. For XNNPACK, serialize `scale` and
any `logit_cap` as **inline FP32 values in four-byte FlexBuffers map slots**.
Its reader takes a raw `float*` into the map. For example, a slot containing
FP64 `0.5` reads as FP32 `0.0` on a little-endian host. Omit `logit_cap`
for uncapped attention. A present FP32 zero enters XNNPACK's divide-by-cap
path. [XNNPACK attribute reader](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/xnnpack/flexbuffers_util.h#L45-L54) ·
[XNNPACK cap lowering](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/xnnpack/xnnpack_delegate.cc#L6715-L6747).

For a positive cap, XNNPACK and YNNPACK apply `cap*tanh(logits/cap)` before
masking and softmax. The MLDrift composite parsers read only `scale`. For
capped attention on MLDrift, implement the cap in the decomposition and
ensure that execution uses that path. The intended math is
`softmax(cap_if_requested(scale*Q@K^T)+mask)@V`.

YNNPACK requires equal Q/K/V head counts for this sequence-major entry point.
Its native GQA path accepts `odml.sdpa_transposed`. Q and output have rank 4.
The sequence-major form also accepts FLOAT16/BFLOAT16 tensors, BOOL masks
(true keeps the entry, false adds -10000), and an optional runtime length.
Quantized INT4/INT8 Q, K, V, and additive masks are supported, as is INT8
output. See the active-length and dot-precision rules under
[`odml.sdpa_transposed`](#odmlsdpa_transposed).

MLDrift now selects a fused tiled prefill kernel for eligible float32/float16
Q/K/V on OpenCL, Metal, and WebGPU: `T>1`, `H % 4 == 0`, `H<=128`, divisible
head counts, compatible mask broadcasting, and device workgroup/memory limits.
This kernel has no active-length operand. Its internal BOOL-mask support does
not establish a public IR contract: the checked IR admission still allows
only floating masks for this name. Keep the additive floating-mask form for
shared exports.

Common PyTorch inputs are head-major `[B,N,T,H]`. Transpose to the serialized
sequence-major contract; tensor names do not change the interpretation.
XNNPACK also transposes internally, so preserving this op alone does not prove
a performance gain. Emit floating attributes and preserve scale application
exactly once in both the native path and decomposition.

Sources: [XNNPACK visitor](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/xnnpack/xnnpack_delegate.cc),
[YNNPACK visitor](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/ynnpack/attention.cc),
[MLDrift composite parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/model_builder.cc),
[IR admission](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/support/support_sdpa.cc),
[IR attributes](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/convert/convert_sdpa.cc),
[fused SDPA contract](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/sdpa_kernel.h),
[fused SDPA scale and math](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/sdpa_kernel.cc).

### `odml.sdpa_transposed`

This preserved attention composite uses head-major tensors and can consume
MLDrift's packed K/V cache. It is distinct from
`odml.scaled_dot_product_attention`. MLDrift, NVIDIA, and YNNPACK handle the
composite under the constraints below. YNNPACK additionally recognizes the
same-name `CUSTOM` form, which needs resolver registration.

| Slot | Tensor | Converter contract |
| --- | --- | --- |
| input 0 | `query` | Logical `[B,Nq,T,H]`, pre-scaled. MLDrift also accepts a symmetric INT8 query. |
| input 1 | `key` | Logical `[B,Nkv,S,H]` for `k_ts_idx=2`; floating or quantized under the backend-specific rules below. |
| input 2 | `value` | Logical `[B,Nkv,H,S]` for `v_ts_idx=3`; floating or quantized under the backend-specific rules below. |
| input 3 | `mask` or `param_tensor` | Four-input form: `INT32` means parameters; otherwise mask. BOOL true=keep, false=replace with `-10000`; floating masks are additive. |
| input 4 | `param_tensor` | Five-input form: mask at slot 3, runtime `INT32 [1,1,1,7]` parameters at slot 4. MLDrift reads active bound at index 2. |
| output 0 | `output` | Floating `[B,Nq,T,H]`. Put decode flattening to `[B,1,Nq*H]` outside the composite. |

MLDrift admits 3–5 runtime inputs and one output. The three-input form is
unmasked attention. For GQA require `Nq % Nkv == 0` and map query head h to
KV head `h // (Nq/Nkv)`. Do not pack heads into T at the composite boundary.

| FlexBuffers attribute | Converter meaning / consumer behavior |
| --- | --- |
| `k_ts_idx=2`, `v_ts_idx=3` | Integer logical time axes used by the torch decomposition. MLDrift does not use these attributes to transpose arbitrary input layouts. |
| `softcap`, `logit_cap` | For capped models shared by MLDrift and YNNPACK, emit **both keys with the same positive float C**. Apply `C*tanh(logits/C)` once before masking. MLDrift prefers `softcap` and accepts `logit_cap` as an alias. YNNPACK reads only `logit_cap`. Omit both for uncapped attention. |
| `is_causal` | Boolean, default false. True promises that explicit mask semantics are fully represented by causal positions and the active cache bound. This is permission to remove the mask on an eligible path. |
| `from_cache_update` | Optional boolean identifying MLDrift cache-produced packed K/V. The parser also infers it from a visible cache-update producer. Do not assert it for ordinary external K/V. |

Set `is_causal=true` only after proving the mask is purely causal. Sliding
windows, bidirectional image spans, arbitrary padding/attention patterns, and
finite additive biases need their explicit mask unless their complete semantics
are otherwise represented. The torch helper defaults false, and the export
path gates global prefill causality by task/model/layer semantics. It also
opts in for single-token parameterized decode, where the valid prefix is
already visible, including its ring-buffer decode path. A direct converter
must prove its own mask equivalence; those frontend shortcuts are not a
general test.

Keep the mask operand and its semantics in the serialized decomposition.
Pruning requires all of `is_causal=true`, float packed-cache origin, runtime
parameters, a BOOL mask, GPU capability information, and the shared
`UsesFusedSdpaKernel` eligibility check. It now covers eligible Metal,
OpenCL, and WebGPU paths. INT8 and INT4 caches retain their mask. Multi-token decode
keeps its mask. The fused path uses `key < active_tokens` and
`key <= q_start + query_index`. Missing `is_causal` retains the mask. Older
runtimes can ignore the extra attribute while still using that mask. Do not
remove host mask construction merely because the attribute was emitted:
that requires runtime confirmation that every relevant consumer can skip it.

Fused kernels require float buffer K/V in the cache writer's packed layout,
`H % 4 == 0`, `S % 4 == 0`, divisible head counts, and device workgroup/local
memory limits. Apple Metal's matrix prefill additionally requires `H<=128`.
OpenCL/WebGPU use the workgroup kernel for prefill when the decomposed score
tensor exceeds the configured budget. Apple WebGPU compares the per-chunk
score size, so long prefills that fit in chunks can use the decomposed graph.
All three APIs support decode subject to their kernel selection criteria,
including WebGPU's GQA grouping constraint. A retained fused-kernel
mask must be shared across heads, have one row or T rows, and cover S entries.
The parser treats `T>2` or batch greater than 1 as prefill. On eligible Metal
devices the wave-SIMD decode selector accepts multiples of four from 64 to
256, and some larger dimensions through 512 depending on the workgroup shape.

For MLDrift quantized K/V, use the same storage type for both tensors, either
symmetric per-tensor INT8 or INT4, batch 1, and time axes 2/3. Each cache
requires a positive finite scale and zero point 0. INT4 additionally requires
H divisible by eight and S divisible by four. K/V must be graph inputs already
in MLDrift's packed layout or outputs of `odml.cache_update`. Keep writer
attributes and tensor scales consistent. The multi-operation graph retains
the explicit mask. For INT4, a separate GPU pass expands the active slots
(up to kernel alignment) into capacity-sized byte tensors for the INT8
matmuls. An INT8 Q requires symmetric per-tensor quantization and this node
as its sole consumer. Output is floating.

Torch now keeps the composite boundary/output rank 4, packs GQA only inside
its decomposition, and applies the final decode reshape after `mark_outputs`.
MLDrift can fuse that following reshape itself. Runtime parameters are retained
through zero-valued dependencies in the torch fallback matmuls; fallback still
computes full extent and relies on the mask for valid-token semantics.
The torch helper also converts a floating mask to `mask == 0` when parameters
are present. A direct converter must preserve finite additive bias instead of
copying that conversion for arbitrary floating masks.

YNNPACK accepts FLOAT32/FLOAT16/BFLOAT16 or quantized INT4/INT8 Q, K, and V
in logical tensor layout. Floating Q and K must match when both are
unquantized. Output may be floating or quantized INT8. Use per-tensor
affine quantization and an even innermost extent for every INT4 tensor.
The delegate widens INT4 operands to INT8. A quantized INT4/INT8 mask is
dequantized and added. INT32/INT64 in the fourth slot denotes runtime
parameters. The delegate reads active length from element 1 (element 0 for a
one-element tensor) and defaults `scale` to 1 for the transposed name.

The delegate option `sdpa_dot_type` controls arithmetic separately from
storage. By default (`kTfLiteNoType`), each dot uses the more precise operand
type, treating quantized inputs as INT8 and probabilities after an integer QK
dot as FP32. A floating override converts/dequantizes both dots to that type.
The INT8 setting uses the K and V types for their respective dots and
dynamically quantizes a floating left operand for an integer dot. BOOL false
still contributes an additive -10000, and the cap key is `logit_cap`.

NVIDIA's FP16-stored cache path admits batch 1, compatible GQA tiling, and
H in {128, 256, 512}. With five inputs it uses a BOOL mask
`[1,1,1 or T,S]`. With four, slot 3 supplies INT32 `[1,1,1,7]`. The compiler
builds a mask using element 2 as the active count and element 0 for the
causal query offset when `is_causal` is set. These float-cache paths do not
apply a logit cap.

NVIDIA's specialized quantized path takes five inputs, pre-scaled INT8 Q,
symmetric per-tensor INT8/INT4 K/V, a BOOL mask, and floating output. The
optimized kernel returns zero for an all-masked row. With 16-bit
activations and `LITERT_NVIDIA_TENSORRT_GENERAL_ATTENTION=1`, a separate
lowering handles eligible forms outside the specialized path, including
FLOAT or INT8 Q, BOOL or additive floating masks, `scale`, `softcap` (alias
`logit_cap`), and UINT8 caches with zero point 128. In this general form slot
3 is the mask. Verify which lowering claims a capped model.

Sources: [torch emitter](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/experimental/composites/sdpa.py),
[task/model gating](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/core/exportable_module.py),
[layer gating](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/core/attention.py),
[legacy GPU parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/sdpa_transposed_parser.cc),
[IR GPU parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/ir/sdpa_transposed_parser.cc),
[GPU kernels and selection](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.cc),
[shared eligibility checks](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/sdpa_transposed_support.cc),
[NVIDIA admission and attention fusion](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc),
[decode reshape fusion](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/fuse_sdpa_transposed_reshape.cc),
[YNNPACK contract](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/ynnpack/attention.cc),
[YNNPACK live length](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/ynnpack/ynnpack_delegate.cc).

### `odml.fused_sdpa_cache_update`

MLDrift implements this preserved composite for sliding-window attention. It
attends to the **pre-write** ring cache plus the current token block, then
optionally writes the cache. Ordering reads before writes is part of the
contract. It is not a rename of cache update followed by transposed SDPA.

Let W be ring capacity, T the new-token block length, and G=Nq/Nkv. All seven
inputs must be runtime operands in the MLDrift legacy parser. The floating
form accepts FLOAT32 or FLOAT16. Use batch 1 and positive W, T, and H,
with W and H divisible by four.

| Slot | Tensor | Shape / meaning |
| --- | --- | --- |
| input 0 | query | `[1,Nkv,G*T,H]`, pre-scaled; each KV head's query heads are packed in G-major order. |
| input 1 | old K cache | `[1,Nkv,W,H]`. |
| input 2 | old V cache | `[1,Nkv,H,W]`. |
| input 3 | new K | `[1,Nkv,T,H]`. |
| input 4 | new V | `[1,Nkv,H,T]`. |
| input 5 | mask | `[1,1,T,W+T]`, BOOL selection with false replaced by `-10000`, or additive float. First W columns describe physical old-cache slots; final T columns describe new tokens. |
| input 6 | parameters | INT32 with at least two elements; normally the live `[1,1,1,7]` signature input. Element 0 is start position; element 1 is end position. |
| output 0 | attention | Floating, same shape as query. |
| outputs 1/2 | updated K/V | Present together in the three-output form; match old-cache shapes/types. The one-output form does not write caches. |

Torch emits integer `cache_size=W`, boolean `update_cache`, and optional
positive float `softcap`. MLDrift derives capacity from shapes and write mode
from output count; keep those redundant attributes consistent. Scores against
old and new keys share one capped/masked softmax over W+T entries; each packed
query row r uses mask row `r % T`. There is no implicit attention scale or
causality attribute.

Use a nonnegative start position and set `valid=clamp(end-start,0,T)`.
The writer stores the last `min(valid,W)` valid new tokens in slots
`(start+i)%W`, preserving untouched slots. Provide
the mask for causal/sliding-window visibility, invalid new tokens, and empty
old slots. The native softmax independently excludes old slots at or beyond
`clamp(start,0,W)`; the decomposition relies on the explicit mask. Test empty
cache and all-masked rows for the intended semantics.

MLDrift also admits a symmetric per-tensor INT8 query and matching INT8 or
INT4 cache types. With quantized caches, fresh K/V may be floating or
symmetric per-tensor INT8. Floating caches require floating updates. INT8 Q
or fresh K/V must have this node as their sole consumer. Quantized caches
must enter as graph inputs in the packed layout or come from
`odml.cache_update`. INT4 requires H divisible by eight. Cache outputs must
preserve their input shapes, types, and quantization scales. The kernel zeros
the old-cache contribution on an empty first step, including when old storage
contains NaNs.

NVIDIA's specialized form takes INT8 Q, INT8 graph-input caches, matching
INT8 updates, a BOOL mask and three outputs. Its opt-in general attention
path also accepts eligible floating, INT4, and UINT8 caches, additive masks,
and either one or three outputs. It derives write mode from output count
in the general path. [NVIDIA fused attention admission](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc#L1869).

Native caches require buffer storage in MLDrift's packed layouts and, when
written, correctly aliased input/output storage. The kernel orders writes
after attention consumes the old cache. The torch fallback includes a data
dependency to preserve that ordering and uses builtin matmuls/ring updates.
Packing, initialization, reset, and aliasing remain executor responsibilities.

Sources: [emitter and decomposition](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/experimental/composites/fused_sdpa_cache_update.py),
[GPU admission](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_parser.cc),
[IR parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/ir/fused_sdpa_cache_update_parser.cc),
[attention and ring-write kernels](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_kernel.cc).

### `odml.qsa_indexer_update`

This Qwen sparse attention (QSA) indexer completes blocks of new keys, updates
a ring and a pool of compressed keys, and selects blocks for each query row.
MLDrift, XNNPACK, and NVIDIA implement the preserved composite. MLDrift also
dispatches the exact custom name.

| Slot | Tensor | Type and logical shape |
| --- | --- | --- |
| 0 | Index queries | Floating `[B,T,Ni,Di]` |
| 1 | New index keys | Floating `[B,T,Di]` |
| 2, 3 | Incomplete-block ring, compressed block pool | Floating `[B,R,Di]`, `[B,M,Di]` |
| 4 | RMSNorm weight | Floating `[Di]` |
| 5, 6 | Block RoPE cosine, sine | Floating `[1,M,Dr]` each |
| 7 | Start position | One INT32 value |
| 8 | Visibility mask | BOOL `[B,1,T,S]` for shared export, with `S >= M*R` |
| outputs 0, 1, 2 | Selected block IDs, updated ring, updated pool | INT32 `[B,T,K]`, then tensors matching inputs 2 and 3 |

Emit integer `compress_ratio=R` and `block_topk=K`, floating `rms_norm_eps`
(default `1e-6`), and the intended floating `scale`. Complete blocks undergo
mean pooling, weighted RMSNorm, and block RoPE. Each query scores a visible
complete block by summing the positive parts of the per-index-head dot products
and multiplying by `scale`. The result contains block IDs with `-1` for unused
slots. Consume it as a selection of blocks. XNNPACK sorts by score, while the
MLDrift kernel emits selected IDs in block order.

For MLDrift, use FLOAT16/FLOAT32, B=1, R=4, `1 <= K <= 2048`, Di divisible
by four and at most 256, and Dr divisible by eight with `0 < Dr <= Di`.
Inputs 0–3 must be runtime tensors. The GPU kernel reads and updates *output*
storage for the ring and pool in place. Initialize both and retain them across
calls. XNNPACK reads explicit state operands. Its attribute parser requires R
and K and defaults the scale to `1/sqrt(Di)`, whereas MLDrift and NVIDIA default
to 1. NVIDIA requires in-place-compatible graph state and applies its CUDA
kernel shape limits. [MLDrift parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/qsa_indexer_update_parser.cc) ·
[GPU state and selection](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/qsa_indexer_update_kernel.cc) ·
[XNNPACK QSA kernels](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/xnnpack/qsa_delegate_kernel.cc) ·
[NVIDIA QSA admission](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc#L2296).

### `odml.sparse_sdpa_transposed`

The QSA attention consumer takes five inputs, in order. Use floating Q
`[B,Nq,T,H]`, floating K `[B,Nkv,S,H]`, floating V `[B,Nkv,H,S]`,
INT32 selected block IDs `[B,T,K]`, and a BOOL visibility mask `[B,1,T,S]`.
**The single floating output is token-major `[B,T,Nq,H]`.** Require
`Nq % Nkv == 0`. Set `compress_ratio` and `block_topk` to the indexer's R and
K, `k_ts_idx=2`, `v_ts_idx=3`, and an explicit floating `scale` for the QK dot.
XNNPACK defaults the scale to `1/sqrt(H)`, MLDrift and NVIDIA to 1.

The softmax includes visible tokens in selected complete blocks and visible
tokens in partially visible boundary blocks. The mask expresses visibility.
For floating masks, XNNPACK treats values greater than -1 as visible and the
MLDrift sparse kernel checks equality to zero. A BOOL mask avoids different
results for values such as -0.5. XNNPACK scans all partial blocks. MLDrift
collects at most eight partial boundary blocks, with at most two candidates
per work item. Use masks within those limits for matching results.

MLDrift accepts the composite or custom spelling at B=1, R=4, S divisible
by four, and H divisible by four up to 256. Use `from_cache_update=true`
only for its packed K/V layout. XNNPACK handles the composite using logical
tensors and accepts batch and query-row mask broadcasting. NVIDIA admits the
composite with eligible FP16-stored graph-input or cache-update caches at
H=64, 128, or 256. [MLDrift admission](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/sparse_sdpa_transposed_parser.cc) ·
[GPU mask and reduction](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/sparse_sdpa_transposed_kernel.cc) ·
[XNNPACK QSA kernels](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/xnnpack/qsa_delegate_kernel.cc) ·
[NVIDIA sparse kernel](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/trtllm/qsa_sparse_attention.cu) ·
[Example builtin decompositions](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/tools/qsa_decompositions.h).

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

The audited litert-torch RMSNorm helper has an export bug when
`with_scale=True`, `enable_hlfb=True`, and `final_scale` differs from one.
It marks only `x` and `w` as inputs. The decomposition computes
`(normalize(x) * final_scale) * w`, while a named-op implementation computes
`normalize(x) * w`. For this form, a converter can expose the normalized
value (using unit gamma) and apply both multiplications outside the composite
in their original order. [RMSNorm layer](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/layers/normalization.py#L76-L99) ·
[Composite boundary](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/layers/normalization.py#L210-L222).

In Google's published Gemma 3n E2B IT INT4 bundle, all **992** RMSNorm
decompositions apply only the exposed gamma after normalization. The vision
encoder has 184 calls, decode has 342, and each of the two prefill signatures
has 233. The vision graph's other multiplier is `1/D`, applied to the sum
of squares to compute the mean before `rsqrt`. [Inspected model revision](https://huggingface.co/google/gemma-3n-E2B-it-litert-lm/blob/c03b6f60b8da6c5400b6838a2cf26420f80c0a01/gemma-3n-E2B-it-int4.litertlm).

MLDrift's IR admission requires opcode version 1, nonconstant x of rank 2–4,
and constant rank-1 gamma when provided. It accepts float32/float16/bfloat16
tensor types; backend precision/storage still has to support them. That IR
path also supports omitting gamma for unweighted normalization. Use the
two-input form when sharing a model with consumers that require gamma.

NVIDIA's `LowerCompositeRmsNorm` reads the floating `epsilon` attribute
and defaults to `1e-6` when absent. It accepts static floating tensors and
optional gamma. `LITERT_NVIDIA_TENSORRT_NATIVE_COMPOSITES` can restrict
native lowering to a comma-separated list of short names, or `none` to inline.

Arm Vulkan ML requires composite **version 0**, two inputs, matching static
input/output dimensions, a channel-vector scale, and positive finite epsilon.
Its normalization admission accepts FP32 and supported quantized INT8/INT16
forms. An explicit `axis` must denote the last dimension. Google Tensor also
lists RMSNorm and standard SDPA for its compiler. [Arm normalization admission](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/arm_vulkan_ml/compiler/special_ops.cc) ·
[Google Tensor composite dispatch](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/google_tensor/compiler/compiler_plugin.cc).

Sources: [MLDrift parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/model_builder.cc),
[IR conversion](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/convert/convert_rms_norm.cc),
[IR admission](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/support/support_rms_norm.cc),
[Qualcomm builder](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/qualcomm/core/builders/rms_norm_op_builder.cc),
[MediaTek dispatch](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/mediatek/compiler/create_model.cc),
[NVIDIA builder](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc),
[Samsung builder](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/samsung/compiler/builders/rms_norm_op_builder.cc).

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
When `sub_type` is absent, legacy dispatch selects LayerNorm if the axes key
is absent and GroupNorm if it is present. The GroupNorm parser accepts either
`[rank-1]` or the full sequence `[1,...,rank-1]` in that case. With explicit
subtype 1, LayerNorm reduces the final channel dimension at each location.
[MLDrift dispatch](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/model_builder.cc#L7720-L7739) ·
[GroupNorm axes check](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/model_builder.cc#L7014-L7033).

Google Tensor requires exactly three inputs (x, scale, bias) and one output.
A bias-free export falls back to its decomposition. Arm Vulkan ML requires
composite **version 1**, positive finite epsilon, and integer `sub_type` and
`num_groups`. Subtype 0 reduces spatial dimensions within each group and
requires G to divide C. Subtype 1 requires G=1 and reduces the final axis.
Provided scale and bias must be channel vectors. [Google Tensor admission](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/google_tensor/compiler/compiler_plugin.cc) ·
[Arm normalization lowering](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/arm_vulkan_ml/compiler/special_ops.cc).

MediaTek's current options-aware path admits `num_groups=1` and lowers to
the `MTKEXT_LAYER_NORMALIZATION` OEM operation, reading `epsilon` and
`channel_axis` (default last dimension). Its older name-only path still
recognizes only RMS/L2 norm.
This is a LayerNorm specialization, not evidence for arbitrary spatial
GroupNorm or G>1; match the reduction axes and decomposition.

Qualcomm has its own group/layer lowering and maps `num_groups==1` to its layer
builder. Check that the chosen reduction dimensions agree with the model and
decomposition before selecting that path. Attributes tolerated by one parser
do not change another backend's normalization axes.

Sources: [MLDrift dispatch and normalization checks](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/model_builder.cc),
[Qualcomm group builder](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/qualcomm/core/builders/group_norm_op_builder.cc),
[MediaTek admission](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/mediatek/compiler/legalizations/common_op_legalization.cc),
[MediaTek LayerNorm lowering](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/mediatek/compiler/legalizations/layer_norm_op_legalization.cc),
[Qualcomm composite mapping](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/qualcomm/compiler/qnn_compose_graph.cc).

### `odml.l2_norm`

Preserved composite with one floating input `x[...,D]` and one same-shaped
output. Source test decomposition uses float attributes `epsilon`, integer
`axis=-1`, and `y=x/sqrt(sum(x*x,axis=-1,keepdims=true)+epsilon)`.

Arm Vulkan ML requires composite **version 0**, one input, matching static
input/output shapes, and positive finite epsilon. It normalizes the last
axis. [Arm normalization lowering](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/arm_vulkan_ml/compiler/special_ops.cc).

Qualcomm's builder consumes epsilon and hardcodes the final axis. MediaTek
maps the composite directly to `NEURON_L2_NORMALIZATION` without forwarding
these composite attributes. Therefore arbitrary axis/epsilon settings are
**not** a cross-backend contract; verify target normalization semantics,
especially at zero and tiny norms. Use float32 unless a target-specific
quantized path is verified. Builtin `L2_NORMALIZATION` has its own contract and
is not automatically interchangeable with every epsilon placement.

Sources: [decomposition](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/test/testdata/l2_norm_composite.mlir),
[Qualcomm builder](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/qualcomm/core/builders/l2_norm_op_builder.cc),
[MediaTek dispatch](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/mediatek/compiler/create_model.cc),
[MediaTek attribute handling](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/mediatek/compiler/legalizations/common_op_legalization.cc).

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
with head-size entries. False uses `scale_k` with cache-size entries. BMM
reads the resolved producer scales. The cache parser uses explicit attributes
when present and can recover missing scales from symmetric output tensor
metadata. A standalone INT8 RHS needs the BMM's own `scale` attribute.

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
kernel's alignment and retain masking of any padded work. The executor must
fill live values on each call. See [shared CPU/GPU attention](#sharing-an-sdpa-graph-between-cpu-and-gpu)
for the alignment and masking rules.

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
Recognized attention chains can now use windowed-cache rewrites, and an
optional static context limit can restrict a prefix. Those compiler
optimizations do not make parameter element 1 or 2 a general NVIDIA runtime
bound for arbitrary `odml.runtime_bmm` nodes.

Sources: [MLDrift parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/runtime_batched_matmul_parser.cc),
[IR parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/ir/runtime_batched_matmul_parser.cc),
[YNNPACK lowering](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/ynnpack/dot.cc),
[live parameter reading](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/ynnpack/ynnpack_delegate.cc),
[NVIDIA implementation](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc),
[torch emitter](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/experimental/composites/runtime_batched_matmul.py).

### `odml.cache_update`

MLDrift implements a paired K/V cache writer; NVIDIA also has a composite
compiler implementation. The MLDrift contract accepts exactly 3 or 7 runtime
inputs and two outputs. `litert-converter` uses the seven-input form so a
builtin decomposition can express explicit state updates.

| Slot | Tensor | Contract |
| --- | --- | --- |
| input 0 | new K | Floating `[1,Bkv,T,H]` by default; `[1,Bkv,H,T]` when `k_update_ts_idx=3`. |
| input 1 | new V | Floating `[1,Bkv,T,H]` by default; `[1,Bkv,H,T]` when `v_update_ts_idx=3`. |
| input 2 | parameters | `INT32`; use `[1,1,1,7]` shared with runtime BMM. |
| inputs 3..6 | fallback state | In the torch seven-input carrier: K cache, V cache, K update indices, V update indices. Their shapes/layouts belong to the decomposition. MLDrift ignores these four operands. |
| output 0 | updated K cache | Floating, INT8, UINT8, or packed INT4 on MLDrift. |
| output 1 | updated V cache | Same storage class; different packed matrix layout from K. |

Required integer attributes are `kv_cache_batch_size`, `cache_size=S`, and
`head_size=H`, all positive. Emit positive finite float `scale_k` and
`scale_v` matching the quantized cache tensors. The writer uses the
attributes when present and can recover per-tensor scales from metadata. Boolean `is_ring_buffer` defaults false.

Layout attributes are integer `k_cache_ts_idx=2`, `v_cache_ts_idx=3`,
`k_update_ts_idx=2`, and `v_update_ts_idx=2` by default. Cache axes are fixed
to 2/3 by MLDrift admission; each update may independently use time axis 2
or 3. Emit them when update tensors are transposed and reproduce those layouts
in the decomposition. These are distinct from SDPA's `k_ts_idx`/`v_ts_idx`.

Linear mode writes token `p=parameters[0]+x` only while `p<S` and
`p<parameters[1]`. Ring mode instead processes `x<parameters[3]` and writes
`p=(parameters[0]+x)%S`. Use nonnegative offsets and an update length within
the supplied slice. A linear dynamic-update-slice fallback is not equivalent
to ring mode; its wraparound, masks, and attention positions must also match.

K is packed as external weights with output size S and input size H; V has
output size H and input size S. They cannot share one assumed physical layout.
Keep the serialized tensor dimensions logical. MLDrift writes INT8 as biased
bytes (+128) and INT4 as biased nibbles (+8), packing two head-channel values
per byte in distinct K/V layouts. INT4 requires `H % 8 == 0` and `S % 4 == 0`.
The cache writer also accepts UINT8 with zero point 128. MLDrift's dense
and fused SDPA readers take signed INT8/INT4 quantized caches with zero
point 0. The writer rejects UINT4 and INT2 cache outputs. For a portable
quantized export, supply one positive scale per tensor. INT4 requires a
resolvable scale for both caches.

For quantized caches, fresh K/V may be FLOAT16/FLOAT32 or per-tensor affine
INT8/UINT8/INT4 from an in-graph QUANTIZE, optionally followed by up to
four supported RESHAPE/TRANSPOSE operations. The writer checks rank,
token count, head extent, and cache dimensions against the attributes.
Fresh K/V must have batch 1 and either one head or one head per cache row.

The shader reads state from the cache destination; ignored fallback inputs do
not initialize or reset it on the native path. Arrange cache allocation,
initialization, aliasing, lifetime, and feedback with the executor. For a
single-cache update, use the corresponding builtin update operations.

NVIDIA also implements seven-input updates into INT8/INT4 graph-input caches
when the new slices have matching quantization and the two start tensors
are INT32 vectors of length four. Its opt-in general attention path admits
additional float or requantized updates and UINT8 caches with zero point 128.

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
prefill semantics. A separate updated-cache-reader specialization now handles
decode and global prefill: cache tensors are subgraph inputs, fresh K/V are
`[1,Nkv,T,H]`, update-index operands are `INT32 [4]`, and the updated caches
have internal readers. It uses those explicit indices. NVIDIA may store
declared float32 caches internally as float16 unless
`LITERT_NVIDIA_TENSORRT_CACHE_STORAGE=native`; this is backend-owned state,
not interchangeable with another runtime's public tensor buffer.

Sources: [MLDrift parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/add_values_to_cache_parser.cc),
[packing and ring-buffer kernel](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/add_values_to_cache_kernel.cc),
[NVIDIA implementation](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc),
[seven-input decomposition](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/experimental/composites/cache_update.py).

### `odml.rope`

MLDrift recognizes `STABLEHLO_COMPOSITE` and same-name `CUSTOM` encodings. The
torch helper emits a composite. The kernel supports two different signatures:

| Form | Inputs, in order | Outputs, in order | Meaning |
| --- | --- | --- | --- |
| Combined tensor | `x`, `positions` | `rotated_x` | Split the channel dimension into rotary pairs, rotate, and concatenate. |
| Split tensors | `left`, `right`, `positions` | `rotated_left`, `rotated_right` | Rotate corresponding left/right components together. These are **paired components**, not independently rotated full Q and K tensors. |

The GPU kernel uses MLDrift's `[B, height, width, channels]` interpretation:
`width` is the position axis. For attention this normally means `[B,N,T,H]`.
The parsers now require positions of rank 2, 3, or 4: `[Bp,T]`, `[Bp,T,1]`,
or `[Bp,1,T,1]`, with `Bp` either 1 or B and matching token length. A rank-1
`[T]` position tensor is rejected by GPU admission even though the torch helper
accepts it. Use INT32 positions with floating activations; IR admission excludes
INT64. Outputs have the corresponding input shapes and floating type.

The MLDrift IR parser can now materialize constant position tensors. ATS
includes a standalone RoPE generator. NVIDIA implements the two-input
FLOAT16 `[1,N,T,H]` form with T INT32 positions only in its matched quantized
projection and RMSNorm chain. Its parser requires even H up to 1024 and
`0 < proportion <= 1`. [IR position handling](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/ir/rope_parser.cc) ·
[RoPE ATS generator](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/test/generators/rope.h) ·
[NVIDIA projection matcher](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc#L2781).

The torch decomposition still selects row 0 from rank-2 positions. Native
batch broadcasting was fixed, but distinct positions per batch therefore need
a corrected matching decomposition. The helper's docstring also permits
`[B,T,N,H]`; that does not make the GPU kernel infer which axis holds tokens.

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

Sources: [emitter](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/experimental/composites/rope.py),
[parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/rope_parser.cc),
[kernel](https://github.com/google-ai-edge/ml-drift/blob/57dc62c9ee54396f242b122f8ce384248647fe14/ml_drift/common/kernels/rope.cc),
[IR shape/type admission](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/support/support_rotary_positional_embedding.cc),
[attribute defaults](https://github.com/google-ai-edge/ml-drift/blob/57dc62c9ee54396f242b122f8ce384248647fe14/ml_drift/common/operations.h).

### `odml.qkv_norm_rope`

Composite or same-name custom op recognized by MLDrift, with a full-QKV
composite lowering in NVIDIA's plugin. It fuses splitting, Q/K RMS
normalization, layout conversion, and RoPE. V can optionally receive
unweighted RMS normalization; it is not rotated.

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
respectively, with configurable `proportion`. Boolean `has_v_norm` defaults
false; when true, V is multiplied by `rsqrt(mean(V*V,-1)+epsilon)` without a
gamma operand. Q/K compute `x*rsqrt(mean(x*x,-1)+epsilon)*weight` before RoPE.
It does not add 1 to gamma; models using an offset scale must supply the
effective gamma themselves.

MLDrift also admits a **query-only** form: three inputs
`q[B,T,Nq*H]`, `position`, `q_weight[H]` and one output `[B,Nq,T,H]`.
Emit `num_kv_heads=0` and `has_v_norm=false`; this form is intended for layers
that reuse K/V from elsewhere. The GPU builder resolves head counts and head
dimension from output shapes and rejects conflicting positive attributes.
Arity acceptance is not full shape validation. Use the
emitter's floating tensor contract and verify batch handling, head packing,
rotary pairing, and positions against its decomposition. As with the standalone
torch RoPE helper, rank-2 positions use row 0 in the decomposition.

MLDrift's IR path now materializes constant Q/K RMSNorm weights as well as
runtime weight inputs. The QKV ATS generator covers both. [IR parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/ir/qkv_norm_rope_parser.cc) ·
[ATS registration](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/ats/register_qkv_norm_rope.cc).

NVIDIA accepts only four inputs/three outputs at batch 1, with INT32 positions
(one per token), constant FLOAT32 Q/K scales, matching static shapes, and
kernel-supported even head dimensions/rotary extents. It reads epsilon and
`has_v_norm`, and casts activations to bfloat16 for its fused plugin. Do not infer query-only
support or float32 execution from name recognition.

Sources: [emitter](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/experimental/composites/qkv_norm_rope.py),
[parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/ir/qkv_norm_rope_parser.cc),
[kernel](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/qkv_norm_rope_kernel.cc),
[NVIDIA admission/lowering](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc),
[NVIDIA kernel constraints](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/trtllm/qkv_norm_rope.cu).

### `odml.swiglu`

Composite or same-name custom op recognized by MLDrift. It performs the
activation/gating step, not either surrounding linear projection.

| Form | Inputs | Output |
| --- | --- | --- |
| Concatenated | floating `gate_up[...,2*D]`, ordered gate then up | floating `y[...,D]` |
| Separate | floating `gate[...,D]`, `up[...,D]` | floating `y[...,D]` |

The default is `y = (gate * sigmoid(gate)) * up`. Optional string `activation`
can select `"gelu_tanh"` for `gelu(gate, approximate="tanh") * up` in the
concatenated form (GeGLU); the GPU also accepts `"gelu_pytorch_tanh"` as an
alias. Missing or unknown values select SiLU, so validate the spelling instead
of relying on parser rejection. The two-input GPU kernel supports SiLU only.

NVIDIA lowers the one-input floating composite with explicit positive
`gate_size` and last dimension `2*gate_size`. Accepted activation strings are
`silu` (also the default), `gelu_tanh`, and `gelu_pytorch_tanh`. [NVIDIA SwiGLU admission](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc#L1170).

Optional integer `gate_size` selects the split point; a nonpositive/unset
kernel value uses half the input channels. The torch emitter always writes
`gate_size` and uses the one-input form, emitting `activation` for GeGLU.
For portable use, both halves have equal size and identical leading dimensions.
The GPU concatenated path splits at a four-channel slice boundary, so choose
`D` divisible by four; do not assume an arbitrary scalar split is implemented.
The parser counts 1 or 2 **runtime** inputs and exactly one output.

Sources: [emitter](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/experimental/composites/swiglu.py),
[parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/ir/glu_parser.cc),
[kernel](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/glu_kernel.cc).

### `odml.short_conv_step`

Composite or same-name custom op recognized by MLDrift. It now covers both
prefill and single-token decode with a causal depthwise convolution and
rolling state. MLDrift requires batch 1 and **channels-last** state and weights.

| Slot | Tensor | Current serialized contract |
| --- | --- | --- |
| input 0 | `in_proj_out` | Floating `[1,T,3*D]`, ordered `b,c,x` when gated; `[1,T,D]` when ungated. |
| input 1 | `conv_state` | Floating `[1,L-1,D]`, oldest to newest. |
| input 2 | `conv_weight` | Floating `[1,L,D]`; fold any source `[D,1,L]` or `[D,L]` weights before marking the composite boundary. |
| optional input 3 | bias or valid count | Floating bias `[D]`, or `INT32 [1] num_valid_tokens` when there is no bias. |
| optional input 4 | valid count | `INT32 [1] num_valid_tokens` when bias occupies slot 3. |
| output 0 | `y` | Floating `[1,T,D]`. |
| output 1 | `next_state` | Floating `[1,L-1,D]`. |

Emit integer `conv_L_cache=L`, with `2<=L<=8` on MLDrift (default 3). Boolean
`is_gated` defaults true; `use_silu` defaults false. `channels_last` is true
when omitted; explicitly false is rejected. Gated math forms `u=b*x`, applies
the causal depthwise convolution and optional bias, optionally applies SiLU,
then multiplies by c. Ungated math convolves the input directly, with no b/c
split. Qwen3.5 uses four taps, `is_gated=false`, and `use_silu=true`.

The optional count denotes a valid prefix in `[0,T]` (the GPU clamps it).
Output y is computed for every position; only state advancement is limited.
`next_state` is the `L-1` values ending at the last valid token in
`concat(state,u)`; count 0 preserves state. The torch decomposition uses
slices/multiply/add and a dynamic gather for this update, avoiding a runtime
convolution-filter operand that XNNPACK cannot delegate.

The GPU parser drops input 1 from the native operation and reads/writes
**output 1 in place**, as with `odml.cache_update`. The executor must initialize
and preserve that destination and arrange aliasing/feedback; passing an
independent input-state buffer alone does not initialize the GPU state. The
TFLite decomposition retains explicit input/output state semantics. Test both
paths over partial prefills and multiple calls.

NVIDIA implements a separate ungated composite form with state
`[1,D,L-1]`, constant FLOAT32 weights `[D,1,L]`, and 2–16 taps. It accepts
an optional one-element runtime INT32 valid length and requires FP32 state
eligible for in-place update. Slot 3 is reserved for the count. Set
`is_gated=false` and `use_silu` explicitly because the two backends have
different defaults. [NVIDIA short-convolution parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc#L2178) ·
[CUDA tap limits](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/trtllm/gated_delta.cu).

Sources: [emitter and decomposition](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/experimental/composites/short_conv.py),
[parser and layout checks](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/short_conv_step_parser.cc),
[attribute defaults](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/short_conv_step_parser.h),
[kernel](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/short_conv_step_kernel.cc).

### `odml.gated_delta_block`

MLDrift implements this composite as the Qwen3.5 block between short
convolution and the output projection. Its factories also accept the same-name
custom form, which has no decomposition. The torch emitter uses a composite
whose fallback contains builtins **and `gated_delta_update`**: CPU execution
still needs that custom registration.

Let Nk/Nv be Q/K and value head counts and Dk/Dv their dimensions.

| Slot | Tensor | Shape / type |
| --- | --- | --- |
| input 0 | `conv_out` | Floating `[1,T,2*Nk*Dk+Nv*Dv]`, concatenated Q, K, V. |
| input 1 | `zba` | Floating `[1,T,Nv*Dv+2*Nv]`, concatenated z, b, a projections. |
| input 2 | `a_log` | FLOAT32 `[Nv]`. |
| input 3 | `dt_bias` | FLOAT32 `[Nv]`. |
| input 4 | `norm_weight` | FLOAT32 `[Dv]`, effective RMSNorm gamma. |
| input 5 | recurrent state | FLOAT32 `[1,Nv,Dk,Dv]`. |
| optional input 6 | valid mask | `[1,T]` in the activation dtype; use 0/1 values. This is a token mask, not the recurrence op's `[B]` length operand. |
| output 0 | gated output | Floating `[1,T,Nv*Dv]`. |
| output 1 | new state | FLOAT32 `[1,Nv,Dk,Dv]`. |

Emit integer `mode` and floating `eps`/`norm_eps` (defaults 0 and `1e-6` /
`1e-6`). Head dimensions/counts are inferred from conv/state shapes. Native
GPU admission requires batch 1, `Dk=128`, Dv a multiple of 16 in `[16,128]`,
and positive Nk with `Nv % Nk == 0`. Torch defaults boundary activations to
float32; its optional lower-precision activation path still keeps state and
parameter vectors float32.

The block computes `beta=sigmoid(b)` and
`g=-exp(a_log)*softplus(a+dt_bias)`, L2-normalizes Q/K with eps inside the square
root, and scales Q by `1/sqrt(Dk)`. It performs the recurrence, then computes
`RMSNorm(recurrence_output,norm_eps)*norm_weight*gate(z)` per value head.
The optional string `output_gate_type` selects `silu` (default) or `sigmoid`.
Torch emits the key for a sigmoid gate. MLDrift also accepts `swish` for
SiLU and reads `gate_activation` if the canonical key is absent. Masked tokens
preserve state and produce zero output. The export adapter can derive its
validity mask from positions, keeping the first row and then rows with
strictly increasing positions. Reproduce the mask, stable softplus, and both
epsilons in the decomposition. [Torch input adaptation](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/core/exportable_module.py).

The native parser drops input state and updates output state in place; the
kernel requires FLOAT32 buffer storage. Supply initialized/aliased persistent
state through the executor, including reset between sessions. The composite
does not include the preceding short convolution or following output
projection.

Sources: [torch emitter and decomposition](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/model_ext/qwen3_5/gated_delta_rule.py),
[contract and defaults](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/gated_delta_update_parser.h),
[GPU admission](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/gated_delta_update_parser.cc),
[IR parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/ir/gated_delta_update_parser.cc),
[GPU implementation](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/gated_delta_update_kernel.cc).

### `odml.blocked_ffw` and `odml.block_ffw`

Both spellings dispatch to the MLDrift feed-forward parser in GraphFloat32
and IR. Use one floating `x[1,T,D]` and one output of identical FLOAT16 or
FLOAT32 type and shape. The remaining inputs are constant weights for four
stages, in this order. Let P be the number of edge partitions, G the number
of middle groups, C the per-group slice width, and F the middle output width.

| Inputs after x | Count | Weight shape |
| --- | --- | --- |
| U1 | P | `[G*C,D/P]` each |
| U2 | 1 | `[G,P*C,F]` |
| D1 | 1 | `[G,F,P*C]` |
| D2 | P | `[D/P,G*C]` each |

There are `3+2*P` total inputs. P must divide D. The parser requires D/P, C,
and F to be multiples of four. It also accepts the last two axes transposed
on middle weights and uses quantization metadata to resolve ambiguous square
matrices. U1 operates on slices of x. Its outputs are regrouped for U2, then
SiLU is applied before D1. The D2 outputs are concatenated to restore D.
The optional string `activation` accepts `silu` or `swish`.

Weights may be floating or INT8/INT4/INT2. Quantized rank-3 weights require
`BlockwiseQuantizationV2` and host data. Rank-2 edge weights also admit the
parser's affine axis-0 layout. Optional positive finite `u1_in_scale`,
`u1_out_scale` and corresponding `u2`, `d1`, `d2` keys describe static
symmetric activation quantization at stage boundaries. Dynamic
`*_quant_spec` attributes are rejected. For eligible single-token graphs,
the GPU can fuse the edge projections and their packing operations. [MLDrift parser and layouts](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/block_ffw_parser.cc) ·
[IR lowering](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/ir/block_ffw_parser.cc) ·
[GPU kernels](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/block_ffw_kernel.cc).

### `odml.factorized_mlp`

NVIDIA's carve tool outlines a fixed two-branch feed-forward graph into this
composite. It takes FLOAT16 `x[1,T,D]` and six constant centered INT2
weights. In input order, these are two U1 matrices `[U,D/2]`, U2 `[G,K,N]`,
D1 `[G,N,K]`, and two D2 matrices `[D/2,U]`. K is twice the per-group slice width U/G. The middle
matrices may have a leading unit dimension.
The single FLOAT16 output has x's shape. The carved graph splits x, applies
the two up stages, applies SiLU, then regroups the down stages and concatenates
the results.

Preserve blockwise weight metadata with positive finite FP32 scales and
FP32 zero points of **-0.5** per output (unit extent along the input axis).
The FlexBuffers map supplies six positive float scales for INT8 activations,
`scale_in`, `scale_up1`, `scale_up2`, `scale_act`, `scale_down1`, and `scale_out`.
Set `activation="silu"` or omit it. NVIDIA requires its quantized-product
path with 16-bit activations. MLDrift's blocked feed-forward parser uses
different stage attributes and weight metadata. [Carve pattern and decomposition](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/tools/carve_builtin_patterns.cc) ·
[NVIDIA parser and lowering](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc#L2873).

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
The 10-input form now admits `MultiAxisQuantization` describing per-output-row
and per-expert scales: axes `[0,1]`, `blocksize=0`, no zero-point tensor, and
the metadata's scale index equal to the corresponding explicit scale operand.
This does not admit arbitrary multi-axis block layouts.

E is inferred from weight axis 1 and A from the last index dimension; require
`0<A<=E`. No composite attributes are consumed. The gate uses approximate
GELU, then multiplies the up projection, applies the down projection, weights
by the route and output scale, and sums selected experts. Quantized execution
dynamically quantizes activations and the intermediate to int8 and dequantizes
int32 dot accumulations. Match fallback/reference tolerances to that arithmetic;
float metadata support alone does not establish identical low-bit numerics.

Sources: [YNNPACK implementation](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/ynnpack/moe.cc),
[MoE tests](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/ynnpack/moe_test.cc).

## Expert and other runtime custom ops

### `moe`

Plain `CUSTOM` routed expert block implemented by MLDrift and XNNPACK, with a
restricted NVIDIA compiler-plugin path. It is distinct from YNNPACK's
`odml.moe_experts` composite. Provide routing outside this op, with `0<A<=E`,
expert IDs in `[0,E)`, and the intended normalized route weights. There are no
expert biases. Torch now requests INT32 router indices and casts before
reshaping them to the custom-op layout.

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

For a Gemma-style model with tanh-approximate GELU, the computation is

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

**Serialize the model's gate activation.** Use `activation="silu"` for SiLU
experts on XNNPACK, MLDrift, and NVIDIA. Torch canonicalizes a model's
`swish` to `silu`. MLDrift and NVIDIA also accept `swish` directly. For a
tanh-approximate GELU gate, emit `gelu_tanh`. Plain `gelu` selects exact GELU
on XNNPACK and NVIDIA and the tanh approximation on MLDrift. The torch
binding obtains the active-expert count from the model configuration.

**Apply per-expert output scale exactly once.** The current torch binding moves
`router.per_expert_scale` into the last custom-op operand and disables router
folding, avoiding an external one-hot/matmul scale lookup. If input route
weights already include that scale, pass ones in the final operand instead.
This scale is independent of the projection quantization scales.

MLDrift supports `weight_type="fp32"`, `"int8"`, and `"int4"`. Quantized modes
require native INT8 or INT4 weight tensors respectively; an INT8 byte container
is not the GPU INT4 contract. Expert scales accept both one scale per
output/expert row and multiple input-axis blocks. For a projection of shape
`[O,E,1,I]` with G blocks:

```text
row = out_channel * E + expert
block_size = I / G
w_real[out_channel, expert, input_channel] =
    w_quantized[out_channel, expert, input_channel] *
    scales[row * G + input_channel // block_size]
```

Use positive finite floating scales. MLDrift accepts float32/float16 scale
operands and symmetric affine, BlockwiseQuantization, MultiAxisQuantization,
or BlockwiseQuantizationV2 metadata. For metadata referencing a zero-point
tensor, a negative index means zero; otherwise the referenced tensor must be
constant INT8 or INT32 containing only zeros. The explicit scale operands
supply the kernel scales; quantization metadata must agree and is not a
replacement.

XNNPACK also supports blockwise int8 scales, but requires float32 activations,
outputs, fp32-mode weights, and all scales. Weights/scales must be read-only
constants (`kTfLiteMmapRo`). It infers block counts from element counts; emit
the canonical shapes above and enforce exact divisibility in the converter
even though some CPU helpers tolerate a ragged final block.

XNNPACK additionally supports `weight_type="int4"` with the ten-input order.
Weights may be native INT4 tensors or INT8 containers holding half as many
packed bytes. Low nibble stores the earlier signed value, high nibble the next.
Use even input widths, with the same row/block scale order and no zero-point
subtraction. Use native INT4 tensors when sharing the model with MLDrift.
YNNPACK's separate composite support does not extend the custom `moe` ABI.

NVIDIA's current custom path requires exactly ten inputs, native constant INT4
weights, and input-axis blocks of 32 for every projection. D and F must be
positive multiples of 32. Projection scales must all be float32 or all float16;
per-expert output scales must be float32 with E elements. Activations, routes,
and output are floating; expert indices are INT32. It distinguishes exact
`"gelu"` from `"gelu_tanh"`. Its CUDA kernels impose additional shape/device
limits. The plugin requires `LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH` for
the weight store, 16-bit activations (BF16 by default), CUDA GEMV
pre-dequantization (the default), and CUDA GEMM enabled.

Torch offers `dynamic_wi8_moe_afp32`, `dynamic_wi4_moe_afp32`, and
`dynamic_wi4b32_moe_afp32` recipes for symmetric multi-axis (0,1)
quantization of custom expert weights. In a LiteRT-LM bundle, setting both
`externalize_weights` and `externalize_moe_weights` makes slots 3–8 of the
ten-input form eligible for the external weight section. Keep the per-expert
output scale at slot 9 inline. The externalizer documents XNNPACK as the
supported execution path for externalized experts. Keep weight and scale
buffers inline when targeting the MLDrift parsers. [Torch quantization recipes](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/core/quantization.py) ·
[Torch export options](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/core/exportable_module_config.py) ·
[LiteRT weight externalizer](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/tools/externalize_tflite_flatbuffer.py).

Sources: [torch options, math, and scale binding](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/layers/moe.py),
[legacy GPU parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/moe_experts_parser.cc),
[IR GPU parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/ir/moe_experts_parser.cc),
[GPU implementation](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/moe_experts_kernel.cc),
[XNNPACK admission and execution](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/xnnpack/moe_delegate_kernel.cc),
[XNNPACK scale layout](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/xnnpack/moe_block_scale.h),
[shared GPU MoE builder](https://github.com/google-ai-edge/ml-drift/blob/57dc62c9ee54396f242b122f8ce384248647fe14/ml_drift/common/gpu_model_builder_moe_util.cc),
[NVIDIA admission and lowering](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc),
[NVIDIA block and shape limits](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/trtllm/int4_block_gemv.cu).

### `gated_delta_update` and `custom_call.gated_delta_update`

Both names have an explicit TFLite CPU registration. MLDrift's custom factory
recognizes the **unprefixed** `gated_delta_update` name. These are `CUSTOM` ops;
they do not acquire a decomposition merely because their math is decomposable.

| Slot | Tensor | Shared CPU/GPU contract |
| --- | --- | --- |
| input 0 | `q` | FLOAT32 `[B,Nk,T,Dk]`. |
| input 1 | `k` | FLOAT32 `[B,Nk,T,Dk]`. |
| input 2 | `v` | FLOAT32 `[B,Nv,T,Dv]`. |
| input 3 | `beta` | FLOAT32 `[B,Nv,T]`, nonnegative update gate for valid tokens. |
| input 4 | `g` | FLOAT32 `[B,Nv,T]`, nonpositive log decay for valid tokens. |
| input 5 | `state` | FLOAT32 `[B,Nv,Dk,Dv]`. |
| optional input 6 | `valid_len` | INT32 `[B]`, number of valid prefix tokens per batch. |
| output 0 | `output` | FLOAT32 `[B,Nv,T,Dv]`. |
| output 1 | `new_state` | FLOAT32 `[B,Nv,Dk,Dv]`. |

Six or seven inputs and two outputs. Require `Nk>0` and `Nv % Nk == 0`.
Value head h uses Q/K head `h // (Nv/Nk)`; beta, g, state, and output remain
indexed by Nv. Do not repeat Q/K in the converter just to match value heads.

For each valid token/head:

```text
S = exp(g) * S
delta = beta * (v - k @ S)
S = S + outer(k, delta)
output = q @ S
```

With `valid_len`, both native GPU and recurrent CPU kernels clamp each length
to `[0,T]`, stop state updates there, and zero the remaining output suffix.
Length 0 preserves initial state and returns zero output. The CPU wrappers
also accept FLOAT32 counts, but emit INT32 `[B]` for the shared contract; the
torch emitter now enforces that one-dimensional batch-sized shape.

**The old sentinel convention is no longer portable.** The CPU implementation
still substitutes decay 1 for `g>0` and beta 0 for `beta<=0`. The current GPU
shaders and torch eager recurrence use raw `exp(g)` and beta for processed
tokens. Thus positive-g/negative-beta sentinels from older exports can change
GPU state. Use `valid_len` for right padding, with normal gate values in the
active prefix. For masked holes inside that prefix, neutralize g/beta and Q
explicitly when the model requires state preservation and zero output.

Optional FlexBuffers integer `mode` defaults to 0; emit string
`state_dtype="float32"` as the torch exporter does and actually keep the state
float32. Both the TFLite wrapper and LiteRT C++ custom-kernel wrapper currently
call the recurrent implementation even for `mode=1`; the existence of a
chunked helper is not evidence that this option dispatches to it. MLDrift has
its own scheduling and admits Dk/Dv that are powers of two, at least 16.
CPU shape/type admission remains incomplete, so validate the full contract.

Normalize/scale Q and K outside the recurrence when required by the model.
CPU state is an explicit input/output pair. The GPU parser now drops input 5
and reads/writes output 1 in place. The executor must initialize and retain
that buffer; an unrelated input-state tensor is insufficient. The separate
[`odml.gated_delta_block`](#odmlgated_delta_block) wraps normalization/gating
around this recurrence, with a different operand list.

NVIDIA implements both custom spellings at batch 1. Its plugin accepts
Dk in {64, 128, 256}, Dv a positive multiple of 32, and an optional one-element
runtime INT32 valid length. The FP32 state must be a single-use subgraph input
and the updated state must have no internal consumers. NVIDIA can fuse the
surrounding gated-delta graph, including a sigmoid output gate, when its
pattern matches. [NVIDIA admission and fusion](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc#L2104) ·
[CUDA recurrence](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/vendors/nvidia/trtllm/gated_delta.cu).

Sources: [CPU registrations](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/experimental/custom_ops/gated_delta_net/gated_delta_update_tflite_op.cc),
[numerical implementation](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/experimental/custom_ops/gated_delta_net/gated_delta_update_impl.cc),
[LiteRT custom-kernel API](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/experimental/custom_ops/gated_delta_net/gated_delta_update_litert_custom_op.cc),
[GPU admission](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/gated_delta_update_parser.cc),
[GPU head mapping and padding](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/gated_delta_update_kernel.cc),
[torch Qwen3.5 emitter](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/model_ext/qwen3_5/gated_delta_rule.py),
[torch export flags](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/model_ext/qwen3_5/patch.py).

### `gdn_tril_inv` and `custom_call.gdn_tril_inv`

`CUSTOM`, explicitly registered by the same gated-delta-net TFLite registerer.
One `FLOAT32` input `A[...,C,C]`, one `FLOAT32` output of the same shape; rank
must be at least two and the last two dimensions equal. Computes
`(I + A)^-1` by forward substitution for a **strictly lower-triangular** A.
This is not a general matrix inverse. No custom options are consumed. No
same-name MLDrift parser is registered in the checked factory.

Sources: [registration and shape checks](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/experimental/custom_ops/gated_delta_net/gated_delta_update_tflite_op.cc),
[forward substitution](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/experimental/custom_ops/gated_delta_net/gated_delta_update_impl.cc).

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

Source: [legacy normalization parsers](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/model_builder.cc).

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

Sources: [parsers](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/model_builder.cc),
[IR names](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/ir_model_builder.cc),
[absolute-position kernel](https://github.com/google-ai-edge/ml-drift/blob/57dc62c9ee54396f242b122f8ce384248647fe14/ml_drift/common/kernels/positional_embedding.cc),
[rotary kernels](https://github.com/google-ai-edge/ml-drift/blob/57dc62c9ee54396f242b122f8ce384248647fe14/ml_drift/common/kernels/rope.cc).

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

Sources: [legacy parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/model_builder.cc),
[IR dispatch](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/ir_model_builder.cc).

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

Sources: [GPU parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/model_builder.cc),
[XNNPACK visitor](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/xnnpack/xnnpack_delegate.cc).

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

Sources: [CPU pooling](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/perception/max_pool_with_argmax.cc),
[CPU unpooling](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/perception/max_unpooling_2d.cc),
[legacy GPU contracts](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/model_builder.cc),
[IR unpooling](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/convert/convert_unpooling2d.cc),
[XNNPACK contracts](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/xnnpack/xnnpack_delegate.cc).

### `Resampler` and `DenseImageWarp`

| Name | Ordered inputs → output | Semantics / support |
| --- | --- | --- |
| `Resampler` | floating image `[B,H,W,C]`, warp `[B,Ho,Wo,2]` → sampled image `[B,Ho,Wo,C]` | GPU/MLDrift custom op. Coordinates contain `(x,y)` **absolute sampling locations**. Bilinear sampling with zero contributions outside the image. No options. MLDrift IR admits float32/float16 and requires both inputs to be nonconstant. |
| `DenseImageWarp` | `FLOAT32 image[B,H,W,C]`, `FLOAT32 flow[B,H,W,2]` → same-shaped float32 image | Explicit perception CPU kernel, no options. Flow contains `(dy,dx)` **displacements**: sample at `(y-dy,x-dx)`, with coordinates clamped to image borders. H and W must be at least 2. |

These names differ in coordinate ordering, absolute-versus-relative coordinates,
and boundary behavior. They cannot be renamed into each other.

Sources: [resampler conversion](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/convert/convert_resampler.cc),
[resampler admission](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/support/support_resampler.cc),
[GPU sampling](https://github.com/google-ai-edge/ml-drift/blob/57dc62c9ee54396f242b122f8ce384248647fe14/ml_drift/common/kernels/resampler.cc),
[CPU warp](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/perception/dense_image_warp.cc).

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
| output 3 | `num_detections` | `FLOAT32 [1]` counting selected anchors in fast NMS or selected box/class rows in regular NMS. |

Here `M=max_detections*max_classes_per_detection`; execution supports batch 1.
The required FlexBuffers keys are integer `max_detections`,
`max_classes_per_detection`, `num_classes`, and float `nms_score_threshold`,
`nms_iou_threshold`, `y_scale`, `x_scale`, `h_scale`, `w_scale`.
Optional integer `detections_per_class` defaults to 100; boolean
`use_regular_nms` defaults to false. Regular NMS and fast NMS select different
class/box suppression algorithms. Supply a positive IoU threshold ≤1 and
nonzero box scales. Preserve the source model's background-class convention
and quantization metadata; this op does not perform classifier calibration.

In fast NMS, each selected anchor occupies a group of
`max_classes_per_detection` rows. The kernel writes the top
`min(max_classes_per_detection, num_classes)` classes into that group and
increments `num_detections` once. For example, two selected anchors with
three classes emitted per anchor produce six rows and `num_detections=2`.
When reading the outputs, index each class row as
`anchor_ordinal * max_classes_per_detection + class_rank`.
[Fast NMS output loop](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/detection_postprocess.cc#L789-L832) ·
[Regular NMS output count](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/detection_postprocess.cc#L707-L739).

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

Sources: [spectrogram](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/audio_spectrogram.cc),
[window/FFT implementation](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/internal/spectrogram.cc),
[MFCC](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/mfcc.cc).

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

Source: [kernel](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/numeric_verify.cc).

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

Source: [implementation](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/hadamard_rotation.cc).

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

Source: [parser kernel and registerer](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/parse_example/parse_example.cc).

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

Sources: [factory declarations](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/custom_ops_register.h),
[atan2](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/atan2_custom.cc),
[sign](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/sign_custom.cc),
[normal RNG](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/random_standard_normal_custom.cc),
[uniform RNG](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/random_uniform_custom.cc),
[categorical RNG](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/multinomial.cc),
[roll](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/roll.cc),
[IRFFT](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/irfft2d.cc),
[3-D pooling](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/pooling3d.cc),
[LUT](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/table.cc),
[broadcast gradient axes](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/kernels/gradient/bcast_grad_args.cc).

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

Sources: [partition processing](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/compiler/plugin/compiler_plugin.cc),
[compiler integration](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/COMPILER_PLUGIN.md),
[dispatch name and metadata](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/core/build_stamp.h),
[Flex integration](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/flex/delegate.cc).

## Active cache length and attention work

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
| MLDrift GPU | Q, K, V, optional mask; no active-length operand. The new fused prefill kernel still follows the supplied K/V extent. | Supports an explicit runtime parameter tensor; element 2 bounds the main matmul/softmax reductions and fused kernels. Some unfused GPU stages still process full capacity; see below. |
| XNNPACK CPU | No active-length operand consumed; work follows the supplied K/V extent. | No handler in the checked implementation. |
| YNNPACK CPU | Supports an optional runtime parameter operand and slices K/V to the active prefix. | Supports the same active-prefix mechanism. |

XNNPACK needs the transposed composite's executable decomposition or a future
native implementation. On YNNPACK, both names can honor active length, so this
particular advantage does not distinguish them; choose the intended model
layout and document the available execution path.

In YNNPACK, slot 3 holds INT32/INT64 parameters when the mask is absent.
With a mask in slot 3, the parameters move to slot 4. YNNPACK reads length
from element 1 (element 0 for a one-element tensor). Positive lengths are
capped at capacity. Nonpositive lengths select the full capacity. MLDrift
and XNNPACK do not consume this extension on standard SDPA.

For both SDPA names, YNNPACK slices K before the first matmul, computes
softmax over the shortened sequence dimension, and slices V before the
second matmul. When runtime parameters are present, YNNPACK sets the active
shapes and requests a runtime reshape on each invocation.

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

Sources: [MLDrift standard parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/tflite/model_builder.cc),
[MLDrift transposed kernels](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.cc),
[XNNPACK attention visitor](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/xnnpack/xnnpack_delegate.cc),
[YNNPACK input decoding and slicing](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/ynnpack/attention.cc),
[YNNPACK live length handling](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/ynnpack/ynnpack_delegate.cc),
[CPU backend selection](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/c/options/litert_cpu_options.h),
[YNNPACK accelerator setup](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/runtime/accelerators/ynnpack/ynnpack_accelerator.cc),
[torch fallback matmul](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/experimental/composites/runtime_batched_matmul.py).

### Sharing an SDPA graph between CPU and GPU

Use the following intended converter contract for new shared models. Native
consumers currently include YNNPACK CPU and MLDrift GPU; XNNPACK can execute
supported builtins in the decomposition but has no transposed-SDPA handler.
Backend implementation gaps do not require a different semantic contract.

| Concern | Intended serialized representation | Current backend status |
| --- | --- | --- |
| Layout/output | Q/output `[B,Nq,T,H]`, K `[B,Nkv,S,H]`, V `[B,Nkv,H,S]`; flatten decode output with a following reshape. | Torch now marks the rank-4 output before flattening. MLDrift can fuse the following decode reshape internally; YNNPACK requires rank 4 at the composite boundary. |
| GQA | Require `Nq % Nkv == 0`; query head h uses KV head `h // (Nq/Nkv)`. Keep heads separate from tokens at the boundary. | Torch packs query heads inside the fallback decomposition to avoid copying K/V. MLDrift and YNNPACK use folding or broadcasting internally; shared masks can remain folded, while per-query-head masks need additional layout handling. |
| Scale | Pre-scale Q once; omit `scale` or use floating `1.0`. | MLDrift expects pre-scaled Q; YNNPACK's transposed default is 1. |
| Runtime bounds | Supply live `INT32 [1,1,1,7]` parameters, an accurate query offset, and a mask covering the semantic active prefix. | YNNPACK reads element 1; MLDrift reads element 2 and some kernels require aligned work bounds. See the alignment caveat below. |
| Logit cap | For shared capped attention, emit both `softcap=C` and `logit_cap=C` with the same positive float and apply the cap once in the decomposition. Omit both keys for uncapped attention. | MLDrift reads `softcap` first and accepts `logit_cap` as an alias. YNNPACK reads only `logit_cap`. |
| Causality | Emit boolean `is_causal`, true only when runtime bounds/positions fully reproduce the causal mask. Keep the explicit mask in the serialized graph/decomposition. | MLDrift may prune a BOOL mask on eligible float-cache Metal/OpenCL/WebGPU paths. YNNPACK follows the supplied mask and does not read `is_causal`. |
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

Sources: [torch composite boundary and fallback](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/experimental/composites/sdpa.py),
[YNNPACK attention](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/ynnpack/attention.cc),
[GPU parser](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/ir/sdpa_transposed_parser.cc),
[GPU kernels](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.cc),
[decode reshape fusion](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/fuse_sdpa_transposed_reshape.cc),
[aligned ATS inputs](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/test/generators/sdpa_transposed.h),
[GPU bound alignment](https://github.com/google-ai-edge/ml-drift/blob/57dc62c9ee54396f242b122f8ce384248647fe14/ml_drift/common/task/gpu_operation.h).

### Correctness status and backend follow-up

The same SDPA export can select different math and storage paths across delegates.

| Area | Status at the checked revisions |
| --- | --- |
| Fused decode masks and softcap | The fast loop now applies masks and softcap as well as the tail. The later unified Metal/OpenCL implementation retains explicit mask handling. |
| Fused prefill additive bias | Additive bias is now multiplied by `1/ln(2)` to match the base-2 softmax logits. |
| Fused prefill causality | `is_causal` defaults false; causal bounds are conditional on the attribute. A noncausal mask is no longer overridden merely because the fused prefill kernel was selected. |
| GPU GQA fallback | Query heads are grouped for matmuls and restored for mask application/output, matching the contiguous head mapping. |
| GPU BOOL mask cost | `SelectBoolMask` is now linkable into the QK epilogue. Shared masks can apply directly to folded GQA logits, avoiding the old mandatory standalone `SelectV2` pass. |
| GPU INT8/INT4 K/V | Both use packed external weights and the multi-operation graph. Flash kernels and causal-mask pruning require floating caches. |
| YNNPACK cap attribute | YNNPACK can claim a `softcap`-only node and execute uncapped attention. Its lowering reads `logit_cap`. |
| YNNPACK BOOL masks | Still converts false to an additive `-10000` penalty. Torch/GPU decomposition semantics replace the logit with `-10000`; extreme logits distinguish these functions. |

Useful converter/backend regression cases are: a masked prefix spanning the
fused decode fast loop; finite additive prefill biases; noncausal prefill with
future keys allowed; `softcap=1.0` with saturating logits; BOOL replacement with
an excluded large positive logit; GQA prefill with different masks per query;
and exact/unaligned active lengths with partial chunks. Include all-masked
rows explicitly if the model permits them: fused kernels use large-negative
sentinels and special handling that should not be assumed identical to a
finite-fill decomposition in every numerical corner case.

Sources: [GPU fixes](https://github.com/google-ai-edge/LiteRT/commit/77a7a2ce9bc88d003273d7bd0e84cf5f4f8123ff),
[GPU GQA fix](https://github.com/google-ai-edge/LiteRT/commit/6d59736316f2e63d215904c152059f8d81aaec95),
[current fused kernels](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.cc),
[current CPU attributes and masking](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/tflite/delegates/ynnpack/attention.cc).

### What the current ATS coverage establishes

The `SdpaTransposed` registration still requests 16 grid cases for each
float32/float16, mask, softcap, and parameter combination (256 before filters
or generation failures). The generator covers the following cases.

| Area | Current coverage and limits |
| --- | --- |
| Layout and GQA | Q is now `[B,Nq,T,H]`; query length is independent of head count. The decomposition groups contiguous query heads correctly. |
| Masks | Actual BOOL masks include holes inside the active prefix; floating masks include finite negative biases and excluded keys. The BOOL reference uses selection rather than an additive penalty. |
| Active cache | Parameters include a query offset and matching lengths in elements 1/2. Where possible, length is below capacity and rounded to a multiple of 32. Unaligned bounds and live changes across calls still need separate tests. |
| Softcap | Cap is now 1.0, and both `softcap` and `logit_cap` are emitted. This improves numerical sensitivity but does not test a `softcap`-only contract on YNNPACK. |
| Packed-cache paths | The standalone generator still uses ordinary K/V. A new `CacheUpdateSdpaTransposed` generator connects `odml.cache_update` to SDPA with `from_cache_update=true`, requesting 128 cases across float type, mask, and cap combinations before filters/failures. It uses offset 0 and does not emit `is_causal`, so it does not cover causal-mask pruning or stateful cache reuse. |
| CPU suite | `YNNPACK_DONT_REGISTER` no longer excludes `SdpaTransposed` or the f16/i4 `RuntimeBmm` cases. Other filters still apply. XNNPACK continues to exclude dense transposed SDPA. |

The device `gpu_ats` suite includes `SdpaTransposed`. The suite configuration
describes which cases can run. Multi-invocation cache reuse, unaligned active
lengths, quantized caches, and causal-mask pruning need separate checks on the
selected native path.

Sources: [ATS registration](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/ats/register_sdpa_transposed.cc),
[grid iteration count](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/ats/register_composite_ops.cc),
[current generator/reference](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/test/generators/sdpa_transposed.h),
[compound registration](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/ats/register_cache_update_sdpa_transposed.cc),
[cache-update-to-attention generator](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/test/generators/cache_update_sdpa_transposed.h),
[suite filters](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/ats/BUILD),
[CPU options](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/litert/ats/configure.cc).

### MLDrift work still tied to cache capacity

`BuildSdpaTransposedGpuGraph` selects fused kernels on supported Metal,
OpenCL, or WebGPU devices, or a multi-operation native GPU graph. These are
delegate implementations, distinct from the TFLite decomposition.

The previous standalone BOOL selection limitation has been addressed:
`SelectBoolMask` is a linkable elementwise operation. A mask shared across
query heads can be applied directly to GQA-folded logits and fused into QK.
Per-query-head masks still require unfolding the logits, which can prevent
that fusion. Inspect the generated graph before counting masking, softcap, or
additive bias as separate passes.

The following generic-path findings still apply when
`from_cache_update=false` and the specified kernel is selected. Supplying the
active bound does not make every stage proportional to it.

| Stage | When it occurs | Work that still depends on capacity S |
| --- | --- | --- |
| K/V transpose and packing | Generic `BatchedMatMul` path (`from_cache_update=false`) when the direct fully connected implementation is not selected. | The RHS transpose and weight conversion receive no runtime bound. They process full-capacity K/V even though the following matmul receives the bound. |
| Final softmax normalization | Generic path when `Softmax` selects its two-pass implementation: score rows per GPU compute unit >= 256. | The reduction receives the bound, but `SoftmaxElementwise` / `CreateSoftmaxFinal` computes `exp(logit-max)/sum` over the full score tensor without it. |
| Redundant softmax reductions | Generic path when `SelectSoftmax` chooses `Softmax1x1` and the full-capacity output requires multiple workgroups. | Dispatch uses the full number of output slices. Every workgroup performs the active-prefix reduction before checking whether its output slice is active, so groups entirely beyond the bound still repeat that reduction. They do not scan inactive keys, but their redundant work depends on S. |

The two-pass final normalization cannot simply link into QK: logits feed both
the reduction and normalization, while the linker requires a sole consumer
connected through input 0. The packed-cache path instead uses bound-aware
external weights and `SoftmaxReduce`, with normalization fused into the value
matmul. Fused prefill/decode kernels avoid these separate passes entirely.

Large decomposed GQA workloads can now be split into whole-query-head chunks,
reusing score storage between chunks. This reduces peak intermediate memory;
it does not shorten the key dimension. The implementation retains a single
chunk for per-query-head masks. Intermediate sizes and some dispatch grids
therefore still depend on capacity even when the reductions honor active
length. Selection requirements are listed under
[`odml.sdpa_transposed`](#odmlsdpa_transposed).

Intermediate allocations and some dispatch grids also retain capacity-sized
shapes; matmul bounds can round up to kernel alignment. Active-prefix
arithmetic therefore does not imply exact-L allocation or zero overhead for
inactive workgroups. Mask construction outside the composite likewise does
not acquire a bound merely because SDPA consumes one. These are source-audit
findings; their latency impact requires profiling the selected path.

Sources: [SDPA graph construction](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.cc),
[fused-kernel and score-chunk selection](https://github.com/google-ai-edge/LiteRT/blob/9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d/ml_drift_delegate/delegate/composite/sdpa_transposed_support.cc),
[generic matmul preparation](https://github.com/google-ai-edge/ml-drift/blob/57dc62c9ee54396f242b122f8ce384248647fe14/ml_drift/common/gpu_model_builder.cc),
[softmax path selection](https://github.com/google-ai-edge/ml-drift/blob/57dc62c9ee54396f242b122f8ce384248647fe14/ml_drift/common/gpu_model_builder.cc),
[unbounded final normalization](https://github.com/google-ai-edge/ml-drift/blob/57dc62c9ee54396f242b122f8ce384248647fe14/ml_drift/common/kernels/softmax.cc),
[Softmax1x1 reduction and dispatch](https://github.com/google-ai-edge/ml-drift/blob/57dc62c9ee54396f242b122f8ce384248647fe14/ml_drift/common/kernels/softmax1x1.cc),
[softmax selector](https://github.com/google-ai-edge/ml-drift/blob/57dc62c9ee54396f242b122f8ce384248647fe14/ml_drift/common/selectors/simple_selectors.cc),
[node linking](https://github.com/google-ai-edge/ml-drift/blob/57dc62c9ee54396f242b122f8ce384248647fe14/ml_drift/common/merge_nodes.cc),
[packed-cache weights conversion](https://github.com/google-ai-edge/ml-drift/blob/57dc62c9ee54396f242b122f8ce384248647fe14/ml_drift/common/gpu_model_builder.cc),
[runtime-bound alignment](https://github.com/google-ai-edge/ml-drift/blob/57dc62c9ee54396f242b122f8ce384248647fe14/ml_drift/common/task/gpu_operation.h).

## Builtin fallbacks affected by recent torch changes

Recent frontend changes also improve decompositions without adding runtime
custom names. These are useful when choosing builtin fallbacks:

| Source operation | Current lowering pattern |
| --- | --- |
| Static contiguous `aten.as_strided` | Flatten, optional static slice for storage offset, reshape. Check row-major strides while ignoring size-1 axes; other views retain the gather path. |
| Static slice with step > 1 | `STRIDED_SLICE`; the entire shape and bounds must be concrete. Avoid an unnecessary `GATHER_ND` for a simple strided slice. |
| PReLU | `relu(x) - weight*relu(-x)` avoids the old comparison/SELECT decomposition. Preserve the source's numerical behavior when adopting the rewrite. |
| Pixel shuffle/unshuffle | Interleave one spatial axis at a time using reshape/transpose with intermediates of rank at most 4. This does not require either legacy pixel-shuffle custom spelling. |
| Noncontiguous rank ≥3 linear input | Flatten batch dimensions into a matrix, use `mm`/`addmm`, and restore the output shape. This avoids broadcasting the weight into a high-rank batched matmul. |
| Static expand/repeat | The FX pass uses `CONCATENATION` for eligible copies along singleton axes through rank 5. Constants and expansions already folded by consumers are left to the converter. |
| Index selection / embedding | The JAX lowering uses `take(mode="clip")` for in-range indices and avoids fill-mode bounds checks. |
| Full-length split-cache attention | `attention_kv_block_size` can split a divisible full-length cache into blocks and combine the partial softmax results. This is an attention decomposition with a separate export option. |
| Short convolution / Qwen3.5 state | The shared channels-last fallback uses tap-wise multiply/add and INT32 `GATHER` along time axis 1 to select state ending at the valid prefix. It avoids a nonconstant depthwise-convolution filter inside the decomposition. |

Sources: [contiguous-view lowering](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/backend/lowerings/_jax_lowerings/torchax_ops.py),
[static slices](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/backend/lowerings/_jax_lowerings/lowerings.py),
[PReLU, shuffle, and linear decompositions](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/backend/lowerings/_decomp_registry.py),
[Static broadcast copies](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/_convert/fx_passes/lower_broadcast_to_concat_pass.py),
[Blocked split-cache attention](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/core/split_cache/attention.py),
[short-convolution fallback](https://github.com/google-ai-edge/litert-torch/blob/7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f/litert_torch/generative/export_hf/experimental/composites/short_conv.py).

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

For RMSNorm, use `x=[[1,-1]]`, `final_scale=[2,3]`, `w=[4,5]`, and
`epsilon=1e-6`. Compare the source layer, the decomposition with delegation
disabled, and a confirmed native RMSNorm path. All three should return
approximately `[[8,-15]]`. A native result near `[[4,-5]]` identifies the
missing `final_scale` described under [`odml.rms_norm`](#odmlrms_norm).

For QSA, compare selected block IDs as a set for attention and check the
ordering separately if the indexer output is observable. Exercise a chunk
that completes a previous partial block, a row with no visible keys, and
a step with fewer complete blocks than `block_topk`. For INT4 cache updates,
check readback across both nibbles, including an unaligned token offset and
a partially filled ring.

Record backend work separately from the intended export contract. For an
execution-ready artifact, a custom node needs its registration/delegate; a
composite can use its equivalent builtins when native support is unavailable.
Do not silently change attributes or model semantics to satisfy an old parser.
An invented name or converter-only allowlist does not make an op executable.
Native support on one backend does not establish that another implements the
same contract; keep unimplemented targets visible as backend follow-up work.

## Maintaining the inventory

To reproduce this audit, use clones of the three repositories in
[Scope and source revisions](#scope-and-source-revisions). Run the commands
from the parent of the three clones, or set `LITERT_DIR`, `LITERT_TORCH_DIR`,
and `ML_DRIFT_DIR` to their paths. The searches read committed Git trees
regardless of the checked-out branches or local edits.

```bash
litert_dir=${LITERT_DIR:-./LiteRT}
torch_dir=${LITERT_TORCH_DIR:-./litert-torch}
drift_dir=${ML_DRIFT_DIR:-./ml-drift}

litert_rev=9e22aadeecc07cdf48e47b8ae9bc008afccf8c8d
torch_rev=7aef351afb7aece516bc0a9fbaf9b098fb0f0b0f
drift_rev=57dc62c9ee54396f242b122f8ce384248647fe14

# Locate runtime registrations, delegate dispatch, and torch emitters.
git -C "$litert_dir" grep -n -E 'AddCustom|custom_name|custom_code|odml\.' "$litert_rev" -- \
  litert/runtime litert/experimental litert/vendors ml_drift_delegate \
  tflite/delegates tflite/kernels tflite/core/kernels tflite/experimental
git -C "$torch_dir" grep -n -E 'odml\.|custom_call\.' "$torch_rev" -- litert_torch

# Trace an exact name across all three repositories.
op_name=odml.runtime_bmm
git -C "$litert_dir" grep -l -F "$op_name" "$litert_rev" --
git -C "$torch_dir" grep -l -F "$op_name" "$torch_rev" --
git -C "$drift_dir" grep -l -F "$op_name" "$drift_rev" --
```

For `odml.runtime_bmm`, the last search returns no matches (Git exit status
1). Follow the parser in LiteRT's MLDrift integration to the lower-level
kernel in `ml-drift`. Apply the same check before removing any name. Emitters
and tests establish the intended math. Trace each runtime claim to the code
that parses and executes the operation.

For a new audit, fetch `main` from each URL in the scope table using
`git -C "$checkout" fetch "$url" main`, then record
`git -C "$checkout" rev-parse FETCH_HEAD`. Update the table, the three hashes
above, and the source links together. Report device execution separately
from findings based on source inspection.
