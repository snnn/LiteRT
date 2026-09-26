# FP16 Runtime Rewrite and LiteRT Integration

This document records how XNNPACK's runtime FP16 rewrite works, how the
TensorFlow Lite/LiteRT XNNPACK delegate enables it, and how FP16 weight-only
models produced by AI Edge Quantizer relate to the feature. The findings are
based on the XNNPACK, LiteRT, LiteRT-Torch, AI Edge Quantizer, and ML Drift
checkouts inspected in July 2026.

## Summary

XNNPACK can rewrite an FP32-oriented runtime subgraph to FP16 during runtime
creation. Two runtime flags control the behavior:

```c
XNN_FLAG_HINT_FP16_INFERENCE
XNN_FLAG_FORCE_FP16_INFERENCE
```

The hint attempts the rewrite only when XNNPACK detects native FP16 arithmetic.
If the hardware or subgraph is incompatible, execution remains in its original
precision. The force flag requires the rewrite to succeed; runtime creation
fails otherwise. On x86, forced FP16 can use emulation and may be substantially
slower than FP32.

The original TensorFlow Lite deployment workflow combines two independent
properties:

1. Static FP32 weights and biases are stored as FP16, with `DEQUANTIZE` nodes
   preserving the model's FP32 semantics.
2. TFLite model metadata declares that FP16 inference and FP16 accumulation are
   accuracy-safe:

   ```text
   reduced_precision_support = fp16accfp16
   ```

The LiteRT XNNPACK delegate recognizes this metadata and passes the hint to
XNNPACK. The XNNPACK rewrite then changes compatible floating activations and
operators to FP16 while retaining FP32 external inputs and outputs.

AI Edge Quantizer can create FP16 weight-only tensors and insert the required
TFLite `DEQUANTIZE` nodes. In the inspected checkout, however, it does not write
the reduced-precision metadata, and it leaves biases in FP32. The latter can
conflict with the LiteRT XNNPACK delegate's requirement that FP16 filters and
biases have matching types for convolution, depthwise convolution, and fully
connected operations.

## XNNPACK runtime flags

The public definitions are in `include/xnnpack.h`:

```c
#define XNN_FLAG_HINT_FP16_INFERENCE  0x00000002
#define XNN_FLAG_FORCE_FP16_INFERENCE 0x00000004
```

They are supplied in the `flags` argument to `xnn_create_runtime_v2`,
`xnn_create_runtime_v3`, or `xnn_create_runtime_v4`.

### Hint behavior

`XNN_FLAG_HINT_FP16_INFERENCE` causes XNNPACK to attempt the rewrite when
`xnn_is_f16_supported_natively` is true. In this checkout, native support is
recognized for:

- ARM/ARM64 NEON FP16 arithmetic, subject to build configuration.
- RISC-V vector FP16 arithmetic, subject to build configuration.

The hint is not a guarantee. A failed compatibility analysis leaves the
subgraph in its original representation.

### Force behavior

`XNN_FLAG_FORCE_FP16_INFERENCE` attempts the rewrite even when FP16 is emulated.
Runtime creation fails if:

- The hardware supports neither a native nor an emulated FP16 path.
- The XNNPACK subgraph contains an incompatible node or datatype combination.

The public header warns that x86 FP16 computation may be emulated at a
substantial performance cost.

### Hardware query

Applications can query native support with:

```c
bool native_fp16 = xnn_is_f16_native_supported();
```

This query does not guarantee that a particular subgraph can be rewritten.

## What the rewrite changes

The rewrite is performed during runtime creation, not on every invocation. Its
implementation is `xnn_subgraph_rewrite_for_fp16` in `src/subgraph.c`.

At a high level, it:

1. Verifies that every node in the XNNPACK subgraph is eligible.
2. Marks compatible FP32/PFP32 values for conversion.
3. Converts internal floating values and applicable static values to FP16.
4. Rewires operators to the FP16 values.
5. Inserts FP32-to-FP16 conversions for external FP32 inputs.
6. Inserts FP16-to-FP32 conversions for external FP32 outputs.

Consequently, the public model interface can remain FP32 while most internal
floating computation uses FP16.

Static convolution and fully connected weights have specialized handling. FP32
weights can be converted while weights are packed, and already-FP16 weights can
remain FP16. Quantized weights can also remain quantized in supported hybrid
paths.

## XNNPACK FP16 node allowlist

The current rewrite accepts these node types before applying additional
datatype-specific checks:

- Average pooling 2D
- Batch matrix multiplication
- Binary elementwise
- Concatenation
- Convert
- Convolution 2D
- Copy
- Deconvolution 2D
- Depth-to-space 2D
- Depthwise convolution 2D
- Even split
- Fully connected
- Global average pooling 2D
- Global sum pooling 2D
- Max pooling 2D
- RoPE
- Softmax
- Space-to-depth 2D
- Static constant pad
- Static mean and mean-squared reductions
- Static reduce-min and reduce-max
- Static reshape
- Static resize-bilinear 2D
- Static slice
- Static sum and sum-squared reductions
- Static transpose
- Unary elementwise
- LHS packing when an X16 packing configuration is available

The check is applied to each XNNPACK runtime subgraph created by the delegate,
not necessarily to every operator in the original TFLite model. Operators that
are not delegated can split a model into multiple delegated partitions.

## LiteRT/TFLite flag mapping

The TFLite XNNPACK delegate assembles XNNPACK runtime flags in
`tflite/delegates/xnnpack/xnnpack_delegate.cc` in the LiteRT checkout.

### Force mapping

LiteRT exposes a dedicated delegate flag:

```c
TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16
```

The delegate maps it to:

```c
XNN_FLAG_FORCE_FP16_INFERENCE
```

For benchmark and delegate-provider tools, the corresponding option is:

```text
--xnnpack_force_fp16=true
```

### Hint mapping

There is no dedicated `TFLITE_XNNPACK_DELEGATE_FLAG_HINT_FP16` constant. The
delegate normally enables the hint by reading the TFLite model's
`reduced_precision_support` metadata. It requires both FP16 inference and FP16
accumulation support before adding:

```c
XNN_FLAG_HINT_FP16_INFERENCE
```

Advanced callers can also pass raw XNNPACK runtime flags through
`TfLiteXNNPackDelegateOptions.runtime_flags`. The benchmark equivalent is:

```text
--xnnpack_runtime_flags=2
```

## Reduced-precision TFLite metadata

The metadata key is:

```text
reduced_precision_support
```

Recognized encodings include:

| Value | Meaning |
| --- | --- |
| `fp16accfp16` | FP16 inference with FP16 accumulation is acceptable. |
| `fp16accfp32` | FP16 storage/inference is acceptable, but accumulation should remain FP32. |
| `bf16accfp32` | BF16 inference with FP32 accumulation is acceptable. |
| `fp16bf16accfp32` | FP16 or BF16 inference with FP32 accumulation is acceptable. |

The LiteRT XNNPACK delegate's automatic FP16 hint requires the equivalent of
`fp16accfp16`. Merely having FP16 weights is insufficient to enable FP16
activation computation automatically.

In the TensorFlow converter workflow, metadata is generated with settings such
as:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec._experimental_supported_accumulation_type = tf.float16
```

The first target setting produces FP16 weight storage. The accumulation setting
declares that FP16 accumulation is acceptable and causes the corresponding
metadata to be emitted.

This metadata is a compatibility declaration, not calibration data. It should
only be attached after accuracy has been evaluated under actual FP16
computation and accumulation.

The historical TensorFlow announcement describing this flow is:

<https://blog.tensorflow.org/2023/11/half-precision-inference-doubles-on-device-inference-performance.html>

## TFLite FP16 weight representation and `DEQUANTIZE`

A conventional TFLite FP16 weight-only graph preserves FP32 model semantics:

```text
FP16 constant weight -> DEQUANTIZE -> FP32 Conv/FC input
```

The XNNPACK core rewrite does not require this `DEQUANTIZE` node. It operates on
an XNNPACK subgraph rather than directly on the TFLite graph.

The TFLite XNNPACK delegate recognizes a static FP16-to-FP32 `DEQUANTIZE`
feeding a convolution, depthwise convolution, or fully connected weight input.
The delegate:

1. Marks the dequantization as a static unpack node.
2. Elides it when constructing the XNNPACK subgraph.
3. Defines the original constant as an XNNPACK FP16 value.
4. Rewires the consuming XNNPACK operator to the FP16 value.

Thus the graph received by XNNPACK is effectively:

```text
FP32 activation + FP16 weight -> FP32 output
```

The runtime FP16 rewrite can then produce:

```text
FP16 activation + FP16 weight -> FP16 output
```

with conversions around external FP32 interfaces.

A fully FP32 TFLite model does not need pre-existing `DEQUANTIZE` nodes for the
force path. XNNPACK can convert or repack compatible FP32 static weights while
creating FP16 operators.

## Interaction with quantized-weight models

### Fully static INT8

A fully static INT8 model such as:

```text
INT8 activation x INT8 weight -> INT8 activation
```

does not benefit from this FP16 rewrite. There is no FP32 activation computation
to replace. With the hint, FP16 compatibility analysis fails and the INT8 path
continues unchanged. With force, runtime creation can fail.

For example, `mobilenet_v2_int8_channelwise.tflite` in
`litert-community/MobileNet-v2` is described as channelwise signed INT8 weights
with asymmetric signed INT8 activations. It is outside the intended scope of
the FP16 rewrite.

### Dynamic or hybrid quantized weights

Current XNNPACK supports hybrid paths in which weights remain quantized while
floating activation boundaries become FP16. A typical dynamic weight-only path
changes from:

```text
FP32 activation
  -> dynamic quantization to QD8
  -> QD8 x QC4/QC8 quantized weights
  -> FP32 output
```

to:

```text
FP16 activation
  -> dynamic quantization to QD8
  -> QD8 x QC4/QC8 quantized weights
  -> FP16 output
```

XNNPACK contains corresponding microkernel families such as
`qd8-f16-qc4w`, `qd8-f16-qc8w`, and `qd8-f16-qb4w`. Therefore, the current
feature is broader than the FP16-weight-only scenario described in the 2023
announcement.

The quantized weights are not converted to FP16. The rewrite changes eligible
floating activation and output values and selects compatible hybrid kernels.

## AI Edge Quantizer FP16 weight support

The inspected AI Edge Quantizer checkout implements FP16 weight-only conversion
through:

```python
AlgorithmName.FLOAT_CASTING
```

Its supported weight-bearing TFLite operators are:

- `FULLY_CONNECTED`
- `CONV_2D`
- `DEPTHWISE_CONV_2D`
- `CONV_2D_TRANSPOSE`
- `EMBEDDING_LOOKUP`

For each eligible static weight, it:

1. Reads the FP32 constant.
2. Casts it with `weight_content.astype(np.float16)`.
3. Stores it as a TFLite `FLOAT16` tensor.
4. Inserts a `DEQUANTIZE` node producing FP32 for the original consumer.

This operation requires no calibration data. It is a deterministic floating
cast and does not estimate activation ranges, scales, or zero points.

Representative data is still recommended for accuracy validation, especially
before declaring `fp16accfp16`, but that is evaluation rather than calibration.

### Operator-list comparison

The AI Edge Quantizer and XNNPACK lists answer different questions:

- The quantizer list identifies operators whose static weights it can store as
  FP16.
- The XNNPACK list identifies computation nodes that can execute with FP16
  floating values.

Four quantizer operators map directly to allowed XNNPACK FP16 nodes:

| AI Edge Quantizer | XNNPACK node |
| --- | --- |
| `FULLY_CONNECTED` | Fully connected |
| `CONV_2D` | Convolution 2D |
| `DEPTHWISE_CONV_2D` | Depthwise convolution 2D |
| `CONV_2D_TRANSPOSE` | Deconvolution 2D |

`EMBEDDING_LOOKUP` is the sole direct mismatch among the five quantizer target
operators. It is not handled by the inspected TFLite XNNPACK delegate, and the
XNNPACK FP16 rewrite allowlist has no corresponding embedding/gather node.

This mismatch does not imply that every other node is compatible. Weightless
operators such as activation functions, elementwise operations, pooling,
softmax, reshape, and transpose must independently be accepted by the delegate
and by the XNNPACK FP16 allowlist.

### Remaining integration gaps

AI Edge Quantizer should not currently be described as implementing the entire
XNNPACK automatic FP16 workflow. The inspected implementation has these gaps:

1. It does not emit `reduced_precision_support=fp16accfp16`.
2. Its float-casting algorithm leaves biases unquantized in FP32.
3. The LiteRT XNNPACK delegate treats an elided FP16-filter dequantization as an
   FP16 filter and requires matching floating filter and bias types for
   convolution, depthwise convolution, and fully connected operations.
4. `EMBEDDING_LOOKUP` has no matching delegate/XNNPACK FP16 path.
5. The quantizer does not prove that every weightless node in a delegated
   partition is supported by the XNNPACK FP16 rewrite.

The bias mismatch means that a graph of the following effective form can fail
the delegate's type check:

```text
FP16 filter + FP32 bias
```

A complete conversion path would need to address eligible biases as well as
weights, write the metadata only after FP16-accumulation validation, and verify
actual XNNPACK delegation rather than only model-level numerical correctness.

## LiteRT-Torch status

The inspected LiteRT-Torch source tree does not directly write
`reduced_precision_support` and contains no encoding of `fp16accfp16` or
`fp16accfp32`.

LiteRT-Torch does have FP16-related facilities, including a full-FP16 recipe and
mixed-precision graph rewriting. Those facilities directly change graph
precision; they are distinct from emitting the TFLite compatibility metadata
consumed by the XNNPACK delegate.

LiteRT-Torch also contains code for specialized TFLite metadata such as
`lora_rank`, showing that metadata can be serialized, but no corresponding
writer for XNNPACK reduced-precision compatibility was found.

## ML Drift status

ML Drift supports FP16 execution, but it does not implement or consume the
TFLite/XNNPACK `reduced_precision_support` mechanism. No references to the
metadata key, its `fp16accfp16`/`fp16accfp32` encodings, or the XNNPACK FP16
runtime flags were found in the inspected checkout.

Instead, callers explicitly select one of ML Drift's GPU calculation modes in
`CreateGpuModelInfo::precision`:

- `CalculationsPrecision::F32`: FP32 data and FP32 math.
- `CalculationsPrecision::F16`: FP16 data, math, and accumulation.
- `CalculationsPrecision::F32_F16`: FP16 storage and math, with FP32
  accumulation for convolution, depthwise convolution, fully connected, and
  transposed convolution.

Within the direct ML Drift API, there is no model flag or environment variable
for `F32_F16`. The application activates it by setting
`CreateGpuModelInfo::precision` before converting the source graph or IR into a
`GpuModel`:

```cpp
CreateGpuModelInfo create_info{};
create_info.precision = CalculationsPrecision::F32_F16;
create_info.storage_type = GetFastestStorageType(gpu_info);

GpuModel gpu_model;
RETURN_IF_ERROR(
    GraphToGpuModel(graph, create_info, gpu_info, &gpu_model));
RETURN_IF_ERROR(
    inference_context.InitFromGpuModel(create_info, &gpu_model, env));
```

The OpenCL convenience path `GraphToInferenceContext`, and the backend-specific
`IrModelToInferenceContext` helpers, forward the same `create_info`, so setting
the field before calling those functions also activates the mode. The precision
must be selected during graph-to-GPU-model conversion: setting it only when
loading an already-built or serialized `GpuModel` cannot regenerate its tensor
types and shaders.

Callers should initialize both scalar fields in `CreateGpuModelInfo`; they have
no in-class defaults. For OpenCL, `Environment::IsSupported(F32_F16)` or
`GetSupportedPrecisions()` can be checked first. Conversion otherwise returns
an error on an OpenCL device for which `GpuInfo::SupportsFP16()` is false.

### LiteRT-LM activation setting and current plumbing gap

LiteRT-LM exposes a related setting, but the inspected end-to-end path does not
currently select ML Drift `F32_F16`.

At the application layer, `activation_data_type` can be set through the C API,
Python API, or the hidden CLI option `--activation-data-type`. Its public values
are `fp32`, `fp16`, `int16`, and `int8`; it has no mixed value. For the GPU
backend, the relevant mappings are:

```text
activation_data_type=fp16 -> GpuOptions::Precision::kFp16
                           -> ML Drift delegate kFp16
                           -> CalculationsPrecision::F16

activation_data_type=fp32 -> GpuOptions::Precision::kFp32
                           -> ML Drift delegate kFp32
                           -> CalculationsPrecision::F32
```

A `.litertlm` TFLite section can separately contain:

```text
prefer_activation_type=fp32_fp16
```

The LiteRT-LM builder documents this as mixed activation. When no explicit
`activation_data_type` override is already present, `EngineSettings` translates
the section hint into:

```text
ActivationDataType::FLOAT32
enable_mixed_precision=true
```

However, the main LLM `CreateCompilationOptions` implementation does not read
`enable_mixed_precision`. It sees `FLOAT32` and sets
`GpuOptions::Precision::kFp32`. The LiteRT ML Drift adapter then maps `kFp32` to
the delegate's `kFp32`, and the OpenCL, WebGPU, Vulkan, OpenGL, and Metal
delegates all map explicit `kFp32` to `CalculationsPrecision::F32`.

Therefore, in the inspected versions:

```text
prefer_activation_type=fp32_fp16
    -> LiteRT-LM records mixed-precision intent
    -> actual explicit ML Drift precision is F32, not F32_F16
```

This appears to be a missing API/plumbing connection. The ML Drift delegate
precision enum has only `kDefault`, `kFp16`, and `kFp32`; it has no value that
LiteRT can use to request `CalculationsPrecision::F32_F16`. The Metal delegate
can select `F32_F16` in one `kDefault` hardware-dependent branch, but that is
not the result of LiteRT-LM's `fp32_fp16` hint.

The computation implementation is already present, so the fix is conceptually
small. A safe implementation should add a distinct precision value such as
`kFp16WithFp32Accum` and propagate it through:

1. LiteRT's public `LiteRtDelegatePrecision` and C++ `GpuOptions::Precision`.
2. GPU option serialization and round-trip tests.
3. The LiteRT-to-ML-Drift adapter and ML Drift delegate precision enum.
4. Each ML Drift delegate backend's precision switch, mapping the new value to
   `CalculationsPrecision::F32_F16`.
5. LiteRT-LM `CreateCompilationOptions`, selecting the new value when
   `IsMixedPrecisionEnabled()` is true.

Existing `kFp32` must not simply be remapped to `F32_F16`, because callers use
it to require genuine FP32 tensors and arithmetic. Likewise, mapping mixed
precision to `kDefault` is insufficient: default normally chooses `F16` on
FP16-capable GPUs. Adding an enum value at the end is likely ABI-preserving for
the C representation, but LiteRT-LM and LiteRT versions still need to agree on
the new value, and older runtimes must not receive an enum value they do not
understand.

An explicit `--activation-data-type` or API activation-type setting also takes
precedence over the bundle hint. Because the public activation enum has no
mixed value, callers cannot request this section-hint behavior through that
option.

For `F16` and `F32_F16`, graph tensors that were FP32 are represented internally
as FP16, apart from graph outputs explicitly requested to remain FP32. Kernel
weights are also packed or uploaded using the datatype deduced from the chosen
precision. Consequently, ML Drift can take FP32 graph weights and construct an
FP16 GPU model at runtime; it does not require an FP16 weight-only TFLite model
or the XNNPACK-specific `DEQUANTIZE` pattern to do so.

This is a backend configuration rather than model metadata. It does not modify
the input `.tflite` file or certify that FP16 accumulation is accurate. It also
does not have XNNPACK hint-style fallback semantics: the OpenCL model builder
returns an error when `F16` or `F32_F16` is requested on a GPU without FP16
support. WebGPU separately detects and requests the `ShaderF16` feature.

ML Drift does not share XNNPACK's FP16-rewrite operator allowlist. Its own GPU
operation selectors and kernels determine whether a graph can be lowered in the
requested precision. In particular, its `F32_F16` mode is not the same thing as
TFLite metadata value `fp16accfp16`: the ML Drift mode deliberately retains
FP32 accumulation for convolution-like operations.

ML Drift does link to XNNPACK, but the relevant calls are utility operations
such as FP32-to-FP16 conversion, padding, and transposition while preparing
weights. No creation of an XNNPACK runtime, or forwarding of
`XNN_FLAG_HINT_FP16_INFERENCE` or `XNN_FLAG_FORCE_FP16_INFERENCE`, was found.

## Inspection of Gemma LiteRT-LM bundles

All Gemma-family `.litertlm` files under `/data/bt/models` were inspected by
parsing every embedded TFLite section directly. The scan covered:

- Seven Gemma 3 1B bundles.
- Seven Gemma 3 270M bundles.
- One FunctionGemma bundle.
- Thirty-seven embedded TFLite sections in total.

None contained the TFLite metadata key:

```text
reduced_precision_support
```

Their LiteRT-LM section headers also contained no
`prefer_activation_type` entry. Thus none of the inspected Gemma files request
the separate LiteRT-LM `fp32_fp16` path either.

Therefore, these bundles do not automatically request
`XNN_FLAG_HINT_FP16_INFERENCE` through model metadata. LiteRT-LM or another
caller could still supply raw XNNPACK runtime flags externally.

The inspected Gemma bundles primarily use Q4/Q8 weight formats. Unlike fully
static INT8 vision models, a quantized-weight language model can potentially
benefit from XNNPACK's hybrid FP16-activation paths if its delegated partitions
pass compatibility analysis. The absence of metadata means this behavior is not
requested automatically by the files themselves.

## Recommended validation sequence

Before enabling this feature for a conversion pipeline:

1. Generate FP16 static filters and matching eligible biases using the standard
   TFLite FP16-constant plus `DEQUANTIZE` representation.
2. Run the model with forced FP16 to measure accuracy under FP16 computation and
   accumulation.
3. Compare against the original FP32 model on representative evaluation data.
4. Confirm that all intended operators are delegated to XNNPACK.
5. Confirm that every XNNPACK partition passes FP16 rewrite analysis.
6. Add `reduced_precision_support=fp16accfp16` only after the accuracy result is
   acceptable.
7. Benchmark on native ARM/ARM64 FP16 hardware; do not use x86 emulation as a
   performance proxy.
8. Verify fallback behavior on hardware without native FP16 support.

For static INT8 models, skip this process: the FP16 rewrite does not improve an
already-INT8 activation graph.

## Relevant source locations

### XNNPACK

- `include/xnnpack.h`: runtime flag definitions and native-support query.
- `src/subgraph.c`: FP16 compatibility analysis and graph rewrite.
- `src/xnnpack/hardware-config.h`: native and emulated FP16 hardware checks.
- `src/runtime.c`: runtime optimization flag forwarding.

### LiteRT/TFLite checkout

- `tflite/delegates/xnnpack/xnnpack_delegate.h`: delegate options and force
  flag.
- `tflite/delegates/xnnpack/xnnpack_delegate.cc`: metadata parsing, runtime flag
  mapping, static `DEQUANTIZE` elision, FP16 filter handling, and type checks.
- `tflite/converter/tools/optimize/reduced_precision_metadata.h`: metadata key,
  bitmask, and string encodings.
- `tflite/converter/python/tf_tfl_flatbuffer_helpers.cc`: converter support-mask
  construction.
- `tflite/converter/tf_to_tfl_flatbuffer.cc`: metadata insertion during export.

### AI Edge Quantizer checkout

- `ai_edge_quantizer/algorithms/nonlinear_quantize/float_casting.py`: supported
  weight operators and FP32-to-FP16 casting.
- `ai_edge_quantizer/transformations/quantize_tensor.py`: TFLite tensor type and
  buffer update.
- `ai_edge_quantizer/transformations/dequant_insert.py`: explicit
  `DEQUANTIZE` insertion.
- `ai_edge_quantizer/recipe_manager.py`: FP16 weight-only recipe configuration.

### LiteRT-Torch checkout

- `litert_torch/_convert/litert_converter.py`: TFLite FlatBuffer export path.
- `litert_torch/generative/quantize/quant_recipes.py`: full-FP16 recipe.
- `litert_torch/generative/export_hf/core/mu/mixed_precision.py`: direct
  mixed-precision graph conversion.

### LiteRT-LM checkout

- `runtime/engine/engine_settings.cc`: translates bundle hint
  `fp32_fp16` into FP32 activation type plus `enable_mixed_precision=true`.
- `runtime/executor/llm_executor_settings_utils.cc`: translates the activation
  type into LiteRT `GpuOptions::Precision`; it does not consume the mixed flag.
- `runtime/executor/executor_settings_base.h`: activation type and mixed flag
  storage.
- `python/litert_lm_builder/litertlm_builder.py`: writes section metadata
  `prefer_activation_type`, including documented `fp32_fp16` support.
- `python/litert_lm_cli/common.py`: user-facing activation-type choices, which
  do not include a mixed value.

### LiteRT ML Drift delegate checkout

- `litert/runtime/accelerators/gpu/ml_drift_delegate_create.cc`: maps LiteRT
  FP16/FP32 options to the ML Drift delegate FP16/FP32 enum.
- `ml_drift_delegate/delegate/delegate_opencl.cc`: maps explicit delegate FP16
  and FP32 to `CalculationsPrecision::F16` and `F32`.
- `ml_drift_delegate/delegate/delegate_webgpu.cc`: equivalent WebGPU mapping.
- `ml_drift_delegate/delegate/delegate_metal.mm`: equivalent explicit mapping;
  its default mode has a separate hardware-dependent `F32_F16` branch.

### ML Drift checkout

- `ml_drift/common/precision.h`: `F32`, `F16`, and `F32_F16` calculation-mode
  definitions and accumulation semantics.
- `ml_drift/common/precision.cc`: maps the selected mode to FP32 or FP16 data.
- `ml_drift/common/gpu_model.h`: caller-facing `CreateGpuModelInfo::precision`.
- `ml_drift/common/gpu_model_util.cc`: FP16 tensor conversion, hardware support
  check, and FP32-accumulator option.
- `ml_drift/common/task/weights_conversion.cc`: host-side XNNPACK conversion and
  packing utilities, including `xnn_run_convert_nc_f32_f16`.
- `ml_drift/webgpu/inference_context_test.cc`: explicit construction of an F16
  GPU model from a graph with an FP32 predefined input.
