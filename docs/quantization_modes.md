<!--
Copyright 2026 The LiteRT Authors.

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

# Quantization modes in LiteRT

Quantization sets the representation of weights and activations for each region
of a model. Backend compilation also selects the arithmetic for regions that
remain in floating point. When quantizing before model conversion, preserve
the source graph's quantization boundaries and explicit casts during export.
If the delegate then executes the remaining FP32 regions in FP32, no
operator-specific workaround is needed to prevent an implicit FP16 downcast.

## Where quantization enters the pipeline

| Starting point | Conversion path | What to preserve or inspect |
| --- | --- | --- |
| Source graph with quantization already specified | Quantize in the framework, then convert to `.tflite`. The converter can import explicit quantize/dequantize (Q/DQ) boundaries or derive them from fake-quant ops used in quantization-aware training. | Preserve the rounding and clipping at each boundary, scale and zero point, per-axis mapping, and the types and casts of float regions. Fusing Q/DQ into a quantized op can preserve these semantics. |
| Floating-point `.tflite` | Convert the source model to an FP32 FlatBuffer, then apply a recipe using LiteRT Quantizer (`litert-quantizer`), the official post-training quantizer for `.tflite` files. | Inspect the rewritten tensors and operators, including the conversions between regions using different recipes. |

Both paths produce a model for backend compilation. For a source-quantized
model, use the framework's quantized execution as the conversion reference.
For FlatBuffer post-training quantization, use the float FlatBuffer to measure
the effect of the recipe. In either case, use the *same exported model* when
testing the effect of a delegate precision setting.
[Torch Q/DQ import](../tflite/converter/stablehlo/transforms/torch/legalize_qdq_custom_call_pass.cc) ·
[TensorFlow fake-quant import](../tflite/converter/utils/fake_quant_utils.h) ·
[LiteRT Quantizer guide](../litert_quantizer/README.md)

## What a LiteRT Quantizer recipe emits

The table shows the same `FULLY_CONNECTED` (FC) with bias after quantizing it
with four stock recipes. The operands are listed in FlatBuffer order. The two
`afp32` recipes both store INT8 weights and expose FP32 activations. Following
the **weight input to FC** distinguishes the two graphs.
[Recipe definitions](../litert_quantizer/litert_quantizer/recipe.py) ·
[FC fixture](../litert_quantizer/litert_quantizer/tests/models/single_fc_bias.tflite)

| Recipe | FC operands → output | Weight path and activation handling |
| --- | --- | --- |
| `static_wi8_ai8` (A8W8) | `INT8, INT8, INT32 → INT8` | FC reads integer weights directly. Calibration fixes the activation parameters before inference. |
| `static_wi8_ai16` (A16W8) | `INT16, INT8, INT64 → INT16` | The static recipe uses symmetric INT16 activations and the quantizer exports INT64 bias for this FC. |
| `dynamic_wi8_afp32` | `FP32, INT8, FP32 → FP32` | The builtin hybrid FC quantizes each input batch row at invocation, computes an integer dot product, and combines the scaled result with float bias. |
| `weight_only_wi8_afp32` | `FP32, FP32, FP32 → FP32` | A `DEQUANTIZE` on the INT8 constant supplies the FP32 weight operand. The recipe requests float arithmetic. |

In the recipe manager, static and dynamic configurations both set
`compute_precision=INTEGER` and `explicit_dequantize=False`. Static also sets
`activation_tensor_config`. Dynamic leaves it unset. Weight-only sets
`compute_precision=FLOAT` and `explicit_dequantize=True`.
These are quantizer configuration fields. To identify the execution path from
a deployed model, inspect the graph and the kernel that accepted it. For
example, builtin `EvalHybridDense` uses an INT8 activation buffer even on paths
that unpack INT4 or INT2 weights. FP16 weight casting is another storage option
and uses floating-point encoding without an affine scale or zero point.
[Recipe manager](../litert_quantizer/litert_quantizer/recipe_manager.py) ·
[Builtin FC dispatch and hybrid kernel](../tflite/kernels/fully_connected.cc)

To inspect the two float-activation recipes, run this from the repository root
with LiteRT Quantizer available. The dynamic model contains one FC. The
weight-only model inserts a `DEQUANTIZE` before it.

```python
from ai_edge_litert.tools import flatbuffer_utils
from litert_quantizer import quantizer, recipe
from litert_quantizer.utils import tfl_flatbuffer_utils

fixture = "litert_quantizer/litert_quantizer/tests/models/single_fc_bias.tflite"
for make_recipe in (recipe.dynamic_wi8_afp32, recipe.weight_only_wi8_afp32):
    qt = quantizer.Quantizer(fixture)
    qt.load_quantization_recipe(make_recipe())
    model = tfl_flatbuffer_utils.read_model(qt.quantize().quantized_model)
    op_names = [flatbuffer_utils.opcode_to_name(model, op.opcodeIndex)
                for op in model.subgraphs[0].operators]
    print(make_recipe.__name__, op_names)
```

### Following a change through LiteRT Quantizer

1. **Resolve the recipe.** A rule selects a scope regex, operator, algorithm,
   and tensor configuration. The quantizer derives scope from output tensor
   names, falling back to input names when needed. The last supported match in
   recipe traversal order wins. With no match, the op gets `NO_QUANTIZE`.
   Inspect the exported tensor names when a converter rename changes coverage.
   [Scope construction](../litert_quantizer/litert_quantizer/utils/tfl_flatbuffer_utils.py) ·
   [Rule resolution](../litert_quantizer/litert_quantizer/recipe_manager.py)
2. **Collect statistics when the recipe needs them.** In the Python API, check
   `qt.need_calibration` after loading the recipe. Pass representative samples
   keyed by signature and input name to `qt.calibrate(...)`, then pass its
   result to `qt.quantize(...)`. The stock static recipes need activation
   statistics. The stock min/max dynamic and weight-only recipes can quantize
   directly. A weight algorithm such as GPTQ can also require samples. If a
   static recipe selects `range_method="histogram_mse"`, the calibrator also
   collects histograms for evaluating candidate clipping ranges.
   [Quantizer API](../litert_quantizer/litert_quantizer/quantizer.py) ·
   [Calibration criteria](../litert_quantizer/litert_quantizer/recipe_manager.py)
3. **Inspect the rewrite.** The parameter generator applies the algorithm to
   each eligible op and reconciles producer and consumer requirements, including
   shared buffers. The model modifier applies the tensor changes and inserts
   conversions. Check each affected subgraph and signature, especially when
   one constant feeds consumers with different recipes.
   [Default operator policy](../litert_quantizer/litert_quantizer/default_policy.py) ·
   [Parameter generator](../litert_quantizer/litert_quantizer/params_generator.py) ·
   [Model modifier](../litert_quantizer/litert_quantizer/model_modifier.py)

### Reading quantization boundaries

Trace the producer and consumer of each conversion before rewriting it. A
`DEQUANTIZE` fed by a weight buffer reconstructs the stored approximation.
Folding it to a float constant can remove work at inference while increasing
the compiled model size and peak preparation memory. A `DEQUANTIZE` fed by an
integer operator converts an activation on each execution unless the compiler
fuses it. A float → integer → float activation round trip includes rounding
and clipping. Two integer tensors of the same dtype can also need requantization
when their scales or zero points differ.
[Quantizer dequantization insertion](../litert_quantizer/litert_quantizer/transformations/dequant_insert.py)

## Integer tensor contracts

For affine quantization, `real = scale * (code - zero_point)`. Check the stored
codes and their parameters together. Per-tensor quantization has one scale.
Per-axis quantization indexes scales along `quantized_dimension`. For example,
the output-channel axis is 0 for standard FC and Conv2D weights and 3 for
DepthwiseConv2D weights. Transposing weights must preserve the channel-to-scale
mapping. Blockwise quantization uses a quantization details table referencing
scale and optional zero-point tensors. Check the block geometry supported by
both writer and reader and that each scale is finite and positive in the
consuming type, including for all-zero blocks.
[FlatBuffer schema](../tflite/converter/schema/schema.fbs) ·
[LiteRT 8-bit quantization specification](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec)

For standard INT8 FC and convolution, symmetric weights have zero point 0 and
codes in `[-127, 127]`. INT8 activations can use `[-128, 127]`. The uniform
quantizer applies narrow-range clipping for symmetric tensors of eight or more
bits. Its signed INT4 and INT2 storage can represent `[-8, 7]` and `[-2, 1]`
respectively. Its packer places the first coefficient in the least significant
bits of each byte. For a dense payload of `N` coefficients, check
`ceil(N * bits / 8)` bytes, including the final partial byte. Exercise the
most negative code on the selected kernel's unpacking or repacking path.
[Uniform quantization](../litert_quantizer/litert_quantizer/algorithms/uniform_quantize/uniform_quantize_tensor.py) ·
[Sub-byte packing](../litert_quantizer/litert_quantizer/transformations/transformation_utils.py)

Operand provenance matters for dot products. The direct YNNPACK delegate can set
`YNN_NODE_FLAG_SYMMETRIC_B` for eligible constant INT8 weights without scanning
the buffer. Such a weight must be safely negatable, which excludes `-128`.
A BatchMatMul right operand produced by the graph can be a full-range
activation. Carry that distinction through rewrites that cache operands or
change a matmul into an FC. [YNNPACK dot lowering](../tflite/delegates/ynnpack/dot.cc)

### Choosing weight granularity for mobile NPUs and desktop GPUs

TFLite's `BlockwiseQuantization` has limited portability across mobile NPU
plugins. The Qualcomm QNN and MediaTek plugins reject
`kLiteRtQuantizationBlockWise` during tensor lowering. Samsung's compiler
accepts only per-tensor and per-channel quantization parameters. For a model
targeting these paths, choose a per-channel or per-tensor weight recipe
supported by the target operator. When adapting blockwise weights, requantize
from the original float weights if available. Changing the metadata alone
would apply different scales to the existing codes.
[Qualcomm tensor lowering](../litert/vendors/qualcomm/compiler/qnn_compose_graph.cc) ·
[MediaTek operand mapping](../litert/vendors/mediatek/compiler/legalizations/operand_map.h) ·
[Samsung quantization parameters](../litert/vendors/samsung/compiler/create_model.cc)

Blockwise INT4 weights are useful for bandwidth-bound matrix multiplication on
desktop GPUs. Each group gets a scale suited to its local range, and a kernel
can read packed weights and dequantize them as it computes. Match the export to
the kernel's block size and axis. For example, native TensorRT INT4 weight-only
quantization supports blocks of 64 or 128. LiteRT's NVIDIA plugin also has a
custom CUDA path for 2-D INT4 weights with blocks of 32 along the input
dimension, FP16 or FP32 scales, and implicit zero points of zero.
[TensorRT weight-only quantization](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html) ·
[LiteRT NVIDIA blockwise FC lowering](../litert/vendors/nvidia/compiler/tensorrt_graph_builder.cc)

### Bias and accumulator width

For a static FC with one input scale and per-output-channel weights, bias is
added in the units of the centered integer dot product. For output channel
`c`, the converter needs

```text
bias_scale[c] = input_scale * weight_scale[c]
bias_zero_point = 0
bias_code[c] = round(real_bias[c] / bias_scale[c])
```

LiteRT Quantizer's uniform bias path quantizes to 32 bits and widens the stored
codes to INT64 for INT16 activations. Check that the rounded bias fits before
clipping, then check the sum in the target kernel's accumulator type. One
conservative bound for a signed `p`-bit accumulator is

```text
abs(bias_code[c]) + max_abs_centered_input * sum(abs(centered_weights[c]))
    <= 2^(p - 1) - 1
```

Use the largest input code allowed by the execution contract. A kernel that
materializes partial sums or zero-point corrections also needs those
intermediates to fit. The C++ converter utility `AdjustWeightsForBiasScale`
illustrates one remedy for an oversized bias code. It expands the affected
weight range to reserve bias headroom. Regenerate both weights and biases after
adjusting scales and recheck the error and the bound.
[Quantizer bias implementation](../litert_quantizer/litert_quantizer/algorithms/uniform_quantize/uniform_quantize_tensor.py) ·
[C++ converter scale adjustment](../tflite/tools/optimize/quantization_utils.cc)

## Backend precision for floating-point regions

For a region exported as FP32, check whether the selected backend permits
implicit FP16 execution. The two delegates expose different controls.

| Delegate | When it narrows FP32 | How to preserve FP32 in this path |
| --- | --- | --- |
| XNNPACK | Its automatic FP16 path requires native FP16 support and model metadata permitting both FP16 inference and FP16 accumulation. The delegate can keep external tensors FP32 while using FP16 internally. | For a model that requires FP32, omit that metadata permission and leave both the runtime flag `TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16` and the build override `XNNPACK_DELEGATE_FORCE_PRECISION_FP16` off. |
| MLDrift GPU | The default depends on the GPU backend and device. For example, WebGPU selects FP16 by default when the device reports FP16 support. | Request `GpuOptions::Precision::kFp32` with `SetPrecision` before compilation. The separate `kFp16WithFp32Accum` option still selects FP16 with wider accumulation. |

These settings apply to either quantization workflow. Explicit FP16 casts and
rounding at integer boundaries are already part of the exported model. Account
for them when comparing its outputs with the original float model.
[XNNPACK partition construction](../tflite/delegates/xnnpack/xnnpack_delegate.cc) ·
[Reduced-precision metadata](../tflite/converter/tools/optimize/reduced_precision_metadata.h) ·
[LiteRT GPU options](../litert/cc/options/litert_gpu_options.h) ·
[MLDrift option mapping](../litert/runtime/accelerators/gpu/ml_drift_delegate_create.cc) ·
[WebGPU precision selection](../ml_drift_delegate/delegate/delegate_webgpu.cc)

When FP16 execution is intentional, check the operands, intermediate results,
and accumulator separately. For example, squaring an input of 256 in FP32
produces 65536. Squaring it in FP16 overflows before a subsequent FP32 sum
can help. This makes a sum of squares, as used in RMSNorm, a small regression
test for unintended narrowing. The same distinction applies to an operation
whose result is rounded to FP16 before the next consumer.

## Verifying the deployed path

1. **Check the exported graph.** At each intended boundary, inspect operand and
   result types, quantization parameters, and explicit casts. For a composite,
   check both the call signature and the decomposition subgraph. The builtin
   runtime can execute the decomposition and a delegate can lower the same call
   directly. [Builtin composite evaluation](../tflite/kernels/stablehlo_composite.cc)
2. **Localize numerical differences.** Run identical inputs through the
   conversion reference and the exported model with builtin kernels. Then run
   that exported model on the target with FP32 selected for its float regions.
   If deploying with relaxed precision, repeat with that setting as the only
   change. Compare tensors at quantization and partition boundaries to find the
   first divergence, then measure task quality on held-out data.
   [Quantizer FC tests](../litert_quantizer/litert_quantizer/tests/end_to_end_tests/fully_connected_test.py)
3. **Measure placement and cost.** Record the recipe, converter and runtime
   revisions, device, backend, options, signature, shape, and any fallbacks.
   For weight-only operators, determine whether the backend expands weights
   during preparation or dequantizes them inside the kernel. Compare source and
   compiled sizes, preparation and persistent memory, initialization time, and
   warm inference time under matched conditions.
