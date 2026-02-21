# XNNPACK Delegate Type Restrictions (TFLite / LiteRT)

This document explains tensor type restrictions enforced by the XNNPACK
delegate implementation in:
`tflite/delegates/xnnpack/xnnpack_delegate.cc`.

It is intended for model exporters and converter authors who need predictable
XNNPACK delegation behavior.

## Scope

- These restrictions are for **XNNPACK delegation** only.
- A model can still run if some nodes violate these restrictions, but those
  nodes will run on non-XNNPACK TFLite kernels (partial delegation).

## Core rule

Only tensor types that map to an XNNPACK datatype can be delegated. In
`GetXNNPackDatatype(...)`, supported base TFLite tensor types are:

- `FLOAT32`
- `FLOAT16`
- `UINT8` (with required affine quantization constraints)
- `INT8` / `INT4` / `INT2` (with required quantization constraints)
- `INT32` (quantized bias-style constraints)

`BOOL` is not mapped and is rejected for delegated tensor values.

## High-level implications for model authors

- Do not rely on `BOOL` tensors inside subgraphs you expect XNNPACK to
  accelerate.
- Keep control/shape/index tensors in the expected integer types
  (`INT32`, or `INT32/INT64` where explicitly allowed).
- For quantized models, type alone is not enough; quantization schema details
  must also match XNNPACK checks.

## Type-check helpers used by delegated ops

The delegate uses helper checks to enforce per-op type contracts:

- `CheckTensorType`: exact match to one expected type.
- `CheckTensorFloat32Type`: `FLOAT32` only.
- `CheckTensorFloatType`: `FLOAT32` or `FLOAT16`.
- `CheckTensorFloat32OrQInt8Type`: `FLOAT32` or signed `INT8` quantized.
- `CheckTensorQInt8OrQUInt8Type`: signed or unsigned 8-bit quantized.
- `CheckTensorFloat32OrQUInt8Type`: `FLOAT32` or 8-bit quantized.
- `CheckTensorFloat32OrQCInt8Type`: `FLOAT32` or channel/per-tensor checked
  `INT8`.
- `CheckTensorFilterType`: filter-specific checks (`FLOAT32/FLOAT16/INT8/INT4/INT2/UINT8`
  with constraints).
- `CheckTensorFloat32OrFloat16OrQCInt32Type`: bias-style
  `FLOAT32/FLOAT16/INT32` with quantization checks.
- `CheckTensorInt32Type`: `INT32` only.
- `CheckTensorInt32OrInt64Type`: `INT32` or `INT64`.

## Control tensor restrictions (frequent source of failures)

These are strict and commonly produce `unsupported type ...` messages:

- `REDUCE_*` axes tensor: `INT32`
- `PAD` paddings tensor: `INT32`
- `RESHAPE` shape tensor (2-input form): `INT32`
- `RESIZE_BILINEAR` size tensor: `INT32`
- `SPLIT` split_dim tensor: `INT32`
- `TRANSPOSE_CONV` output_shape tensor: `INT32`
- `STRIDED_SLICE` begin/end/strides tensors: `INT32`
- `SLICE` begin/size tensors: `INT32` or `INT64`
- `EXPAND_DIMS` axis tensor: `INT32` or `INT64`

## Mixed-type restrictions by operator family

Some ops require same-type IO/filter combinations (with a few explicit
exceptions):

- `BATCH_MATMUL`: rejects unsupported mixed types, except dynamic quantization
  path (`input_a` `FLOAT32`, `input_b` `INT8`).
- `CONV_2D`: input/output/filter types must match unless dynamic quantization
  path is used.
- `DEPTHWISE_CONV_2D`: input/output/filter types must match.
- `FULLY_CONNECTED`: input/output types must match; filter mismatch allowed only
  for dynamic quantization or SRQ paths as implemented.
- `BINARY` ops:
  - `ADD/MUL/SUB`: `FLOAT32` or 8-bit quantized checked paths.
  - `DIV/MAXIMUM/MINIMUM/PRELU/SQUARED_DIFFERENCE`: `FLOAT32` only.

## Quantization restrictions (important)

Common requirements enforced in checks include:

- Expected quantization type (`Affine` vs `Blockwise` depending on tensor role).
- Valid scale arrays and positive/normal scale values.
- Valid zero-point values and size rules.
- Required `quantized_dimension` values (often `0`, or op-specific channel dim).
- Per-channel parameter count must match target channel dimension.
- Additional INT4/INT2 constraints:
  - `INT4`/`INT2` filter channel alignment and per-channel requirements.
  - Blockwise quantization currently allowed only for `INT4` in mapped path.

## BOOL-specific guidance

- `BOOL` tensor support is not implemented in XNNPACK delegate type mapping.
- If a delegated op input/output/control tensor is `BOOL`, that node is not
  delegated.
- If your model has logical masks, prefer converting them to numeric mask forms
  before the delegated compute region.

## Typical failure logs and what they mean

- `CheckTensorType: unsupported type BOOL ...`
  - A tensor required to be a specific type (often `INT32` or `FLOAT32`) is
    `BOOL`.
- `unsupported datatype (...) of tensor ... in XNNPACK delegate`
  - Tensor type cannot be mapped to an XNNPACK datatype.
- `unsupported mixed types in ... operator`
  - Op-specific type matching rule failed.
- `unsupported quantization type ...`
  - Quantization schema does not match required form for that tensor role.

## Practical checklist before exporting `.tflite`

- Keep delegated compute tensors as supported numeric types.
- Keep shape/index/control tensors in required integer types.
- Verify quantized tensors have compatible quantization metadata.
- Avoid introducing `BOOL` tensors in subgraphs targeted for XNNPACK.
- Expect partial delegation when unsupported ops/types are present.
