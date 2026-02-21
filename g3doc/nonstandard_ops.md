# Non-Standard ODML Operators (StableHLO Composite / TFLite Custom)

LiteRT/TFLite supports a small set of **named, non-standard operators** that
originate as `stablehlo.composite` ops in StableHLO MLIR.

These are important for LLM performance on the default CPU backend (notably the
XNNPACK delegate), but they are **not part of the StableHLO spec**: the meaning
is defined by LiteRT/TFLite code.

This document is intended for people working on **converters** that need to
emit these ops in a form that LiteRT/TFLite recognizes.

## How These Ops Appear In TFLite FlatBuffers

Depending on the conversion pipeline, the same “ODML op name” may end up in a
`.tflite` model as either:

- A builtin `STABLEHLO_COMPOSITE` op (TFLite schema: `StableHLOCompositeOptions`)
  - `name` (string)
  - `version` (int)
  - `decomposition_subgraph_index` (int)
  - `composite_attributes` (bytes; **flexbuffers map**)
- A `CUSTOM` op with `custom_code == <name>`
  - `custom_options` / `custom_initial_data` is typically a **flexbuffers map**

Runtime handling differs by backend:

- Generic CPU fallback:
  - `STABLEHLO_COMPOSITE` can be executed by invoking its decomposition
    subgraph (`tflite/kernels/stablehlo_composite.cc`).
  - Optionally, the interpreter can inline composite nodes into the main graph
    (`InterpreterOptions::SetShloCompositeInlining(true)`, implemented in
    `tflite/core/subgraph.cc`).
- XNNPACK delegate:
  - Can recognize certain composite/custom op names and replace them with
    optimized XNNPACK subgraphs.

## “ODML” Op Names That Matter For CPU (XNNPACK)

### `odml.scaled_dot_product_attention`

This is the primary `stablehlo.composite`-originated op that is recognized in
the XNNPACK delegate today.

Accepted forms in a `.tflite` model:

- `STABLEHLO_COMPOSITE` with `StableHLOCompositeOptions.name ==
  "odml.scaled_dot_product_attention"`.
- `CUSTOM` with `custom_code == "odml.scaled_dot_product_attention"`.

Where it is handled:

- XNNPACK delegation logic:
  - `tflite/delegates/xnnpack/xnnpack_delegate.cc`
  - Looks for `kTfLiteBuiltinStablehloComposite` with `params->name ==
    "odml.scaled_dot_product_attention"`.
  - Also supports `kTfLiteBuiltinCustom` with `custom_name ==
    "odml.scaled_dot_product_attention"`.

Attributes encoding (flexbuffers map):

- `scale`: float32 scalar (optional)
  - If absent, XNNPACK delegate falls back to `1 / sqrt(head_dim)`.
- `logit_cap`: float32 scalar (optional)
  - If present, XNNPACK applies a `tanh`-based logit capping.

Notes for converter authors:

- XNNPACK currently reads **pointers** to float scalars inside the flexbuffer
  blob (see `tflite/delegates/xnnpack/flexbuffers_util.h`). In practice, this
  means you should serialize `scale` / `logit_cap` as **flexbuffer float
  scalars**, not e.g. strings.
- XNNPACK’s SDPA path is currently implemented for `float32` inputs/outputs and
  expects rank-4 Q/K/V tensors; see shape checks in `VisitDotAttentionNode` in
  `tflite/delegates/xnnpack/xnnpack_delegate.cc`.

## Composite Names That Are Lowered To TFLite Custom Ops

Some StableHLO composite names are *not* kept as `STABLEHLO_COMPOSITE` in the
TFLite dialect; instead the converter legalizes them into `tfl.custom` ops with
the same `custom_code`, and serializes the composite attributes into a
flexbuffers map.

Source of truth:

- `tflite/converter/stablehlo/transforms/legalize_stablehlo_composite_to_tfl_custom.cc`

Currently legalized composite names:

- `odml.update_kv_cache`
  - The pass also injects attributes `num_layers` and `layer_index` (int32).
- `odml.update_external_kv_cache`
- `odml.quantize_and_dequantize`
- `odml.detector`

If your converter emits these as `stablehlo.composite` ops, you generally
should ensure the composite attributes are representable as flexbuffers (the
pass currently serializes integer, float, and string attributes).

## How To Keep This List Up To Date

This repo does not currently have a single authoritative registry file; the
“spec” is the code.

Useful commands:

```bash
# Composite names present in MLIR testdata (not necessarily supported by XNNPACK).
rg -n --pcre2 'stablehlo\\.composite\\s+\"([^\"]+)\"' litert tflite

# Composite names recognized by XNNPACK (look for strcmp(...) on params->name).
rg -n 'kTfLiteBuiltinStablehloComposite' tflite/delegates/xnnpack/xnnpack_delegate.cc

# StableHLO composite names that are legalized into tfl.custom.
rg -n 'IsSupportedComposite\\(' tflite/converter/stablehlo/transforms/legalize_stablehlo_composite_to_tfl_custom.cc
```

