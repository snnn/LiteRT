# RMSNorm StableHLO Composite Segfault Repro

This document provides a minimal reproducible example for a LiteRT runtime
segmentation fault observed while loading a generated `.tflite` model.

## Symptom

`CompiledModel.from_file(...)` crashes with `SIGSEGV` in:

- `tflite::ParseStablehloComposite`
- top frame often shows:
  - `flatbuffers::Vector<unsigned char, unsigned int>::size(this=0x0)`

## Repro Script

Use:

- `g3doc/instructions/repro_rms_norm_stablehlo_composite_segv.py`

The script can:

1. generate a minimal model containing RMSNorm composite lowering
2. load that model via `ai_edge_litert` runtime

## Prerequisites

1. LiteRT debug wheel installed in the Python environment used for repro.
2. `litert-converter` repo available (default expected path):
   - `/home/chasun/src/litert-converter`

## Repro Steps

### 1) Generate repro model

```bash
/home/chasun/src/litert-converter/.venv/bin/python \
  g3doc/instructions/repro_rms_norm_stablehlo_composite_segv.py \
  --mode generate \
  --out /tmp/rms_norm_stablehlo_composite_crash_repro.tflite
```

### 2) Reproduce crash (direct run)

```bash
/home/chasun/src/litert-converter/.venv/bin/python \
  g3doc/instructions/repro_rms_norm_stablehlo_composite_segv.py \
  --mode load \
  --model /tmp/rms_norm_stablehlo_composite_crash_repro.tflite
```

On affected builds, this call crashes.

### 3) Reproduce under gdb and capture backtrace

```bash
gdb -q \
  -ex "handle SIGSEGV stop print nopass" \
  -ex run \
  -ex "bt" \
  -ex "thread apply all bt" \
  -ex quit \
  --args /home/chasun/src/litert-converter/.venv/bin/python \
    g3doc/instructions/repro_rms_norm_stablehlo_composite_segv.py \
    --mode load \
    --model /tmp/rms_norm_stablehlo_composite_crash_repro.tflite
```

## Notes

- This repro intentionally uses an `aten.rms_norm.default` model to keep the
  graph small while still triggering the problematic runtime parse path.
- If needed, use `--mode both` to generate + load in one command.

## Debug Findings

Using a breakpoint in `tflite::ParseStablehloComposite` shows:

- `op->builtin_options_2_type() == BuiltinOptions2_StableHLOCompositeOptions`
- `op->builtin_options_2_as_StableHLOCompositeOptions()->name()` is non-null
- `op->builtin_options_2_as_StableHLOCompositeOptions()->decomposition_subgraph_index() == -1`
- `op->builtin_options_2_as_StableHLOCompositeOptions()->composite_attributes() == nullptr`
- crash occurs at:
  - `schema_params->composite_attributes()->size()`
  - file: `tflite/core/api/flatbuffer_conversions.cc`

This indicates runtime-side null dereference in parser code.

## Differential Check (Helpful for Triage)

Two minimal RMSNorm models were compared:

- model A: has `StableHLOCompositeOptions.composite_attributes` bytes
  - contains strings like `epsilon`, `inv_dim`
  - `CompiledModel.from_file(...)` loads successfully
- model B: missing `composite_attributes` (null vector)
  - does not contain `epsilon`/`inv_dim` in flatbuffer string table
  - `CompiledModel.from_file(...)` segfaults in `ParseStablehloComposite`

This narrows crash cause to null-handling in parser for optional composite
attribute payload.

## Suggested Runtime Hardening

In `ParseStablehloComposite`, guard nullable fields before dereference and
return `kTfLiteError` with a diagnostic instead of segfaulting.

Example direction:

```cc
const auto* attrs = schema_params->composite_attributes();
if (attrs == nullptr) {
  TF_LITE_REPORT_ERROR(
      error_reporter,
      "'stablehlo.composite' missing composite_attributes.");
  return kTfLiteError;
}
params->attributes = attrs->data();
params->attributes_size = attrs->size();
```

Optionally, also guard `schema_params->name()` similarly.
