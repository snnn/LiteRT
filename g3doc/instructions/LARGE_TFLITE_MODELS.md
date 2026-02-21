# Large `.tflite` Models and Weight Storage Options

This guide is for:
- developers building apps with LiteRT/TFLite runtimes
- developers writing converters that generate `.tflite` models

## Why this guide exists

FlatBuffers have a practical size limit of about 2GB. Large models can exceed
this limit if all constant tensors are embedded directly in FlatBuffer `Buffer`
data.

LiteRT/TFLite codebase supports two patterns to work around this:
- `buffer_offset` mode: keep one `.tflite` file, but store constant payloads
  outside the FlatBuffer region in the same file.
- external buffer mode: store weights in one or more separate files and
  reference them from the model.

## Option 1: `buffer_offset` mode (single-file deployment)

### What it is

In this mode, FlatBuffer metadata stays inside the normal model region, and
large constant/custom-op payloads are appended outside the FlatBuffer blob.
Each `Buffer` then uses `offset`/`size` fields instead of inline `data`.

### When to use it

- You want one deployable file.
- Your pipeline already uses TFLite converter flags.
- You do not need independent weight files.

### Converter guidance

- Python converter path: set `converter._experimental_use_buffer_offset = True`.
- Flag/API path: set `use_buffer_offset = True` in converter flags.

Notes:
- Export can also auto-retry with `use_buffer_offset` when model size exceeds
  the FlatBuffer limit.
- This mode still produces one `.tflite` file.

### Runtime guidance

- Load from file/mmap path so runtime can access both FlatBuffer and appended
  payload bytes.
- If you pass only partial in-memory bytes (FlatBuffer region only), constant
  tensors that rely on offsets cannot be resolved.

### Do users need a different loading API in `buffer_offset` mode?

Usually, no. You can keep using standard model-loading APIs.

What matters is not API name, but whether the runtime sees the full file bytes
(including appended payload region).

Safe patterns:
- C++: `FlatBufferModel::BuildFromFile(...)` /
  `FlatBufferModel::VerifyAndBuildFromFile(...)`
- Python: `Interpreter(model_path="...")`
- In-memory APIs (`BuildFromBuffer`, `model_content=...`) only if the byte
  array contains the entire `.tflite` file, not just the FlatBuffer prefix.

Unsafe pattern:
- Passing truncated bytes that stop at the FlatBuffer end while using
  `Buffer.offset/size` references.

## Option 2: External buffer files (multi-file deployment)

### What it is

Weights are stored outside the model in separate files. Model tensors point to
`external_buffer` IDs, and the model root contains:
- `external_buffer_groups`
- `external_buffers` (id/group/offset/length/packing)

### When to use it

- You want dedicated weight files.
- You want to shard/replace weight assets independently.
- You are implementing custom converter logic or MLIR-based export with
  `tfl.external_const`.

### Converter guidance

If you emit FlatBuffer directly:
- set `Tensor.external_buffer` to a non-zero ID for external constants
- populate `Model.external_buffer_groups`
- populate `Model.external_buffers`
- for externally stored constants, set `Tensor.buffer = 0` (sentinel empty
  buffer)

If you emit MLIR:
- use `tfl.external_const` with `#tfl.external_buffer<group_name, offset,
  length, packing>`
- TFLite FlatBuffer export will materialize `external_buffer_groups` and
  `external_buffers`

Practical rules:
- external buffer IDs must be unique in the model
- `group` must be a valid index into `external_buffer_groups`
- `offset/length` must describe a valid slice inside the target weight file
- `packing` is metadata; define semantics in your own converter/runtime flow

## Runtime compatibility and behavior

### Core TFLite runtime

Core interpreter parsing supports `tensor.external_buffer` fields and tracks
the IDs, but does not generally load external files by itself. If your runtime
path is pure TFLite interpreter, you need your own external-weight resolution
layer.

### LiteRT runtime

LiteRT contains an external weight loader path that:
- reads external buffer metadata from model
- resolves group names to files
- mmaps/reads slices by `offset/length`
- restores tensor data pointers before execution

This path is guarded by build configuration (`LITERT_WITH_EXTERNAL_WEIGHT_LOADER`).

## Packaging patterns

### Pattern A: one `.tflite` file

Use `buffer_offset` mode.

Pros:
- simplest deployment artifact model

Cons:
- still a single very large file

### Pattern B: `.tflite` + multiple weight files

Use external buffer groups and buffers.

Pros:
- easy sharding
- can update weight files independently

Cons:
- app packaging must keep all referenced files together

### Pattern C: `.tflite` + one packed weight file with sections

LiteRT `Options::SetExternalWeightScopedFile(...)` supports mapping external
buffer groups to regions in one packed file.

Pros:
- single weight payload file
- avoids many filesystem entries

Cons:
- requires section map maintenance

## Validation checklist for converter authors

- Model size path:
  - if using `buffer_offset`, verify metadata and `Buffer.offset/size` fields
  - if using external files, verify `external_buffer_groups/external_buffers`
- Tensor consistency:
  - external constants should use `Tensor.buffer = 0`
  - `Tensor.external_buffer` should be non-zero and resolvable
- File slice validity:
  - every `(group, offset, length)` points to valid bytes
- Runtime test:
  - load model with target runtime path and run one known-good inference

## Common pitfalls

- Passing only partial model bytes to runtime in `buffer_offset` mode.
- Emitting `external_buffer` metadata but not shipping referenced files.
- Reusing external buffer IDs unintentionally.
- Using variable tensors with external constant storage.

## Summary

For models approaching or exceeding 2GB:
- first choice for simple deployment: `use_buffer_offset`
- first choice for explicit weight file separation: external buffers

If you need true separate weight files today, prefer LiteRT runtime paths that
enable external weight loading, or provide your own loader layer on top of core
TFLite interpreter.
