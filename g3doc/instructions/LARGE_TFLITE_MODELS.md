# Large `.tflite` Models and Weight Storage Options

This guide is for developers loading LiteRT/TFLite models and developers writing
converters. It describes the current source tree; check the runtime and converter
versions shipped with your application when using these features.

## Why this guide exists

The TFLite FlatBuffer structure must fit below approximately 2 GiB. Embedding all
constant tensors in `Buffer.data` can exceed that limit. The code supports two
ways to move payloads outside the FlatBuffer structure:

| Storage | Deployment | References in the model |
| --- | --- | --- |
| `buffer_offset` | One `.tflite` file, with appended payloads | `Buffer.offset/size`; `Operator.large_custom_options_offset/size` |
| External buffers | `.tflite` plus weight files or application-provided storage | `Tensor.external_buffer`, `Model.external_buffers`, `Model.external_buffer_groups` |

Both allow the total payload to exceed the FlatBuffer limit. Neither removes the
limit on the FlatBuffer structure itself or the target device's memory limits.
External buffers describe immutable tensor data; appended payloads can also hold
custom-op options. See the [schema](../../tflite/converter/schema/schema.fbs).

## Option 1: `buffer_offset` mode (single-file deployment)

### Representation

The graph and metadata remain in a FlatBuffer at the start of the `.tflite` file.
Constant payloads are appended after it. For each appended constant:

- `Tensor.buffer` still indexes `Model.buffers`.
- `Buffer.data` is absent; `Buffer.offset` and `Buffer.size` are 64-bit byte
  values. The offset is relative to the beginning of the `.tflite` file and must
  be greater than 1. Values 0 and 1 are not valid appended-payload offsets.
- Appended custom-op options use the operator's
  `large_custom_options_offset/size` fields, with the same offset convention.

The exporter aligns appended constants to 16 bytes and adds `buffer_location`
metadata with the value `outside flatbuffers`. Preserve this metadata when
rewriting a model: converter utilities and `BuildFromModel` use it to recognize
this storage mode. The payload locations themselves come from the offset fields.

### Converter guidance

- Python TFLite converter: set
  `converter._experimental_use_buffer_offset = True` before `convert()`.
- Converter flags: set `use_buffer_offset = True` (C++:
  `converter_flags.set_use_buffer_offset(true)`).
- The MLIR translation command-line flag is `--use-buffer-offset`.

The current [FlatBuffer exporter](../../tflite/converter/flatbuffer_export.cc)
also enables this mode proactively when the estimated module size exceeds the
FlatBuffer limit minus 512 MiB. If an initial export reports the FlatBuffer size
limit error, it retries once with offsets enabled. These are exporter behaviors,
not a guarantee that every conversion or postprocessing step can handle large
models. The [StableHLO Python conversion pipeline](../../tflite/converter/python/stablehlo_tfl_pipeline.cc)
explicitly enables buffer offsets even for small models.

### Runtime guidance

Standard loading APIs work when the runtime receives the complete file allocation,
including appended bytes:

- TFLite C++: `FlatBufferModel::BuildFromFile(...)` or
  `FlatBufferModel::VerifyAndBuildFromFile(...)`, then construct the interpreter
  from the `FlatBufferModel`.
- TFLite Python: `Interpreter(model_path="model.tflite")`.
- LiteRT C++: `Model::CreateFromFile(env, ...)` or the filename overload of
  `CompiledModel::Create(env, ..., options)`.
- In-memory loading (`BuildFromBuffer`, `model_content=...`, or LiteRT buffer
  overloads) requires the entire `.tflite` byte sequence and its full size. Keep
  caller-owned storage alive for its consumers.

File loading allows mmap where supported and avoids first reading the whole model
into a Python bytes object. Passing only the FlatBuffer prefix loses the appended
data. Also avoid `FlatBufferModel::BuildFromModel(const tflite::Model*)`, which
rejects models marked with `buffer_location`. The raw-model `InterpreterBuilder`
overload needs an explicit backing `Allocation` to resolve appended offsets.
See [model loading](../../tflite/converter/core/model_builder_base.h) and
[interpreter parsing](../../tflite/core/interpreter_builder.cc).

A plain FlatBuffer unpack/repack does not preserve appended payloads or relocate
their offsets. Any tool that rewrites the model must handle both constant buffers
and large custom-op options explicitly.

## Option 2: External buffers (separate weight storage)

Use external buffers when weight assets need separate packaging or sharding. The
model records group names and slices; the loader supplies their backing storage.

### Emitting FlatBuffers directly

For each external constant:

- Set `Tensor.external_buffer` to a nonzero `ExternalBuffer.id`. This is an ID,
  not an index into `Model.external_buffers`.
- Set `Tensor.buffer = 0`, and keep `Model.buffers[0]` empty. Do not attach inline
  or appended constant data to the same tensor, or mark it as variable.
- Add an `ExternalBuffer` with a unique ID, a `group` index into
  `Model.external_buffer_groups`, and 64-bit byte `offset`/`length` fields.
- Give the referenced `ExternalBufferGroup` a name identifying its backing
  storage. Group index 0 is valid; it is not the tensor-buffer sentinel.

The current exporter assigns IDs as `0x80000000 | external_buffer_index`, setting
the high bit to distinguish them from ordinary TFLite buffer indices. Follow that
convention when generating models for the same runtime/delegate paths.

The `packing` string records layout information. The built-in loader exposes it
to consumers but does not decode arbitrary compression or packing formats. For
CPU execution, provide bytes in the tensor's expected representation; the
externalization tool below writes `packing = "unpacked"`.

For host access, the resolved tensor address must satisfy
`LITERT_HOST_MEMORY_BUFFER_ALIGNMENT` (currently 64 bytes). Align file slice
offsets accordingly. For a packed file, align `section.offset + buffer.offset`;
for in-memory groups, align `group_base + buffer.offset`. The length must cover
the tensor's required bytes. These requirements come from
[host tensor buffer validation](../../litert/runtime/tensor_buffer.cc).

### Emitting MLIR

Use `tfl.external_const` with an `external_buffer` attribute. The attribute uses
named fields, for example:

```mlir
%weights = "tfl.external_const"() <{
  external_buffer = #tfl.external_buffer<
    group_name = "weights.bin", offset = 0, length = 64, packing = "unpacked">
}> : () -> tensor<4x4xf32>
```

The exporter materializes the groups, buffers, and tensor references. Your
converter must separately write the referenced bytes. A `tfl.external_const`
with only `buffer_index` is a different form used to reference an existing
FlatBuffer constant; it does not create a separate weight file. See the
[op definition](../../tflite/converter/ir/tfl_ops.td),
[attribute definition](../../tflite/converter/ir/tfl_op_enums.td), and
[exporter](../../tflite/converter/flatbuffer_export.cc).

### Externalizing an existing `.tflite` model

[`litert/tools/externalize_tflite_flatbuffer.py`](../../litert/tools/externalize_tflite_flatbuffer.py)
writes `model.tflite` and one weight blob into an output directory. From the
repository root, in a Python environment with `flatbuffers`, NumPy, and the
generated `litert.python.schema_py_generated` module available:

```bash
python3 -m litert.tools.externalize_tflite_flatbuffer \
  --input_model=/path/to/input.tflite \
  --output_dir=/path/to/output \
  --group_name=tflite_weights \
  --num_elements_threshold=256
```

The generated module must include the external-buffer schema fields. It is not
checked in at that import path, and this tool's Python Bazel targets are currently
commented out in [the OSS BUILD file](../../litert/tools/BUILD). With `flatc`
available, generate the module from the current schema before running the command:

```bash
flatc --python --gen-onefile --gen-object-api \
  --filename-suffix _py_generated -o litert/python \
  tflite/converter/schema/schema.fbs
```

The tool is scoped to weights that the LiteRT-LM external-weight path can consume:

- It selects tensors used at input 1 of `FULLY_CONNECTED`, `CONV_2D`,
  `DEPTHWISE_CONV_2D`, or `EMBEDDING_LOOKUP`, with **more than** the threshold
  number of elements. The threshold counts elements, not bytes.
- It skips subgraph inputs, bias tensors, variables, and tensors already using
  external buffers. Other constants remain embedded.
- It deduplicates identical newly externalized payloads and aligns their offsets
  to 64 bytes. It clears old buffers only when remaining tensors no longer
  reference them.
- It reads appended `Buffer.offset/size` payloads before processing. Unselected
  payloads, such as biases, are repacked inline in the output model.

This is not a general streaming conversion for arbitrarily large files: it reads
the input into memory, and the remaining inline model must fit in a FlatBuffer.
It does not relocate appended `Operator.large_custom_options_*` payloads. Models
using those fields need a tool that handles them. Existing external weight files
must also be preserved; the Python `externalize(..., existing_weights=...)`
argument can copy an existing blob before appending, but the CLI does not expose
that argument.

## Runtime compatibility and behavior

### Core TFLite interpreter

The [interpreter builder](../../tflite/core/interpreter_builder.cc) records
`Tensor.external_buffer` IDs, but does not resolve the external groups into files
or restore their data. A plain `Interpreter(model_path=...)` therefore does not
automatically load separate weights. Use LiteRT's compiled-model path, or supply
an integration that resolves external constants before kernels and delegates
prepare them.

### LiteRT compiled-model runtime

The current [compiled-model runtime](../../litert/runtime/compiled_model.cc)
creates a [weight loader](../../weight_loader/external_weight_loader_litert.cc)
automatically unless the client supplies one. The loader is a direct dependency
in [Bazel](../../litert/runtime/BUILD) and is included in the
[CMake runtime sources](../../litert/runtime/CMakeLists.txt). There is no
`LITERT_WITH_EXTERNAL_WEIGHT_LOADER` guard in the current tree.

When CPU is requested, the runtime prepares host access and restores external
tensor pointers as immutable `kTfLiteMmapRo` data **before applying delegates**, so
CPU kernels and XNNPACK can use them. The loader is also passed to accelerator
options. GPU-only loading depends on the delegate's weight-loader integration;
the presence of schema fields alone does not guarantee support on every backend.
On Web, CPU pointer restoration runs only when CPU is requested without GPU/NPU,
to support GPU weight streaming.

For a model and its weight files in the same directory, no external-weight option
is needed. For example, inside a function returning `litert::Expected<...>`, with
an existing `litert::Environment env`:

```cpp
LITERT_ASSIGN_OR_RETURN(auto options, litert::Options::Create());
LITERT_RETURN_IF_ERROR(
    options.SetHardwareAccelerators(litert::HwAccelerators::kCpu));
LITERT_ASSIGN_OR_RETURN(
    auto compiled_model,
    litert::CompiledModel::Create(env, "/models/model.tflite", options));
```

If the model's group name is `weights.bin`, this loads `/models/weights.bin`.
Keep the environment alive for the compiled model and its executions.

### How group names resolve

The built-in loader checks these sources in order for each group:

| Priority | Source | Meaning of `ExternalBuffer.offset` |
| --- | --- | --- |
| 1 | Matching entry in `Options::SetWeightInMemoryMap(...)` | Offset into the group's host-memory span |
| 2 | Matching section in `Options::SetExternalWeightScopedFile(...)` | Offset within the section; file position is `section.offset + buffer.offset` |
| 3 | `ExternalBufferGroup.name` as a filesystem path | Offset from the start of that file |

For the filesystem fallback, absolute paths are used unchanged. Relative paths
are resolved against the model's source directory when available. Models loaded
from memory or a file descriptor have no source directory, so relative names are
passed to the filesystem as-is (relative to the process working directory).
Provide explicit storage mappings when that is unsuitable. The native filesystem
fallback does not fetch URLs, even though the schema describes groups as file/URI
references.

### Packed weight files and client-owned storage

`Options::SetExternalWeightScopedFile(...)` maps group names to sections in one
open file. Before the `CompiledModel::Create` call above, add, for example:

```cpp
LITERT_ASSIGN_OR_RETURN(auto weight_file,
                       litert::ScopedFile::Open("/models/weights.pack"));
litert::Options::ScopedWeightSectionMap sections;
sections.emplace("weights.bin", litert::ScopedWeightSection{4096, 8192});
LITERT_RETURN_IF_ERROR(
    options.SetExternalWeightScopedFile(weight_file, std::move(sections)));
```

This example maps `weights.bin` to an 8192-byte section starting at byte 4096.
An external buffer at offset 64 then starts at file byte 4160. Add one map entry
per group backed by the packed file. Each section must have positive length and
fit inside the file, and every buffer slice must fit inside its section.

Although the setter takes `ScopedFile&`, it **moves the file handle** into the
options; `weight_file` is no longer usable after a successful call. Include
`litert/cc/internal/scoped_file.h` and
`litert/cc/internal/scoped_weight_source.h` for these types. The public
[Options header](../../litert/cc/litert_options.h) defines the setters.

For application-owned weights, `SetWeightInMemoryMap(...)` borrows the group map
and its backing memory; both must remain valid for the compiled model's lifetime.
This setter is unavailable when `LITERT_NO_ABSL` is defined. `SetWeightLoader(...)`
instead supplies a client-owned loader, which must also outlive its consumers.

The [`run_model` tool](../../litert/tools/run_model.cc) exposes a single section
mapping through `--scoped_weight_file`, `--scoped_weight_group`,
`--scoped_weight_offset` (default 0), and `--scoped_weight_length` (default -1,
meaning the rest of the file). Multiple sections require the C++ map API.

## Validation checklist for converter authors

- Keep the graph, metadata, and all remaining inline data below the FlatBuffer
  size limit. Use generated bindings that include every field being emitted.
- For appended payloads, check `buffer_location`, offset values greater than 1,
  alignment, and the full allocation length. Preserve or recompute offsets for
  both constant buffers and custom-op options after every rewrite.
- For external constants, check the empty buffer sentinel, nonzero and unique
  external-buffer IDs, valid group indices, immutable tensor metadata, and
  representation/length matching the tensor type and shape.
- Validate slice bounds without integer overflow: require `offset <= size` and
  `length <= size - offset`. For packed files, validate both section and tensor
  ranges, and check the final host address alignment.
- Ship all referenced weights or provide matching storage mappings. Replacing a
  weight asset requires compatible shapes, types, quantization parameters,
  offsets, and packing metadata in the model.
- Run a known-good inference on the intended runtime and accelerator combination,
  including CPU fallback if needed. FlatBuffer schema verification alone cannot
  verify the contents of separate weight files or backend compatibility.
