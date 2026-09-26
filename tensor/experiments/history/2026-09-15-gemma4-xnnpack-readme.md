# Gemma4 on the XNNPACK tensor runner

This local example runs the text portion of Gemma4 E2B/E4B through tensor
expression graphs and XNNPACK CPU kernels. It reads safetensors directly;
no `.tflite` conversion or LiteRT interpreter is involved. Image/audio towers
and their inputs are outside this example.

## Components and execution

| Component | Responsibility |
| --- | --- |
| [xnnpack_main.cc](xnnpack_main.cc) | Command-line input, checkpoint loading, host embedding lookup, separate prefill/decode runners, masks and RoPE, growing KV buffers, greedy generation, timings, and logit dumps. |
| [gemma4_config.h](gemma4_config.h) | E2B/E4B dimensions, local/global attention pattern, RoPE parameters, and KV-sharing presets. |
| [gemma4_weights.cc](gemma4_weights.cc) | Hugging Face text-weight names to graph-weight names. |
| [gemma4_graph.h](gemma4_graph.h) | Embedding scaling, per-layer input projection, transformer stack and shared KV routing, final normalization, head projection, and logit softcap. |
| [helpers](helpers/README.md) | Attention, MLP, RMSNorm/RoPE integration, quantized embedding lookup, and mobile fully connected layers. |
| [reference_mobile.py](reference_mobile.py) | Independent Transformers CPU reference using the same mobile checkpoint, dequantized FP32 weights, static activation quantization, and lazy embedding rows. |
| [text_io.py](text_io.py) | Offline host tokenization and output decoding using the checkpoint's Hugging Face tokenizer. |
| [compare_logits.py](compare_logits.py) | Compare finite, complete logit vectors with matching token histories; report argmax, cosine, RMSE, and maximum absolute error. |
| `*_test.cc` and [helper references](helpers/reference/README.md) | Small numerical and metadata tests, including full prefill versus repeated cached decode. |

The driver maps the checkpoint, translates weight names, and creates token and
per-layer embedding accessors. It slices the combined per-layer projection into
one matrix per layer. Model variant detection uses the final normalization
width; this driver targets the supplied E2B/E4B presets. Actual loaded MLP
matrix shapes supply E2B's wider shared-layer MLPs.

Prefill receives all prompt embeddings, constant causal/sliding masks, and
position-zero RoPE tables. Its last-position logits predict the first token.
The driver then copies the owning layers' KV outputs into host buffers and
feeds one predicted token at a time to the decode runner. Each step binds new
embedding/mask/RoPE data, grows the bound cache shape, and appends new K/V
values. Shared layers reuse the last owning layer of the same attention type.
The current driver supports batch size one and retains growing FP32 KV caches,
including for sliding attention; it does not implement a ring buffer.

The native packed-weights cache is created before both runners, populated by
both `PrepareRuntime()` calls, and finalized before execution. It outlives the
runtimes. Source constants and embedding accessors also remain alive during
inference. Borrowed host input views are rebound for each call and their
storage remains live through `Run()`.

## Checkpoints and quantization

Google publishes mobile compressed-tensors checkpoints:

- [google/gemma-4-E2B-it-qat-mobile-ct](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-ct)
- [google/gemma-4-E4B-it-qat-mobile-ct](https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-ct)

Use a local snapshot containing `model.safetensors`, `config.json`, and tokenizer
files. Keep `config.json` beside the weights, including on Android. The
[loader](../utils/README.md) resolves each module's exact/regex
quantization targets, physical packed shape, scale tensors, and activation
scales. These mobile checkpoints mix 2/4/8-bit weights. Two-bit weights expand
to signed INT4 storage without changing their quantized values; this is a
format adaptation, not another quantization pass. E2B's per-layer embedding is
4-bit groupwise; E4B's is 2-bit groupwise. Both use groups of 256 there.

Static INT8 input/output activation scales are applied to the corresponding
linear layers. Residuals, normalization, attention, softmax, and KV storage use
FP32. Ordinary BF16/FP16 weights used by those graph operations are converted
to FP32. An explicitly stored `lm_head.weight` is preferred over tied embedding
weights. The head preserves FP32 activations with a local XNNPACK FC extension;
its INT4 weights are converted to the unsigned nibble representation expected
by that kernel while remaining packed. No multimodal execution or quantized
KV-cache scheme is implemented.

The native FP32 head path requires an even number of input channels for INT4
weights. It also requires at least two output channels and symmetric per-channel
weight scales on axis zero. Both supplied model presets satisfy these
constraints; other shapes are rejected during graph construction.

Google's mobile snapshots contain a BPE `tokenizer.json`, which is not a
SentencePiece model. Use `text_io.py` on the host with `--token_ids` in C++.
The retained `--tokenizer` path accepts a real, compatible SentencePiece
`tokenizer.model`; it has not been validated against these mobile snapshots.
`--token_ids` supplies the exact input sequence and adds no BOS or chat wrapper.

## Build and synthetic tests

From the repository root, follow [tensor build/test setup](../../README.md#build-and-test).
Enable the local example in either the Linux or Android CMake configuration:

```sh
cmake -S litert/tensor -B build/gemma4-linux -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLITERT_TENSOR_BUILD_GEMMA4=ON
cmake --build build/gemma4-linux --parallel 16
ctest --test-dir build/gemma4-linux -L gemma4 --output-on-failure

cmake -S litert/tensor -B build/gemma4-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_HOME/ndk/27.3.13750724/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 \
  -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLITERT_TENSOR_BUILD_GEMMA4=ON
cmake --build build/gemma4-android --parallel 16
GEMMA_SERIAL=replace-with-the-intended-device-serial
python3 scripts/run-litert-tensor-tests-android.py \
  --build-dir build/gemma4-android --serial "$GEMMA_SERIAL" \
  --filter 'gemma4|safetensor_loader' \
  --output-dir "build/gemma4-android-results-$(date +%Y%m%d-%H%M%S)" \
  --cleanup
```

Set `GEMMA_SERIAL` to the intended serial from `adb devices -l`. The synthetic
tests need no model. The driver is `bin/gemma4_xnnpack_main` in the build root;
example GoogleTest binaries live under `examples/bin/`. Builds use the pinned
SentencePiece runtime with bundled protobuf sources, static project libraries,
and static libc++ on Android. No Android-side Python environment is needed.

## Linux model run and independent comparison

Set `GEMMA_MODEL` to the local mobile-CT snapshot. Set `GEMMA_PYTHON` to a Python
interpreter with PyTorch, Transformers with Gemma4 support, safetensors, and
NumPy. The reference loads active linear weights in FP32 on the host; allow
substantially more host RAM than the compressed checkpoint size.

Use a fresh output directory or prefix for each run:

```sh
GEMMA_MODEL=/absolute/path/to/mobile-ct-snapshot
GEMMA_PYTHON=/absolute/path/to/reference-env/bin/python
mkdir -p build/gemma4-results

# Raw completion: BOS + "The capital of France is".
build/gemma4-linux/bin/gemma4_xnnpack_main \
  --weights="$GEMMA_MODEL/model.safetensors" \
  --token_ids=2,818,5279,529,7001,563 --max_tokens=3 --num_threads=2 \
  --dump_logits=build/gemma4-results/linux

"$GEMMA_PYTHON" litert/tensor/examples/gemma4/reference_mobile.py \
  --model-dir "$GEMMA_MODEL" --input-ids=2,818,5279,529,7001,563 \
  --decode-steps=2 --threads=2 \
  --output-prefix=build/gemma4-results/reference
"$GEMMA_PYTHON" litert/tensor/examples/gemma4/compare_logits.py \
  build/gemma4-results/reference build/gemma4-results/linux --steps=3
"$GEMMA_PYTHON" litert/tensor/examples/gemma4/text_io.py \
  --model "$GEMMA_MODEL" decode --report=build/gemma4-results/linux.json
```

`--max_tokens` includes the token predicted by prefill. Stop tokens are recorded
in the report but not printed as text. Dump files are little-endian FP32 vectors
of length 262144: `.prefill.f32` contains the final prompt position, then
`.decode_0001.f32`, etc. The adjacent JSON records prompt/generated IDs,
filenames, dimensions, and the stop reason. A reference can request forwards
beyond a stop token; compare only steps present in both reports with identical
preceding input IDs. The comparison command fails on token disagreement,
nonfinite/missing data, or exceeded numeric thresholds.

For a different prompt, obtain exact IDs through the checkpoint's own tokenizer:

```sh
"$GEMMA_PYTHON" litert/tensor/examples/gemma4/text_io.py \
  --model "$GEMMA_MODEL" encode --chat --prompt 'What is the capital of France?'
```

Pass the comma-separated output to `--token_ids`. Without `--chat`, the utility
encodes a raw completion and explicitly prepends BOS. Both paths are offline.

## E2B consistent-arithmetic diagnostics

E2B's single KV head broadcasts directly in the two attention matrix products.
It therefore does not require the explicit broadcast rewrite that XNNPACK
disables under `XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC`. The driver exposes this
runtime flag as `--consistent_arithmetic`, disabled by default, and passes it
to both the prefill and decode runners. Each JSON report records
`consistent_arithmetic` and `xnnpack_runtime_flags`.

To compare the two modes, repeat a model run with the same token IDs and a
fresh output prefix:

```sh
build/gemma4-linux/bin/gemma4_xnnpack_main \
  --weights="$GEMMA_MODEL/model.safetensors" \
  --token_ids=2,818,5279,529,7001,563 --max_tokens=3 --num_threads=2 \
  --consistent_arithmetic --dump_intermediates \
  --dump_logits=build/gemma4-results/linux-consistent
"$GEMMA_PYTHON" litert/tensor/examples/gemma4/compare_logits.py \
  build/gemma4-results/reference build/gemma4-results/linux-consistent --steps=3
```

For Android, add the same flag to the device command in the section below and
pull the new report and dumps before comparing them. Keep the device serial
explicit. The flag attempts consistency among paths in one compiled XNNPACK
library; it does not promise equality between Linux and Android builds.
Compare full logits and intermediate values against the reference, and compare
later decode steps only while preceding generated IDs agree.

This diagnostic option currently requires a single KV head. E4B's multi-head
tiling still needs the general broadcast rewrite, so the driver rejects E4B
with `--consistent_arithmetic`. Default E4B execution retains its existing path.

## Numerical validation and intermediate dumps

Strict end-to-end comparison against the independent reference has **not
passed**. The comparator's defaults, cosine at least `0.999` and RMSE at most
`0.05`, are diagnostic checks; do not relax them to claim numerical equivalence.
Matching greedy token IDs alone does not establish that the logits agree.

### Six-token E2B reference comparison

An investigation of the six-token E2B prefill above found bitwise-identical
first-layer Q/K/V projection outputs. A later attention-context value differed
by only approximately `5.96e-7`, but dividing by the static input scale moved it
from `11.499996` to `11.500016`. This crossed a rounding boundary at the attention
output projection (`o_proj`), changing its INT8 input code from `11` to `12` and
one quantized output value. Subsequent normalization and quantized projections
amplified the difference. All nine first-layer fully connected operations run
in isolation with the reference inputs matched all 118,272 reference output
values exactly. This shows how rounding is amplified in E2B's first layer;
it does not account for all end-to-end drift or establish the cause of E4B
differences.

### BOS-only E2B across platforms

For `--token_ids=2 --max_tokens=1`, Linux and Android decoder-layer outputs
agree within about `1.8e-7` relative L2 error through layer 28. Layer 29's
output then differs by about 2.5%. Probes that expose the FP32 and INT8 inputs
and outputs of that layer's linear operations preserve the original logits
bit-for-bit on each platform.

The consequential difference is channel 387 of the normalized input shared
by `mlp.gate_proj` and `mlp.up_proj`, whose static input scale is
`0.032805636525154114`:

| Runtime | FP32 input | Input divided by scale (FP32) | INT8 code |
| --- | --- | --- | --- |
| Python reference | 0.7053208947 | 21.49999046 | 21 |
| Linux | 0.7053210139 | 21.49999428 | 21 |
| Android | 0.7053211927 | 21.5 | 22 |

Nearest-even rounding sends the Android value to 22. The changed input code
changes 220 gate outputs and 202 up outputs by one output quantization level.
When each platform runs these two FCs in isolation with identical captured
FP32 inputs, all **49,152 outputs** agree bit-for-bit across architectures,
including raw INT8 inputs/outputs and dequantized FP32 outputs. Reconstructing
the preceding normalization and residual operations reproduces Android's
captured MLP input from its own incoming state; the input difference is
consistent with accumulated FP32 rounding.

A temporary diagnostic binary reduced only channel 387 of those two MLP
inputs by three FP32 ULPs (`1.7881393432617188e-7`). This restored Android's
predicted token to `568` and reduced full-vocabulary RMSE against the reference
from `1.6198` to `5.18e-6`; cosine became approximately `1.0`. Later layer
outputs returned to approximately `1e-7` relative error against Linux. This
controlled intervention establishes the cause of this BOS-only mismatch.
The correction exists only in the temporary diagnostic binary; the normal
driver retains the checkpoint's computations and its original results.

This diagnosis covers one E2B input. Other prompts and E4B still require
separate investigation before claiming model numerical equivalence or quality.

### First FP32 difference: reciprocal square root

Tracing the BOS-only graph earlier locates its first FP32 difference in layer
0's initial RMSNorm. The embedding bytes, embedding scale, scaled input,
normalization weights, and epsilon are identical across platforms. An isolated
RMSNorm reproduces each platform's full-graph output bit-for-bit. Its captured
stages show:

| Stage | Linux x86_64 | Android arm64 |
| --- | --- | --- |
| Mean of squared input | 0.9646416902542114 | 0.9646416902542114 |
| Mean plus epsilon | 0.9646427035331726 | 0.9646427035331726 |
| Reciprocal square root | 1.0181616544723511 (`0x3f82531f`) | 1.0181617736816406 (`0x3f825320`) |

The reciprocal-square-root results differ by one FP32 ULP. That changes 715
of the 1,536 normalized output values, with maximum absolute error
`1.52587890625e-5`. Squaring, reduction, and epsilon addition produce identical
results for this input, despite the platforms using different reduction
kernels.

Actual runtime function pointers, resolved against the executable symbols,
identify the selected implementations:

- Linux uses `xnn_f32_vrsqrt_ukernel__avx_rsqrt_u16`: an x86 `VRSQRTPS`
  estimate followed by one Newton refinement with separate FP32 arithmetic.
  See the [AVX kernel](../../../../src/f32-vrsqrt/gen/f32-vrsqrt-avx-rsqrt.c).
- Android arm64 uses `xnn_f32_vrsqrt_ukernel__scalar_sqrt_u4`: square root
  followed by division. Disassembly of this one-element invocation shows
  `FSQRT` then `FDIV`. See the
  [scalar kernel](../../../../src/f32-vrsqrt/gen/f32-vrsqrt-scalar-sqrt.c).

The [dispatch configuration](../../../../src/configs/unary-elementwise-config.c)
selects these paths with the runner's default runtime flags. Calling both
actual kernels on the **same Linux CPU**, with the same input bytes, exactly
reproduces the two platform results. This identifies a difference in the
selected arithmetic algorithms. Both observed values satisfy the existing
[FP32 rsqrt test tolerance](../../../../test/unary-ops.h), which is relative
error of twice FP32 epsilon for this operator.

This establishes the source of the first FP32 difference. The layer-29
experiment separately establishes where a small input difference changes an
INT8 code and the final token. Other floating-point operations can contribute
rounding differences between those points; replacing only this first
reciprocal-square-root operation has not been shown to restore model parity.

### Capturing layer outputs

To inspect layer values, add `--dump_intermediates` to the C++ command above;
it requires a nonempty `--dump_logits` prefix. Add `--dump-intermediates` to
the Python reference command, which uses its `--output-prefix`. Both write
little-endian FP32 files for each complete decoder-layer output, including its
layer scalar, and for the final normalized hidden state before the head:

- `.prefill.layer_000.f32`, `.prefill.layer_001.f32`, and so on, plus
  `.prefill.norm.f32`.
- `.decode_0001.layer_000.f32`, subsequent layer files, and
  `.decode_0001.norm.f32`; later decode passes increment the step number.

These files retain the full row-major shape: `[1, prompt_length, hidden_size]`
for prefill and `[1, 1, hidden_size]` for decode. The JSON `intermediates` entries
record each suffix, shape, and dtype. C++ exposes these additional runtime
outputs only when the flag is enabled. `compare_logits.py` compares logits;
use the intermediate metadata to locate earlier differences separately.

## Run the same model on Android

Use the Android binary built above, the same model snapshot, and the same IDs.
Choose a device directory dedicated to this experiment:

```sh
GEMMA_ADB="$ANDROID_HOME/platform-tools/adb"
GEMMA_REMOTE=/data/local/tmp/gemma4-local-test
"$GEMMA_ADB" -s "$GEMMA_SERIAL" shell mkdir -p "$GEMMA_REMOTE"
"$GEMMA_ADB" -s "$GEMMA_SERIAL" push \
  build/gemma4-android/bin/gemma4_xnnpack_main \
  "$GEMMA_MODEL/model.safetensors" "$GEMMA_MODEL/config.json" "$GEMMA_REMOTE/"
"$GEMMA_ADB" -s "$GEMMA_SERIAL" shell chmod 755 "$GEMMA_REMOTE/gemma4_xnnpack_main"
"$GEMMA_ADB" -s "$GEMMA_SERIAL" shell \
  "$GEMMA_REMOTE/gemma4_xnnpack_main" \
  --weights="$GEMMA_REMOTE/model.safetensors" \
  --token_ids=2,818,5279,529,7001,563 --max_tokens=3 --num_threads=2 \
  --dump_logits="$GEMMA_REMOTE/android"
"$GEMMA_ADB" -s "$GEMMA_SERIAL" pull "$GEMMA_REMOTE/android.json" build/gemma4-results/
"$GEMMA_ADB" -s "$GEMMA_SERIAL" pull "$GEMMA_REMOTE/android.prefill.f32" build/gemma4-results/
"$GEMMA_ADB" -s "$GEMMA_SERIAL" pull "$GEMMA_REMOTE/android.decode_0001.f32" build/gemma4-results/
"$GEMMA_ADB" -s "$GEMMA_SERIAL" pull "$GEMMA_REMOTE/android.decode_0002.f32" build/gemma4-results/
"$GEMMA_PYTHON" litert/tensor/examples/gemma4/compare_logits.py \
  build/gemma4-results/reference build/gemma4-results/android --steps=3
```

Packed weights, converted constants, source buffers, runtime caches, and KV
storage all contribute to memory use. The safetensors file size alone is not
the runtime memory requirement. Avoid running both model processes at once on
a memory-constrained phone. Timing output describes this example's execution;
short correctness runs are not a benchmark of an optimized production runtime.

## Validation results

Local validation on 2026-09-12 used Linux x86_64 with Clang 22.1.8 and a
TECNO LJ9 arm64 phone, CMake 4.3.3, and `RelWithDebInfo` builds. Android used
NDK 27.3.13750724, API 24, and static libc++. All device commands selected the
TECNO serial explicitly. The Python CPU reference used Python 3.13.8,
PyTorch 2.12.0+rocm7.1, and Transformers 5.12.1.

The 11 synthetic Gemma4/helper/loader test executables passed all **103 cases**
on both platforms, with no skips or failures. These cover graph execution,
cached decode, singleton-KV broadcasting with default and consistent arithmetic,
embedding formats, static activation rounding, FP32-input
quantized-weight projections, and malformed checkpoint/buffer handling.
The separate core tensor suite's results are recorded in the
[tensor test guide](../../README.md#interpreting-results).

The three-step raw-completion run above produced identical greedy IDs on
Linux, Android, and the reference:

| Mobile-CT checkpoint | Hugging Face snapshot revision | Generated IDs |
| --- | --- | --- |
| E2B | `594b1c7919e98f8275aefdc844c306d709e61727` | `9079,236761,106` (Paris, period, end of turn) |
| E4B | `3624117cf04528e099519f93839f0f0b7a18913d` | `506,5279,529` |

Comparing all 262,144 logits at each of those steps gave the following ranges
against the independent reference. **Every row fails the default numerical
thresholds**; token agreement is a smoke-test result, not model equivalence.

| Model and runtime | Cosine range | RMSE range |
| --- | --- | --- |
| E2B Linux | 0.9834–0.9914 | 0.8561–0.9664 |
| E2B Android | 0.9857–0.9912 | 0.7317–1.0165 |
| E4B Linux | 0.6737–0.9882 | 0.9625–3.5023 |
| E4B Android | 0.9927–0.9937 | 0.4925–0.6486 |

Linux-to-Android comparisons also exceed those thresholds. The large E4B
Linux difference occurs in the second cached decode step: the first six
decoder-layer outputs remain close (relative error around `1e-7`), then drift
starts at layer 6 and grows in later layers. Review found no specific KV-growth
or shared-source routing defect; the large final difference remains unresolved.

An E2B boundary check with `--token_ids=2 --max_tokens=1` completed on both
platforms and wrote only the prefill vector, but its predicted token differed:
Linux and the reference produced `568`, Android `236761`. Linux matched this
single-step reference closely (RMSE `5.81e-6`); Android's cosine was `0.9803`
and RMSE `1.6198`. Numerical differences can therefore change generated
tokens. Longer generation, broader prompt coverage, and model quality have
not been validated. The
[intermediate-dump investigation](#numerical-validation-and-intermediate-dumps)
documents the localized E2B finding without attributing all remaining drift
to that one cause.

Peak process RSS during these short C++ runs was approximately 6.14 GiB for
E2B on both platforms, 10.18 GiB for E4B on Linux, and 7.38 GiB for E4B on
Android. File-backed pages can be reclaimed, so these measurements are not
minimum device-memory requirements. The FP32 Python references peaked near
10.44 GiB and 20.62 GiB respectively.

### E2B broadcast-fix recheck

After replacing singleton-KV tiling with implicit BMM broadcasting, the full
tensor/example suites passed on Linux and TECNO: 652 cases per platform,
626 passed, 26 expected skips, and no failures. Seven new cases cover runtime
flags and moves, singleton-head attention prefill, and repeated cached decode
with growing history. The attention expectations use an independent scalar
calculation with distinct query-head results.

The same E2B checkpoint was run before and after the change on four prompts:

| Input | Exact input token IDs | Generated steps compared |
| --- | --- | --- |
| BOS only | `2` | 1 |
| The capital of France is | `2,818,5279,529,7001,563` | 3 |
| The opposite of hot is | `2,818,11688,529,3425,563` | 3 |
| Chat: What is 2 + 2? | `2,105,2364,107,3689,563,236743,236778,900,236743,236778,236881,106,107,105,4368,107` | 3 |

Default-mode results are bitwise unchanged by the fix on both platforms:
20 logit vectors and 720 decoder-layer/final-norm tensors match the original
binaries exactly. Fresh references also reproduce the earlier BOS and France
reference dumps bit-for-bit.

Both default and consistent arithmetic complete all four prompts, including
two cached decode steps for each multi-token prompt. Each runtime is compared
against the independent reference over ten generated positions:

| Runtime | Matching greedy IDs | Passing strict logit checks | Worst RMSE | Lowest cosine |
| --- | --- | --- | --- | --- |
| Linux, default | 10/10 | 1/10 | 1.48214 | 0.842790 |
| Linux, consistent | 10/10 | 1/10 | 1.40674 | 0.853174 |
| TECNO, default | 9/10 | 0/10 | 1.76416 | 0.927968 |
| TECNO, consistent | 9/10 | 0/10 | 1.61977 | 0.927453 |

Strict checks retain cosine at least `0.999`, RMSE at most `0.05`, and matching
argmax. All ten positions have matching preceding token histories for each
comparison; a differing current prediction is counted as a failure. Shapes,
exact file lengths, finite values, argmax metadata, and all 35 layer outputs
plus final normalization were validated.

Consistent arithmetic lowers reference RMSE at seven positions and raises it
at three on each platform. The BOS mismatch remains: Linux predicts `568`,
TECNO `236761`; the first material cross-platform layer difference remains
layer 29. The option removes the broadcast execution blocker, but does not
establish model numerical parity. Default E4B math is unchanged; the E4B
consistent-mode rejection was checked on Linux, and no new E4B model inference
was performed in this recheck.

### E2B full-model arithmetic controls

Further local checks used the same E2B snapshot and consistent arithmetic on
Linux and TECNO. These separate kernel behavior from differences propagated
through the model; the original checkpoint's strict results above remain
unchanged.

All **277 active linear operations** were replayed with captured six-token
France reference inputs, covering all 35 layers, the global per-layer
projection, and the LM head. Both platforms produced bitwise-identical
5,963,776 FP32 outputs, 4,164,096 INT8 input codes, and 5,647,872 INT8 output
codes. Every INT8 input code also matched the reference. Only 14 reference
output codes differed, each by one level. Independent weight decoding,
integer accumulation, scale calculations, and reproduced reference matmuls
explain all 14 as rounding near half-integers. The two projections with FP32
activations were also checked against FP64 dot products; their maximum errors
were `4.38e-7` and `2.79e-5`. These results validate the captured cases, not
every possible input or full-model equivalence.

The same replay was repeated for all 277 operations in the first cached
France decode step, using its captured reference inputs. All 1,212,416 FP32
outputs, 694,016 INT8 inputs, and 941,312 INT8 outputs agree bit-for-bit across
platforms; every INT8 input also matches the reference. The sole reference
output-code difference is explained by rounding around `-14.5`; exact integer
accumulation and scaling agree with XNNPACK. Together, prefill and decode cover
554 FC invocations and 7,176,192 bitwise-identical cross-platform FP32 outputs.

The first France attention divergence under consistent arithmetic was traced
further: all 288 attention scores match an independent sequential FP32 FMA
calculation bit-for-bit. Score accumulation rounding alone is sufficient to
move one context value across the `11.5` quantization boundary. An 80-digit
calculation and substitutions of individual attention stages confirm this
cause for that coordinate.

Two separate full-model controls tested the effect of activation quantization:

- Retaining static INT8 QDQ while promoting arithmetic and the original FP32
  scales to FP64 inside the Python reference changes France prefill logits
  substantially: RMSE `0.782839`, cosine `0.990522`, with the same predicted
  token. FP32 RoPE tables were retained. This demonstrates arithmetic
  sensitivity within the reference itself; it does not replace the reference.
- Removing static input/output activation QDQ from all 275 active linear
  operations on both sides,
  while retaining checkpoint weights and the remaining graph, gives close
  agreement on all four prompts. Native FCs consequently use the existing
  FP32-activation quantized-weight kernels. This is a **different diagnostic
  model**, not a mode of the normal driver or a passing result for the mobile model.

The activation-QDQ-disabled control compares ten positions per pairing,
including prefill and two cached decode steps for each multi-token prompt:

| Comparison | Strict logit checks passed | Worst logit RMSE | Worst layer/norm relative L2 |
| --- | --- | --- | --- |
| Reference vs Linux | 10/10 | `1.67241e-5` | `4.12231e-6` |
| Reference vs TECNO | 10/10 | `1.71949e-5` | `4.70144e-6` |
| Linux vs TECNO | 10/10 | `1.65625e-5` | `4.21557e-6` |

Each pairing includes 360 layer/final-norm tensors. All predicted IDs and
preceding token histories match within this control. The existing comparator
passed all twelve case/pairing invocations without threshold changes; file
sizes, shapes, finite values, and argmax metadata were also checked.
These controls support the unquantized activation path and explain specific
quantization-sensitive differences. They do not establish original E2B/E4B
model parity or broader model quality.

### E2B sliding-window boundary control

A separate test prefills 511 tokens: BOS followed by the five France-prompt
tokens repeated 102 times. It then feeds fixed tokens `9079,236761,106` at
absolute positions 511, 512, and 513. Predictions are observations only;
they never determine subsequent inputs, and EOS does not stop the schedule.
This keeps all four passes comparable even when predictions differ.

With activation QDQ disabled, all four strict logit checks pass in each
pairing. Worst RMSE is `3.81744e-5` for reference/Linux, `4.37547e-5` for
reference/TECNO, and `7.99665e-5` across platforms. Each pairing checks 144
layer/final-norm tensors. Some intermediate relative errors exceed the
`1e-5` diagnostic marker, reaching about `6.39e-5`; the logit thresholds remain
unchanged. This extends the diagnostic model's coverage across the 512-token
local-attention window while global attention retains the complete history.

With original activation QDQ retained, both native runs complete, but every
pairing fails all four strict logit checks (RMSE range approximately
`0.623`–`1.314`). Linux predicts `106` after the first forced decode input;
the reference and TECNO predict `236761`. This additional quantized-model
difference remains unresolved.

In both modes and on both platforms, byte comparisons confirm that the two
cache appends preserve the complete existing history and copy the emitted
key/value deltas exactly. The reference's observed cache lengths and storage
shapes were also checked. Mask bounds follow the unchanged mask-construction
code; runtime mask arrays were not dumped. These are bounded cache/window
checks, not exhaustive long-context or model-quality validation.

## Extending the example

Add or change checkpoint names in `gemma4_weights.cc` and the loader, and verify
shape/type constraints before graph construction. Change model mathematics in
`gemma4_graph.h` or the helper that owns the operation. Keep the independent
Transformers reference separate from the C++ implementation and compare both
prefill and multiple cached decode steps after changes. For other variants,
update preset detection, attention/KV sharing, dimensions, and weight mapping
together. Longer context, multimodal inputs, sampling, and production memory
optimizations need separate implementation and validation.
