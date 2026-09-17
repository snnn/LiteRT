<!-- Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Local YNNPACK INT8-KV SDPA extension

This local LiteRT delegate change allows FP32 queries and output with
per-tensor affine INT8 K/V in `odml.scaled_dot_product_attention` and
`odml.sdpa_transposed`. It reuses existing YNN mixed-type matrix support;
no new YNN microkernel or converter change is required to implement the
delegate path.

The [mobile analysis](native_mobile_decode_analysis.md) identified the
measured baseline's floating-point-only SDPA admission as an obstacle to
authoring tighter attention regions. This patch removes that specific
obstacle. The published Gemma4 bundle still contains runtime-BMM regions;
rebuilding its runtime with this patch does not automatically replace them
with SDPA. None of the earlier performance measurements includes this change.

## Supported contract

| Item | Contract |
| --- | --- |
| Query and output | FP32, rank four. Existing all-floating SDPA support is preserved. |
| K and V | Both INT8, each with one finite positive scale and one zero point in `[-128,127]`. Their quantization parameters may differ. |
| Sequence-major composite | Q `[B,Q,H,D]`, K/V `[B,S,H,D]`. |
| Transposed composite | Q `[B,H,Q,D]`, K `[B,H,S,D]`, V `[B,H,D,S]`. |
| Active extent | Existing SDPA live-prefix bound and mask slicing. The allocated capacity can exceed the active prefix. |
| Masks and attributes | Existing Boolean/additive masking, attention scale and optional logit cap. |
| Quantized range | Includes cache code `-128`, for mutable and constant caches. |

This is not general support for every attention dtype or grouped-query
configuration. Per-channel KV quantization, mixed floating/INT8 K/V, and
FP16/BF16 Q/output with INT8 KV are outside the new contract. No arbitrary
local-window begin, cache append, ownership change or online/Flash attention
algorithm is added. Callers retain responsibility for valid shapes, bounds,
causal masks and cache contents.

## How it works

[attention.cc](../../tflite/delegates/ynnpack/attention.cc) validates the
quantization metadata and keeps the existing layout conversion and live-prefix
slicing. For INT8 caches, both matrix stages call a small helper exposed by
[dot.h](../../tflite/delegates/ynnpack/dot.h) and implemented in
[dot.cc](../../tflite/delegates/ynnpack/dot.cc):

1. Scale Q and dynamically quantize each reduction row to INT8.
2. Compute QK with the INT8 cache, INT32 accumulation, affine zero-point
   correction and FP32 dequantization of the result.
3. Apply the optional logit cap, mask and softmax over the active score domain.
4. Dynamically quantize the probabilities and compute PV through the same
   mixed-type machinery, producing FP32 output.

The lowering does not insert a full-cache INT8-to-FP32 conversion. Layout
transposes, packing and workspace may still be needed; keeping the stored
and matrix-input KV dtype INT8 is not proof of zero copies or a measured
memory-traffic reduction.

Two details matter. The floating-point short-query optimization reverses
matrix operands; the INT8 path keeps Q/probabilities on the left so dynamic
quantization reduces the correct axis. Also, the shared dot helper's
constant-weight optimization assumes codes in `[-127,127]`. The new caller
disables that assumption because caches may contain `-128`, including
constant test buffers. Existing callers retain their previous behavior.

## Numerical meaning and validation

Dynamic quantization of Q and probabilities follows the existing YNN
runtime-BMM approach. It differs from the native runner's FP32/QC8 arithmetic
and from an unquantized attention reference. This patch therefore does not
promise bitwise parity with either. Changing the active probability domain
can also change quantization statistics compared with a capacity-shaped
runtime-BMM graph.

The new [INT8 tests](../../tflite/delegates/ynnpack/int8_sdpa_test.cc) check
delegation explicitly; their fallback invocation deliberately fails. An
independent double-precision reference dequantizes K/V and computes attention
without calling the production helper. The tests cover:

- Both layouts and query lengths 1, 32 and 33, including the boundary of the
  existing floating-point short-query optimization.
- Active prefixes 13, 47, 64 and 13 again at capacity 64; poisoning inactive
  cache positions must leave the output unchanged.
- Distinct K/V scales, zero and nonzero zero points, additive and Boolean
  masks, attention scaling and logit caps.
- Constant and mutable caches containing `-128`, including zero-point-zero
  cases, checked against a constant-output oracle. A separate constant-cache
  case combines `-128`/`-127` keys with nonconstant values so an error in QK
  changes the final output. Distinct data in each head checks head addressing.
- Rejection of 32 unsupported or malformed K/V metadata/type combinations.

The general reference threshold is absolute error `0.01` for this bounded
fixture, and the constant-output threshold is `0.001`. Those are unit-test
tolerances, not full-model logit acceptance criteria. Each general-reference
test records its observed maximum error in the GoogleTest XML output.

Validation results and exact build provenance are retained in
[the local artifacts](/data/bt/tmp/ynnpack-int8-sdpa-20260916).
Final validation on September 16, 2026:

| Platform | New INT8 tests | Existing floating-point tests | Largest recorded reference error |
| --- | ---: | ---: | ---: |
| Linux x86_64 | 10/10 pass | 9/9 pass | 0.007181 |
| TECNO LJ9, Android ARM64 | 10/10 pass | 9/9 pass | 0.007181 |
| Pixel 8, Android ARM64 | 10/10 pass | 9/9 pass | 0.007181 |

All runs completed with zero failures, errors or disabled tests. The INT8
target contains 66 positive invocations and 32 rejection checks within its
10 GoogleTest cases. Equal recorded maxima do not imply bitwise equality of
every output. The Android tests used a single cross-compiled binary pair
on both devices; the Samsung phone was not used for this validation.

The tested source starts from LiteRT commit
`345852646f25bc3e8d3935fa589243bad8bc5288` with the preexisting local development
changes plus this delegate patch. It uses the repository-selected XNNPACK/YNN
archive `d89ef6669a14db203b3b7935b1b3862cb63fb6df`; no dependency override or
upgrade was introduced for this patch. Exact delegate source snapshots,
patch, hashes, build configuration and test XML are preserved with the logs.

Full-model replacement correctness remains open: the prior YNN full-logit
gate described in the [mobile analysis](native_mobile_decode_analysis.md#numerical-scope)
has not been resolved by these unit tests. No new bundle or performance
claim is part of this patch. A subsequent authoring experiment needs its
own delegation, intermediate/logit and mobile-latency checks.

## Reproduction

From the LiteRT repository root, the host targets are:

```bash
mkdir -p .cache .bazelisk-cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" test -c opt --jobs=8 \
  --cxxopt=-ffp-contract=off --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  --test_output=errors \
  //tflite/delegates/ynnpack:int8_sdpa_test \
  //tflite/delegates/ynnpack:attention_test
```

For Android, with `ANDROID_HOME` set and NDK 28.2.13676358 installed:

```bash
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" build -c opt --jobs=8 \
  --cxxopt=-ffp-contract=off --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  --repo_env=ANDROID_SDK_HOME="$ANDROID_HOME" \
  --repo_env=ANDROID_NDK_HOME="$ANDROID_HOME/ndk/28.2.13676358" \
  --repo_env=ANDROID_NDK_VERSION=28 \
  --repo_env=ANDROID_SDK_API_LEVEL=35 \
  --repo_env=ANDROID_NDK_API_LEVEL=31 \
  --repo_env=ANDROID_BUILD_TOOLS_VERSION=35.0.0 \
  --config=android_arm64 \
  //tflite/delegates/ynnpack:int8_sdpa_test \
  //tflite/delegates/ynnpack:attention_test
```

The resulting test executables are under
`bazel-bin/tflite/delegates/ynnpack/`. The validated ARM64 binaries depend
only on Android's `libm`, `libdl`, `liblog` and `libc`; no extra shared library
deployment was needed. Push both to a device test directory,
make them executable, and run each with
`--gtest_output=xml:<device-path-to-results.xml>`. Pull the XML and retain
stdout/stderr, the exit status, device identity and binary hashes. These are
correctness tests; their elapsed times are not model benchmarks.
