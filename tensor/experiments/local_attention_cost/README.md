# Attention cost probe

This local experiment measures an isolated Gemma4-shaped FP32/QC8 attention
stage. It repeats QK, mask addition, softmax, and PV with fixed inputs and a
reused Tensor API runner. Only the context is a graph output, so diagnostic
intermediate outputs do not constrain fusion.

The matrix contains five attention widths (544, 1088, 2048, 4096, 8448), head
dimensions 256 and 512, eight query heads folded into one matrix or broadcast
as eight batches, and token-major or transposed INT8 V storage. K always uses
token-major storage. Quantization scales come from the published E2B owner
metadata. The valid prefix is `min(width,1088)`, with later positions masked.

Both the native runner and the published LiteRT-LM decomposition already use
folded `[1,1,8,D]` queries. The unfolded variant is a sensitivity control, not
evidence that the LM graph missed folding.

Build the manual binary target with the same validated XNNPACK source override
as the native runner:

```sh
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
bazelisk --output_base="$PWD/.bazel-output-tensor-native-consolidation" \
  build -c opt --cxxopt=-ffp-contract=off \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  --override_repository=XNNPACK="$PWD/.native-tensor-build/host/_deps/litert_native_xnnpack-src" \
  //tensor/experiments/local_attention_cost:attention_cost_probe

bazel-bin/tensor/experiments/local_attention_cost/attention_cost_probe \
  --threads=4 --ms=200 --passes=3 > attention.jsonl
```

`--depths=544,1088` and `--dims=256` select smaller matrices. Android builds add
the repository's Android SDK/NDK settings and `--config=android_arm64`; exact
successful host and Android build commands and binary hashes are preserved in
[the experiment archive](/data/bt/tmp/gemma4-mobile-decode-attribution-20260916/source-audit/probe/build-manifest.json).

Every case checks a double-precision reference, cross-layout numerical error
(maximum absolute error below 2e-5), immutable inputs, and actual runtime
operators: two F32/QC8 BMMs, two QS8-to-QC8 adapters, and one softmax. It rejects
materialized FP32 KV or dynamically quantized attention. JSONL records include
workspace sizes, BMM input shapes and actual compute loop ranges. A successful
complete run prints a `PASS` message to stderr and exits zero.

Timing covers `runner.Run()`, including its shape propagation, setup, and
invocation. Construction, input binding, reference checks, input hashing and
JSON writes are outside the interval. Each variant warms up five times before
each timed batch; variant order rotates between passes. These are hot-cache
microbenchmarks with unchanged inputs and no cache flushing. They do not
measure full-model decode, DRAM traffic, cache append cost, LM fallback loops,
or FC layers, and should not be treated as a decomposition of model speedup.
