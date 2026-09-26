# YNNPACK attention-layout experiment

This preserves the local FP32 decode-layout experiment from XNNPACK
`bf3ee43b63070284f85a4a882f298b6ca5273f01`. It compares K stored as
`[group, channel, position]` with K stored as `[group, position, channel]`,
using eight groups, two query heads per group and head dimension 128.
The benchmark checks outputs against a scalar reference before timing.
It is a separate YNNPACK experiment; the optimized Gemma4 runner uses XNNPACK.

[attention_bench.cc](attention_bench.cc) is the exact preserved source.
[attention_layout.patch](attention_layout.patch) also preserves its Bazel
integration. The source retains the [XNNPACK BSD license](LICENSE.XNNPACK).

To resume this kernel experiment, create a scratch XNNPACK checkout under
LiteRT's `.native-tensor-build/ynnpack-layout`, select the revision above, then
apply this patch with `git apply`. Build the XNNPACK target
`//ynnpack/subgraph/test:attention_bench` using that repository's instructions.
The patch changes only that target and source. Keep the patch/source here as
the development record. Do not apply it to the native runner's fetched dependency.

The migration checked patch applicability against the recorded base. It did
not build or run this unrelated benchmark, and performed no host performance tests.
