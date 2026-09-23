# Standalone Tensor API and native Gemma4 build

This build compiles LiteRT's `tensor/` sources directly against XNNPACK. It
includes core tests, Gemma4 model/loader tests, and the
[optimized matched-bundle runner](../examples/gemma4/native/README.md).
The executable does not load the LiteRT/TFLite runtime. The repository's normal
Bazel and integrated CMake builds remain available for other Tensor API users.

Dependencies are fetched into the chosen build directory. No XNNPACK source
checkout or retained static archives from earlier experiments are required.
The initial migration pins XNNPACK `bf3ee43b63070284f85a4a882f298b6ca5273f01`
(the validated native revision), Abseil `20260107.1`, GoogleTest `8eff9e3366`,
FXdiv `b408327ac2` and SentencePiece `0.2.0`. Archive SHA256 values are in
[CMakeLists.txt](CMakeLists.txt) and [examples.cmake](examples.cmake).
XNNPACK also fetches its pinned pthreadpool dependency.

## Linux build and unit tests

From the LiteRT repository root, with CMake 3.28 or later, Ninja and Clang:

```bash
cmake -S tensor/standalone -B .native-tensor-build/host -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build .native-tensor-build/host -j 3
ctest --test-dir .native-tensor-build/host --output-on-failure -j 1
```

CTest uses synthetic tensors and small models, without downloading model
weights. It does not run full-model inference or performance tests. To build
only the Tensor API core, configure with `-DLITERT_TENSOR_BUILD_GEMMA4=OFF`.
To omit only the optimized matched-bundle runner and its tests, use
`-DLITERT_TENSOR_BUILD_NATIVE=OFF`.

The standalone build disables Perfetto scopes in the shared loader and the
TFLite disk-cache-specific graph test with `LITERT_TENSOR_STANDALONE`.
The regular Bazel targets retain both. Native weight-cache, runtime flag,
workspace, cache lifetime and numerical regression tests remain included.

## Android build and unit tests

Assume `ANDROID_HOME` points to the installed Android SDK. The initial
consolidation uses NDK `27.3.13750724`, Android API 24, static libc++, and
`RelWithDebInfo` (`-O2`). Tensor C++ sources use `-ffp-contract=off` to retain
the arithmetic configuration used by the frozen runner.

```bash
cmake -S tensor/standalone -B .native-tensor-build/android -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_HOME/ndk/27.3.13750724/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 \
  -DANDROID_STL=c++_static
cmake --build .native-tensor-build/android -j 3
"$ANDROID_HOME/platform-tools/adb" devices -l
python3 tensor/standalone/run_android_tests.py \
  --build-dir .native-tensor-build/android \
  --serial DEVICE_SERIAL \
  --output-dir .native-tensor-build/android-test-results
```

Choose a connected device that is available for testing. The output directory
must be new. The launcher checks the ABI, system-library dependencies and
Android API, deploys the manifest's binaries to a unique remote directory,
and records logs, exit codes, GoogleTest XML where applicable, and a summary.
Use `--filter 'gemma4_native|safetensor_loader'` for a focused rerun.
Self-check executables with their own `main` are validated by their exit code;
the launcher does not claim they emitted GoogleTest XML.

## Full-model migration checks

Use a [verified matched-bundle export](../examples/gemma4/native/README.md)
and identical fixed token histories. Raw compressed-tensors checkpoints and
the matched LiteRT-LM bundle have different numerical policies; they are
separate validation paths. Deploy the bundle once and keep its provenance.

The helper below compares a newly built Android executable with a frozen
reference executable on the **same phone**. Both executables are local files
that it deploys into a unique temporary directory. The exported model already
resides at `--remote-bundle-dir`. No host model inference runs.

```bash
python3 tensor/standalone/compare_native_android.py \
  --serial DEVICE_SERIAL --affinity-mask CPU_MASK \
  --binary .native-tensor-build/android/bin/gemma4_native_runner \
  --reference-binary /path/to/frozen-native.android \
  --remote-bundle-dir /data/local/tmp/gemma4/matched-bundle \
  --cases-file /path/to/capacity_smoke_8_1.tsv \
  --output-dir .native-tensor-build/smoke-comparison
```

Correctness is the default: compare every float in each 262144-entry logits
vector bitwise, verify token histories and argmax, and check the recorded
INT2/KV/workspace settings. Repeat with the boundary fixture spanning prompt
position 4095 and subsequent decode positions. Use the deeper logit/cache
comparison tools in `native/tools` to investigate differences.

After correctness passes, `--mode performance` selects one warmup and three
measured sessions without logit/memory/trace diagnostics. Use the 128, 1024
and 4096 prompt fixtures, each followed by 64 forced decode inputs. Alternate
`--order reference-first` and `--order candidate-first` between captures, allow
cooldown, and inspect recorded thermal/process state before interpreting a
speed difference. The helper reports all repetitions and medians, not a
statistical significance claim. Timing setup and weight packing is separate
from the reported prefill/decode intervals.

Both paths explicitly enable 60 compact INT2 weights, shared workspace,
128-row reusable prefill, INT8 token-major KV, and attention alignment 32;
capacity defaults to 8448. Affinity masks depend on the device. The migration
used `f0` on TECNO LJ9 and `1e0` on Pixel 8 with four threads. These masks are
examples for those devices, not portable defaults.

## Build changes

Keep the explicit CMake test list aligned with the corresponding Bazel targets.
The optimized executable's Bazel label is
`//tensor/examples/gemma4/native:gemma4_native_runner`. The initial standalone
pin intentionally preserves the previously validated dependency; the normal
LiteRT dependency pin can differ. For a Bazel numerical comparison, pass
`--cxxopt=-ffp-contract=off` to propagate the arithmetic setting to dependency
translation units, and explicitly select the same XNNPACK source revision.
A default Bazel build is not covered by the standalone binary's parity result.
Private XNNPACK struct access in native
runtime audits and memory accounting requires revalidation after upgrades.
Build outputs belong outside `tensor/`, for example `.native-tensor-build/`.

See the [consolidation report](../NATIVE_DEVELOPMENT_MIGRATION.md) for executed
validation, preserved source snapshots, and remaining limitations.
